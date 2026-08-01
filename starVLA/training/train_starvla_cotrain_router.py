# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""Unified router cotraining for WallX.

This trainer uses one prompt for both branches:
- ``<|pred_action|>``: train the VLM router token and the action expert.
- ``<|pred_bbox|><point>[x1, y1, x2, y2]</point>``: train VLM bbox generation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
def _install_wallx_v2_lerobot_compat() -> None:
    from lerobot.datasets import lerobot_dataset as _lerobot_dataset

    if getattr(_lerobot_dataset, "_starvla_wallx_v2_compat", False):
        return
    original_check = _lerobot_dataset.check_version_compatibility

    def check_version_compatibility(repo_id, version_to_check, current_version, enforce_breaking_major=True):
        repo = str(repo_id)
        version = str(version_to_check).lstrip("v")
        current = str(current_version).lstrip("v")
        if repo.startswith("dzb/lerobot_ego_") and version.startswith("2.") and current.startswith("3."):
            return
        return original_check(repo_id, version_to_check, current_version, enforce_breaking_major)

    _lerobot_dataset.check_version_compatibility = check_version_compatibility
    _lerobot_dataset._starvla_wallx_v2_compat = True


from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    normalize_dotlist_args,
)

IGNORE_INDEX = -100

def build_accelerator(cfg) -> Accelerator:
    def cfg_get(section, key, default=None):
        if section is None:
            return default
        if hasattr(section, "get"):
            return section.get(key, default)
        return getattr(section, key, default)

    trainer_cfg = getattr(cfg, "trainer", None)
    cfg_steps = cfg_get(trainer_cfg, "gradient_accumulation_steps", None)
    gradient_accumulation_steps = int(cfg_steps or os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1") or "1")
    ds_cfg = cfg_get(trainer_cfg, "deepspeed", None)
    zero_stage = cfg_get(ds_cfg, "zero_stage", None)
    zero_stage = int(zero_stage) if zero_stage is not None else None
    gradient_clipping = cfg_get(trainer_cfg, "gradient_clipping", None)
    gradient_clipping = float(gradient_clipping) if gradient_clipping is not None else None
    offload_optimizer_device = cfg_get(ds_cfg, "offload_optimizer_device", None)
    offload_param_device = cfg_get(ds_cfg, "offload_param_device", None)
    zero3_save_16bit_model = cfg_get(ds_cfg, "zero3_save_16bit_model", None)
    deepspeed_plugin = DeepSpeedPlugin(
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clipping=gradient_clipping,
        zero_stage=zero_stage,
        offload_optimizer_device=offload_optimizer_device,
        offload_param_device=offload_param_device,
        zero3_save_16bit_model=zero3_save_16bit_model,
    )
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    accelerator.print(accelerator.state)
    return accelerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_bool(cfg: Any, key: str, default: bool = False) -> bool:
    value = _cfg_get(cfg, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _append_csv_value(current: Any, value: str) -> str:
    values = []
    if isinstance(current, str):
        values = [item.strip() for item in current.split(",") if item.strip()]
    elif current:
        values = [str(item).strip() for item in current if str(item).strip()]
    if value not in values:
        values.append(value)
    return ",".join(values)


def _apply_router_framework_overrides(cfg: Any) -> None:
    framework_cfg = _cfg_get(cfg, "framework", None)
    router_framework_cfg = _cfg_get(framework_cfg, "router", None)
    datasets_cfg = _cfg_get(cfg, "datasets", None)
    router_data_cfg = _cfg_get(datasets_cfg, "router_data", None)
    if router_data_cfg is None:
        return

    action_supervision = _cfg_get(router_framework_cfg, "action_supervision", None)
    if action_supervision is not None:
        router_data_cfg.action_supervision = str(action_supervision)
    elif _cfg_get(router_data_cfg, "action_supervision", None) is None:
        router_data_cfg.action_supervision = "flow_matching"

    for key in ("action_route_format", "subtask_start_token", "subtask_end_token"):
        value = _cfg_get(router_framework_cfg, key, None)
        if value is not None:
            setattr(router_data_cfg, key, value)

    action_tokenizer_cfg = _cfg_get(framework_cfg, "action_tokenizer", None)
    if action_tokenizer_cfg is not None:
        router_data_cfg.action_tokenizer = _clone_cfg(action_tokenizer_cfg)


def _validate_go2_training_config(cfg: Any) -> None:
    datasets_cfg = _cfg_get(cfg, "datasets", None)
    router_data_cfg = _cfg_get(datasets_cfg, "router_data", None)
    if str(_cfg_get(router_data_cfg, "dataset_py", "")) != "go2_waypoint_router_dataset":
        return

    framework_cfg = _cfg_get(cfg, "framework", None)
    router_cfg = _cfg_get(framework_cfg, "router", None)
    action_cfg = _cfg_get(framework_cfg, "action_model", None)
    trainer_cfg = _cfg_get(cfg, "trainer", None)

    data_horizon = int(_cfg_get(router_data_cfg, "action_horizon", 0))
    model_horizon = int(_cfg_get(action_cfg, "action_horizon", 0))
    future_window = int(_cfg_get(action_cfg, "future_action_window_size", -1))
    if data_horizon <= 0 or model_horizon != data_horizon or future_window + 1 != data_horizon:
        raise ValueError(
            "Go2 action horizon mismatch: expected "
            "datasets.router_data.action_horizon == framework.action_model.action_horizon == "
            "framework.action_model.future_action_window_size + 1, got "
            f"{data_horizon}, {model_horizon}, {future_window} + 1"
        )

    if int(_cfg_get(action_cfg, "action_dim", 0)) != 10 or int(_cfg_get(action_cfg, "state_dim", 0)) != 10:
        raise ValueError(
            "Go2 action_dim and state_dim must both be 10: three NAV dimensions plus "
            "seven base-frame Cartesian arm/gripper dimensions."
        )

    main_routes = [str(route) for route in _cfg_get(router_data_cfg, "main_routes", [])]
    expected_routes = ["nav", "grasp", "place", "done", "recover"]
    if set(main_routes) != set(expected_routes) or len(main_routes) != len(expected_routes):
        raise ValueError(f"Go2 main_routes must contain exactly {expected_routes}, got {main_routes}")

    include_routes_cfg = _cfg_get(router_data_cfg, "include_routes", expected_routes)
    if isinstance(include_routes_cfg, str):
        include_routes = [route.strip() for route in include_routes_cfg.split(",") if route.strip()]
    else:
        include_routes = [str(route) for route in include_routes_cfg]
    if not include_routes or not set(include_routes).issubset(expected_routes):
        raise ValueError(
            f"Go2 include_routes must be a non-empty subset of {expected_routes}, got {include_routes}"
        )

    route_tokens_cfg = _cfg_get(router_data_cfg, "route_tokens", None)
    route_tokens = {route: str(_cfg_get(route_tokens_cfg, route, "")) for route in expected_routes}
    if any(not token for token in route_tokens.values()) or len(set(route_tokens.values())) != len(route_tokens):
        raise ValueError(f"Go2 route tokens must be present and unique, got {route_tokens}")

    special_cfg = _cfg_get(_cfg_get(framework_cfg, "qwenvl", None), "special_tokens", None)
    if str(_cfg_get(special_cfg, "policy", "")) != "auto_add":
        raise ValueError("Go2 training from the clean Qwen base requires special_tokens.policy=auto_add.")
    special_tokens = {str(token) for token in _cfg_get(special_cfg, "router_tokens", [])}
    required_tokens = set(route_tokens.values()) | {
        str(_cfg_get(router_data_cfg, "subtask_start_token", "")),
        str(_cfg_get(router_data_cfg, "subtask_end_token", "")),
    }
    if "" in required_tokens or not required_tokens.issubset(special_tokens):
        raise ValueError(f"Qwen special tokens are missing Go2 route/subtask tokens: {sorted(required_tokens - special_tokens)}")

    action_supervision = str(_cfg_get(router_cfg, "action_supervision", ""))
    data_action_supervision = str(_cfg_get(router_data_cfg, "action_supervision", ""))
    if action_supervision != "flow_matching" or data_action_supervision != "flow_matching":
        raise ValueError("Go2 training requires flow_matching action supervision in framework and dataset config.")
    if str(_cfg_get(router_data_cfg, "action_route_format", "")) != "route_subtask":
        raise ValueError("Go2 training requires datasets.router_data.action_route_format=route_subtask.")

    bbox_cfg = _cfg_get(router_data_cfg, "bbox", None)
    if any(
        _cfg_bool(bbox_cfg, key, False)
        for key in ("allow_route_prediction", "train_enabled", "evaluation_enabled", "fallback_enabled")
    ):
        raise ValueError("Go2 main training requires bbox route, training, evaluation, and fallback to stay disabled.")

    sft_multi_cfg = _cfg_get(datasets_cfg, "sft_multi", None)
    loss_scale_cfg = _cfg_get(trainer_cfg, "loss_scale", None)
    if not _cfg_bool(sft_multi_cfg, "enable", False) and float(_cfg_get(loss_scale_cfg, "sft_vlm", 0.0)) != 0.0:
        raise ValueError("trainer.loss_scale.sft_vlm must be 0 when datasets.sft_multi.enable=false.")
    vlm_scale = float(_cfg_get(loss_scale_cfg, "vlm", 0.0))
    action_scale = float(_cfg_get(loss_scale_cfg, "action", 0.0))
    if vlm_scale < 0.0 or action_scale < 0.0 or (vlm_scale == 0.0 and action_scale == 0.0):
        raise ValueError("Go2 VLM/action loss scales must be non-negative and at least one must be positive.")

    stage = str(_cfg_get(trainer_cfg, "stage", "joint"))
    if stage not in {"vlm", "action", "joint"}:
        raise ValueError(f"trainer.stage must be vlm, action, or joint, got {stage!r}")
    qwenvl_frozen = _cfg_bool(_cfg_get(framework_cfg, "qwenvl", None), "freeze", False)
    action_frozen = _cfg_bool(action_cfg, "freeze", False)
    action_grad_to_vlm = _cfg_bool(router_cfg, "action_loss_grad_to_vlm", True)
    if stage == "vlm" and not (vlm_scale > 0.0 and action_scale == 0.0 and not qwenvl_frozen and action_frozen):
        raise ValueError("VLM stage requires VLM loss only, trainable Qwen, and a frozen action model.")
    if stage == "action":
        if not (vlm_scale == 0.0 and action_scale > 0.0 and qwenvl_frozen and not action_frozen):
            raise ValueError("Action stage requires action loss only, frozen Qwen, and a trainable action model.")
        if action_grad_to_vlm:
            raise ValueError("Action stage requires framework.router.action_loss_grad_to_vlm=false.")
        if not str(_cfg_get(trainer_cfg, "pretrained_checkpoint", "")).strip():
            raise ValueError("Action stage requires trainer.pretrained_checkpoint from the VLM stage.")
    if stage == "joint" and (vlm_scale <= 0.0 or action_scale <= 0.0 or qwenvl_frozen or action_frozen):
        raise ValueError("Joint stage requires positive VLM/action losses and both modules trainable.")

    if int(_cfg_get(router_data_cfg, "per_device_batch_size", 0)) <= 0:
        raise ValueError("datasets.router_data.per_device_batch_size must be positive.")
    if int(_cfg_get(router_data_cfg, "num_workers", -1)) < 0:
        raise ValueError("datasets.router_data.num_workers must be non-negative.")
    for key in ("rdp_epsilon_m", "yaw_metric_scale_m_per_rad", "max_translation_m", "max_yaw_deg"):
        if float(_cfg_get(router_data_cfg, key, 0.0)) <= 0.0:
            raise ValueError(f"datasets.router_data.{key} must be positive.")

    max_steps = int(_cfg_get(trainer_cfg, "max_train_steps", 0))
    warmup_steps = int(_cfg_get(trainer_cfg, "num_warmup_steps", 0))
    if max_steps <= 0 or not 0 <= warmup_steps < max_steps:
        raise ValueError(f"Invalid Go2 training steps: max_train_steps={max_steps}, num_warmup_steps={warmup_steps}")
    for key in ("save_interval", "eval_interval", "logging_frequency", "gradient_accumulation_steps"):
        if int(_cfg_get(trainer_cfg, key, 0)) <= 0:
            raise ValueError(f"trainer.{key} must be positive for Go2 training.")
    if float(_cfg_get(trainer_cfg, "gradient_clipping", 0.0)) <= 0.0:
        raise ValueError("trainer.gradient_clipping must be positive.")


def _sync_framework_freeze_modules(cfg: Any) -> None:
    framework_cfg = _cfg_get(cfg, "framework", None)
    qwenvl_cfg = _cfg_get(framework_cfg, "qwenvl", None)
    action_model_cfg = _cfg_get(framework_cfg, "action_model", None)

    freeze_modules = _cfg_get(cfg.trainer, "freeze_modules", "")
    if _cfg_bool(qwenvl_cfg, "freeze", False):
        freeze_modules = _append_csv_value(freeze_modules, "qwen_vl_interface")
    if _cfg_bool(action_model_cfg, "freeze", False):
        freeze_modules = _append_csv_value(freeze_modules, "action_model")
    cfg.trainer.freeze_modules = freeze_modules


def _clone_cfg(cfg: Any):
    if isinstance(cfg, AccessTrackedConfig):
        return OmegaConf.create(cfg.to_dict(resolve=True))
    if hasattr(cfg, "deepcopy"):
        return cfg.deepcopy()
    if OmegaConf.is_config(cfg):
        return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    return OmegaConf.create(cfg)


def _dataset_total_episodes(data_cfg: Any) -> int:
    root = str(_cfg_get(data_cfg, "root", ""))
    if not root:
        raise ValueError("data_cfg.root is required when datasets.split.enable is true.")
    if str(_cfg_get(data_cfg, "dataset_py", "")) == "go2_waypoint_router_dataset":
        info_path = Path(root) / "meta" / "info.json"
        return int(json.loads(info_path.read_text())["total_episodes"])
    _install_wallx_v2_lerobot_compat()
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    repo_id = str(_cfg_get(data_cfg, "repo_id", "dzb/lerobot_ego_data"))
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    return int(meta.total_episodes)


def _validate_no_manual_episode_window(data_cfg: Any, data_name: str) -> None:
    num_episodes = _cfg_get(data_cfg, "num_episodes", None)
    episode_start = int(_cfg_get(data_cfg, "episode_start", 0))
    if num_episodes is not None or episode_start != 0:
        raise ValueError(
            f"datasets.split.enable=true conflicts with manual {data_name}.episode_start/num_episodes. "
            "Please unset manual episode window overrides."
        )


def _resolve_router_episode_split(cfg: Any) -> Optional[dict[str, Any]]:
    datasets_cfg = _cfg_get(cfg, "datasets", None)
    split_cfg = _cfg_get(datasets_cfg, "split", None)
    if not split_cfg or not bool(_cfg_get(split_cfg, "enable", False)):
        return None

    mode = str(_cfg_get(split_cfg, "mode", "contiguous"))
    if mode != "contiguous":
        raise ValueError(f"Unsupported datasets.split.mode `{mode}`. Only `contiguous` is supported.")

    train_ratio = float(_cfg_get(split_cfg, "train_ratio", 0.8))
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"datasets.split.train_ratio must be in (0,1), got {train_ratio}")

    router_cfg = datasets_cfg.router_data
    _validate_no_manual_episode_window(router_cfg, "datasets.router_data")

    total_episodes = _dataset_total_episodes(router_cfg)
    train_num_episodes = int(np.floor(total_episodes * train_ratio))
    train_num_episodes = max(1, min(train_num_episodes, total_episodes - 1))
    eval_num_episodes = total_episodes - train_num_episodes
    if eval_num_episodes <= 0:
        raise ValueError(
            f"Invalid split result: total={total_episodes}, train_num={train_num_episodes}, eval_num={eval_num_episodes}"
        )

    train_start = 0
    eval_start = train_num_episodes
    return {
        "enable": True,
        "mode": mode,
        "train_ratio": train_ratio,
        "total_episodes": total_episodes,
        "train_episode_start": train_start,
        "train_num_episodes": train_num_episodes,
        "train_episode_range": [train_start, train_start + train_num_episodes - 1],
        "eval_episode_start": eval_start,
        "eval_num_episodes": eval_num_episodes,
        "eval_episode_range": [eval_start, eval_start + eval_num_episodes - 1],
    }


def _apply_episode_window(data_cfg: Any, episode_start: int, num_episodes: int) -> None:
    data_cfg.episode_start = int(episode_start)
    data_cfg.num_episodes = int(num_episodes)


def _disable_photometric_augmentation(data_cfg: Any) -> None:
    aug_cfg = _cfg_get(data_cfg, "photometric_augmentation", None)
    if aug_cfg is not None:
        aug_cfg.enabled = False


def _is_sft_multi_enabled(cfg: Any) -> bool:
    datasets_cfg = _cfg_get(cfg, "datasets", None)
    sft_multi_cfg = _cfg_get(datasets_cfg, "sft_multi", None)
    return bool(_cfg_get(sft_multi_cfg, "enable", False))


def _iter_enabled_sft_sources(cfg: Any) -> list[tuple[str, Any]]:
    datasets_cfg = _cfg_get(cfg, "datasets", None)
    sft_multi_cfg = _cfg_get(datasets_cfg, "sft_multi", None)
    sources_cfg = _cfg_get(sft_multi_cfg, "sources", None)
    if sources_cfg is None:
        return []

    if not hasattr(sources_cfg, "items"):
        raise ValueError("datasets.sft_multi.sources must be a mapping of source_name -> source_config")

    enabled_sources: list[tuple[str, Any]] = []
    for source_name, source_cfg in sources_cfg.items():
        if bool(_cfg_get(source_cfg, "enabled", True)):
            enabled_sources.append((str(source_name), source_cfg))
    return enabled_sources


def _merge_sft_source_cfg(cfg: Any, source_cfg: Any):
    datasets_cfg = _cfg_get(cfg, "datasets", None)
    base_vlm_cfg = _cfg_get(datasets_cfg, "vlm_data", None)

    if base_vlm_cfg is None:
        base_cfg_dict: dict[str, Any] = {}
    elif OmegaConf.is_config(base_vlm_cfg):
        base_cfg_dict = OmegaConf.to_container(base_vlm_cfg, resolve=True)
    else:
        base_cfg_dict = dict(base_vlm_cfg)

    if OmegaConf.is_config(source_cfg):
        source_cfg_dict = OmegaConf.to_container(source_cfg, resolve=True)
    else:
        source_cfg_dict = dict(source_cfg)

    source_cfg_dict = dict(source_cfg_dict)
    source_cfg_dict.pop("enabled", None)
    source_cfg_dict.pop("weight", None)

    return OmegaConf.merge(
        OmegaConf.create(base_cfg_dict),
        OmegaConf.create(source_cfg_dict),
    )


def _build_single_sft_source_dataloader(cfg: Any, source_cfg: Any) -> DataLoader:
    source_run_cfg = _clone_cfg(cfg)
    source_run_cfg.datasets.vlm_data = _merge_sft_source_cfg(cfg, source_cfg)

    dataset_py = str(_cfg_get(source_run_cfg.datasets.vlm_data, "dataset_py", ""))
    if not dataset_py:
        raise ValueError("SFT source config must provide `dataset_py`.")

    return build_dataloader(cfg=source_run_cfg, dataset_py=dataset_py)


def _build_sft_multi_dataloaders(cfg: Any) -> tuple[dict[str, DataLoader], dict[str, float]]:
    enabled_sources = _iter_enabled_sft_sources(cfg)
    if not enabled_sources:
        raise ValueError("datasets.sft_multi.enable=true, but no enabled source found in datasets.sft_multi.sources")

    train_dataloaders: dict[str, DataLoader] = {}
    source_weights: dict[str, float] = {}
    for source_name, source_cfg in enabled_sources:
        weight = float(_cfg_get(source_cfg, "weight", 1.0))
        if weight < 0:
            raise ValueError(f"datasets.sft_multi.sources.{source_name}.weight must be >= 0, got {weight}")
        if weight == 0:
            continue

        train_dataloaders[source_name] = _build_single_sft_source_dataloader(cfg, source_cfg)
        source_weights[source_name] = weight

    if sum(source_weights.values()) <= 0:
        raise ValueError(
            "Sum of datasets.sft_multi source weights must be > 0. "
            f"Current weights: {source_weights}"
        )

    return train_dataloaders, source_weights


def prepare_data(
    cfg,
    accelerator,
    output_dir,
) -> Tuple[
    DataLoader,
    Optional[DataLoader],
    Optional[dict[str, Any]],
    Optional[dict[str, DataLoader]],
    Optional[dict[str, float]],
]:
    logger.info("Creating unified router dataset: %s", cfg.datasets.router_data.dataset_py)
    _apply_router_framework_overrides(cfg)

    split_summary = _resolve_router_episode_split(cfg)
    if split_summary is None:
        router_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.router_data.dataset_py)
        router_eval_dataloader = None
    else:
        if accelerator.is_main_process:
            logger.info(
                "Episode split enabled: mode=%s ratio=%.4f total=%d train=%s eval=%s",
                split_summary["mode"],
                split_summary["train_ratio"],
                split_summary["total_episodes"],
                split_summary["train_episode_range"],
                split_summary["eval_episode_range"],
            )

        train_cfg = _clone_cfg(cfg)
        eval_cfg = _clone_cfg(cfg)
        _apply_episode_window(
            train_cfg.datasets.router_data,
            split_summary["train_episode_start"],
            split_summary["train_num_episodes"],
        )
        _apply_episode_window(
            eval_cfg.datasets.router_data,
            split_summary["eval_episode_start"],
            split_summary["eval_num_episodes"],
        )
        eval_cfg.datasets.router_data.shuffle = False
        _disable_photometric_augmentation(eval_cfg.datasets.router_data)

        router_train_dataloader = build_dataloader(cfg=train_cfg, dataset_py=train_cfg.datasets.router_data.dataset_py)
        router_eval_dataloader = build_dataloader(cfg=eval_cfg, dataset_py=eval_cfg.datasets.router_data.dataset_py)

    sft_train_dataloaders = None
    sft_source_weights = None
    if _is_sft_multi_enabled(cfg):
        sft_train_dataloaders, sft_source_weights = _build_sft_multi_dataloaders(cfg)
        if accelerator.is_main_process:
            logger.info(
                "SFT multi-source enabled with sources=%s, weights=%s",
                list(sft_train_dataloaders.keys()),
                sft_source_weights,
            )

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return router_train_dataloader, router_eval_dataloader, split_summary, sft_train_dataloaders, sft_source_weights


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer_kwargs = {}
    foreach = _cfg_get(cfg.trainer.optimizer, "foreach", None)
    if foreach is not None:
        optimizer_kwargs["foreach"] = bool(foreach)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        **optimizer_kwargs,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )
    return optimizer, lr_scheduler


class VLARouterTrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        router_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        router_eval_dataloader: Optional[DataLoader] = None,
        split_summary: Optional[dict[str, Any]] = None,
        sft_train_dataloaders: Optional[dict[str, DataLoader]] = None,
        sft_source_weights: Optional[dict[str, float]] = None,
    ):
        self.config = cfg
        self.model = model
        self.router_train_dataloader = router_train_dataloader
        self.router_eval_dataloader = router_eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.split_summary = split_summary
        self.sft_train_dataloaders = sft_train_dataloaders or {}
        self.sft_source_weights = sft_source_weights or {}

        loss_scale_cfg = _cfg_get(self.config.trainer, "loss_scale", None)
        self.loss_scale_vlm = float(_cfg_get(loss_scale_cfg, "vlm", 1.0))
        self.loss_scale_action = float(_cfg_get(loss_scale_cfg, "action", 1.0))
        self.loss_scale_sft_vlm = float(_cfg_get(loss_scale_cfg, "sft_vlm", 1.0))
        framework_cfg = _cfg_get(self.config, "framework", None)
        router_framework_cfg = _cfg_get(framework_cfg, "router", None)
        self.router_action_supervision = str(
            _cfg_get(
                router_framework_cfg,
                "action_supervision",
                _cfg_get(self.config.datasets.router_data, "action_supervision", "flow_matching"),
            )
        )
        if self.router_action_supervision not in {"flow_matching", "fast_token_ce"}:
            raise ValueError(
                "framework.router.action_supervision must be `flow_matching` or `fast_token_ce`, "
                f"got `{self.router_action_supervision}`"
            )
        self.action_loss_grad_to_vlm = bool(
            _cfg_get(router_framework_cfg, "action_loss_grad_to_vlm", True)
        )

        self.is_sft_multi = bool(self.sft_train_dataloaders)
        self.sft_source_names: list[str] = []
        self.sft_source_probs_tensor: Optional[torch.Tensor] = None
        self.sft_source_prob_map: dict[str, float] = {}
        self.sync_sft_source_choice = False
        self.sft_source_generator = torch.Generator(device="cpu")
        self.tb_writer = None

        if self.is_sft_multi:
            self.sft_source_names = list(self.sft_train_dataloaders.keys())
            weights = []
            for source_name in self.sft_source_names:
                weight = float(self.sft_source_weights.get(source_name, 1.0))
                if weight < 0:
                    raise ValueError(f"SFT source weight must be >= 0, got {source_name}={weight}")
                weights.append(weight)

            total_weight = float(sum(weights))
            if total_weight <= 0:
                raise ValueError(f"Sum of SFT source weights must be > 0, got {weights}")

            self.sft_source_probs_tensor = torch.tensor(weights, dtype=torch.float32) / total_weight
            self.sft_source_prob_map = {
                name: float(self.sft_source_probs_tensor[idx].item())
                for idx, name in enumerate(self.sft_source_names)
            }

            datasets_cfg = _cfg_get(self.config, "datasets", None)
            sft_multi_cfg = _cfg_get(datasets_cfg, "sft_multi", None)
            self.sync_sft_source_choice = bool(_cfg_get(sft_multi_cfg, "sync_source_choice_across_ranks", True))

            default_seed = int(_cfg_get(self.config, "seed", 42))
            source_seed = int(_cfg_get(sft_multi_cfg, "source_seed", default_seed))
            self.sft_source_generator.manual_seed(source_seed)

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def _calculate_total_batch_size(self):
        return (
            self.config.datasets.router_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        components = [self.model, self.optimizer, self.router_train_dataloader]
        if self.is_sft_multi:
            components.extend([self.sft_train_dataloaders[name] for name in self.sft_source_names])
        has_eval = self.router_eval_dataloader is not None
        if has_eval:
            components.append(self.router_eval_dataloader)

        prepared = self.setup_distributed_training(self.accelerator, *components)
        if not isinstance(prepared, tuple):
            prepared = (prepared,)

        self.model = prepared[0]
        self.optimizer = prepared[1]
        self.router_train_dataloader = prepared[2]
        idx = 3
        if self.is_sft_multi:
            for source_name in self.sft_source_names:
                self.sft_train_dataloaders[source_name] = prepared[idx]
                idx += 1
        if has_eval:
            self.router_eval_dataloader = prepared[idx]

        self._init_wandb()
        self._init_tensorboard()
        self._init_checkpointing()

    def _init_wandb(self):
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-router-train",
            )

    def _init_tensorboard(self):
        if not self.accelerator.is_main_process:
            return
        enable_tb = bool(_cfg_get(self.config.trainer, "enable_tensorboard", False))
        if not enable_tb:
            return
        if SummaryWriter is None:
            logger.warning("TensorBoard requested but torch.utils.tensorboard is unavailable.")
            return
        tb_dir = Path(self.config.output_dir) / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info("TensorBoard logging enabled at %s", tb_dir)

    def _init_checkpointing(self):
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _load_checkpoint(self, checkpoint_path):
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps({"steps": self.completed_steps}) + "\n")

            if isinstance(self.config, AccessTrackedConfig):
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            self.accelerator.print(f"Checkpoint saved at {checkpoint_path}")

        self.accelerator.wait_for_everyone()

    def _finalize_training(self):
        if not _cfg_bool(self.config.trainer, "save_final_model", True):
            if self.accelerator.is_main_process:
                logger.info("Skipping final model save because trainer.save_final_model=false")
            self.accelerator.wait_for_everyone()
            return
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            final_dir = Path(self.config.output_dir) / "final_model"
            final_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, final_dir / "pytorch_model.pt")
            if isinstance(self.config, AccessTrackedConfig):
                self.config.save_accessed_config(Path(self.config.output_dir) / "config.yaml", use_original_values=False)
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        base_group_lr = None
        for idx, group in enumerate(self.optimizer.param_groups):
            group_name = str(group.get("name", f"group_{idx}"))
            group_lr = float(group["lr"])
            metrics[f"learning_rate/{group_name}"] = group_lr
            if group_name == "base":
                base_group_lr = group_lr
        if base_group_lr is None and self.optimizer.param_groups:
            base_group_lr = float(self.optimizer.param_groups[0]["lr"])
        metrics["learning_rate"] = base_group_lr
        metrics["epoch"] = round(self.completed_steps / len(self.router_train_dataloader), 2)

        wandb.log(metrics, step=self.completed_steps)
        if self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(str(key), float(value), self.completed_steps)
            self.tb_writer.flush()
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            logger.info(f"Step {self.completed_steps}, Metrics: {metrics}")

    def _create_data_iterators(self):
        self.router_iter = iter(self.router_train_dataloader)
        self.router_epoch_count = 0
        if self.is_sft_multi:
            self.sft_iters = {}
            self.sft_epoch_count = {}
            for source_name in self.sft_source_names:
                self.sft_iters[source_name] = iter(self.sft_train_dataloaders[source_name])
                self.sft_epoch_count[source_name] = 0
        if self.router_eval_dataloader is not None:
            self.router_eval_iter = iter(self.router_eval_dataloader)
            self.router_eval_epoch_count = 0

    def _next_from_loader(self, iter_name: str, loader, epoch_counter_name: str):
        data_iter = getattr(self, iter_name)
        try:
            return next(data_iter)
        except StopIteration:
            epoch_count = getattr(self, epoch_counter_name, 0)
            data_iter, epoch_count = TrainerUtils._reset_dataloader(loader, epoch_count)
            setattr(self, iter_name, data_iter)
            setattr(self, epoch_counter_name, epoch_count)
            return next(data_iter)

    def _get_next_batch(self):
        return self._next_from_loader("router_iter", self.router_train_dataloader, "router_epoch_count")

    def _draw_sft_source_index(self) -> int:
        if not self.is_sft_multi:
            return 0
        if self.sft_source_probs_tensor is None:
            raise RuntimeError("SFT source probs are not initialized.")

        sampled_idx = 0
        if not (dist.is_initialized() and self.sync_sft_source_choice):
            sampled_idx = int(
                torch.multinomial(
                    self.sft_source_probs_tensor,
                    num_samples=1,
                    replacement=True,
                    generator=self.sft_source_generator,
                ).item()
            )
            return sampled_idx

        if dist.get_rank() == 0:
            sampled_idx = int(
                torch.multinomial(
                    self.sft_source_probs_tensor,
                    num_samples=1,
                    replacement=True,
                    generator=self.sft_source_generator,
                ).item()
            )

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        idx_tensor = torch.tensor([sampled_idx], dtype=torch.long, device=device)
        dist.broadcast(idx_tensor, src=0)
        return int(idx_tensor.item())

    def _get_next_sft_batch_from_source(self, source_name: str):
        if source_name not in self.sft_iters:
            raise KeyError(f"Unknown SFT source: {source_name}. Available={list(self.sft_iters.keys())}")
        try:
            return next(self.sft_iters[source_name])
        except StopIteration:
            self.sft_iters[source_name], self.sft_epoch_count[source_name] = TrainerUtils._reset_dataloader(
                self.sft_train_dataloaders[source_name],
                self.sft_epoch_count[source_name],
            )
            return next(self.sft_iters[source_name])

    def _get_next_sft_train_batch(self) -> tuple[Optional[str], Any]:
        if not self.is_sft_multi:
            return None, None
        source_idx = self._draw_sft_source_index()
        source_name = self.sft_source_names[source_idx]
        return source_name, self._get_next_sft_batch_from_source(source_name)

    def _get_next_eval_batch(self):
        if self.router_eval_dataloader is None:
            return self._get_next_batch()
        return self._next_from_loader("router_eval_iter", self.router_eval_dataloader, "router_eval_epoch_count")

    def _unwrap_model(self):
        if hasattr(self.model, "action_loss_from_hidden_states"):
            return self.model
        return self.accelerator.unwrap_model(self.model)

    def _get_qwen_vl_interface(self):
        if hasattr(self.model, "qwen_vl_interface"):
            return self.model.qwen_vl_interface
        return self.accelerator.unwrap_model(self.model).qwen_vl_interface

    def _prepare_router_batch(self, batch_router):
        if not isinstance(batch_router, list):
            return batch_router

        images = [sample["image"] for sample in batch_router]
        instructions = [sample["lang"] for sample in batch_router]
        solutions = [sample["solution"] for sample in batch_router]
        qwen_vl_interface = self._get_qwen_vl_interface()
        return qwen_vl_interface.build_qwenvl_inputs(
            images=images,
            instructions=instructions,
            solutions=solutions,
        )

    def _prepare_sft_batch(self, batch_sft):
        if not isinstance(batch_sft, list):
            return batch_sft

        images = [sample["image"] for sample in batch_sft]
        instructions = [sample["lang"] for sample in batch_sft]
        solutions = [sample.get("solution") for sample in batch_sft]
        if any(solution is None for solution in solutions):
            solutions = None

        qwen_vl_interface = self._get_qwen_vl_interface()
        return qwen_vl_interface.build_qwenvl_inputs(
            images=images,
            instructions=instructions,
            solutions=solutions,
        )

    def _configured_route_tokens(self, *, include_bbox: bool = False) -> dict[str, str]:
        router_cfg = self.config.datasets.router_data
        configured = _cfg_get(router_cfg, "route_tokens", None)
        if configured:
            main_routes = [str(route) for route in _cfg_get(router_cfg, "main_routes", configured.keys())]
            tokens = {route: str(_cfg_get(configured, route)) for route in main_routes}
            if include_bbox:
                bbox_token = _cfg_get(router_cfg, "pred_bbox_token", None)
                if bbox_token:
                    tokens["bbox"] = str(bbox_token)
            return tokens
        return {
            "action": str(_cfg_get(router_cfg, "pred_action_token", "<|pred_action|>")),
            "bbox": str(_cfg_get(router_cfg, "pred_bbox_token", "<|pred_bbox|>")),
        }

    def _bbox_flag(self, key: str, default: bool = False) -> bool:
        bbox_cfg = _cfg_get(self.config.datasets.router_data, "bbox", None)
        return bool(_cfg_get(bbox_cfg, key, default))

    def _filter_bbox_samples(self, batch_router, *, training: bool):
        enabled = self._bbox_flag("train_enabled" if training else "evaluation_enabled", False)
        if enabled or not isinstance(batch_router, list):
            return batch_router
        return [sample for sample in batch_router if str(sample.get("route", "")) != "bbox"]

    def _route_counts(self, batch_router):
        counts = {route: 0 for route in self._configured_route_tokens(include_bbox=False)}
        counts["unknown"] = 0
        for sample in batch_router:
            route = str(sample.get("route", "unknown"))
            if route not in counts:
                route = "unknown"
            counts[route] += 1
        return counts

    @staticmethod
    def _action_indices(batch_router):
        return [
            idx
            for idx, sample in enumerate(batch_router)
            if "action" in sample
        ]

    @staticmethod
    def _any_rank_has_action(local_has_action: bool, device: torch.device) -> bool:
        if not dist.is_initialized():
            return bool(local_has_action)
        flag = torch.tensor([1 if local_has_action else 0], device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _build_dummy_action_example(self) -> dict[str, Any]:
        router_cfg = self.config.datasets.router_data
        action_cfg = self.config.framework.action_model
        action_horizon = int(_cfg_get(router_cfg, "action_horizon", 16))
        action_dim = int(_cfg_get(action_cfg, "action_dim", 6))
        state_dim = int(_cfg_get(action_cfg, "state_dim", 6))

        example: dict[str, Any] = {
            "action": np.zeros((action_horizon, action_dim), dtype=np.float16),
            "action_mask": np.ones((action_horizon,), dtype=np.float16),
            "action_dim_mask": np.ones((action_dim,), dtype=np.float16),
        }
        if bool(_cfg_get(router_cfg, "include_state", False)):
            example["state"] = np.zeros((1, state_dim), dtype=np.float16)
        return example

    def _compute_router_action_loss(
        self,
        qwen_hidden_states,
        batch_router,
        action_indices,
        *,
        allow_dummy: bool = True,
    ) -> tuple[torch.Tensor, bool, bool]:
        hidden_device = qwen_hidden_states[-1].device
        local_has_action = bool(action_indices)
        global_has_action = self._any_rank_has_action(local_has_action, hidden_device)
        zero = qwen_hidden_states[-1].new_zeros(())

        if not global_has_action:
            return zero, local_has_action, global_has_action

        train_model = self._unwrap_model()
        if local_has_action:
            action_index_tensor = torch.tensor(
                action_indices,
                device=hidden_device,
                dtype=torch.long,
            )
            action_examples = [batch_router[idx] for idx in action_indices]
            if os.environ.get("STARVLA_DEBUG_ROUTER_SHAPES", "0") == "1":
                rank = dist.get_rank() if dist.is_initialized() else 0
                route_counts = self._route_counts(batch_router)
                action_shapes = [np.asarray(example["action"]).shape for example in action_examples]
                print(
                    f"[rank{rank}][router_action_loss] "
                    f"batch_size={len(batch_router)} "
                    f"route_counts={route_counts} "
                    f"action_indices={action_indices} "
                    f"hidden_last_shape={tuple(qwen_hidden_states[-1].shape)} "
                    f"action_shapes={action_shapes}",
                    flush=True,
                )
            action_loss = train_model.action_loss_from_hidden_states(
                qwen_hidden_states,
                action_examples,
                indices=action_index_tensor,
                detach_vlm_hidden_states=not self.action_loss_grad_to_vlm,
            )
            return action_loss, local_has_action, global_has_action

        if not allow_dummy:
            return zero, local_has_action, global_has_action

        dummy_index_tensor = torch.tensor([0], device=hidden_device, dtype=torch.long)
        dummy_examples = [self._build_dummy_action_example()]
        action_loss = train_model.action_loss_from_hidden_states(
            qwen_hidden_states,
            dummy_examples,
            indices=dummy_index_tensor,
            detach_vlm_hidden_states=not self.action_loss_grad_to_vlm,
        )
        return action_loss * 0.0, local_has_action, global_has_action

    def _route_token_ids(self):
        tokenizer = self._get_qwen_vl_interface().processor.tokenizer
        result = {}
        include_bbox = self._bbox_flag("evaluation_enabled", False)
        for route, token in self._configured_route_tokens(include_bbox=include_bbox).items():
            token_ids = tokenizer(token, add_special_tokens=False).input_ids
            if len(token_ids) != 1:
                return None
            result[int(token_ids[0])] = route
        return result

    def _compute_route_token_metrics(self, qwen_output, batch_inputs):
        token_id_to_route = self._route_token_ids()
        if token_id_to_route is None or getattr(qwen_output, "logits", None) is None:
            return {}

        labels = batch_inputs.get("labels")
        if labels is None:
            return {}

        logits = qwen_output.logits.detach()
        route_total = 0
        route_correct = 0
        route_names = list(self._configured_route_tokens(include_bbox=self._bbox_flag("evaluation_enabled", False)))
        pred_counts = {route: 0 for route in route_names}
        pred_counts["other"] = 0

        for i in range(labels.shape[0]):
            label_positions = torch.nonzero(labels[i] != IGNORE_INDEX, as_tuple=False).reshape(-1)
            if label_positions.numel() == 0:
                continue
            pos = int(label_positions[0].item())
            if pos <= 0:
                continue
            target_id = int(labels[i, pos].item())
            if target_id not in token_id_to_route:
                continue
            pred_id = int(torch.argmax(logits[i, pos - 1]).item())
            pred_route = token_id_to_route.get(pred_id, "other")
            pred_counts[pred_route] = pred_counts.get(pred_route, 0) + 1
            route_total += 1
            if pred_id == target_id:
                route_correct += 1

        if route_total == 0:
            return {}
        metrics = {
            "router_token_accuracy": route_correct / route_total,
            "router_token_eval_count": route_total,
            "router_pred_other_count": pred_counts.get("other", 0),
        }
        for route in route_names:
            metrics[f"router_pred_{route}_count"] = pred_counts.get(route, 0)
        return metrics

    def _compute_router_ce_breakdown_metrics(self, qwen_output, batch_inputs, batch_router):
        if getattr(qwen_output, "logits", None) is None:
            return {}
        labels = batch_inputs.get("labels")
        if labels is None or not isinstance(batch_router, list):
            return {}

        router_cfg = self.config.datasets.router_data
        action_route_format = str(_cfg_get(router_cfg, "action_route_format", "token")).strip()
        subtask_start_token = str(_cfg_get(router_cfg, "subtask_start_token", "<|subtask|>"))
        subtask_end_token = str(_cfg_get(router_cfg, "subtask_end_token", "<|end_subtask|>"))
        tokenizer = self._get_qwen_vl_interface().processor.tokenizer

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}

        def add_values(name: str, values: torch.Tensor) -> None:
            if values.numel() == 0:
                return
            values = values.detach().float()
            sums[name] = sums.get(name, 0.0) + float(values.sum().item())
            counts[name] = counts.get(name, 0) + int(values.numel())

        def token_count(text: str) -> int:
            return len(tokenizer(str(text), add_special_tokens=False).input_ids)

        logits = qwen_output.logits.detach()
        with torch.no_grad():
            for i, sample in enumerate(batch_router):
                if i >= labels.shape[0]:
                    break
                label_positions = torch.nonzero(labels[i] != IGNORE_INDEX, as_tuple=False).reshape(-1)
                label_positions = label_positions[label_positions > 0]
                if label_positions.numel() == 0:
                    continue

                targets = labels[i, label_positions].long()
                token_losses = F.cross_entropy(
                    logits[i, label_positions - 1, :].float(),
                    targets,
                    reduction="none",
                )
                token_correct = (torch.argmax(logits[i, label_positions - 1, :], dim=-1) == targets).float()
                add_values("router_vlm_token_ce", token_losses)
                add_values("router_vlm_token_accuracy", token_correct)
                add_values("router_route_token_ce", token_losses[:1])

                route = str(sample.get("route", "unknown"))
                if route == "bbox":
                    add_values("router_bbox_text_ce", token_losses[1:])
                    continue

                route_tokens = self._configured_route_tokens(include_bbox=False)
                if route not in route_tokens:
                    continue

                route_prefix_len = 1
                if action_route_format == "route_subtask":
                    subtask_text = str(sample.get("subtask_text", "")).strip()
                    if subtask_text:
                        route_prefix = (
                            f"{route_tokens[route]}"
                            f"{subtask_start_token}"
                            f"{subtask_text}"
                            f"{subtask_end_token}"
                        )
                        route_prefix_len = max(1, token_count(route_prefix))
                    subtask_end = min(route_prefix_len, int(token_losses.numel()))
                    add_values("router_action_subtask_ce", token_losses[1:subtask_end])
                    add_values("router_action_subtask_token_accuracy", token_correct[1:subtask_end])
                    add_values("router_action_route_prefix_ce", token_losses[:subtask_end])

                    if self.router_action_supervision == "fast_token_ce":
                        add_values("router_fast_action_token_ce", token_losses[subtask_end:])
                    else:
                        add_values("router_action_extra_text_ce", token_losses[subtask_end:])
                elif self.router_action_supervision == "fast_token_ce":
                    add_values("router_fast_action_token_ce", token_losses[1:])

        metrics: dict[str, float | int] = {}
        for name, total in sums.items():
            count = counts[name]
            metrics[name] = total / max(count, 1)
            metrics[f"{name}_count"] = count
        return metrics

    def _append_batch_route_metrics(self, log_dict, batch_router):
        counts = self._route_counts(batch_router)
        total = max(sum(counts.values()), 1)
        for route, count in counts.items():
            log_dict[f"batch_route_{route}_count"] = count
            log_dict[f"batch_route_{route}_fraction"] = count / total

    def _append_action_dim_loss_metrics(self, log_dict, prefix: str = "action_dim_loss"):
        action_model = getattr(self._unwrap_model(), "action_model", None)
        per_dim_loss = getattr(action_model, "latest_action_dim_loss", None)
        if per_dim_loss is None:
            return
        per_dim_loss = per_dim_loss.detach().float().cpu().view(-1)
        for dim_idx, value in enumerate(per_dim_loss.tolist()):
            log_dict[f"{prefix}/dim_{dim_idx}"] = float(value)

    def _append_rtc_delay_metrics(self, log_dict, prefix: str = "rtc_delay"):
        action_model = getattr(self._unwrap_model(), "action_model", None)
        delay = getattr(action_model, "latest_rtc_delay", None)
        if delay is None:
            return
        delay = delay.detach().float().cpu().view(-1)
        if delay.numel() == 0:
            return
        log_dict[f"{prefix}_mean"] = float(delay.mean())
        log_dict[f"{prefix}_min"] = float(delay.min())
        log_dict[f"{prefix}_max"] = float(delay.max())

    def _train_step(self, batch_router, batch_sft=None, sft_source_name: Optional[str] = None):
        log_dict = {}
        batch_router = self._filter_bbox_samples(batch_router, training=True)
        if not batch_router:
            raise RuntimeError("Router batch is empty after excluding disabled bbox samples.")
        action_indices = self._action_indices(batch_router)

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            batch_inputs = self._prepare_router_batch(batch_router)
            qwen_vl_interface = self._get_qwen_vl_interface()
            use_flow_action_loss = (
                self.router_action_supervision == "flow_matching" and self.loss_scale_action > 0.0
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                qwen_output = qwen_vl_interface(
                    **batch_inputs,
                    output_attentions=False,
                    output_hidden_states=use_flow_action_loss,
                    return_dict=True,
                )
                if getattr(qwen_output, "loss", None) is None:
                    raise RuntimeError("Router VLM forward did not return loss. Ensure `solution` is present.")
                vlm_loss_raw = qwen_output.loss
                vlm_loss = vlm_loss_raw * self.loss_scale_vlm

            if use_flow_action_loss:
                action_loss_raw, _, _ = self._compute_router_action_loss(
                    qwen_output.hidden_states,
                    batch_router,
                    action_indices,
                )
            else:
                action_loss_raw = vlm_loss_raw.new_zeros(())
            action_loss = action_loss_raw * self.loss_scale_action
            router_total_loss = vlm_loss + action_loss
            route_metrics = self._compute_route_token_metrics(qwen_output, batch_inputs)
            ce_breakdown_metrics = self._compute_router_ce_breakdown_metrics(qwen_output, batch_inputs, batch_router)
            router_total_loss_value = router_total_loss.detach().float().item()
            vlm_loss_raw_value = vlm_loss_raw.detach().float().item()
            vlm_loss_value = vlm_loss.detach().float().item()
            action_loss_raw_value = action_loss_raw.detach().float().item()
            action_loss_value = action_loss.detach().float().item()

            skip_no_grad_batches = bool(_cfg_get(self.config.trainer, "skip_no_grad_batches", False))
            if batch_sft is None and skip_no_grad_batches and not router_total_loss.requires_grad:
                log_dict.update(
                    {
                        "loss": router_total_loss_value,
                        "vlm_loss_raw": vlm_loss_raw_value,
                        "vlm_loss": vlm_loss_value,
                        "action_dit_loss_raw": action_loss_raw_value,
                        "action_dit_loss": action_loss_value,
                        "action_batch_size": len(action_indices),
                        "sft_vlm_loss_raw": 0.0,
                        "sft_vlm_loss": 0.0,
                        "skipped_no_grad_batch": 1,
                        "_skip_optimizer_step": True,
                    }
                )
                log_dict.update(route_metrics)
                log_dict.update(ce_breakdown_metrics)
                self._append_batch_route_metrics(log_dict, batch_router)
                if len(action_indices) > 0:
                    self._append_action_dim_loss_metrics(log_dict)
                    self._append_rtc_delay_metrics(log_dict)
                del qwen_output, batch_inputs, router_total_loss, vlm_loss_raw, vlm_loss, action_loss_raw, action_loss
                return log_dict

            self.accelerator.backward(router_total_loss)
            del qwen_output, batch_inputs, router_total_loss, vlm_loss_raw, vlm_loss, action_loss_raw, action_loss

            sft_vlm_loss_raw_value = 0.0
            sft_vlm_loss_value = 0.0
            if batch_sft is not None:
                batch_sft_inputs = self._prepare_sft_batch(batch_sft)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    sft_output = qwen_vl_interface(
                        **batch_sft_inputs,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    if getattr(sft_output, "loss", None) is None:
                        raise RuntimeError("SFT VLM forward did not return loss. Ensure labels are present in SFT batch.")
                    sft_vlm_loss_raw = sft_output.loss
                    sft_vlm_loss = sft_vlm_loss_raw * self.loss_scale_sft_vlm

                sft_vlm_loss_raw_value = sft_vlm_loss_raw.detach().float().item()
                sft_vlm_loss_value = sft_vlm_loss.detach().float().item()
                self.accelerator.backward(sft_vlm_loss)
                del batch_sft_inputs, sft_output, sft_vlm_loss_raw, sft_vlm_loss

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            self.lr_scheduler.step()

            log_dict.update(
                {
                    "loss": router_total_loss_value + sft_vlm_loss_value,
                    "vlm_loss_raw": vlm_loss_raw_value,
                    "vlm_loss": vlm_loss_value,
                    "action_dit_loss_raw": action_loss_raw_value,
                    "action_dit_loss": action_loss_value,
                    "action_batch_size": len(action_indices),
                    "sft_vlm_loss_raw": sft_vlm_loss_raw_value,
                    "sft_vlm_loss": sft_vlm_loss_value,
                }
            )
            if sft_source_name is not None:
                log_dict[f"sft_vlm_loss/source={sft_source_name}"] = sft_vlm_loss_value
                log_dict[f"sft_vlm_loss_raw/source={sft_source_name}"] = sft_vlm_loss_raw_value
            log_dict.update(route_metrics)
            log_dict.update(ce_breakdown_metrics)
            self._append_batch_route_metrics(log_dict, batch_router)
            if len(action_indices) > 0:
                self._append_action_dim_loss_metrics(log_dict)
                self._append_rtc_delay_metrics(log_dict)

        return log_dict

    def eval_router_model(self, step_metrics: dict = None):
        if step_metrics is None:
            step_metrics = {}

        batch_router = self._get_next_eval_batch()
        batch_router = self._filter_bbox_samples(batch_router, training=False)
        if not batch_router:
            return step_metrics
        action_indices = self._action_indices(batch_router)
        batch_inputs = self._prepare_router_batch(batch_router)
        qwen_vl_interface = self._get_qwen_vl_interface()
        use_flow_action_loss = (
            self.router_action_supervision == "flow_matching" and self.loss_scale_action > 0.0
        )

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                qwen_output = qwen_vl_interface(
                    **batch_inputs,
                    output_attentions=False,
                    output_hidden_states=use_flow_action_loss,
                    return_dict=True,
                )

            action_loss_raw = None
            if use_flow_action_loss:
                action_loss_raw_sync, local_has_action, global_has_action = self._compute_router_action_loss(
                    qwen_output.hidden_states,
                    batch_router,
                    action_indices,
                )
                if global_has_action and local_has_action:
                    action_loss_raw = action_loss_raw_sync.detach().float()

            if self.accelerator.is_main_process:
                if getattr(qwen_output, "loss", None) is not None:
                    vlm_loss_raw = qwen_output.loss.detach().float()
                    step_metrics["vlm_loss_eval_raw"] = vlm_loss_raw.item()
                    step_metrics["vlm_loss_eval"] = (vlm_loss_raw * self.loss_scale_vlm).item()

                if action_loss_raw is not None:
                    step_metrics["action_dit_loss_eval_raw"] = action_loss_raw.item()
                    step_metrics["action_dit_loss_eval"] = (action_loss_raw * self.loss_scale_action).item()
                    self._append_action_dim_loss_metrics(step_metrics, prefix="action_dim_loss_eval")
                    self._append_rtc_delay_metrics(step_metrics, prefix="rtc_delay_eval")

                route_metrics = self._compute_route_token_metrics(qwen_output, batch_inputs)
                ce_metrics = self._compute_router_ce_breakdown_metrics(qwen_output, batch_inputs, batch_router)
                step_metrics.update({f"{key}_eval": value for key, value in route_metrics.items()})
                step_metrics.update({f"{key}_eval": value for key, value in ce_metrics.items()})
                eval_batch_metrics: dict[str, float] = {}
                self._append_batch_route_metrics(eval_batch_metrics, batch_router)
                step_metrics.update({f"{key}_eval": value for key, value in eval_batch_metrics.items()})

        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        if self.accelerator.is_main_process:
            router_cfg = self.config.datasets.router_data
            logger.info("***** Router Training Configuration *****")
            logger.info(f"  Training stage = {_cfg_get(self.config.trainer, 'stage', 'joint')}")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device router batch size = {router_cfg.per_device_batch_size}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            logger.info(f"  Router action supervision = {self.router_action_supervision}")
            logger.info(f"  Action loss grad to VLM = {self.action_loss_grad_to_vlm}")
            logger.info(
                "  Loss scales: router_vlm=%.4f, action=%.4f, sft_vlm=%.4f",
                self.loss_scale_vlm,
                self.loss_scale_action,
                self.loss_scale_sft_vlm,
            )
            if self.is_sft_multi:
                logger.info(f"  SFT source probs = {self.sft_source_prob_map}")
            logger.info("  Route tokens: %s", self._configured_route_tokens(include_bbox=False))
            logger.info("  Included routes = %s", _cfg_get(router_cfg, "include_routes", "all"))
            logger.info(
                "  BBox enabled: train=%s evaluation=%s",
                self._bbox_flag("train_enabled", False),
                self._bbox_flag("evaluation_enabled", False),
            )
            if self.split_summary:
                logger.info(
                    "  Episode split = mode=%s ratio=%.4f total=%d train=%s eval=%s",
                    self.split_summary.get("mode"),
                    float(self.split_summary.get("train_ratio", 0.0)),
                    int(self.split_summary.get("total_episodes", 0)),
                    self.split_summary.get("train_episode_range"),
                    self.split_summary.get("eval_episode_range"),
                )

    def train(self):
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_router = self._get_next_batch()
            sft_source_name, batch_sft = self._get_next_sft_train_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_router, batch_sft=batch_sft, sft_source_name=sft_source_name)
            t_end_model = time.perf_counter()

            skipped_optimizer_step = bool(step_metrics.pop("_skip_optimizer_step", False))

            if self.accelerator.sync_gradients and not skipped_optimizer_step:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                postfix = {
                    "data_times": f"{t_end_data - t_start_data:.3f}",
                    "model_times": f"{t_end_model - t_start_model:.3f}",
                }
                if sft_source_name is not None:
                    postfix["sft_source"] = sft_source_name
                progress_bar.set_postfix(postfix)

            if not skipped_optimizer_step and self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_router_model(step_metrics)

            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()
                if dist.is_initialized():
                    dist.barrier()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()
        if self.tb_writer is not None:
            self.tb_writer.close()


def main(cfg) -> None:
    cfg = wrap_config(cfg)
    _apply_router_framework_overrides(cfg)
    _validate_go2_training_config(cfg)
    _sync_framework_freeze_modules(cfg)
    accelerator = build_accelerator(cfg)
    logger.info("VLA Router Training :: Warming Up")
    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    (
        router_train_dataloader,
        router_eval_dataloader,
        split_summary,
        sft_train_dataloaders,
        sft_source_weights,
    ) = prepare_data(
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLARouterTrainer(
        cfg=cfg,
        model=vla,
        router_train_dataloader=router_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        router_eval_dataloader=router_eval_dataloader,
        split_summary=split_summary,
        sft_train_dataloaders=sft_train_dataloaders,
        sft_source_weights=sft_source_weights,
    )
    trainer.prepare_training()
    trainer.train()

    logger.info("Router training finished.")
    if dist.is_initialized():
        try:
            dist.barrier()
            dist.destroy_process_group()
        except RuntimeError as exc:
            # ZeRO-3 may leave CUDA memory fully occupied after the final
            # checkpoint gather. A late NCCL teardown OOM must not invalidate
            # a training run whose synchronized save has already completed.
            logger.warning("Distributed cleanup failed after successful training: %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
