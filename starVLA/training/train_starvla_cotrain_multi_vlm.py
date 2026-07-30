# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).


CUDA_VISIBLE_DEVICES=0,1,2,3 \
/beijing-c/workspace/hxj/miniconda3/envs/starvla/bin/accelerate launch \
--config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
--num_processes 4 \
--main_process_port 29523 \
starVLA/training/train_starvla_cotrain.py \
--config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi.yaml \
--datasets.vla_data.data_mix wallx_dzb

cd /beijing-c/wallx_workspace/starVLA

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/beijing-c/workspace/hxj/miniconda3/envs/starvla/bin/accelerate launch \
--config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
--num_processes 4 \
--main_process_port 29523 \
starVLA/training/train_starvla_cotrain.py \
--config_yaml starVLA/config/training/starvla_cotrain_wallx_qwenpi.yaml \
--datasets.vla_data.data_mix wallx_dzb \
--datasets.vla_data.episode_start 0 \
--datasets.vla_data.num_episodes 0 \
--datasets.vlm_data.episode_start 0 \
--datasets.vlm_data.num_episodes 0 \
--trainer.loss_scale.vlm 0.0



--trainer.pretrained_checkpoint /path/to/checkpoints/steps_5000_pytorch_model.pt

"""

# Standard Library
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
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


def _clone_cfg(cfg: Any):
    if hasattr(cfg, "deepcopy"):
        return cfg.deepcopy()
    if OmegaConf.is_config(cfg):
        return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    return OmegaConf.create(cfg)


def _dataset_total_episodes(data_cfg: Any) -> int:
    repo_id = str(_cfg_get(data_cfg, "repo_id", "dzb/lerobot_ego_data"))
    root = str(_cfg_get(data_cfg, "root", ""))
    if not root:
        raise ValueError("data_cfg.root is required when datasets.split.enable is true.")
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


def _resolve_wallx_episode_split(cfg: Any) -> Optional[dict[str, Any]]:
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

    vla_data_cfg = datasets_cfg.vla_data
    vlm_data_cfg = datasets_cfg.vlm_data

    vlm_multi_cfg = _cfg_get(datasets_cfg, "vlm_multi", None)
    is_vlm_multi = bool(_cfg_get(vlm_multi_cfg, "enable", False))

    _validate_no_manual_episode_window(vla_data_cfg, "datasets.vla_data")

    # In multi-VLM mode we only split VLA contiguous episodes. VLM split is applied
    # selectively to WallX VLM source(s) inside prepare_data.
    if not is_vlm_multi:
        _validate_no_manual_episode_window(vlm_data_cfg, "datasets.vlm_data")

    total_vla = _dataset_total_episodes(vla_data_cfg)
    if not is_vlm_multi:
        total_vlm = _dataset_total_episodes(vlm_data_cfg)
        if total_vla != total_vlm:
            raise ValueError(
                "VLA/VLM datasets have different total episodes, cannot apply unified ratio split: "
                f"vla={total_vla}, vlm={total_vlm}"
            )

    total_episodes = total_vla
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


def _is_vlm_multi_enabled(cfg: Any) -> bool:
    datasets_cfg = _cfg_get(cfg, "datasets", None)
    vlm_multi_cfg = _cfg_get(datasets_cfg, "vlm_multi", None)
    return bool(_cfg_get(vlm_multi_cfg, "enable", False))


def _iter_enabled_vlm_sources(cfg: Any) -> list[tuple[str, Any]]:
    datasets_cfg = _cfg_get(cfg, "datasets", None)
    vlm_multi_cfg = _cfg_get(datasets_cfg, "vlm_multi", None)
    sources_cfg = _cfg_get(vlm_multi_cfg, "sources", None)
    if sources_cfg is None:
        return []

    if not hasattr(sources_cfg, "items"):
        raise ValueError("datasets.vlm_multi.sources must be a mapping of source_name -> source_config")

    enabled_sources: list[tuple[str, Any]] = []
    for source_name, source_cfg in sources_cfg.items():
        if bool(_cfg_get(source_cfg, "enabled", True)):
            enabled_sources.append((str(source_name), source_cfg))
    return enabled_sources


def _merge_vlm_source_cfg(base_vlm_cfg: Any, source_cfg: Any):
    if OmegaConf.is_config(base_vlm_cfg):
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


def _build_single_vlm_source_dataloader(
    cfg: Any,
    source_cfg: Any,
    split_summary: Optional[dict[str, Any]],
    split_part: str,
) -> DataLoader:
    source_run_cfg = _clone_cfg(cfg)
    merged_vlm_cfg = _merge_vlm_source_cfg(cfg.datasets.vlm_data, source_cfg)
    source_run_cfg.datasets.vlm_data = merged_vlm_cfg

    dataset_py = str(_cfg_get(source_run_cfg.datasets.vlm_data, "dataset_py", ""))
    if not dataset_py:
        raise ValueError("VLM source config must provide `dataset_py` (or inherit one from datasets.vlm_data)")

    # Split logic only applies to WallX LeRobot VLM dataset.
    if split_summary is not None and dataset_py == "wallx_vlm_dataset":
        if split_part == "train":
            _apply_episode_window(
                source_run_cfg.datasets.vlm_data,
                split_summary["train_episode_start"],
                split_summary["train_num_episodes"],
            )
        elif split_part == "eval":
            _apply_episode_window(
                source_run_cfg.datasets.vlm_data,
                split_summary["eval_episode_start"],
                split_summary["eval_num_episodes"],
            )
        else:
            raise ValueError(f"split_part must be train/eval, got: {split_part}")

    return build_dataloader(cfg=source_run_cfg, dataset_py=dataset_py)


def _build_multi_vlm_dataloaders(
    cfg: Any,
    split_summary: Optional[dict[str, Any]],
) -> tuple[Dict[str, DataLoader], Optional[DataLoader], Dict[str, float], Optional[str]]:
    enabled_sources = _iter_enabled_vlm_sources(cfg)
    if not enabled_sources:
        raise ValueError("datasets.vlm_multi.enable=true, but no enabled source found in datasets.vlm_multi.sources")

    train_dataloaders: Dict[str, DataLoader] = {}
    source_weights: Dict[str, float] = {}

    for source_name, source_cfg in enabled_sources:
        weight = float(_cfg_get(source_cfg, "weight", 1.0))
        if weight < 0:
            raise ValueError(f"datasets.vlm_multi.sources.{source_name}.weight must be >= 0, got {weight}")

        train_dataloaders[source_name] = _build_single_vlm_source_dataloader(
            cfg=cfg,
            source_cfg=source_cfg,
            split_summary=split_summary,
            split_part="train",
        )
        source_weights[source_name] = weight

    if sum(source_weights.values()) <= 0:
        raise ValueError(
            "Sum of datasets.vlm_multi source weights must be > 0. "
            f"Current weights: {source_weights}"
        )

    datasets_cfg = _cfg_get(cfg, "datasets", None)
    vlm_multi_cfg = _cfg_get(datasets_cfg, "vlm_multi", None)
    eval_source_name = str(_cfg_get(vlm_multi_cfg, "eval_source", "dzb"))

    source_cfg_map = {name: source_cfg for name, source_cfg in enabled_sources}
    if eval_source_name not in source_cfg_map:
        raise ValueError(
            f"datasets.vlm_multi.eval_source `{eval_source_name}` not found among enabled sources: "
            f"{list(source_cfg_map.keys())}"
        )

    eval_dataloader = _build_single_vlm_source_dataloader(
        cfg=cfg,
        source_cfg=source_cfg_map[eval_source_name],
        split_summary=split_summary,
        split_part="eval",
    )

    return train_dataloaders, eval_dataloader, source_weights, eval_source_name


def prepare_data(cfg, accelerator, output_dir) -> Tuple[
    DataLoader,
    Union[DataLoader, Dict[str, DataLoader]],
    Optional[DataLoader],
    Optional[DataLoader],
    Optional[dict[str, Any]],
    Optional[dict[str, float]],
    Optional[str],
]:
    """Prepare co-training data."""
    data_mix = _cfg_get(cfg.datasets.vla_data, "data_mix", "N/A")
    logger.info(f"Creating VLA Dataset with Mixture `{data_mix}`")

    split_summary = _resolve_wallx_episode_split(cfg)

    if split_summary is not None and accelerator.is_main_process:
        logger.info(
            "Episode split enabled: mode=%s ratio=%.4f total=%d train=%s eval=%s",
            split_summary["mode"],
            split_summary["train_ratio"],
            split_summary["total_episodes"],
            split_summary["train_episode_range"],
            split_summary["eval_episode_range"],
        )

    # 1) VLA branch remains unchanged. Split only affects VLA LeRobot windows.
    if split_summary is None:
        vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
        vla_eval_dataloader = None
    else:
        vla_train_cfg = _clone_cfg(cfg)
        vla_eval_cfg = _clone_cfg(cfg)

        _apply_episode_window(
            vla_train_cfg.datasets.vla_data,
            split_summary["train_episode_start"],
            split_summary["train_num_episodes"],
        )
        _apply_episode_window(
            vla_eval_cfg.datasets.vla_data,
            split_summary["eval_episode_start"],
            split_summary["eval_num_episodes"],
        )

        vla_train_dataloader = build_dataloader(cfg=vla_train_cfg, dataset_py=vla_train_cfg.datasets.vla_data.dataset_py)
        vla_eval_dataloader = build_dataloader(cfg=vla_eval_cfg, dataset_py=vla_eval_cfg.datasets.vla_data.dataset_py)

    # 2) VLM branch supports either single-source (legacy) or multi-source routing.
    vlm_source_weights: Optional[dict[str, float]] = None
    vlm_eval_source_name: Optional[str] = None

    if _is_vlm_multi_enabled(cfg):
        vlm_train_dataloader, vlm_eval_dataloader, vlm_source_weights, vlm_eval_source_name = _build_multi_vlm_dataloaders(
            cfg=cfg,
            split_summary=split_summary,
        )
        if accelerator.is_main_process:
            logger.info(
                "Multi-VLM enabled with sources=%s, weights=%s, eval_source=%s",
                list(vlm_train_dataloader.keys()),
                vlm_source_weights,
                vlm_eval_source_name,
            )
    else:
        if split_summary is None:
            vlm_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vlm_data.dataset_py)
            vlm_eval_dataloader = None
        else:
            vlm_train_cfg = _clone_cfg(cfg)
            vlm_eval_cfg = _clone_cfg(cfg)

            _apply_episode_window(
                vlm_train_cfg.datasets.vlm_data,
                split_summary["train_episode_start"],
                split_summary["train_num_episodes"],
            )
            _apply_episode_window(
                vlm_eval_cfg.datasets.vlm_data,
                split_summary["eval_episode_start"],
                split_summary["eval_num_episodes"],
            )

            vlm_train_dataloader = build_dataloader(cfg=vlm_train_cfg, dataset_py=vlm_train_cfg.datasets.vlm_data.dataset_py)
            vlm_eval_dataloader = build_dataloader(cfg=vlm_eval_cfg, dataset_py=vlm_eval_cfg.datasets.vlm_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()

    return (
        vla_train_dataloader,
        vlm_train_dataloader,
        vla_eval_dataloader,
        vlm_eval_dataloader,
        split_summary,
        vlm_source_weights,
        vlm_eval_source_name,
    )
def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and learning rate scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
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


class VLAMTrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        vlm_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        vla_eval_dataloader: Optional[DataLoader] = None,
        vlm_eval_dataloader: Optional[DataLoader] = None,
        split_summary: Optional[dict[str, Any]] = None,
        vlm_source_weights: Optional[dict[str, float]] = None,
        vlm_eval_source_name: Optional[str] = None,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vlm_train_dataloader = vlm_train_dataloader
        self.vla_eval_dataloader = vla_eval_dataloader
        self.vlm_eval_dataloader = vlm_eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.split_summary = split_summary

        self.vlm_source_weights = vlm_source_weights or {}
        self.vlm_eval_source_name = vlm_eval_source_name

        self.is_multi_vlm = isinstance(self.vlm_train_dataloader, dict)
        self.vlm_source_names: list[str] = []
        self.vlm_source_probs_tensor: Optional[torch.Tensor] = None
        self.vlm_source_prob_map: dict[str, float] = {}

        if self.is_multi_vlm:
            self.vlm_source_names = list(self.vlm_train_dataloader.keys())
            if not self.vlm_source_names:
                raise ValueError("Multi-VLM mode enabled but no VLM train dataloader was built.")

            weights = []
            for source_name in self.vlm_source_names:
                weight = float(self.vlm_source_weights.get(source_name, 1.0))
                if weight < 0:
                    raise ValueError(f"VLM source weight must be >= 0, got {source_name}={weight}")
                weights.append(weight)

            total_weight = float(sum(weights))
            if total_weight <= 0:
                raise ValueError(f"Sum of VLM source weights must be > 0, got {weights}")

            self.vlm_source_probs_tensor = torch.tensor(weights, dtype=torch.float32) / total_weight
            self.vlm_source_prob_map = {
                name: float(self.vlm_source_probs_tensor[idx].item())
                for idx, name in enumerate(self.vlm_source_names)
            }

            datasets_cfg = _cfg_get(self.config, "datasets", None)
            vlm_multi_cfg = _cfg_get(datasets_cfg, "vlm_multi", None)
            self.sync_vlm_source_choice = bool(_cfg_get(vlm_multi_cfg, "sync_source_choice_across_ranks", True))

            default_seed = int(_cfg_get(self.config, "seed", 42))
            source_seed = int(_cfg_get(vlm_multi_cfg, "source_seed", default_seed))
            self.vlm_source_generator = torch.Generator(device="cpu")
            self.vlm_source_generator.manual_seed(source_seed)

            if not self.vlm_eval_source_name:
                self.vlm_eval_source_name = self.vlm_source_names[0]
        else:
            self.sync_vlm_source_choice = False

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

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

        if self.is_multi_vlm:
            train_vlm_components = [self.vlm_train_dataloader[name] for name in self.vlm_source_names]

            components = [self.model, self.optimizer, self.vla_train_dataloader]
            components.extend(train_vlm_components)
            has_vla_eval = self.vla_eval_dataloader is not None
            has_vlm_eval = self.vlm_eval_dataloader is not None
            if has_vla_eval:
                components.append(self.vla_eval_dataloader)
            if has_vlm_eval:
                components.append(self.vlm_eval_dataloader)

            prepared = self.setup_distributed_training(self.accelerator, *components)
            if not isinstance(prepared, tuple):
                prepared = (prepared,)

            idx = 0
            self.model = prepared[idx]
            idx += 1
            self.optimizer = prepared[idx]
            idx += 1
            self.vla_train_dataloader = prepared[idx]
            idx += 1

            for source_name in self.vlm_source_names:
                self.vlm_train_dataloader[source_name] = prepared[idx]
                idx += 1

            if has_vla_eval:
                self.vla_eval_dataloader = prepared[idx]
                idx += 1
            if has_vlm_eval:
                self.vlm_eval_dataloader = prepared[idx]
                idx += 1
        else:
            if self.vla_eval_dataloader is not None and self.vlm_eval_dataloader is not None:
                (
                    self.model,
                    self.optimizer,
                    self.vla_train_dataloader,
                    self.vlm_train_dataloader,
                    self.vla_eval_dataloader,
                    self.vlm_eval_dataloader,
                ) = self.setup_distributed_training(
                    self.accelerator,
                    self.model,
                    self.optimizer,
                    self.vla_train_dataloader,
                    self.vlm_train_dataloader,
                    self.vla_eval_dataloader,
                    self.vlm_eval_dataloader,
                )
            else:
                self.model, self.optimizer, self.vla_train_dataloader, self.vlm_train_dataloader = (
                    self.setup_distributed_training(
                        self.accelerator,
                        self.model,
                        self.optimizer,
                        self.vla_train_dataloader,
                        self.vlm_train_dataloader,
                    )
                )

        self._init_wandb()
        self._init_checkpointing()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        if pretrained_checkpoint and is_resume:
            self._load_checkpoint(self.config.resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
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

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
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
        metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

        wandb.log(metrics, step=self.completed_steps)

        if "vla_history_frames_total" in metrics or "vlm_history_frames_total" in metrics:
            logger.info(
                "Step %s history_frames | vla(mean/max/min/total)=%.2f/%.0f/%.0f/%.0f, "
                "vlm(mean/max/min/total)=%.2f/%.0f/%.0f/%.0f",
                self.completed_steps,
                metrics.get("vla_history_frames_mean", 0.0),
                metrics.get("vla_history_frames_max", 0.0),
                metrics.get("vla_history_frames_min", 0.0),
                metrics.get("vla_history_frames_total", 0.0),
                metrics.get("vlm_history_frames_mean", 0.0),
                metrics.get("vlm_history_frames_max", 0.0),
                metrics.get("vlm_history_frames_min", 0.0),
                metrics.get("vlm_history_frames_total", 0.0),
            )

        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            logger.info(f"Step {self.completed_steps}, Loss: {metrics}")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

        if self.is_multi_vlm:
            self.vlm_iters = {}
            self.vlm_epoch_count = {}
            for source_name in self.vlm_source_names:
                self.vlm_iters[source_name] = iter(self.vlm_train_dataloader[source_name])
                self.vlm_epoch_count[source_name] = 0
        else:
            self.vlm_iter = iter(self.vlm_train_dataloader)

        if self.vla_eval_dataloader is not None:
            self.vla_eval_iter = iter(self.vla_eval_dataloader)
            self.vla_eval_epoch_count = 0

        if self.vlm_eval_dataloader is not None:
            self.vlm_eval_iter = iter(self.vlm_eval_dataloader)
            self.vlm_eval_epoch_count = 0

    def _draw_vlm_source_index(self) -> int:
        if not self.is_multi_vlm:
            return 0

        if self.vlm_source_probs_tensor is None:
            raise RuntimeError("VLM source probs are not initialized for multi-VLM mode.")

        sampled_idx = 0

        if not (dist.is_initialized() and self.sync_vlm_source_choice):
            sampled_idx = int(
                torch.multinomial(
                    self.vlm_source_probs_tensor,
                    num_samples=1,
                    replacement=True,
                    generator=self.vlm_source_generator,
                ).item()
            )
            return sampled_idx

        if dist.get_rank() == 0:
            sampled_idx = int(
                torch.multinomial(
                    self.vlm_source_probs_tensor,
                    num_samples=1,
                    replacement=True,
                    generator=self.vlm_source_generator,
                ).item()
            )

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        idx_tensor = torch.tensor([sampled_idx], dtype=torch.long, device=device)
        dist.broadcast(idx_tensor, src=0)
        return int(idx_tensor.item())

    def _get_next_vla_train_batch(self):
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)
        return batch_vla

    def _get_next_vla_eval_batch(self):
        if self.vla_eval_dataloader is None:
            return self._get_next_vla_train_batch()

        try:
            batch_vla = next(self.vla_eval_iter)
        except StopIteration:
            self.vla_eval_iter, self.vla_eval_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_eval_dataloader, self.vla_eval_epoch_count
            )
            batch_vla = next(self.vla_eval_iter)
        return batch_vla

    def _get_next_vlm_batch_from_source(self, source_name: str):
        if not self.is_multi_vlm:
            try:
                return next(self.vlm_iter)
            except StopIteration:
                if not hasattr(self, "vlm_epoch_count"):
                    self.vlm_epoch_count = 0
                self.vlm_iter, self.vlm_epoch_count = self._reset_dataloader(
                    self.vlm_train_dataloader,
                    self.vlm_epoch_count,
                )
                return next(self.vlm_iter)

        if source_name not in self.vlm_iters:
            raise KeyError(f"Unknown VLM source: {source_name}. Available={list(self.vlm_iters.keys())}")

        try:
            batch_vlm = next(self.vlm_iters[source_name])
        except StopIteration:
            self.vlm_iters[source_name], self.vlm_epoch_count[source_name] = self._reset_dataloader(
                self.vlm_train_dataloader[source_name],
                self.vlm_epoch_count[source_name],
            )
            batch_vlm = next(self.vlm_iters[source_name])
        return batch_vlm

    def _get_next_vlm_train_batch(self) -> tuple[str, Any]:
        if not self.is_multi_vlm:
            return "single", self._get_next_vlm_batch_from_source("single")

        source_idx = self._draw_vlm_source_index()
        source_name = self.vlm_source_names[source_idx]
        batch_vlm = self._get_next_vlm_batch_from_source(source_name)
        return source_name, batch_vlm

    def _get_next_batch(self) -> tuple[Any, Any, str]:
        """Get next train batch (automatically handle data loop)."""
        batch_vla = self._get_next_vla_train_batch()
        source_name, batch_vlm = self._get_next_vlm_train_batch()
        return batch_vla, batch_vlm, source_name

    def _get_next_eval_batch(self) -> tuple[Any, Any, str]:
        """Get next eval batch. In multi-VLM mode eval defaults to configured dzb source."""
        batch_vla = self._get_next_vla_eval_batch()

        if self.vlm_eval_dataloader is not None:
            try:
                batch_vlm = next(self.vlm_eval_iter)
            except StopIteration:
                self.vlm_eval_iter, self.vlm_eval_epoch_count = TrainerUtils._reset_dataloader(
                    self.vlm_eval_dataloader,
                    self.vlm_eval_epoch_count,
                )
                batch_vlm = next(self.vlm_eval_iter)
            eval_source_name = self.vlm_eval_source_name or "eval"
            return batch_vla, batch_vlm, eval_source_name

        if self.is_multi_vlm:
            eval_source_name = self.vlm_eval_source_name or self.vlm_source_names[0]
            batch_vlm = self._get_next_vlm_batch_from_source(eval_source_name)
            return batch_vla, batch_vlm, eval_source_name

        source_name, batch_vlm = self._get_next_vlm_train_batch()
        return batch_vla, batch_vlm, source_name

    def _get_qwen_vl_interface(self):
        if hasattr(self.model, "qwen_vl_interface"):
            return self.model.qwen_vl_interface
        return self.accelerator.unwrap_model(self.model).qwen_vl_interface

    def _prepare_vlm_batch(self, batch_vlm):
        if not isinstance(batch_vlm, list):
            return batch_vlm

        vlm_images = [sample["image"] for sample in batch_vlm]
        vlm_instructions = [sample["lang"] for sample in batch_vlm]
        vlm_solutions = [sample.get("solution") for sample in batch_vlm]
        if any(sol is None for sol in vlm_solutions):
            vlm_solutions = None

        qwen_vl_interface = self._get_qwen_vl_interface()
        return qwen_vl_interface.build_qwenvl_inputs(
            images=vlm_images,
            instructions=vlm_instructions,
            solutions=vlm_solutions,
        )

    @staticmethod
    def _extract_history_frame_counts(batch_examples):
        counts = []
        if isinstance(batch_examples, dict):
            images = batch_examples.get("image")
            if isinstance(images, (list, tuple)):
                if images and isinstance(images[0], (list, tuple)):
                    return [max(len(sample_images) - 1, 0) for sample_images in images]
                return [max(len(images) - 1, 0)]
            if torch.is_tensor(images) and images.ndim >= 2:
                # Common batched tensor format: [B, T, ...], where T includes current frame.
                batch_size = int(images.shape[0]) if images.ndim >= 1 else 0
                history_count = max(int(images.shape[1]) - 1, 0)
                return [history_count for _ in range(batch_size)]
            return counts

        if not isinstance(batch_examples, list):
            return counts

        for sample in batch_examples:
            images = sample.get("image") if isinstance(sample, dict) else None
            if isinstance(images, (list, tuple)):
                counts.append(max(len(images) - 1, 0))
            else:
                counts.append(0)
        return counts

    @staticmethod
    def _append_history_metrics(log_dict, prefix, counts):
        if not counts:
            log_dict[f"{prefix}_history_frames_mean"] = 0.0
            log_dict[f"{prefix}_history_frames_max"] = 0.0
            log_dict[f"{prefix}_history_frames_min"] = 0.0
            log_dict[f"{prefix}_history_frames_total"] = 0.0
            return

        arr = np.asarray(counts, dtype=np.float32)
        log_dict[f"{prefix}_history_frames_mean"] = float(arr.mean())
        log_dict[f"{prefix}_history_frames_max"] = float(arr.max())
        log_dict[f"{prefix}_history_frames_min"] = float(arr.min())
        log_dict[f"{prefix}_history_frames_total"] = float(arr.sum())

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla, batch_vlm, vlm_source_name = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla, batch_vlm, vlm_source_name)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "vlm_source": vlm_source_name,
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

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

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Evaluate action prediction and VLM loss with current model."""
        if step_metrics is None:
            step_metrics = {}

        examples, batch_vlm, vlm_eval_source_name = self._get_next_eval_batch()

        # Run models on all processes for ZeRO-3 compatibility
        output_dict = self.model.predict_action(examples=examples)

        batch_vlm_eval = self._prepare_vlm_batch(batch_vlm)
        qwen_vl_interface = self._get_qwen_vl_interface()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                vlm_output = qwen_vl_interface(**batch_vlm_eval)

        if self.accelerator.is_main_process:
            actions = [example["action"] for example in examples]
            normalized_actions = output_dict["normalized_actions"]

            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

            if getattr(vlm_output, "loss", None) is not None:
                vlm_eval_loss_raw = vlm_output.loss.detach().float()
                vlm_eval_loss = (vlm_eval_loss_raw * self.config.trainer.loss_scale.vlm).item()
                step_metrics["vlm_loss_eval_raw"] = vlm_eval_loss_raw.item()
                step_metrics["vlm_loss_eval"] = vlm_eval_loss
                step_metrics[f"vlm_loss_eval/source={vlm_eval_source_name}"] = vlm_eval_loss

        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

            if self.is_multi_vlm:
                logger.info(f"  Multi-VLM source probs = {self.vlm_source_prob_map}")
                logger.info(f"  Multi-VLM eval source = {self.vlm_eval_source_name}")

            if self.split_summary:
                logger.info(
                    "  Episode split = mode=%s ratio=%.4f total=%d train=%s eval=%s",
                    self.split_summary.get("mode"),
                    float(self.split_summary.get("train_ratio", 0.0)),
                    int(self.split_summary.get("total_episodes", 0)),
                    self.split_summary.get("train_episode_range"),
                    self.split_summary.get("eval_episode_range"),
                )

    def _train_step(self, batch_vla, batch_vlm, vlm_source_name: str):
        """Execute single training step."""
        log_dict = {}
        vla_history_counts = self._extract_history_frame_counts(batch_vla)
        vlm_history_counts = self._extract_history_frame_counts(batch_vlm)

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss
            self.accelerator.backward(total_loss)

            batch_vlm = self._prepare_vlm_batch(batch_vlm)
            qwen_vl_interface = self._get_qwen_vl_interface()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                vlm_output = qwen_vl_interface(**batch_vlm)
                if getattr(vlm_output, "loss", None) is None:
                    raise RuntimeError("VLM forward did not return loss. Ensure labels/solutions are present in VLM batch.")
                vlm_loss_raw = vlm_output.loss
                vlm_loss = vlm_loss_raw * self.config.trainer.loss_scale.vlm
            self.accelerator.backward(vlm_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            self.lr_scheduler.step()

            log_dict.update(
                {
                    "action_dit_loss": action_loss.item(),
                    "vlm_loss": vlm_loss.item(),
                    "vlm_loss_raw": vlm_loss_raw.item(),
                    f"vlm_loss/source={vlm_source_name}": vlm_loss.item(),
                    f"vlm_loss_raw/source={vlm_source_name}": vlm_loss_raw.item(),
                }
            )
            self._append_history_metrics(log_dict, "vla", vla_history_counts)
            self._append_history_metrics(log_dict, "vlm", vlm_history_counts)

        return log_dict

    def _finalize_training(self):
        """Training end processing."""
        # Ensure all processes participate in state_dict collection to avoid ZeRO-3 deadlocks.
        state_dict = self.accelerator.get_state_dict(self.model)

        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    (
        vla_train_dataloader,
        vlm_train_dataloader,
        vla_eval_dataloader,
        vlm_eval_dataloader,
        split_summary,
        vlm_source_weights,
        vlm_eval_source_name,
    ) = prepare_data(
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLAMTrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vlm_train_dataloader=vlm_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        vla_eval_dataloader=vla_eval_dataloader,
        vlm_eval_dataloader=vlm_eval_dataloader,
        split_summary=split_summary,
        vlm_source_weights=vlm_source_weights,
        vlm_eval_source_name=vlm_eval_source_name,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/starvla_cotrain_wallx_qwenpi_multi_vlm.yaml",
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
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
