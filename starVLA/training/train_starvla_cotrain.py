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
from typing import Any, Optional, Tuple

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
    vlm_bbox_data_cfg = datasets_cfg.vlm_data
    vlm_signal_data_cfg = _cfg_get(datasets_cfg, "vlm_signal_data", None)

    _validate_no_manual_episode_window(vla_data_cfg, "datasets.vla_data")
    _validate_no_manual_episode_window(vlm_bbox_data_cfg, "datasets.vlm_data")
    if vlm_signal_data_cfg is not None:
        _validate_no_manual_episode_window(vlm_signal_data_cfg, "datasets.vlm_signal_data")

    total_vla = _dataset_total_episodes(vla_data_cfg)
    total_vlm_bbox = _dataset_total_episodes(vlm_bbox_data_cfg)
    if total_vla != total_vlm_bbox:
        raise ValueError(
            "VLA/VLM-bbox datasets have different total episodes, cannot apply unified ratio split: "
            f"vla={total_vla}, vlm_bbox={total_vlm_bbox}"
        )

    if vlm_signal_data_cfg is not None:
        total_vlm_signal = _dataset_total_episodes(vlm_signal_data_cfg)
        if total_vla != total_vlm_signal:
            raise ValueError(
                "VLA/VLM-signal datasets have different total episodes, cannot apply unified ratio split: "
                f"vla={total_vla}, vlm_signal={total_vlm_signal}"
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


def prepare_data(cfg, accelerator, output_dir) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    Optional[DataLoader],
    Optional[DataLoader],
    Optional[DataLoader],
    Optional[dict[str, Any]],
]:
    """Prepare co-training data for VLA + VLM-bbox + VLM-signal."""
    data_mix = _cfg_get(cfg.datasets.vla_data, "data_mix", "N/A")
    logger.info(f"Creating VLA Dataset with Mixture `{data_mix}`")

    vlm_signal_cfg = _cfg_get(cfg.datasets, "vlm_signal_data", None)
    if vlm_signal_cfg is None:
        raise ValueError("datasets.vlm_signal_data is required for VLM signal task.")

    split_summary = _resolve_wallx_episode_split(cfg)

    if split_summary is None:
        vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
        vlm_bbox_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vlm_data.dataset_py)
        vlm_signal_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vlm_signal_data.dataset_py)

        vla_eval_dataloader = None
        vlm_bbox_eval_dataloader = None
        vlm_signal_eval_dataloader = None
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
            train_cfg.datasets.vla_data,
            split_summary["train_episode_start"],
            split_summary["train_num_episodes"],
        )
        _apply_episode_window(
            train_cfg.datasets.vlm_data,
            split_summary["train_episode_start"],
            split_summary["train_num_episodes"],
        )
        _apply_episode_window(
            train_cfg.datasets.vlm_signal_data,
            split_summary["train_episode_start"],
            split_summary["train_num_episodes"],
        )

        _apply_episode_window(
            eval_cfg.datasets.vla_data,
            split_summary["eval_episode_start"],
            split_summary["eval_num_episodes"],
        )
        _apply_episode_window(
            eval_cfg.datasets.vlm_data,
            split_summary["eval_episode_start"],
            split_summary["eval_num_episodes"],
        )
        _apply_episode_window(
            eval_cfg.datasets.vlm_signal_data,
            split_summary["eval_episode_start"],
            split_summary["eval_num_episodes"],
        )

        vla_train_dataloader = build_dataloader(cfg=train_cfg, dataset_py=train_cfg.datasets.vla_data.dataset_py)
        vlm_bbox_train_dataloader = build_dataloader(cfg=train_cfg, dataset_py=train_cfg.datasets.vlm_data.dataset_py)
        vlm_signal_train_dataloader = build_dataloader(cfg=train_cfg, dataset_py=train_cfg.datasets.vlm_signal_data.dataset_py)

        vla_eval_dataloader = build_dataloader(cfg=eval_cfg, dataset_py=eval_cfg.datasets.vla_data.dataset_py)
        vlm_bbox_eval_dataloader = build_dataloader(cfg=eval_cfg, dataset_py=eval_cfg.datasets.vlm_data.dataset_py)
        vlm_signal_eval_dataloader = build_dataloader(cfg=eval_cfg, dataset_py=eval_cfg.datasets.vlm_signal_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()

    return (
        vla_train_dataloader,
        vlm_bbox_train_dataloader,
        vlm_signal_train_dataloader,
        vla_eval_dataloader,
        vlm_bbox_eval_dataloader,
        vlm_signal_eval_dataloader,
        split_summary,
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
        vlm_bbox_train_dataloader,
        vlm_signal_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        vla_eval_dataloader: Optional[DataLoader] = None,
        vlm_bbox_eval_dataloader: Optional[DataLoader] = None,
        vlm_signal_eval_dataloader: Optional[DataLoader] = None,
        split_summary: Optional[dict[str, Any]] = None,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vlm_bbox_train_dataloader = vlm_bbox_train_dataloader
        self.vlm_signal_train_dataloader = vlm_signal_train_dataloader
        self.vla_eval_dataloader = vla_eval_dataloader
        self.vlm_bbox_eval_dataloader = vlm_bbox_eval_dataloader
        self.vlm_signal_eval_dataloader = vlm_signal_eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.split_summary = split_summary

        loss_scale_cfg = _cfg_get(self.config.trainer, "loss_scale", None)
        self.loss_scale_vlm_total = float(_cfg_get(loss_scale_cfg, "vlm", 1.0))
        self.loss_scale_vlm_bbox = float(_cfg_get(loss_scale_cfg, "vlm_bbox", 1.0))
        self.loss_scale_vlm_signal = float(_cfg_get(loss_scale_cfg, "vlm_signal", 1.0))

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

        has_eval = all(
            dl is not None
            for dl in [self.vla_eval_dataloader, self.vlm_bbox_eval_dataloader, self.vlm_signal_eval_dataloader]
        )

        components = [
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.vlm_bbox_train_dataloader,
            self.vlm_signal_train_dataloader,
        ]
        if has_eval:
            components.extend(
                [self.vla_eval_dataloader, self.vlm_bbox_eval_dataloader, self.vlm_signal_eval_dataloader]
            )

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
        self.vlm_bbox_train_dataloader = prepared[idx]
        idx += 1
        self.vlm_signal_train_dataloader = prepared[idx]
        idx += 1

        if has_eval:
            self.vla_eval_dataloader = prepared[idx]
            idx += 1
            self.vlm_bbox_eval_dataloader = prepared[idx]
            idx += 1
            self.vlm_signal_eval_dataloader = prepared[idx]
            idx += 1

        self._init_wandb()
        self._init_checkpointing()

    def _calculate_total_batch_size(self):
        """Calculate global batch size using VLA branch as reference."""
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

        if (
            "vla_history_frames_total" in metrics
            or "vlm_bbox_history_frames_total" in metrics
            or "vlm_signal_history_frames_total" in metrics
        ):
            logger.info(
                "Step %s history_frames | "
                "vla(mean/max/min/total)=%.2f/%.0f/%.0f/%.0f, "
                "vlm_bbox(mean/max/min/total)=%.2f/%.0f/%.0f/%.0f, "
                "vlm_signal(mean/max/min/total)=%.2f/%.0f/%.0f/%.0f",
                self.completed_steps,
                metrics.get("vla_history_frames_mean", 0.0),
                metrics.get("vla_history_frames_max", 0.0),
                metrics.get("vla_history_frames_min", 0.0),
                metrics.get("vla_history_frames_total", 0.0),
                metrics.get("vlm_bbox_history_frames_mean", 0.0),
                metrics.get("vlm_bbox_history_frames_max", 0.0),
                metrics.get("vlm_bbox_history_frames_min", 0.0),
                metrics.get("vlm_bbox_history_frames_total", 0.0),
                metrics.get("vlm_signal_history_frames_mean", 0.0),
                metrics.get("vlm_signal_history_frames_max", 0.0),
                metrics.get("vlm_signal_history_frames_min", 0.0),
                metrics.get("vlm_signal_history_frames_total", 0.0),
            )

        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            logger.info(f"Step {self.completed_steps}, Loss: {metrics}")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vlm_bbox_iter = iter(self.vlm_bbox_train_dataloader)
        self.vlm_signal_iter = iter(self.vlm_signal_train_dataloader)

        if all(
            dl is not None
            for dl in [self.vla_eval_dataloader, self.vlm_bbox_eval_dataloader, self.vlm_signal_eval_dataloader]
        ):
            self.vla_eval_iter = iter(self.vla_eval_dataloader)
            self.vlm_bbox_eval_iter = iter(self.vlm_bbox_eval_dataloader)
            self.vlm_signal_eval_iter = iter(self.vlm_signal_eval_dataloader)

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
        """Get next train batch from all task dataloaders."""
        batch_vla = self._next_from_loader("vla_iter", self.vla_train_dataloader, "vla_epoch_count")
        batch_vlm_bbox = self._next_from_loader(
            "vlm_bbox_iter", self.vlm_bbox_train_dataloader, "vlm_bbox_epoch_count"
        )
        batch_vlm_signal = self._next_from_loader(
            "vlm_signal_iter", self.vlm_signal_train_dataloader, "vlm_signal_epoch_count"
        )
        return batch_vla, batch_vlm_bbox, batch_vlm_signal

    def _get_next_eval_batch(self):
        """Get next eval batch from eval-only dataloaders."""
        if not all(
            dl is not None
            for dl in [self.vla_eval_dataloader, self.vlm_bbox_eval_dataloader, self.vlm_signal_eval_dataloader]
        ):
            return self._get_next_batch()

        batch_vla = self._next_from_loader("vla_eval_iter", self.vla_eval_dataloader, "vla_eval_epoch_count")
        batch_vlm_bbox = self._next_from_loader(
            "vlm_bbox_eval_iter", self.vlm_bbox_eval_dataloader, "vlm_bbox_eval_epoch_count"
        )
        batch_vlm_signal = self._next_from_loader(
            "vlm_signal_eval_iter", self.vlm_signal_eval_dataloader, "vlm_signal_eval_epoch_count"
        )
        return batch_vla, batch_vlm_bbox, batch_vlm_signal

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
            batch_vla, batch_vlm_bbox, batch_vlm_signal = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla, batch_vlm_bbox, batch_vlm_signal)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
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
        """Evaluate action prediction and both VLM task losses with current model."""
        if step_metrics is None:
            step_metrics = {}

        examples, batch_vlm_bbox, batch_vlm_signal = self._get_next_eval_batch()

        output_dict = self.model.predict_action(examples=examples)

        batch_vlm_bbox_eval = self._prepare_vlm_batch(batch_vlm_bbox)
        batch_vlm_signal_eval = self._prepare_vlm_batch(batch_vlm_signal)
        qwen_vl_interface = self._get_qwen_vl_interface()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                vlm_bbox_output = qwen_vl_interface(**batch_vlm_bbox_eval)
                vlm_signal_output = qwen_vl_interface(**batch_vlm_signal_eval)

        if self.accelerator.is_main_process:
            actions = [example["action"] for example in examples]
            normalized_actions = output_dict["normalized_actions"]

            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

            vlm_bbox_loss_eval_raw = None
            if getattr(vlm_bbox_output, "loss", None) is not None:
                vlm_bbox_loss_eval_raw = vlm_bbox_output.loss.detach().float()
                step_metrics["vlm_bbox_loss_eval_raw"] = vlm_bbox_loss_eval_raw.item()
                step_metrics["vlm_bbox_loss_eval"] = (
                    vlm_bbox_loss_eval_raw * self.loss_scale_vlm_bbox
                ).item()

            vlm_signal_loss_eval_raw = None
            if getattr(vlm_signal_output, "loss", None) is not None:
                vlm_signal_loss_eval_raw = vlm_signal_output.loss.detach().float()
                step_metrics["vlm_signal_loss_eval_raw"] = vlm_signal_loss_eval_raw.item()
                step_metrics["vlm_signal_loss_eval"] = (
                    vlm_signal_loss_eval_raw * self.loss_scale_vlm_signal
                ).item()

            if (vlm_bbox_loss_eval_raw is not None) and (vlm_signal_loss_eval_raw is not None):
                vlm_eval_raw = (
                    vlm_bbox_loss_eval_raw * self.loss_scale_vlm_bbox
                    + vlm_signal_loss_eval_raw * self.loss_scale_vlm_signal
                )
                step_metrics["vlm_loss_eval_raw"] = vlm_eval_raw.item()
                step_metrics["vlm_loss_eval"] = (vlm_eval_raw * self.loss_scale_vlm_total).item()

        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device VLA batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Per device VLM-bbox batch size = {self.config.datasets.vlm_data.per_device_batch_size}")
            logger.info(
                "  Per device VLM-signal batch size = %s",
                self.config.datasets.vlm_signal_data.per_device_batch_size,
            )
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size (VLA reference) = {self.total_batch_size}")
            logger.info(
                "  Loss scales: vlm_total=%.4f, vlm_bbox=%.4f, vlm_signal=%.4f",
                self.loss_scale_vlm_total,
                self.loss_scale_vlm_bbox,
                self.loss_scale_vlm_signal,
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

    def _train_step(self, batch_vla, batch_vlm_bbox, batch_vlm_signal):
        """Execute single training step."""
        log_dict = {}
        vla_history_counts = self._extract_history_frame_counts(batch_vla)
        vlm_bbox_history_counts = self._extract_history_frame_counts(batch_vlm_bbox)
        vlm_signal_history_counts = self._extract_history_frame_counts(batch_vlm_signal)

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
            self.accelerator.backward(action_loss)

            qwen_vl_interface = self._get_qwen_vl_interface()

            batch_vlm_bbox_inputs = self._prepare_vlm_batch(batch_vlm_bbox)
            batch_vlm_signal_inputs = self._prepare_vlm_batch(batch_vlm_signal)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                vlm_bbox_output = qwen_vl_interface(**batch_vlm_bbox_inputs)
                if getattr(vlm_bbox_output, "loss", None) is None:
                    raise RuntimeError(
                        "VLM bbox forward did not return loss. Ensure labels/solutions are present in bbox batch."
                    )
                vlm_bbox_loss_raw = vlm_bbox_output.loss

                vlm_signal_output = qwen_vl_interface(**batch_vlm_signal_inputs)
                if getattr(vlm_signal_output, "loss", None) is None:
                    raise RuntimeError(
                        "VLM signal forward did not return loss. Ensure labels/solutions are present in signal batch."
                    )
                vlm_signal_loss_raw = vlm_signal_output.loss

                vlm_bbox_loss_weighted = vlm_bbox_loss_raw * self.loss_scale_vlm_bbox
                vlm_signal_loss_weighted = vlm_signal_loss_raw * self.loss_scale_vlm_signal
                vlm_loss_raw = vlm_bbox_loss_weighted + vlm_signal_loss_weighted
                vlm_loss = vlm_loss_raw * self.loss_scale_vlm_total

            self.accelerator.backward(vlm_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            self.lr_scheduler.step()

            log_dict.update(
                {
                    "action_dit_loss": action_loss.item(),
                    "vlm_bbox_loss_raw": vlm_bbox_loss_raw.item(),
                    "vlm_signal_loss_raw": vlm_signal_loss_raw.item(),
                    "vlm_bbox_loss": vlm_bbox_loss_weighted.item(),
                    "vlm_signal_loss": vlm_signal_loss_weighted.item(),
                    "vlm_loss_raw": vlm_loss_raw.item(),
                    "vlm_loss": vlm_loss.item(),
                }
            )
            self._append_history_metrics(log_dict, "vla", vla_history_counts)
            self._append_history_metrics(log_dict, "vlm_bbox", vlm_bbox_history_counts)
            self._append_history_metrics(log_dict, "vlm_signal", vlm_signal_history_counts)

        return log_dict

    def _finalize_training(self):
        """Training end processing."""
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
        vlm_bbox_train_dataloader,
        vlm_signal_train_dataloader,
        vla_eval_dataloader,
        vlm_bbox_eval_dataloader,
        vlm_signal_eval_dataloader,
        split_summary,
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
        vlm_bbox_train_dataloader=vlm_bbox_train_dataloader,
        vlm_signal_train_dataloader=vlm_signal_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        vla_eval_dataloader=vla_eval_dataloader,
        vlm_bbox_eval_dataloader=vlm_bbox_eval_dataloader,
        vlm_signal_eval_dataloader=vlm_signal_eval_dataloader,
        split_summary=split_summary,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/starvla_cotrain_oxe.yaml",
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
