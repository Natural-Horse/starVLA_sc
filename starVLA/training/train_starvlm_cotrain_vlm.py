# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
VLM-only trainer aligned with cotrain VLM branch behavior.

Key points compared with train_starvlm.py:
1) Supports WallX list-style VLM batches by converting them through
   qwen_vl_interface.build_qwenvl_inputs(...).
2) Supports optional contiguous episode split (train/eval) for WallX LeRobot data.
3) Uses ZeRO-3-safe state_dict collection (all ranks participate) to avoid deadlocks.
"""

# Standard Library
import argparse
import json
import os
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
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

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


def _resolve_wallx_vlm_episode_split(cfg: Any) -> Optional[dict[str, Any]]:
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

    vlm_data_cfg = datasets_cfg.vlm_data
    _validate_no_manual_episode_window(vlm_data_cfg, "datasets.vlm_data")

    total_episodes = _dataset_total_episodes(vlm_data_cfg)
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


def prepare_data(
    cfg,
    accelerator,
    output_dir,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[dict[str, Any]]]:
    """Prepare VLM training/eval data with optional contiguous split."""
    dataset_name = _cfg_get(cfg.datasets.vlm_data, "dataset_use", _cfg_get(cfg.datasets.vlm_data, "repo_id", "N/A"))
    logger.info(f"Creating VLM Dataset `{dataset_name}`")

    split_summary = _resolve_wallx_vlm_episode_split(cfg)

    if split_summary is None:
        vlm_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vlm_data.dataset_py)
        vlm_eval_dataloader = None
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
            train_cfg.datasets.vlm_data,
            split_summary["train_episode_start"],
            split_summary["train_num_episodes"],
        )
        _apply_episode_window(
            eval_cfg.datasets.vlm_data,
            split_summary["eval_episode_start"],
            split_summary["eval_num_episodes"],
        )

        vlm_train_dataloader = build_dataloader(cfg=train_cfg, dataset_py=train_cfg.datasets.vlm_data.dataset_py)
        vlm_eval_dataloader = build_dataloader(cfg=eval_cfg, dataset_py=eval_cfg.datasets.vlm_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vlm_train_dataloader, vlm_eval_dataloader, split_summary


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
        vlm_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        vlm_eval_dataloader: Optional[DataLoader] = None,
        split_summary: Optional[dict[str, Any]] = None,
    ):
        self.config = cfg
        self.model = model
        self.vlm_train_dataloader = vlm_train_dataloader
        self.vlm_eval_dataloader = vlm_eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.split_summary = split_summary

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

        freeze_modules = self.config.trainer.freeze_modules if hasattr(self.config.trainer, "freeze_modules") else None
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        if self.vlm_eval_dataloader is not None:
            self.model, self.optimizer, self.vlm_train_dataloader, self.vlm_eval_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vlm_train_dataloader,
                self.vlm_eval_dataloader,
            )
        else:
            self.model, self.optimizer, self.vlm_train_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vlm_train_dataloader,
            )

        self._init_wandb()
        self._init_checkpointing()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        per_device_bs = getattr(self.config.datasets.vlm_data, "per_device_batch_size", 1)
        return per_device_bs * self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vlm-train",
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
        """Save current training state (ZeRO-3 safe: all ranks participate in state_dict gather)."""
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

        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            if hasattr(self.vlm_train_dataloader, "__len__"):
                dataloader_length = len(self.vlm_train_dataloader)
                if dataloader_length:
                    metrics["epoch"] = round(self.completed_steps / dataloader_length, 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Metrics: {metrics}")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vlm_iter = iter(self.vlm_train_dataloader)
        if self.vlm_eval_dataloader is not None:
            self.vlm_eval_iter = iter(self.vlm_eval_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            return next(self.vlm_iter)
        except StopIteration:
            if not hasattr(self, "vlm_epoch_count"):
                self.vlm_epoch_count = 0
            self.vlm_iter, self.vlm_epoch_count = self._reset_dataloader(self.vlm_train_dataloader, self.vlm_epoch_count)
            return next(self.vlm_iter)

    def _get_next_eval_batch(self):
        """Get next eval batch from eval-only dataloader."""
        if self.vlm_eval_dataloader is None:
            return self._get_next_batch()

        try:
            return next(self.vlm_eval_iter)
        except StopIteration:
            if not hasattr(self, "vlm_eval_epoch_count"):
                self.vlm_eval_epoch_count = 0
            self.vlm_eval_iter, self.vlm_eval_epoch_count = self._reset_dataloader(
                self.vlm_eval_dataloader, self.vlm_eval_epoch_count
            )
            return next(self.vlm_eval_iter)

    def _get_qwen_vl_interface(self):
        if hasattr(self.model, "qwen_vl_interface"):
            return self.model.qwen_vl_interface
        return self.accelerator.unwrap_model(self.model).qwen_vl_interface

    def _prepare_vlm_batch(self, batch_vlm):
        # WallX style: collate returns list[dict(image, lang, solution)]
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

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            batch_vlm = self._get_next_batch()
            step_metrics = self._train_step(batch_vlm)

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()
                dist.barrier()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics=None):
        """Evaluate VLM loss on eval split if available."""
        if step_metrics is None:
            step_metrics = {}

        if self.vlm_eval_dataloader is None:
            return step_metrics

        batch_vlm = self._get_next_eval_batch()
        batch_vlm = self._prepare_vlm_batch(batch_vlm)
        qwen_vl_interface = self._get_qwen_vl_interface()

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                vlm_output = qwen_vl_interface(**batch_vlm)

        if self.accelerator.is_main_process and getattr(vlm_output, "loss", None) is not None:
            vlm_eval_loss_raw = vlm_output.loss.detach().float()
            step_metrics["vlm_loss_eval_raw"] = vlm_eval_loss_raw.item()
            step_metrics["vlm_loss_eval"] = (vlm_eval_loss_raw * self.config.trainer.loss_scale.vlm).item()

        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            per_device_bs = getattr(self.config.datasets.vlm_data, "per_device_batch_size", "N/A")
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {per_device_bs}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            if self.split_summary:
                logger.info(
                    "  Episode split = mode=%s ratio=%.4f total=%d train=%s eval=%s",
                    self.split_summary.get("mode"),
                    float(self.split_summary.get("train_ratio", 0.0)),
                    int(self.split_summary.get("total_episodes", 0)),
                    self.split_summary.get("train_episode_range"),
                    self.split_summary.get("eval_episode_range"),
                )

    def _train_step(self, batch_vlm):
        """Execute single training step."""
        log_dict = {}
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

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
            log_dict["vlm_loss"] = vlm_loss.item()
            log_dict["vlm_loss_raw"] = vlm_loss_raw.item()

        return log_dict

    def _finalize_training(self):
        """Training end processing (ZeRO-3 safe)."""
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
    logger.info("VLM Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vlm = build_framework(cfg)
    vlm_train_dataloader, vlm_eval_dataloader, split_summary = prepare_data(
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vlm, cfg=cfg)

    trainer = VLAMTrainer(
        cfg=cfg,
        model=vlm,
        vlm_train_dataloader=vlm_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        vlm_eval_dataloader=vlm_eval_dataloader,
        split_summary=split_summary,
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
