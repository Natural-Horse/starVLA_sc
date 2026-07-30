# Copyright 2025 starVLA community. rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""Subtask-aware training for WallX.

Replaces the router with subtask primitives (Search / Fly to).
Key differences from router trainer:
- All samples are action route (no bbox branch)
- Knowledge Isolation: hidden_states detached before action loss
- VLM CE loss trains on full subtask_text instead of route token
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
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from peft import LoraConfig, get_peft_model

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    normalize_dotlist_args,
)

IGNORE_INDEX = -100

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)


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


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def _dataset_total_episodes(data_cfg: Any) -> int:
    repo_id = str(_cfg_get(data_cfg, "repo_id", "dzb/lerobot_ego_data_subtask"))
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


def _resolve_subtask_episode_split(cfg: Any) -> Optional[dict[str, Any]]:
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

    subtask_cfg = datasets_cfg.subtask_data
    _validate_no_manual_episode_window(subtask_cfg, "datasets.subtask_data")

    total_episodes = _dataset_total_episodes(subtask_cfg)
    train_num_episodes = int(np.floor(total_episodes * train_ratio))
    train_num_episodes = max(1, min(train_num_episodes, total_episodes - 1))
    eval_num_episodes = total_episodes - train_num_episodes
    if eval_num_episodes <= 0:
        raise ValueError(
            f"Invalid split result: total={total_episodes}, train_num={train_num_episodes}, eval={eval_num_episodes}"
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


def prepare_data(
    cfg,
    accelerator,
    output_dir,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[dict[str, Any]]]:
    logger.info("Creating WallX subtask dataset")

    split_summary = _resolve_subtask_episode_split(cfg)
    if split_summary is None:
        subtask_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.subtask_data.dataset_py)
        subtask_eval_dataloader = None
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
            train_cfg.datasets.subtask_data,
            split_summary["train_episode_start"],
            split_summary["train_num_episodes"],
        )
        _apply_episode_window(
            eval_cfg.datasets.subtask_data,
            split_summary["eval_episode_start"],
            split_summary["eval_num_episodes"],
        )
        _disable_photometric_augmentation(eval_cfg.datasets.subtask_data)

        subtask_train_dataloader = build_dataloader(cfg=train_cfg, dataset_py=train_cfg.datasets.subtask_data.dataset_py)
        subtask_eval_dataloader = build_dataloader(cfg=eval_cfg, dataset_py=eval_cfg.datasets.subtask_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return subtask_train_dataloader, subtask_eval_dataloader, split_summary


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    # Filter out frozen params (LoRA base weights, frozen action_model, etc.)
    for group in param_groups:
        group["params"] = [p for p in group["params"] if p.requires_grad]
    param_groups = [g for g in param_groups if g["params"]]
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


class VLA_SubtaskTrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        subtask_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        subtask_eval_dataloader: Optional[DataLoader] = None,
        split_summary: Optional[dict[str, Any]] = None,
    ):
        self.config = cfg
        self.model = model
        self.subtask_train_dataloader = subtask_train_dataloader
        self.subtask_eval_dataloader = subtask_eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.split_summary = split_summary

        loss_scale_cfg = _cfg_get(self.config.trainer, "loss_scale", None)
        self.loss_scale_vlm = float(_cfg_get(loss_scale_cfg, "vlm", 1.0))
        self.loss_scale_action = float(_cfg_get(loss_scale_cfg, "action", 1.0))

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def _calculate_total_batch_size(self):
        return (
            self.config.datasets.subtask_data.per_device_batch_size
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

        components = [self.model, self.optimizer, self.subtask_train_dataloader]
        has_eval = self.subtask_eval_dataloader is not None
        if has_eval:
            components.append(self.subtask_eval_dataloader)

        prepared = self.setup_distributed_training(self.accelerator, *components)
        if not isinstance(prepared, tuple):
            prepared = (prepared,)

        self.model = prepared[0]
        self.optimizer = prepared[1]
        self.subtask_train_dataloader = prepared[2]
        if has_eval:
            self.subtask_eval_dataloader = prepared[3]

        self._init_wandb()
        self._init_checkpointing()

    def _init_wandb(self):
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-subtask-train",
            )

    def _init_checkpointing(self):
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

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
        metrics["epoch"] = round(self.completed_steps / len(self.subtask_train_dataloader), 2)

        wandb.log(metrics, step=self.completed_steps)
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            logger.info(f"Step {self.completed_steps}, Metrics: {metrics}")

    def _create_data_iterators(self):
        self.subtask_iter = iter(self.subtask_train_dataloader)
        self.subtask_epoch_count = 0
        if self.subtask_eval_dataloader is not None:
            self.subtask_eval_iter = iter(self.subtask_eval_dataloader)
            self.subtask_eval_epoch_count = 0

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
        return self._next_from_loader("subtask_iter", self.subtask_train_dataloader, "subtask_epoch_count")

    def _get_next_eval_batch(self):
        if self.subtask_eval_dataloader is None:
            return self._get_next_batch()
        return self._next_from_loader("subtask_eval_iter", self.subtask_eval_dataloader, "subtask_eval_epoch_count")

    def _unwrap_model(self):
        if hasattr(self.model, "action_loss_from_hidden_states"):
            return self.model
        return self.accelerator.unwrap_model(self.model)

    def _get_qwen_vl_interface(self):
        if hasattr(self.model, "qwen_vl_interface"):
            return self.model.qwen_vl_interface
        return self.accelerator.unwrap_model(self.model).qwen_vl_interface

    def _prepare_subtask_batch(self, batch_subtask):
        if not isinstance(batch_subtask, list):
            return batch_subtask

        images = [sample["image"] for sample in batch_subtask]
        instructions = [sample["lang"] for sample in batch_subtask]
        solutions = [sample["solution"] for sample in batch_subtask]
        qwen_vl_interface = self._get_qwen_vl_interface()
        return qwen_vl_interface.build_qwenvl_inputs(
            images=images,
            instructions=instructions,
            solutions=solutions,
        )

    def _compute_subtask_action_loss(self, hidden_states, batch_subtask):
        """Compute action loss with Knowledge Isolation (detached hidden states).
        All samples are action route.
        """
        train_model = self._unwrap_model()
        hidden_device = hidden_states[-1].device

        # All samples are action — no route filtering needed
        action_examples = batch_subtask
        action_index_tensor = torch.arange(
            len(batch_subtask), device=hidden_device, dtype=torch.long,
        )

        # Knowledge Isolation: detach hidden states
        detached_hidden = [h.detach() for h in hidden_states]

        action_loss = train_model.action_loss_from_hidden_states(
            detached_hidden,
            action_examples,
            indices=action_index_tensor,
        )
        return action_loss

    def _train_step(self, batch_subtask):
        log_dict = {}

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            batch_inputs = self._prepare_subtask_batch(batch_subtask)
            qwen_vl_interface = self._get_qwen_vl_interface()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                qwen_output = qwen_vl_interface(
                    **batch_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                if getattr(qwen_output, "loss", None) is None:
                    raise RuntimeError("Subtask VLM forward did not return loss. Ensure `solution` is present.")
                vlm_loss_raw = qwen_output.loss
                vlm_loss = vlm_loss_raw * self.loss_scale_vlm

            # Knowledge Isolation: detached hidden_states for action loss
            action_loss_raw = self._compute_subtask_action_loss(
                qwen_output.hidden_states,
                batch_subtask,
            )
            action_loss = action_loss_raw * self.loss_scale_action

            total_loss = vlm_loss + action_loss
            total_loss_value = total_loss.detach().float().item()
            vlm_loss_raw_value = vlm_loss_raw.detach().float().item()
            action_loss_raw_value = action_loss_raw.detach().float().item()

            self.accelerator.backward(total_loss)
            del qwen_output, batch_inputs, total_loss, vlm_loss_raw, vlm_loss, action_loss_raw, action_loss

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            self.lr_scheduler.step()

            log_dict.update(
                {
                    "loss": total_loss_value,
                    "subtask_vlm_loss": vlm_loss_raw_value,
                    "subtask_vlm_loss_scaled": vlm_loss_raw_value * self.loss_scale_vlm,
                    "action_dit_loss_raw": action_loss_raw_value,
                    "action_dit_loss": action_loss_raw_value * self.loss_scale_action,
                    "batch_size": len(batch_subtask),
                }
            )

        return log_dict

    def eval_subtask_model(self, step_metrics: dict = None):
        if step_metrics is None:
            step_metrics = {}

        batch_subtask = self._get_next_eval_batch()
        batch_inputs = self._prepare_subtask_batch(batch_subtask)
        qwen_vl_interface = self._get_qwen_vl_interface()

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                qwen_output = qwen_vl_interface(
                    **batch_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

            if self.accelerator.is_main_process:
                if getattr(qwen_output, "loss", None) is not None:
                    vlm_loss_raw = qwen_output.loss.detach().float()
                    step_metrics["subtask_vlm_loss_eval"] = vlm_loss_raw.item()
                    step_metrics["subtask_vlm_loss_eval_scaled"] = (vlm_loss_raw * self.loss_scale_vlm).item()

                # Action eval with detached hidden states
                detached_hidden = [h.detach() for h in qwen_output.hidden_states]
                train_model = self._unwrap_model()
                action_loss_raw = train_model.action_loss_from_hidden_states(
                    detached_hidden,
                    batch_subtask,
                    indices=torch.arange(len(batch_subtask), device=detached_hidden[-1].device, dtype=torch.long),
                )
                step_metrics["action_dit_loss_eval_raw"] = action_loss_raw.detach().float().item()
                step_metrics["action_dit_loss_eval"] = (action_loss_raw.detach().float() * self.loss_scale_action).item()

        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        if self.accelerator.is_main_process:
            subtask_cfg = self.config.datasets.subtask_data
            logger.info("***** Subtask Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {subtask_cfg.per_device_batch_size}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            logger.info(
                "  Loss scales: vlm=%.4f, action=%.4f",
                self.loss_scale_vlm,
                self.loss_scale_action,
            )
            logger.info("  Knowledge Isolation: hidden_states detached before action loss")
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
            batch_subtask = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_subtask)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    data_time=f"{t_end_data - t_start_data:.3f}",
                    model_time=f"{t_end_model - t_start_model:.3f}",
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_subtask_model(step_metrics)

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


def main(cfg) -> None:
    logger.info("VLA Subtask Training :: Warming Up")

    cfg = wrap_config(cfg)
    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)

    # Apply LoRA to VLM backbone (freeze all base params, train only adapters)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    vla.qwen_vl_interface.model = get_peft_model(vla.qwen_vl_interface.model, lora_cfg)
    vla.qwen_vl_interface.model.print_trainable_parameters()

    subtask_train_dataloader, subtask_eval_dataloader, split_summary = prepare_data(
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLA_SubtaskTrainer(
        cfg=cfg,
        model=vla,
        subtask_train_dataloader=subtask_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        subtask_eval_dataloader=subtask_eval_dataloader,
        split_summary=split_summary,
    )
    trainer.prepare_training()
    trainer.train()

    logger.info("Subtask training finished.")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/starvla_subtask_wallx.yaml",
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
