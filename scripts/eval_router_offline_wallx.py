#!/usr/bin/env python3
"""Offline open-loop evaluation for one WallX router checkpoint.

This is a router-aware evaluator written specifically for the unified
action-vs-bbox WallX checkpoints. It does not import helpers from the older
``eval_open_loop_wallx.py`` script.

Two modes are supported:

1) batch
   Evaluate selected episodes and only report aggregate metrics such as:
   - route accuracy
   - action MSE, computed only when router predicted the correct action route
   - bbox IoU, computed only when router predicted the correct bbox route

2) detail
   Evaluate selected episodes frame-by-frame and save visualizations:
   - action frames: current RGB image + GT/pred action-horizon comparison
   - bbox frames: GT/pred bbox overlay on the original image

Examples
--------
Batch evaluation on episodes 10..30:

CUDA_VISIBLE_DEVICES=0 python \
  /diff/wallx_workspace/starVLA/scripts/eval_router_offline_wallx.py \
  --mode batch \
  --config_yaml \
    /diff/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml \
  --checkpoint \
    /diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_unified_router_small_liangdu_new/checkpoints/steps_12000_pytorch_model.pt \
  --output_dir /diff/wallx_workspace/starVLA/results/router_offline_eval_batch_12000_800_880 \
  --episode_range 800 880 \
  --frame_stride 1 \
  --max_frames_per_episode 0 \
  --device cuda:0

Detailed evaluation of one episode:

CUDA_VISIBLE_DEVICES=0 python \
  /diff/wallx_workspace/starVLA/scripts/eval_router_offline_wallx.py \
  --mode detail \
  --config_yaml \
    /diff/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml \
  --checkpoint \
    /diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_unified_router_small_liangdu_new/checkpoints/steps_15000_pytorch_model.pt \
  --output_dir /diff/wallx_workspace/starVLA/results/router_offline_eval_detail_ep43 \
  --episode_indices 43 \
  --frame_stride 1 \
  --max_frames_per_episode 0 \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import PartialState
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.dataloader.wallx_cotrain_datasets import WallXRouterDataset
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import resize_images
from transformers import StoppingCriteriaList

from scripts.serve_wallx_router_policy import (
    FastActionDecodeError,
    FastActionDecoder,
    TokenSequenceStoppingCriteria,
)


POINT_PATTERN = re.compile(
    r"<point>\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*</point>",
    re.IGNORECASE,
)


def parse_action_route_solution(
    generated_text: str,
    *,
    pred_action_token: str,
    action_route_format: str,
    subtask_start_token: str,
    subtask_end_token: str,
) -> tuple[str, str | None]:
    text = str(generated_text or "").strip()
    if not text.startswith(pred_action_token):
        text = f"{pred_action_token}{text}"
    if action_route_format != "route_subtask":
        return pred_action_token, None

    start_idx = text.find(subtask_start_token)
    if start_idx < 0:
        return pred_action_token, None
    subtask_start = start_idx + len(subtask_start_token)
    end_idx = text.find(subtask_end_token, subtask_start)
    if end_idx < 0:
        return pred_action_token, None
    subtask_text = text[subtask_start:end_idx].strip()
    route_solution = f"{pred_action_token}{subtask_start_token}{subtask_text}{subtask_end_token}"
    return route_solution, subtask_text or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline open-loop evaluation for WallX router checkpoints.")
    parser.add_argument("--mode", choices=("batch", "detail"), required=True, help="Evaluation mode.")
    parser.add_argument("--config_yaml", type=str, required=True, help="Training config yaml for this checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file or checkpoint run directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save summaries and visualizations.")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda / cuda:0 / cpu.")
    parser.add_argument(
        "--episode_indices",
        type=int,
        nargs="*",
        default=None,
        help="Specific episode indices to evaluate.",
    )
    parser.add_argument(
        "--episode_range",
        type=int,
        nargs=2,
        default=None,
        metavar=("START_EP", "END_EP"),
        help="Inclusive episode range.",
    )
    parser.add_argument("--start_episode", type=int, default=0, help="Start episode when no explicit indices are given.")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes when no explicit indices are given.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Sample one frame every N frames inside each episode.")
    parser.add_argument(
        "--max_frames_per_episode",
        type=int,
        default=0,
        help="Maximum frames per episode; <=0 means no limit.",
    )
    parser.add_argument(
        "--dedupe_effective_indices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate frames that snap to the same effective training index.",
    )
    parser.add_argument(
        "--route_mode",
        type=str,
        default="first_token",
        choices=("first_token", "generate"),
        help="Router decision mode.",
    )
    parser.add_argument("--route_max_new_tokens", type=int, default=32, help="Max new tokens for route generation.")
    parser.add_argument(
        "--route_do_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sampling for route generation.",
    )
    parser.add_argument("--route_temperature", type=float, default=0.2, help="Route generation temperature.")
    parser.add_argument("--route_top_p", type=float, default=0.95, help="Route generation top-p.")
    parser.add_argument(
        "--bbox_max_new_tokens",
        type=int,
        default=32,
        help="Max new tokens for forced bbox generation.",
    )
    parser.add_argument(
        "--bbox_do_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sampling for forced bbox generation.",
    )
    parser.add_argument("--bbox_temperature", type=float, default=0.2, help="BBox generation temperature.")
    parser.add_argument("--bbox_top_p", type=float, default=0.95, help="BBox generation top-p.")
    parser.add_argument(
        "--route_confidence_threshold",
        type=float,
        default=None,
        help="Optional threshold for first-token router confidence.",
    )
    parser.add_argument(
        "--action_decode_mode",
        choices=("flow", "fast"),
        default="flow",
        help="Decode action-route outputs with the flow action expert or autoregressive FAST tokens.",
    )
    parser.add_argument(
        "--checkpoint_modules",
        default=None,
        help=(
            "Comma-separated module paths to load from the checkpoint. "
            "Defaults to qwen_vl_interface for FAST eval and full-model loading for flow eval."
        ),
    )
    parser.add_argument("--base_vlm", default=None, help="Override framework.qwenvl.base_vlm before model load.")
    parser.add_argument(
        "--special_tokens_policy",
        choices=("auto_add", "strict", "none"),
        default=None,
        help="Override framework.qwenvl.special_tokens.policy before model load.",
    )
    parser.add_argument(
        "--require_fast_action_tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override framework.qwenvl.special_tokens.require_fast_action_tokens before model load.",
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=None,
        help="Override router/action-model action horizon and FAST decode shape.",
    )
    parser.add_argument("--fast_tokenizer_path", default=None, help="Override framework.action_tokenizer.path.")
    parser.add_argument("--fast_token_prefix", default=None, help="Override FAST VLM token prefix.")
    parser.add_argument("--fast_token_count", type=int, default=None, help="Override FAST VLM token count.")
    parser.add_argument("--fast_max_new_tokens", type=int, default=256, help="FAST action generation budget.")
    parser.add_argument(
        "--fast_decode_attempts",
        type=int,
        default=1,
        help="Maximum FAST action generation attempts before recording a decode error.",
    )
    parser.add_argument(
        "--fast_retry_do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sampling for FAST retries after the first greedy attempt.",
    )
    parser.add_argument("--fast_retry_temperature", type=float, default=0.7, help="FAST retry sampling temperature.")
    parser.add_argument("--fast_retry_top_p", type=float, default=0.95, help="FAST retry sampling top-p.")
    parser.add_argument(
        "--fast_debug_dir",
        default=None,
        help="Optional directory for per-frame FAST generation/decode debug JSON files.",
    )
    parser.add_argument(
        "--save_per_frame_records",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-frame CSV/JSON records.",
    )
    parser.add_argument(
        "--max_action_vis_images",
        type=int,
        default=0,
        help="For detail mode: maximum action visualizations per model; <=0 means unlimited.",
    )
    parser.add_argument(
        "--max_bbox_vis_images",
        type=int,
        default=0,
        help="For detail mode: maximum bbox visualizations per model; <=0 means unlimited.",
    )
    return parser.parse_args()


def to_int(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def to_bool(x: Any) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(x.item())
    return bool(x)


def tensor_image_to_pil_original(image_chw: torch.Tensor) -> Image.Image:
    arr = image_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr_u8 = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr_u8)


def parse_bbox_from_text(text: str) -> list[float] | None:
    match = POINT_PATTERN.search(str(text))
    if not match:
        return None
    return [float(match.group(i)) for i in range(1, 5)]


def is_valid_bbox_xyxy(bbox: list[float] | tuple[float, float, float, float] | np.ndarray | None) -> bool:
    if bbox is None:
        return False
    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if arr.size != 4:
        return False
    x1, y1, x2, y2 = arr.tolist()
    if x1 <= 0 or y1 <= 0 or x2 <= 0 or y2 <= 0:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    return True


def clip_bbox_xyxy(bbox: list[float], w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    return [x1, y1, x2, y2]


def resized_to_original_bbox(bbox_resized: list[float], src_w: int, src_h: int, dst_w: int, dst_h: int) -> list[float]:
    sx = float(src_w) / float(dst_w)
    sy = float(src_h) / float(dst_h)
    x1, y1, x2, y2 = bbox_resized
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def should_save_visual(saved_count: int, max_to_save: int) -> bool:
    return max_to_save <= 0 or saved_count < max_to_save


def resolve_checkpoint_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    candidates = []
    for p in path.rglob("*"):
        if p.is_file() and (
            p.name.endswith("pytorch_model.pt")
            or p.name.endswith("model.safetensors")
            or p.suffix in {".pt", ".safetensors"}
        ):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found under directory: {path}")

    def score(p: Path) -> tuple[int, float]:
        match = re.search(r"steps_(\d+)", p.name)
        step = int(match.group(1)) if match else -1
        return (step, p.stat().st_mtime)

    candidates.sort(key=score)
    return candidates[-1]


def model_alias_from_input(checkpoint_input: str, resolved_checkpoint: Path) -> str:
    in_path = Path(checkpoint_input)
    if in_path.is_dir():
        return in_path.name
    if resolved_checkpoint.parent.name == "checkpoints":
        return resolved_checkpoint.parent.parent.name
    return resolved_checkpoint.stem


def disable_router_photometric_aug(cfg: Any) -> None:
    aug_cfg = getattr(cfg.datasets.router_data, "photometric_augmentation", None)
    if aug_cfg is not None:
        aug_cfg.enabled = False


def cfg_set(cfg: Any, key: str, value: Any) -> None:
    OmegaConf.update(cfg, key, value, merge=True)


def apply_cli_overrides(cfg: Any, args: argparse.Namespace) -> None:
    if args.action_decode_mode == "fast":
        cfg_set(cfg, "framework.router.action_supervision", "fast_token_ce")
        cfg_set(cfg, "datasets.router_data.action_supervision", "fast_token_ce")
    if args.base_vlm:
        cfg_set(cfg, "framework.qwenvl.base_vlm", args.base_vlm)
    if args.special_tokens_policy:
        cfg_set(cfg, "framework.qwenvl.special_tokens.policy", args.special_tokens_policy)
    if args.require_fast_action_tokens is not None:
        cfg_set(
            cfg,
            "framework.qwenvl.special_tokens.require_fast_action_tokens",
            bool(args.require_fast_action_tokens),
        )
    if args.action_horizon is not None:
        if args.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {args.action_horizon}")
        cfg_set(cfg, "datasets.router_data.action_horizon", int(args.action_horizon))
        cfg_set(cfg, "framework.action_model.action_horizon", int(args.action_horizon))
        cfg_set(cfg, "framework.action_model.future_action_window_size", int(args.action_horizon) - 1)
    if args.fast_tokenizer_path:
        cfg_set(cfg, "framework.action_tokenizer.path", args.fast_tokenizer_path)
        cfg_set(cfg, "datasets.router_data.action_tokenizer.path", args.fast_tokenizer_path)
    if args.fast_token_prefix:
        cfg_set(cfg, "framework.action_tokenizer.token_prefix", args.fast_token_prefix)
        cfg_set(cfg, "datasets.router_data.action_tokenizer.token_prefix", args.fast_token_prefix)
        cfg_set(cfg, "framework.qwenvl.special_tokens.action_token_prefix", args.fast_token_prefix)
    if args.fast_token_count is not None:
        if args.fast_token_count <= 0:
            raise ValueError(f"fast_token_count must be positive, got {args.fast_token_count}")
        cfg_set(cfg, "framework.action_tokenizer.token_count", int(args.fast_token_count))
        cfg_set(cfg, "datasets.router_data.action_tokenizer.token_count", int(args.fast_token_count))
        cfg_set(cfg, "framework.qwenvl.special_tokens.action_token_count", int(args.fast_token_count))


def get_action_shape(cfg: Any) -> tuple[int, int]:
    action_cfg = cfg.framework.action_model
    action_horizon = getattr(cfg.datasets.router_data, "action_horizon", None)
    if action_horizon is None:
        action_horizon = getattr(action_cfg, "action_horizon", None)
    if action_horizon is None:
        action_horizon = int(getattr(action_cfg, "future_action_window_size", 15)) + 1
    return int(action_horizon), int(getattr(action_cfg, "action_dim", 6))


def load_model(cfg: Any, checkpoint: Path, device: str, *, checkpoint_modules: str | None = None):
    model = build_framework(cfg=cfg)
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint))
    else:
        state_dict = torch.load(str(checkpoint), map_location="cpu")

    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "module"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    def maybe_resize_qwen_embeddings(module: torch.nn.Module, weights: dict[str, torch.Tensor]) -> None:
        qwen = getattr(module, "qwen_vl_interface", None)
        if qwen is None and hasattr(module, "processor") and hasattr(module, "model"):
            qwen = module
        if qwen is None:
            return

        embed_key = None
        for candidate in (
            "qwen_vl_interface.model.model.language_model.embed_tokens.weight",
            "model.model.language_model.embed_tokens.weight",
        ):
            if candidate in weights:
                embed_key = candidate
                break
        if embed_key is None:
            return

        target_rows = int(weights[embed_key].shape[0])
        current_rows = int(qwen.model.get_input_embeddings().weight.shape[0])
        if target_rows == current_rows:
            return

        tokenizer_len = len(qwen.processor.tokenizer)
        if target_rows < tokenizer_len:
            raise RuntimeError(
                "Checkpoint Qwen embedding rows are smaller than tokenizer length: "
                f"checkpoint_rows={target_rows}, tokenizer_len={tokenizer_len}"
            )
        print(
            "[Info] Resizing Qwen token embeddings to match checkpoint: "
            f"current_rows={current_rows} checkpoint_rows={target_rows} tokenizer_len={tokenizer_len}"
        )
        qwen.model.resize_token_embeddings(target_rows)

    if checkpoint_modules:
        loaded = []
        for module_path in [part.strip() for part in checkpoint_modules.split(",") if part.strip()]:
            module = model
            for attr in module_path.split("."):
                module = getattr(module, attr)
            prefix = module_path + "."
            sub_state_dict = {
                key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
            }
            if not sub_state_dict:
                raise RuntimeError(f"No checkpoint parameters found for module `{module_path}`")
            maybe_resize_qwen_embeddings(module, sub_state_dict)
            missing, unexpected = module.load_state_dict(sub_state_dict, strict=False)
            print(
                f"[Info] Loaded module `{module_path}` from checkpoint: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
            if missing:
                print("  first missing keys:", missing[:10])
            if unexpected:
                print("  first unexpected keys:", unexpected[:10])
            loaded.append(module_path)
        print(f"[Info] Loaded checkpoint modules: {loaded}")
    else:
        maybe_resize_qwen_embeddings(model, state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Warn] Missing keys when loading checkpoint {checkpoint}: {len(missing)}")
            print("  first missing keys:", missing[:10])
        if unexpected:
            print(f"[Warn] Unexpected keys when loading checkpoint {checkpoint}: {len(unexpected)}")
            print("  first unexpected keys:", unexpected[:10])

    model = model.to(device)
    model.eval()
    return model


def release_unused_flow_action_model(model: Any) -> None:
    action_model = getattr(model, "action_model", None)
    if action_model is None:
        return
    action_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def choose_episodes(router_ds: WallXRouterDataset, args: argparse.Namespace) -> list[int]:
    total_episodes = len(router_ds.dataset.episode_data_index["from"])
    if args.episode_indices:
        episodes = [int(ep) for ep in args.episode_indices]
    elif args.episode_range is not None:
        start_ep, end_ep = int(args.episode_range[0]), int(args.episode_range[1])
        lo, hi = min(start_ep, end_ep), max(start_ep, end_ep)
        episodes = list(range(lo, hi + 1))
    else:
        start = max(0, int(args.start_episode))
        end = min(total_episodes, start + max(1, int(args.num_episodes)))
        episodes = list(range(start, end))
    episodes = [ep for ep in episodes if 0 <= ep < total_episodes]
    if not episodes:
        raise ValueError("No valid episode index selected.")
    return episodes


def collect_eval_indices(router_ds: WallXRouterDataset, episodes: list[int], max_frames: int, stride: int) -> list[int]:
    from_arr = router_ds.dataset.episode_data_index["from"]
    to_arr = router_ds.dataset.episode_data_index["to"]
    indices: list[int] = []
    stride = max(1, int(stride))
    for ep in episodes:
        start = to_int(from_arr[ep])
        end = to_int(to_arr[ep])
        ep_indices = list(range(start, end, stride))
        if max_frames > 0:
            ep_indices = ep_indices[:max_frames]
        indices.extend(ep_indices)
    return indices


def dedupe_effective_indices(router_ds: WallXRouterDataset, indices: list[int]) -> list[int]:
    deduped: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        effective_local_index, _, _ = router_ds._resolve_training_sample(idx)
        if effective_local_index in seen:
            continue
        seen.add(int(effective_local_index))
        deduped.append(int(idx))
    return deduped


def prepare_router_eval_sample(router_ds: WallXRouterDataset, index: int) -> tuple[dict[str, Any] | None, dict[str, Any], int, int]:
    effective_local_index, source_index, raw_sample = router_ds._resolve_training_sample(index)
    router_sample = router_ds._make_router_sample(index)
    return router_sample, raw_sample, int(effective_local_index), int(source_index)


def force_generate_bbox_text(
    model,
    example: dict[str, Any],
    pred_bbox_token: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    qwen_vl_interface = model.qwen_vl_interface
    token_ids = model._single_router_token_ids()
    if token_ids is None:
        raise RuntimeError("Router bbox token is not a single tokenizer id; cannot force bbox generation cleanly.")

    images = [to_pil_preserve(example["image"])]
    train_obs_image_size = model._train_image_size()
    if train_obs_image_size:
        images = resize_images(images, target_size=train_obs_image_size)
    instructions = [example["lang"]]

    qwen_inputs = qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
    forced_inputs = {key: value for key, value in qwen_inputs.items() if key != "labels"}
    input_ids = forced_inputs["input_ids"]
    bbox_route_ids = torch.tensor(
        [[token_ids["bbox"]]],
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    forced_inputs["input_ids"] = torch.cat([input_ids, bbox_route_ids], dim=1)
    if "attention_mask" in forced_inputs:
        forced_inputs["attention_mask"] = torch.cat(
            [forced_inputs["attention_mask"], torch.ones_like(bbox_route_ids)],
            dim=1,
        )

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
        )

    generated = qwen_vl_interface.generate(**forced_inputs, **generation_kwargs)
    gen_ids = generated.sequences if hasattr(generated, "sequences") else generated
    prompt_len = int(qwen_inputs["input_ids"].shape[1])
    new_ids = gen_ids[:, prompt_len:]
    text = qwen_vl_interface.processor.batch_decode(
        new_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    text = text.strip()
    if not text.startswith(pred_bbox_token):
        text = f"{pred_bbox_token}{text}"
    return text


def save_bbox_overlay(
    image_orig: Image.Image,
    gt_bbox_orig: list[float] | None,
    pred_bbox_orig: list[float] | None,
    save_path: Path,
    caption: str,
) -> None:
    img = image_orig.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    if gt_bbox_orig is not None and is_valid_bbox_xyxy(gt_bbox_orig):
        x1, y1, x2, y2 = gt_bbox_orig
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1, max(0, y1 - 14)), "GT", fill=(0, 255, 0))

    if pred_bbox_orig is not None and is_valid_bbox_xyxy(pred_bbox_orig):
        x1, y1, x2, y2 = pred_bbox_orig
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, max(0, y1 - 14)), "Pred", fill=(255, 0, 0))

    caption = caption[:150]
    draw.rectangle([(0, 0), (min(img.width, 900), 28)], fill=(0, 0, 0))
    draw.text((4, 4), caption, fill=(255, 255, 255))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)


def save_action_comparison_figure(
    image_orig: Image.Image,
    gt_action: np.ndarray,
    pred_action: np.ndarray,
    save_path: Path,
    caption: str,
) -> None:
    if gt_action.ndim != 2 or pred_action.ndim != 2:
        return
    t = min(gt_action.shape[0], pred_action.shape[0])
    d = min(gt_action.shape[1], pred_action.shape[1])
    if t <= 0 or d <= 0:
        return

    gt_action = gt_action[:t, :d]
    pred_action = pred_action[:t, :d]

    fig = plt.figure(figsize=(16, 9))
    grid = gridspec.GridSpec(3, 3, figure=fig, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1.0, 1.0, 1.0])

    ax_img = fig.add_subplot(grid[:, 0])
    ax_img.imshow(image_orig)
    ax_img.set_title("Current frame")
    ax_img.axis("off")

    ncols = 2
    nrows = math.ceil(d / ncols)
    xs = list(range(t))
    for dim in range(d):
        row = dim // ncols
        col = dim % ncols
        ax = fig.add_subplot(grid[row, col + 1])
        ax.plot(xs, gt_action[:, dim], color="tab:green", marker="o", linewidth=1.5, markersize=3, label="GT")
        ax.plot(xs, pred_action[:, dim], color="tab:red", marker="o", linewidth=1.5, markersize=3, label="Pred")
        ax.set_title(f"action[{dim}]")
        ax.set_xlabel("Horizon step")
        ax.set_ylabel("Normalized value")
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(loc="best", fontsize=8)

    total_axes = nrows * ncols
    for dim in range(d, total_axes):
        row = dim // ncols
        col = dim % ncols
        ax = fig.add_subplot(grid[row, col + 1])
        ax.axis("off")

    fig.suptitle(caption[:180], fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def summarize_records(records: list[dict[str, Any]], *, checkpoint: str, config_yaml: str, episodes: list[int]) -> dict[str, Any]:
    action_mses = [float(r["action_mse"]) for r in records if r.get("action_mse") is not None]
    bbox_ious = [float(r["bbox_iou"]) for r in records if r.get("bbox_iou") is not None]
    route_matches = [bool(r["route_match"]) for r in records if r.get("route_match") is not None]
    gt_action_records = [r for r in records if r.get("gt_route") == "action"]
    gt_bbox_records = [r for r in records if r.get("gt_route") == "bbox"]
    action_route_matches = [bool(r["route_match"]) for r in gt_action_records if r.get("route_match") is not None]
    bbox_route_matches = [bool(r["route_match"]) for r in gt_bbox_records if r.get("route_match") is not None]
    fast_raw_counts = [int(r["fast_raw_token_count"]) for r in records if r.get("fast_raw_token_count") is not None]
    fast_span_counts = [int(r["fast_span_token_count"]) for r in records if r.get("fast_span_token_count") is not None]
    fast_decode_counts = [int(r["fast_decode_token_count"]) for r in records if r.get("fast_decode_token_count") is not None]
    fast_input_coeff_counts = [
        int(r["fast_input_coeff_count"]) for r in records if r.get("fast_input_coeff_count") is not None
    ]
    fast_decode_errors = [r for r in records if r.get("fast_decode_error")]

    return {
        "checkpoint": checkpoint,
        "config_yaml": config_yaml,
        "episodes": episodes,
        "num_records": len(records),
        "num_action_records": len(gt_action_records),
        "num_bbox_records": len(gt_bbox_records),
        "route_accuracy": float(np.mean(route_matches)) if route_matches else None,
        "route_action_accuracy": float(np.mean(action_route_matches)) if action_route_matches else None,
        "route_bbox_accuracy": float(np.mean(bbox_route_matches)) if bbox_route_matches else None,
        "action_mse_mean": float(np.mean(action_mses)) if action_mses else None,
        "action_mse_median": float(np.median(action_mses)) if action_mses else None,
        "action_metric_count": len(action_mses),
        "bbox_iou_mean": float(np.mean(bbox_ious)) if bbox_ious else None,
        "bbox_iou_median": float(np.median(bbox_ious)) if bbox_ious else None,
        "bbox_iou_count": len(bbox_ious),
        "fast_action_generation_count": len(fast_span_counts),
        "fast_decode_error_count": len(fast_decode_errors),
        "fast_raw_token_count_mean": float(np.mean(fast_raw_counts)) if fast_raw_counts else None,
        "fast_raw_token_count_median": float(np.median(fast_raw_counts)) if fast_raw_counts else None,
        "fast_raw_token_count_min": int(np.min(fast_raw_counts)) if fast_raw_counts else None,
        "fast_raw_token_count_max": int(np.max(fast_raw_counts)) if fast_raw_counts else None,
        "fast_span_token_count_mean": float(np.mean(fast_span_counts)) if fast_span_counts else None,
        "fast_span_token_count_median": float(np.median(fast_span_counts)) if fast_span_counts else None,
        "fast_span_token_count_min": int(np.min(fast_span_counts)) if fast_span_counts else None,
        "fast_span_token_count_max": int(np.max(fast_span_counts)) if fast_span_counts else None,
        "fast_decode_token_count_mean": float(np.mean(fast_decode_counts)) if fast_decode_counts else None,
        "fast_decode_token_count_median": float(np.median(fast_decode_counts)) if fast_decode_counts else None,
        "fast_decode_token_count_min": int(np.min(fast_decode_counts)) if fast_decode_counts else None,
        "fast_decode_token_count_max": int(np.max(fast_decode_counts)) if fast_decode_counts else None,
        "fast_input_coeff_count_mean": float(np.mean(fast_input_coeff_counts)) if fast_input_coeff_counts else None,
        "fast_input_coeff_count_median": float(np.median(fast_input_coeff_counts)) if fast_input_coeff_counts else None,
        "fast_input_coeff_count_min": int(np.min(fast_input_coeff_counts)) if fast_input_coeff_counts else None,
        "fast_input_coeff_count_max": int(np.max(fast_input_coeff_counts)) if fast_input_coeff_counts else None,
    }


def evaluate_one_model(
    *,
    model_alias: str,
    cfg_path: Path,
    checkpoint_input: str,
    checkpoint_path: Path,
    args: argparse.Namespace,
    episodes: list[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = OmegaConf.load(str(cfg_path))
    apply_cli_overrides(cfg, args)
    disable_router_photometric_aug(cfg)

    router_ds = WallXRouterDataset(cfg.datasets.router_data)
    if episodes is None:
        episodes = choose_episodes(router_ds, args)

    eval_indices = collect_eval_indices(router_ds, episodes, args.max_frames_per_episode, args.frame_stride)
    if args.dedupe_effective_indices:
        eval_indices = dedupe_effective_indices(router_ds, eval_indices)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    checkpoint_modules = args.checkpoint_modules
    if checkpoint_modules is None and args.action_decode_mode == "fast":
        checkpoint_modules = "qwen_vl_interface"
    model = load_model(cfg=cfg, checkpoint=checkpoint_path, device=args.device, checkpoint_modules=checkpoint_modules)
    fast_decoder = None
    if args.action_decode_mode == "fast":
        action_horizon, action_dim = get_action_shape(cfg)
        fast_debug_dir = None
        if args.fast_debug_dir:
            fast_debug_dir = str(Path(args.fast_debug_dir).expanduser() / model_alias)
        fast_decoder = FastActionDecoder(
            model,
            cfg,
            pred_action_token=str(router_ds.pred_action_token),
            tokenizer_path=args.fast_tokenizer_path,
            token_prefix=args.fast_token_prefix,
            token_count=args.fast_token_count,
            action_horizon=action_horizon,
            action_dim=action_dim,
            max_new_tokens=args.fast_max_new_tokens,
            decode_attempts=args.fast_decode_attempts,
            retry_do_sample=args.fast_retry_do_sample,
            retry_temperature=args.fast_retry_temperature,
            retry_top_p=args.fast_retry_top_p,
            debug_dir=fast_debug_dir,
        )
        release_unused_flow_action_model(model)
    pred_action_token = router_ds.pred_action_token
    pred_bbox_token = router_ds.pred_bbox_token
    action_route_format = str(getattr(router_ds, "action_route_format", "token"))
    subtask_start_token = str(getattr(router_ds, "subtask_start_token", "<|subtask|>"))
    subtask_end_token = str(getattr(router_ds, "subtask_end_token", "<|end_subtask|>"))
    dst_w, dst_h = [int(x) for x in router_ds.common.image_size]

    route_kwargs = {
        "max_new_tokens": int(args.route_max_new_tokens),
        "do_sample": bool(args.route_do_sample),
        "temperature": float(args.route_temperature),
        "top_p": float(args.route_top_p),
        "route_mode": str(args.route_mode),
        "continue_bbox": False,
        "continue_action": action_route_format == "route_subtask",
        "route_confidence_threshold": args.route_confidence_threshold,
    }
    if action_route_format == "route_subtask":
        stop_ids = model.qwen_vl_interface.processor.tokenizer(
            subtask_end_token,
            add_special_tokens=False,
        ).input_ids
        if stop_ids:
            route_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [TokenSequenceStoppingCriteria([stop_ids])]
            )

    records: list[dict[str, Any]] = []
    model_output_dir = Path(args.output_dir) / model_alias
    action_vis_dir = model_output_dir / "detail_action_vis"
    bbox_vis_dir = model_output_dir / "detail_bbox_vis"
    action_vis_saved = 0
    bbox_vis_saved = 0

    iterator = tqdm(eval_indices, desc=f"Eval {model_alias}", leave=False)
    for requested_index in iterator:
        router_sample, raw_sample, effective_index, source_index = prepare_router_eval_sample(router_ds, requested_index)
        if router_sample is None:
            continue

        episode_index = to_int(raw_sample["episode_index"])
        frame_index = to_int(raw_sample["frame_index"])
        task_text = str(raw_sample.get("task", ""))
        gt_signal = str(router_sample.get("pred_signal", "")).strip()
        gt_route = "bbox" if gt_signal == router_ds.pred_signal_stop else "action"
        raw_img = tensor_image_to_pil_original(raw_sample["video.front"])
        src_h = int(raw_sample["video.front"].shape[-2])
        src_w = int(raw_sample["video.front"].shape[-1])

        route_output = model.predict_route(
            examples=[router_sample],
            **route_kwargs,
        )
        route_info = route_output["routes"][0]
        pred_route = str(route_info.get("route", "unknown"))
        route_match = bool(pred_route == gt_route)

        gt_bbox_orig = np.asarray(raw_sample.get("bbox", []), dtype=np.float32).reshape(-1).tolist()
        if not is_valid_bbox_xyxy(gt_bbox_orig):
            gt_bbox_orig = None

        pred_action = None
        action_mse = None
        action_step0_mse = None
        action_visual_path = None
        fast_raw_token_count = None
        fast_span_token_count = None
        fast_decode_token_count = None
        fast_input_coeff_count = None
        fast_decode_coeff_count = None
        fast_expected_coeff_count = None
        fast_trimmed = None
        fast_decode_error = None
        fast_decode_attempt = None
        fast_generated_text = None
        fast_decode_info = None
        fast_debug_context = None
        pred_subtask_text = None
        action_route_solution = pred_action_token
        forced_bbox_text = None
        pred_bbox_orig = None
        bbox_iou = None
        bbox_visual_path = None

        if pred_route == "action":
            action_route_solution, pred_subtask_text = parse_action_route_solution(
                str(route_info.get("generated_text", "")),
                pred_action_token=pred_action_token,
                action_route_format=action_route_format,
                subtask_start_token=subtask_start_token,
                subtask_end_token=subtask_end_token,
            )
            if args.action_decode_mode == "fast":
                assert fast_decoder is not None
                try:
                    action_output = fast_decoder.predict(
                        examples=[router_sample],
                        route_prefix=action_route_solution,
                        debug_context={
                            "model_alias": model_alias,
                            "requested_index": int(requested_index),
                            "effective_index": int(effective_index),
                            "source_index": int(source_index),
                            "episode_index": int(episode_index),
                            "frame_index": int(frame_index),
                            "gt_route": gt_route,
                            "pred_route": pred_route,
                            "route_match": route_match,
                        },
                    )
                    raw_ids = action_output.get("raw_fast_token_ids", [[]])[0]
                    decode_ids = action_output.get("fast_token_ids", [[]])[0]
                    decode_infos = action_output.get("decode_infos", [{}])
                    fast_decode_info = decode_infos[0] if decode_infos else {}
                    fast_raw_token_count = int(len(raw_ids))
                    fast_span_token_count = int(fast_decode_info.get("input_token_count", len(decode_ids)))
                    fast_decode_token_count = int(len(decode_ids))
                    fast_input_coeff_count = fast_decode_info.get("input_coeff_count")
                    fast_decode_coeff_count = fast_decode_info.get("decode_coeff_count")
                    fast_expected_coeff_count = fast_decode_info.get("expected_coeff_count")
                    fast_trimmed = bool(fast_decode_info.get("trimmed", False))
                    fast_decode_attempt = int(action_output.get("attempt", 1))
                    fast_generated_text = str(action_output.get("generated_text", [""])[0])
                    pred_action = np.asarray(action_output["normalized_actions"][0], dtype=np.float32)
                except FastActionDecodeError as exc:
                    payload = exc.debug_payload
                    raw_ids = payload.get("raw_action_fast_token_ids", [[]])[0]
                    span_ids = payload.get("span_action_fast_token_ids", [[]])[0]
                    fast_raw_token_count = int(len(raw_ids))
                    fast_span_token_count = int(len(span_ids))
                    span_coeff_counts = payload.get("span_action_coeff_counts") or []
                    if span_coeff_counts:
                        fast_input_coeff_count = int(span_coeff_counts[0])
                    fast_expected_coeff_count = payload.get("expected_coeff_count")
                    fast_generated_text = str((payload.get("generated_text") or [""])[0])
                    fast_decode_error = str(exc)
                    fast_debug_context = payload
                except Exception as exc:
                    fast_decode_error = repr(exc)
            else:
                action_output = model.predict_action_with_route_token(
                    examples=[router_sample],
                    route_token=action_route_solution,
                )
                pred_action = np.asarray(action_output["normalized_actions"][0], dtype=np.float32)

            if pred_action is not None and gt_route == "action" and route_match:
                gt_action = np.asarray(router_sample["action"], dtype=np.float32)
                t = min(pred_action.shape[0], gt_action.shape[0])
                d = min(pred_action.shape[1], gt_action.shape[1])
                pred_action = pred_action[:t, :d]
                gt_action = gt_action[:t, :d]
                if pred_action.size > 0:
                    diff = pred_action - gt_action
                    action_mse = float(np.mean(diff ** 2))
                    action_step0_mse = float(np.mean((pred_action[0] - gt_action[0]) ** 2)) if t > 0 else None

            if (
                pred_action is not None
                and gt_route == "action"
                and route_match
                and args.mode == "detail"
                and should_save_visual(action_vis_saved, args.max_action_vis_images)
            ):
                caption = (
                    f"model={model_alias} ep={episode_index} frame={frame_index} "
                    f"requested={requested_index} effective={effective_index} "
                    f"route_pred={pred_route} gt_route={gt_route} mse={action_mse}"
                )
                action_visual_path = action_vis_dir / f"episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
                save_action_comparison_figure(
                    image_orig=raw_img,
                    gt_action=gt_action,
                    pred_action=pred_action,
                    save_path=action_visual_path,
                    caption=caption,
                )
                if action_visual_path.exists():
                    action_vis_saved += 1

        elif gt_route == "bbox" and route_match:
            forced_bbox_text = force_generate_bbox_text(
                model,
                router_sample,
                pred_bbox_token=pred_bbox_token,
                max_new_tokens=args.bbox_max_new_tokens,
                do_sample=args.bbox_do_sample,
                temperature=args.bbox_temperature,
                top_p=args.bbox_top_p,
            )
            pred_bbox_resized = parse_bbox_from_text(forced_bbox_text)
            if pred_bbox_resized is not None:
                pred_bbox_orig = resized_to_original_bbox(pred_bbox_resized, src_w=src_w, src_h=src_h, dst_w=dst_w, dst_h=dst_h)
                pred_bbox_orig = clip_bbox_xyxy(pred_bbox_orig, w=src_w, h=src_h)
                if not is_valid_bbox_xyxy(pred_bbox_orig):
                    pred_bbox_orig = None

            if gt_bbox_orig is not None and pred_bbox_orig is not None:
                bbox_iou = bbox_iou_xyxy(gt_bbox_orig, pred_bbox_orig)

            if args.mode == "detail" and should_save_visual(bbox_vis_saved, args.max_bbox_vis_images):
                caption = (
                    f"model={model_alias} ep={episode_index} frame={frame_index} "
                    f"requested={requested_index} effective={effective_index} "
                    f"route_pred={pred_route} gt_route={gt_route} iou={bbox_iou}"
                )
                bbox_visual_path = bbox_vis_dir / f"episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
                save_bbox_overlay(
                    image_orig=raw_img,
                    gt_bbox_orig=gt_bbox_orig,
                    pred_bbox_orig=pred_bbox_orig,
                    save_path=bbox_visual_path,
                    caption=caption,
                )
                if bbox_visual_path.exists():
                    bbox_vis_saved += 1

        elif args.mode == "detail":
            # Route mismatch: keep branch-specific metrics empty. The record still
            # captures route correctness so aggregate MSE / IoU remain conditional
            # on correct router decisions only.
            pass

        records.append(
            {
                "model_alias": model_alias,
                "requested_index": int(requested_index),
                "effective_index": int(effective_index),
                "source_index": int(source_index),
                "episode_index": int(episode_index),
                "frame_index": int(frame_index),
                "task": task_text,
                "gt_signal": gt_signal,
                "gt_route": gt_route,
                "pred_route": pred_route,
                "route_match": route_match,
                "route_generated_text": route_info.get("generated_text"),
                "action_route_solution": action_route_solution,
                "gt_subtask_text": router_sample.get("subtask_text"),
                "pred_subtask_text": pred_subtask_text,
                "route_confidence": route_info.get("route_confidence"),
                "route_action_prob": route_info.get("route_action_prob"),
                "route_bbox_prob": route_info.get("route_bbox_prob"),
                "action_mse": action_mse,
                "action_step0_mse": action_step0_mse,
                "pred_action": pred_action.tolist() if isinstance(pred_action, np.ndarray) else None,
                "gt_action": json_safe(router_sample.get("action")) if gt_route == "action" else None,
                "action_decode_mode": args.action_decode_mode,
                "fast_raw_token_count": fast_raw_token_count,
                "fast_span_token_count": fast_span_token_count,
                "fast_decode_token_count": fast_decode_token_count,
                "fast_input_coeff_count": fast_input_coeff_count,
                "fast_decode_coeff_count": fast_decode_coeff_count,
                "fast_expected_coeff_count": fast_expected_coeff_count,
                "fast_trimmed": fast_trimmed,
                "fast_decode_error": fast_decode_error,
                "fast_decode_attempt": fast_decode_attempt,
                "fast_generated_text": fast_generated_text,
                "fast_decode_info": fast_decode_info,
                "fast_debug_context": fast_debug_context,
                "forced_bbox_text": forced_bbox_text,
                "gt_bbox_orig": gt_bbox_orig,
                "pred_bbox_orig": pred_bbox_orig,
                "bbox_iou": bbox_iou,
                "action_visual_path": str(action_visual_path) if action_visual_path is not None else None,
                "bbox_visual_path": str(bbox_visual_path) if bbox_visual_path is not None else None,
            }
        )

    summary = summarize_records(
        records,
        checkpoint=str(checkpoint_path),
        config_yaml=str(cfg_path),
        episodes=episodes,
    )
    summary["model_alias"] = model_alias
    summary["requested_checkpoint"] = checkpoint_input
    summary["num_selected_indices"] = len(eval_indices)
    summary["action_vis_saved_count"] = int(action_vis_saved)
    summary["bbox_vis_saved_count"] = int(bbox_vis_saved)
    summary["action_decode_mode"] = args.action_decode_mode

    if args.save_per_frame_records:
        write_json(model_output_dir / "records.json", records)
        write_csv(model_output_dir / "records.csv", records)
    write_json(model_output_dir / "summary.json", summary)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, records


def main() -> None:
    args = parse_args()
    PartialState()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_input = str(args.checkpoint)
    checkpoint_path = resolve_checkpoint_path(checkpoint_input)
    cfg_path = Path(args.config_yaml)

    reference_cfg = OmegaConf.load(str(cfg_path))
    apply_cli_overrides(reference_cfg, args)
    disable_router_photometric_aug(reference_cfg)
    reference_ds = WallXRouterDataset(reference_cfg.datasets.router_data)
    episodes = choose_episodes(reference_ds, args)

    model_alias = model_alias_from_input(checkpoint_input, checkpoint_path)

    print(f"[Info] Mode: {args.mode}")
    print(f"[Info] Episodes: {episodes}")
    print(f"[Info] Model alias: {model_alias}")
    print(f"[Info] Checkpoint: {checkpoint_path}")
    print(f"[Info] Config: {cfg_path}")

    summary, _ = evaluate_one_model(
        model_alias=model_alias,
        cfg_path=cfg_path,
        checkpoint_input=checkpoint_input,
        checkpoint_path=checkpoint_path,
        args=args,
        episodes=episodes,
    )
    print(
        f"[Done] {model_alias}: "
        f"action_mse_mean={summary.get('action_mse_mean')} "
        f"bbox_iou_mean={summary.get('bbox_iou_mean')} "
        f"route_accuracy={summary.get('route_accuracy')} "
        f"fast_span_token_count_mean={summary.get('fast_span_token_count_mean')} "
        f"fast_decode_error_count={summary.get('fast_decode_error_count')}"
    )


if __name__ == "__main__":
    main()
