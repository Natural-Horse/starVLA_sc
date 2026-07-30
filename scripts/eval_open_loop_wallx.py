#!/usr/bin/env python3
"""Open-loop evaluation for WallX cotrain tasks on StarVLA.

This script evaluates three branches on selected frames:
1) VLM signal prediction (pred_action vs bbox).
2) VLA action generation gated by signal prediction.
3) Legacy VLM bbox prediction task on the same frame.

Outputs:
- Printed per-sample signal/action/bbox predictions.
- Action error curve figure.
- Bbox overlay visualizations on original-resolution images
  (green: ground truth, red: prediction).
- JSON summary + per-sample records.

CUDA_VISIBLE_DEVICES=2 /beijing-c/workspace/hxj/miniconda3/envs/starvla/bin/python \
/beijing-c/wallx_workspace/starVLA/scripts/eval_open_loop_wallx.py \
  --config_yaml /beijing-c/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi.yaml \
  --checkpoint /beijing-c/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_cotrain_2/checkpoints/steps_500_pytorch_model.pt \
  --output_dir /beijing-c/wallx_workspace/starVLA/results/open_loop_eval_ep10_11 \
  --episode_indices 10 11 \
  --max_frames_per_episode 30 \
  --frame_stride 1 \
  --device cuda:0

"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from omegaconf import OmegaConf

# Make `import starVLA...` work when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.dataloader.wallx_cotrain_datasets import WallXVlaDataset, WallXVlmBboxDataset, WallXVlmSignalDataset
from starVLA.model.framework import build_framework


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-loop signal-gated action+bbox evaluation on WallX training episodes.")
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="/beijing-c/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi.yaml",
        help="Training config yaml used to build model and datasets.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint file (.pt/.safetensors) or checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation figures and json records.",
    )
    parser.add_argument(
        "--episode_indices",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to evaluate, e.g. --episode_indices 10 11 12",
    )
    parser.add_argument(
        "--episode_range",
        type=int,
        nargs=2,
        default=None,
        metavar=("START_EP", "END_EP"),
        help="Inclusive episode range, e.g. --episode_range 10 20 (means episodes 10..20).",
    )
    parser.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="Start episode index when --episode_indices is not provided.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=3,
        help="How many episodes to evaluate when --episode_indices is not provided.",
    )
    parser.add_argument(
        "--max_frames_per_episode",
        type=int,
        default=20,
        help="Maximum number of frames sampled per episode.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Sample one frame every N frames inside each episode.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max new tokens for VLM bbox generation.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling for bbox generation (default: greedy).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature when --do_sample is enabled.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p when --do_sample is enabled.",
    )
    parser.add_argument(
        "--signal_max_new_tokens",
        type=int,
        default=16,
        help="Max new tokens for VLM signal generation.",
    )
    parser.add_argument(
        "--signal_do_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sampling for signal generation (default: deterministic).",
    )
    parser.add_argument(
        "--signal_temperature",
        type=float,
        default=0.2,
        help="Signal sampling temperature when --signal_do_sample is enabled.",
    )
    parser.add_argument(
        "--signal_top_p",
        type=float,
        default=0.95,
        help="Signal top-p when --signal_do_sample is enabled.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device, e.g. cuda / cuda:0 / cpu",
    )
    parser.add_argument(
        "--bbox_only_valid_gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, only evaluate bbox on frames with valid GT bbox. Use --no-bbox_only_valid_gt to force prediction on invalid-GT frames.",
    )
    parser.add_argument(
        "--max_action_vis_images",
        type=int,
        default=200,
        help="Maximum number of per-step action comparison images to save. <=0 means unlimited.",
    )
    parser.add_argument(
        "--max_bbox_vis_images",
        type=int,
        default=200,
        help="Maximum number of bbox overlay images to save. <=0 means unlimited.",
    )
    return parser.parse_args()


def _to_int(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def tensor_image_to_pil_original(image_chw: torch.Tensor) -> Image.Image:
    arr = image_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr_u8 = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr_u8)


def _to_bool(x: Any) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(x.item())
    return bool(x)


def parse_task_string(task_str: str) -> tuple[str, str, str]:
    m = re.search(r"(.*)Catch:\s*(.*)\.\s*Put:\s*(.*)", task_str)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return task_str.strip(), "", ""


def build_vlm_infer_sample_from_raw(vlm_ds: WallXVlmBboxDataset, index: int, raw: dict[str, Any]) -> dict[str, Any]:
    front = torch.as_tensor(raw["video.front"])
    src = front.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    src_u8 = (src * 255.0).astype(np.uint8)

    dst_w, dst_h = vlm_ds.common.image_size
    current_image = Image.fromarray(src_u8).resize((dst_w, dst_h), Image.BILINEAR)
    history_images = vlm_ds._collect_history_images(index, raw)
    images = [current_image] + history_images

    task_str = str(raw.get("task", ""))
    instruction, catch_target, put_target = parse_task_string(task_str)
    grasp = _to_bool(raw.get("grasp", False))
    target_name = put_target if grasp else catch_target
    if not target_name:
        target_name = "target object"

    lang = vlm_ds.bbox_prompt_template.format(
        instruction=instruction if instruction else task_str,
        target_name=target_name,
    )
    return {"image": images, "lang": lang}


def normalize_target_name(target_name: str) -> str:
    text = str(target_name).strip()
    while text.endswith("."):
        text = text[:-1].rstrip()
    return text


def build_vlm_signal_infer_sample_from_raw(
    signal_ds: WallXVlmSignalDataset,
    raw: dict[str, Any],
) -> dict[str, Any]:
    front = torch.as_tensor(raw["video.front"])
    src = front.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    src_u8 = (src * 255.0).astype(np.uint8)

    dst_w, dst_h = signal_ds.common.image_size
    current_image = Image.fromarray(src_u8).resize((dst_w, dst_h), Image.BILINEAR)

    task_str = str(raw.get("task", ""))
    _, catch_target, put_target = parse_task_string(task_str)
    grasp = _to_bool(raw.get("grasp", False))

    target_name = put_target if grasp else catch_target
    target_name = normalize_target_name(target_name)
    if not target_name:
        target_name = "target object"

    operation = "put" if grasp else "grasp"
    lang = signal_ds.signal_prompt_template.format(
        target_name=target_name,
        operation=operation,
        pred_action_token=signal_ds.pred_action_token,
        instruction="",
    )

    return {
        "image": [current_image],
        "lang": lang,
        "target_name": target_name,
        "operation": operation,
    }


def parse_signal_response_type(pred_text: str, pred_action_token: str) -> str:
    if parse_bbox_from_text(pred_text) is not None:
        return "bbox"

    lower = pred_text.lower()
    if (pred_action_token in pred_text) or ("pred_action" in lower):
        return "pred_action"

    # Conservative fallback for control: keep moving by predicting action branch.
    return "pred_action"


def is_valid_bbox_xyxy(bbox: list[float] | tuple[float, float, float, float] | np.ndarray | None) -> bool:
    if bbox is None:
        return False
    b = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if b.size != 4:
        return False
    x1, y1, x2, y2 = b.tolist()
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


POINT_PATTERN = re.compile(
    r"<point>\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*</point>",
    re.IGNORECASE,
)


def parse_bbox_from_text(text: str) -> list[float] | None:
    m = POINT_PATTERN.search(text)
    if not m:
        return None
    vals = [float(m.group(i)) for i in range(1, 5)]
    return vals


def resized_to_original_bbox(bbox_resized: list[float], src_w: int, src_h: int, dst_w: int, dst_h: int) -> list[float]:
    sx = float(src_w) / float(dst_w)
    sy = float(src_h) / float(dst_h)
    x1, y1, x2, y2 = bbox_resized
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def draw_bbox_overlay(
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

    # top-left caption (kept compact to avoid covering too much image)
    draw.rectangle([(0, 0), (min(img.width, 720), 24)], fill=(0, 0, 0))
    draw.text((4, 4), caption[:120], fill=(255, 255, 255))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)


def should_save_visual(saved_count: int, max_to_save: int) -> bool:
    return max_to_save <= 0 or saved_count < max_to_save


def save_action_horizon_compare(
    pred_action: np.ndarray,
    gt_action: np.ndarray,
    save_path: Path,
    caption: str,
) -> None:
    if pred_action.ndim != 2 or gt_action.ndim != 2:
        return

    t = min(pred_action.shape[0], gt_action.shape[0])
    d = min(pred_action.shape[1], gt_action.shape[1])
    if t <= 0 or d <= 0:
        return

    xs = list(range(t))
    ncols = 3 if d >= 3 else d
    nrows = (d + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
    for j in range(d):
        ax = axes[j // ncols][j % ncols]
        ax.plot(xs, gt_action[:t, j], color="tab:green", marker="o", linewidth=1.5, markersize=3, label="GT")
        ax.plot(xs, pred_action[:t, j], color="tab:red", marker="o", linewidth=1.5, markersize=3, label="Pred")
        ax.set_title(f"action[{j}]")
        ax.set_xlabel("Horizon step")
        ax.set_ylabel("Normalized value")
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.legend(loc="best", fontsize=8)

    total_axes = nrows * ncols
    for j in range(d, total_axes):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(caption[:140], fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def write_text_report(report_path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines: list[str] = []

    lines.append("Open-loop Evaluation Report")
    lines.append("=" * 120)
    lines.append(f"checkpoint: {summary.get('checkpoint')}")
    lines.append(f"config_yaml: {summary.get('config_yaml')}")
    lines.append(f"episodes: {summary.get('episodes')}")
    lines.append(f"episode_range: {summary.get('episode_range')}")
    lines.append(f"num_selected_frames: {summary.get('num_selected_frames')}")
    lines.append(f"num_evaluated_records: {summary.get('num_evaluated_records')}")
    lines.append(f"action_mse_mean: {summary.get('action_mse_mean')}")
    lines.append(f"action_mse_median: {summary.get('action_mse_median')}")
    lines.append(f"num_action_evaluated_records: {summary.get('num_action_evaluated_records')}")
    lines.append(f"num_action_skipped_by_signal: {summary.get('num_action_skipped_by_signal')}")
    lines.append(f"signal_type_accuracy: {summary.get('signal_type_accuracy')}")
    lines.append(f"signal_bbox_iou_mean: {summary.get('signal_bbox_iou_mean')}")
    lines.append(f"bbox_iou_mean: {summary.get('bbox_iou_mean')}")
    lines.append(f"bbox_iou_median: {summary.get('bbox_iou_median')}")
    lines.append(f"bbox_eval_count: {summary.get('bbox_eval_count')}")
    lines.append(f"max_action_vis_images: {summary.get('max_action_vis_images')}")
    lines.append(f"action_vis_saved_count: {summary.get('action_vis_saved_count')}")
    lines.append(f"max_bbox_vis_images: {summary.get('max_bbox_vis_images')}")
    lines.append(f"bbox_vis_saved_count: {summary.get('bbox_vis_saved_count')}")
    lines.append("=" * 120)

    total = len(records)
    for i, r in enumerate(records, start=1):
        lines.append(
            f"[{i}/{total}] ep={r['episode_index']} frame={r['frame_index']} idx={r['dataset_index']} "
            f"task={r.get('task', '')}"
        )
        lines.append(
            f"  signal_pred_type={r.get('signal_pred_type')} signal_gt_type={r.get('signal_gt_type')} "
            f"signal_type_match={r.get('signal_type_match')} signal_bbox_iou={r.get('signal_bbox_iou')}"
        )
        lines.append(
            f"  action_mse={r.get('action_mse')} "
            f"action_step0_mse={r.get('action_step0_mse')} "
            f"action_train_style_score={r.get('action_train_style_score')} "
            f"action_eval_skipped_by_signal={r.get('action_eval_skipped_by_signal')}"
        )
        lines.append(f"  action_pred={json.dumps(r.get('pred_action', []), ensure_ascii=False)}")
        lines.append(f"  action_gt={json.dumps(r.get('gt_action', []), ensure_ascii=False)}")
        lines.append(f"  action_skip_reason={r.get('action_skip_reason')}")
        lines.append(f"  action_vis={r.get('action_visual_path')}")
        lines.append(f"  action_vis_skip_reason={r.get('action_visual_skip_reason')}")
        lines.append(f"  signal_pred_text={r.get('signal_pred_text')}")
        lines.append(f"  bbox_pred_text={r.get('bbox_pred_text')}")
        lines.append(f"  bbox_gt_orig={r.get('bbox_gt_orig')}")
        lines.append(f"  bbox_pred_orig={r.get('bbox_pred_orig')}")
        lines.append(f"  bbox_iou={r.get('bbox_iou')}")
        lines.append(f"  bbox_vis={r.get('bbox_visual_path')}")
        lines.append(f"  bbox_vis_skip_reason={r.get('bbox_visual_skip_reason')}")
        lines.append("-" * 120)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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

    # Prefer files with step number; fallback to latest mtime.
    def score(p: Path) -> tuple[int, float]:
        m = re.search(r"steps_(\d+)", p.name)
        step = int(m.group(1)) if m else -1
        return (step, p.stat().st_mtime)

    candidates.sort(key=score)
    return candidates[-1]


def load_model(cfg, checkpoint: Path, device: str):
    model = build_framework(cfg=cfg)

    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint))
    else:
        state_dict = torch.load(str(checkpoint), map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warn] Missing keys when loading checkpoint: {len(missing)}")
        print("  first missing keys:", missing[:10])
    if unexpected:
        print(f"[Warn] Unexpected keys when loading checkpoint: {len(unexpected)}")
        print("  first unexpected keys:", unexpected[:10])

    model = model.to(device)
    model.eval()
    return model


def choose_episodes(vla_ds: WallXVlaDataset, args: argparse.Namespace) -> list[int]:
    ep_count = len(vla_ds.dataset.episode_data_index["from"])
    if args.episode_indices:
        episodes = args.episode_indices
    elif args.episode_range is not None:
        start_ep, end_ep = int(args.episode_range[0]), int(args.episode_range[1])
        lo, hi = min(start_ep, end_ep), max(start_ep, end_ep)
        episodes = list(range(lo, hi + 1))
    else:
        start = max(0, args.start_episode)
        end = min(ep_count, start + max(1, args.num_episodes))
        episodes = list(range(start, end))

    episodes = [ep for ep in episodes if 0 <= ep < ep_count]
    if not episodes:
        raise ValueError("No valid episode index selected.")
    return episodes


def collect_eval_indices(vla_ds: WallXVlaDataset, episodes: list[int], max_frames: int, stride: int) -> list[int]:
    idxs: list[int] = []
    from_arr = vla_ds.dataset.episode_data_index["from"]
    to_arr = vla_ds.dataset.episode_data_index["to"]

    for ep in episodes:
        start = _to_int(from_arr[ep])
        end = _to_int(to_arr[ep])
        ep_indices = list(range(start, end, max(1, stride)))
        if max_frames > 0:
            ep_indices = ep_indices[:max_frames]
        idxs.extend(ep_indices)
    return idxs


def decode_vlm_generation(qwen_vl_interface, qwen_inputs: dict[str, torch.Tensor], gen_kwargs: dict[str, Any]) -> str:
    generated = qwen_vl_interface.generate(**qwen_inputs, **gen_kwargs)
    gen_ids = generated.sequences if hasattr(generated, "sequences") else generated

    # batch size = 1 in this script.
    prompt_len = int(qwen_inputs["input_ids"].shape[1])
    new_tokens = gen_ids[:, prompt_len:]
    text = qwen_vl_interface.processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def main() -> None:
    args = parse_args()

    np.set_printoptions(precision=4, suppress=True, linewidth=180)

    cfg = OmegaConf.load(args.config_yaml)
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Using checkpoint: {checkpoint}")
    print(f"[Info] Output directory: {output_dir}")

    print("[Info] Building datasets...")
    vla_ds = WallXVlaDataset(cfg.datasets.vla_data)
    vlm_ds = WallXVlmBboxDataset(cfg.datasets.vlm_data)
    signal_ds = WallXVlmSignalDataset(cfg.datasets.vlm_signal_data)

    episodes = choose_episodes(vla_ds, args)
    eval_indices = collect_eval_indices(vla_ds, episodes, args.max_frames_per_episode, args.frame_stride)

    print(f"[Info] Selected episodes: {episodes}")
    print(f"[Info] Total selected frames: {len(eval_indices)}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    print("[Info] Loading model...")
    model = load_model(cfg=cfg, checkpoint=checkpoint, device=device)
    qwen_vl_interface = model.qwen_vl_interface

    dst_w, dst_h = [int(x) for x in cfg.datasets.vlm_data.image_size]

    generation_kwargs_bbox = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
    }
    if args.do_sample:
        generation_kwargs_bbox["temperature"] = float(args.temperature)
        generation_kwargs_bbox["top_p"] = float(args.top_p)

    generation_kwargs_signal = {
        "max_new_tokens": int(args.signal_max_new_tokens),
        "do_sample": bool(args.signal_do_sample),
    }
    if args.signal_do_sample:
        generation_kwargs_signal["temperature"] = float(args.signal_temperature)
        generation_kwargs_signal["top_p"] = float(args.signal_top_p)

    records: list[dict[str, Any]] = []
    action_vis_saved = 0
    bbox_vis_saved = 0
    action_vis_dir = output_dir / "action_horizon_vis"
    action_vis_dir.mkdir(parents=True, exist_ok=True)

    print("[Info] Starting open-loop evaluation...")
    signal_dst_w, signal_dst_h = [int(x) for x in signal_ds.common.image_size]

    with torch.inference_mode():
        for data_idx in eval_indices:
            # Raw sample for current frame metadata / GT.
            raw = vlm_ds.dataset[data_idx]
            episode_index = _to_int(raw["episode_index"])
            frame_index = _to_int(raw["frame_index"])
            task_text = str(raw.get("task", ""))
            src_h = int(raw["video.front"].shape[-2])
            src_w = int(raw["video.front"].shape[-1])
            raw_img = tensor_image_to_pil_original(raw["video.front"])
            gt_bbox_orig = np.asarray(raw.get("bbox", []), dtype=np.float32).reshape(-1).tolist()
            gt_bbox_orig = gt_bbox_orig if is_valid_bbox_xyxy(gt_bbox_orig) else None

            # -----------------------------------------------------------------
            # 1) Signal branch (same prompt semantics as training)
            # -----------------------------------------------------------------
            signal_eval = build_vlm_signal_infer_sample_from_raw(signal_ds, raw)
            qwen_signal_inputs = qwen_vl_interface.build_qwenvl_inputs(
                images=[signal_eval["image"]],
                instructions=[signal_eval["lang"]],
            )
            signal_pred_text = decode_vlm_generation(
                qwen_vl_interface=qwen_vl_interface,
                qwen_inputs=qwen_signal_inputs,
                gen_kwargs=generation_kwargs_signal,
            )
            signal_response_type = parse_signal_response_type(signal_pred_text, signal_ds.pred_action_token)

            signal_pred_bbox_resized = parse_bbox_from_text(signal_pred_text)
            signal_pred_bbox_orig = None
            if signal_pred_bbox_resized is not None:
                signal_pred_bbox_orig = resized_to_original_bbox(
                    signal_pred_bbox_resized,
                    src_w=src_w,
                    src_h=src_h,
                    dst_w=signal_dst_w,
                    dst_h=signal_dst_h,
                )
                signal_pred_bbox_orig = clip_bbox_xyxy(signal_pred_bbox_orig, w=src_w, h=src_h)
                if not is_valid_bbox_xyxy(signal_pred_bbox_orig):
                    signal_pred_bbox_orig = None

            signal_gt_raw = str(raw.get("pred_signal", "")).strip()
            if signal_gt_raw in {signal_ds.pred_signal_pred_action, signal_ds.pred_action_token}:
                signal_gt_type = "pred_action"
            elif signal_gt_raw == signal_ds.pred_signal_stop:
                signal_gt_type = "bbox"
            else:
                signal_gt_type = None

            signal_type_match = None
            if signal_gt_type is not None:
                signal_type_match = bool(signal_response_type == signal_gt_type)

            signal_bbox_iou = None
            if signal_gt_type == "bbox" and signal_pred_bbox_orig is not None and gt_bbox_orig is not None:
                signal_bbox_iou = bbox_iou_xyxy(gt_bbox_orig, signal_pred_bbox_orig)

            # -----------------------------------------------------------------
            # 2) Action branch (gated by signal prediction)
            # -----------------------------------------------------------------
            action_eval_skipped_by_signal = signal_response_type == "bbox"
            action_skip_reason = None

            pred_action = np.zeros((0, 0), dtype=np.float32)
            gt_action = np.zeros((0, 0), dtype=np.float32)
            pred_action_step0 = np.zeros((0,), dtype=np.float32)
            gt_action_step0 = np.zeros((0,), dtype=np.float32)
            action_mse = None
            train_style_score = None
            action_step0_mse = None
            t = 0
            d = 0

            action_visual_path = None
            action_visual_skip_reason = None

            if not action_eval_skipped_by_signal:
                vla_sample = vla_ds[data_idx]
                action_out = model.predict_action(examples=[vla_sample])
                pred_action = np.asarray(action_out["normalized_actions"][0], dtype=np.float32)
                gt_action = np.asarray(vla_sample["action"], dtype=np.float32)

                t = min(pred_action.shape[0], gt_action.shape[0])
                d = min(pred_action.shape[1], gt_action.shape[1])
                pred_action = pred_action[:t, :d]
                gt_action = gt_action[:t, :d]

                action_diff = pred_action - gt_action
                action_mse = float(np.mean(action_diff ** 2)) if action_diff.size > 0 else None
                train_style_score = (
                    float(np.linalg.norm(action_diff) / max(np.prod(action_diff.shape), 1))
                    if action_diff.size > 0
                    else None
                )

                pred_action_step0 = pred_action[0] if t > 0 else np.zeros((d,), dtype=np.float32)
                gt_action_step0 = gt_action[0] if t > 0 else np.zeros((d,), dtype=np.float32)
                action_step0_diff = pred_action_step0 - gt_action_step0
                action_step0_mse = float(np.mean(action_step0_diff ** 2)) if d > 0 else None

                if should_save_visual(action_vis_saved, args.max_action_vis_images):
                    action_ep_dir = action_vis_dir / f"episode_{episode_index:06d}"
                    action_visual_path = action_ep_dir / f"frame_{frame_index:06d}_idx_{data_idx:06d}.png"
                    action_caption = (
                        f"ep={episode_index} frame={frame_index} idx={data_idx} "
                        f"horizon={t} dim={d} action_mse={action_mse if action_mse is not None else 'NA'}"
                    )
                    save_action_horizon_compare(
                        pred_action=pred_action,
                        gt_action=gt_action,
                        save_path=action_visual_path,
                        caption=action_caption,
                    )
                    if action_visual_path.exists():
                        action_vis_saved += 1
                    else:
                        action_visual_path = None
                        action_visual_skip_reason = "invalid_action_shape"
                else:
                    action_visual_skip_reason = "max_action_vis_reached"
            else:
                action_skip_reason = "signal_predicted_bbox"
                action_visual_skip_reason = "skipped_by_signal_bbox"

            # -----------------------------------------------------------------
            # 3) BBox branch (legacy bbox task evaluation, independent)
            # -----------------------------------------------------------------
            vlm_eval = vlm_ds._make_vlm_sample(data_idx)
            if vlm_eval is None and not args.bbox_only_valid_gt:
                vlm_eval = build_vlm_infer_sample_from_raw(vlm_ds, data_idx, raw)

            pred_text = ""
            pred_bbox_resized = None
            pred_bbox_orig = None
            iou = None
            bbox_visual_path = None
            bbox_visual_skip_reason = None
            bbox_eval_skipped = False

            if vlm_eval is None:
                bbox_eval_skipped = True
                reason = "invalid_gt_bbox" if args.bbox_only_valid_gt else "sample_build_failed"
                bbox_visual_skip_reason = reason
                print(f"[BBox Skip] ep={episode_index} frame={frame_index} idx={data_idx} reason={reason}")
            else:
                qwen_inputs = qwen_vl_interface.build_qwenvl_inputs(
                    images=[vlm_eval["image"]],
                    instructions=[vlm_eval["lang"]],
                )
                pred_text = decode_vlm_generation(
                    qwen_vl_interface=qwen_vl_interface,
                    qwen_inputs=qwen_inputs,
                    gen_kwargs=generation_kwargs_bbox,
                )
                pred_bbox_resized = parse_bbox_from_text(pred_text)

                if pred_bbox_resized is not None:
                    pred_bbox_orig = resized_to_original_bbox(
                        pred_bbox_resized,
                        src_w=src_w,
                        src_h=src_h,
                        dst_w=dst_w,
                        dst_h=dst_h,
                    )
                    pred_bbox_orig = clip_bbox_xyxy(pred_bbox_orig, w=src_w, h=src_h)
                    if not is_valid_bbox_xyxy(pred_bbox_orig):
                        pred_bbox_orig = None

                if gt_bbox_orig is not None and pred_bbox_orig is not None:
                    iou = bbox_iou_xyxy(gt_bbox_orig, pred_bbox_orig)

                if should_save_visual(bbox_vis_saved, args.max_bbox_vis_images):
                    ep_dir = output_dir / f"episode_{episode_index:06d}" / "bbox_vis"
                    bbox_visual_path = ep_dir / f"frame_{frame_index:06d}_idx_{data_idx:06d}.png"
                    caption = f"ep={episode_index} frame={frame_index} idx={data_idx} iou={iou if iou is not None else 'NA'}"
                    draw_bbox_overlay(
                        image_orig=raw_img,
                        gt_bbox_orig=gt_bbox_orig,
                        pred_bbox_orig=pred_bbox_orig,
                        save_path=bbox_visual_path,
                        caption=caption,
                    )
                    bbox_vis_saved += 1
                else:
                    bbox_visual_skip_reason = "max_bbox_vis_reached"

            # -----------------------------------------------------------------
            # 4) Logging / records
            # -----------------------------------------------------------------
            print("=" * 120)
            print(f"[Sample] ep={episode_index} frame={frame_index} idx={data_idx}")
            print(f"[Task] {task_text}")
            print(f"[Signal] pred_type={signal_response_type} gt_type={signal_gt_type} type_match={signal_type_match}")
            print(f"[Signal] pred_text: {signal_pred_text}")
            print(f"[Signal] pred_bbox_orig: {signal_pred_bbox_orig}")
            print(f"[Signal] bbox_iou(vs gt): {signal_bbox_iou}")
            if action_eval_skipped_by_signal:
                print(f"[Action] skipped_by_signal=True reason={action_skip_reason}")
            else:
                print(f"[Action] mse={action_mse:.6f}" if action_mse is not None else "[Action] mse=NA")
                print(
                    f"[Action] train_style_score(L2/numel)={train_style_score:.6f}"
                    if train_style_score is not None
                    else "[Action] train_style_score=NA"
                )
                print(
                    f"[Action] step0_mse(current frame)={action_step0_mse:.6f}"
                    if action_step0_mse is not None
                    else "[Action] step0_mse=NA"
                )
                print("[Action] pred_step0 (current frame):")
                print(pred_action_step0)
                print("[Action] gt_step0 (current frame):")
                print(gt_action_step0)
                print("[Action] pred (normalized):")
                print(pred_action)
                print("[Action] gt (normalized):")
                print(gt_action)
            if action_visual_path is not None:
                print(f"[Action] vis saved: {action_visual_path}")
            elif action_visual_skip_reason is not None:
                print(f"[Action] vis skipped: {action_visual_skip_reason}")

            print(f"[BBox] pred_text: {pred_text}")
            print(f"[BBox] gt_orig: {gt_bbox_orig}")
            print(f"[BBox] pred_resized: {pred_bbox_resized}")
            print(f"[BBox] pred_orig: {pred_bbox_orig}")
            print(f"[BBox] iou: {iou}")
            if bbox_visual_path is not None:
                print(f"[BBox] vis saved: {bbox_visual_path}")
            elif bbox_visual_skip_reason is not None:
                print(f"[BBox] vis skipped: {bbox_visual_skip_reason}")

            records.append(
                {
                    "dataset_index": int(data_idx),
                    "episode_index": int(episode_index),
                    "frame_index": int(frame_index),
                    "task": task_text,
                    "signal_pred_text": signal_pred_text,
                    "signal_pred_type": signal_response_type,
                    "signal_pred_bbox_resized": signal_pred_bbox_resized,
                    "signal_pred_bbox_orig": signal_pred_bbox_orig,
                    "signal_gt_raw": signal_gt_raw,
                    "signal_gt_type": signal_gt_type,
                    "signal_type_match": signal_type_match,
                    "signal_bbox_iou": signal_bbox_iou,
                    "signal_prompt": signal_eval["lang"],
                    "action_eval_skipped_by_signal": bool(action_eval_skipped_by_signal),
                    "action_skip_reason": action_skip_reason,
                    "action_mse": action_mse,
                    "action_train_style_score": train_style_score,
                    "action_step0_mse": action_step0_mse,
                    "pred_action_step0": pred_action_step0.tolist(),
                    "gt_action_step0": gt_action_step0.tolist(),
                    "pred_action": pred_action.tolist(),
                    "gt_action": gt_action.tolist(),
                    "action_eval_horizon": int(t),
                    "action_dim": int(d),
                    "action_visual_path": str(action_visual_path) if action_visual_path is not None else None,
                    "action_visual_skip_reason": action_visual_skip_reason,
                    "bbox_pred_text": pred_text,
                    "bbox_gt_orig": gt_bbox_orig,
                    "bbox_pred_resized": pred_bbox_resized,
                    "bbox_pred_orig": pred_bbox_orig,
                    "bbox_iou": iou,
                    "bbox_visual_path": str(bbox_visual_path) if bbox_visual_path is not None else None,
                    "bbox_visual_skip_reason": bbox_visual_skip_reason,
                    "bbox_eval_skipped": bbox_eval_skipped,
                }
            )

    # Save records
    records_path = output_dir / "open_loop_records.json"
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Plot action error curves by episode (chunk-level and current-frame step0).
    action_records = [r for r in records if r.get("action_mse") is not None]
    unique_eps = sorted({r["episode_index"] for r in action_records})

    action_chunk_curve_path = output_dir / "action_chunk_mse_curve.png"
    action_step0_curve_path = output_dir / "action_step0_mse_curve.png"
    action_train_style_curve_path = output_dir / "action_train_style_curve.png"
    action_step0_dim_dir = output_dir / "action_step0_dim_compare"
    action_step0_dim_dir.mkdir(parents=True, exist_ok=True)

    if action_records:
        # 1) Chunk-level MSE curve: compares full predicted chunk vs full GT chunk.
        plt.figure(figsize=(12, 6))
        for ep in unique_eps:
            rec_ep = [r for r in action_records if r["episode_index"] == ep]
            rec_ep.sort(key=lambda x: x["frame_index"])
            xs = [r["frame_index"] for r in rec_ep]
            ys = [r["action_mse"] for r in rec_ep]
            plt.plot(xs, ys, marker="o", linewidth=1.6, markersize=3, label=f"ep {ep}")

        plt.title("Open-loop Action Chunk Error Curve (MSE)")
        plt.xlabel("Frame Index")
        plt.ylabel("Chunk MSE (normalized space)")
        plt.grid(True, alpha=0.25)
        if len(unique_eps) <= 20:
            plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(action_chunk_curve_path, dpi=160)
        plt.close()

        # 2) Current-frame (step0) MSE curve.
        plt.figure(figsize=(12, 6))
        for ep in unique_eps:
            rec_ep = [r for r in action_records if r["episode_index"] == ep]
            rec_ep.sort(key=lambda x: x["frame_index"])
            xs = [r["frame_index"] for r in rec_ep]
            ys = [r["action_step0_mse"] for r in rec_ep]
            plt.plot(xs, ys, marker="o", linewidth=1.6, markersize=3, label=f"ep {ep}")

        plt.title("Open-loop Current-Frame Action Error Curve (Step0 MSE)")
        plt.xlabel("Frame Index")
        plt.ylabel("Step0 MSE (normalized space)")
        plt.grid(True, alpha=0.25)
        if len(unique_eps) <= 20:
            plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(action_step0_curve_path, dpi=160)
        plt.close()

        # 3) Trainer-style score curve.
        plt.figure(figsize=(12, 6))
        for ep in unique_eps:
            rec_ep = [r for r in action_records if r["episode_index"] == ep]
            rec_ep.sort(key=lambda x: x["frame_index"])
            xs = [r["frame_index"] for r in rec_ep]
            ys = [r["action_train_style_score"] for r in rec_ep]
            plt.plot(xs, ys, marker="o", linewidth=1.6, markersize=3, label=f"ep {ep}")

        plt.title("Open-loop Trainer-Style Action Score Curve (L2 / numel)")
        plt.xlabel("Frame Index")
        plt.ylabel("Trainer-style score")
        plt.grid(True, alpha=0.25)
        if len(unique_eps) <= 20:
            plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(action_train_style_curve_path, dpi=160)
        plt.close()

        # 4) Per-episode, per-dimension current-frame GT vs Pred curves.
        for ep in unique_eps:
            rec_ep = [r for r in action_records if r["episode_index"] == ep]
            rec_ep.sort(key=lambda x: x["frame_index"])
            if not rec_ep:
                continue

            dim = len(rec_ep[0].get("pred_action_step0", []))
            if dim <= 0:
                continue

            xs = [r["frame_index"] for r in rec_ep]
            ncols = 3
            nrows = (dim + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)

            for j in range(dim):
                ax = axes[j // ncols][j % ncols]
                gt_vals = [r["gt_action_step0"][j] for r in rec_ep]
                pred_vals = [r["pred_action_step0"][j] for r in rec_ep]
                ax.plot(xs, gt_vals, color="tab:green", marker="o", linewidth=1.5, markersize=3, label="GT")
                ax.plot(xs, pred_vals, color="tab:red", marker="o", linewidth=1.5, markersize=3, label="Pred")
                ax.set_title(f"Episode {ep} | Action dim {j}")
                ax.set_xlabel("Frame Index")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.25)
                if j == 0:
                    ax.legend(loc="best", fontsize=8)

            total_axes = nrows * ncols
            for j in range(dim, total_axes):
                axes[j // ncols][j % ncols].axis("off")

            fig.tight_layout()
            fig.savefig(action_step0_dim_dir / f"episode_{ep:06d}_step0_dim_compare.png", dpi=160)
            plt.close(fig)
    else:
        # Produce minimal placeholder figures when all frames are skipped by signal.
        for out_path, title in [
            (action_chunk_curve_path, "Open-loop Action Chunk Error Curve (MSE)"),
            (action_step0_curve_path, "Open-loop Current-Frame Action Error Curve (Step0 MSE)"),
            (action_train_style_curve_path, "Open-loop Trainer-Style Action Score Curve (L2 / numel)"),
        ]:
            plt.figure(figsize=(10, 4))
            plt.title(title)
            plt.text(0.5, 0.5, "No action-evaluated frames (all skipped by signal)", ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out_path, dpi=160)
            plt.close()

    # Backward-compatible alias key used by earlier script versions.
    action_curve_path = action_chunk_curve_path

    # Summary
    action_mses = [r["action_mse"] for r in records if r.get("action_mse") is not None]
    action_step0_mses = [r["action_step0_mse"] for r in records if r.get("action_step0_mse") is not None]
    action_train_style_scores = [
        r["action_train_style_score"] for r in records if r.get("action_train_style_score") is not None
    ]
    bbox_ious = [r["bbox_iou"] for r in records if r.get("bbox_iou") is not None]

    signal_type_matches = [r["signal_type_match"] for r in records if r.get("signal_type_match") is not None]
    signal_bbox_ious = [r["signal_bbox_iou"] for r in records if r.get("signal_bbox_iou") is not None]
    signal_pred_action_count = int(sum(1 for r in records if r.get("signal_pred_type") == "pred_action"))
    signal_pred_bbox_count = int(sum(1 for r in records if r.get("signal_pred_type") == "bbox"))
    action_skipped_count = int(sum(1 for r in records if r.get("action_eval_skipped_by_signal", False)))

    episode_range = [int(min(episodes)), int(max(episodes))] if episodes else None
    text_report_path = output_dir / "open_loop_metrics.txt"
    summary = {
        "checkpoint": str(checkpoint),
        "config_yaml": str(args.config_yaml),
        "episodes": episodes,
        "episode_range": episode_range,
        "num_selected_frames": len(eval_indices),
        "num_evaluated_records": len(records),
        "num_action_evaluated_records": int(len(action_records)),
        "num_action_skipped_by_signal": int(action_skipped_count),
        "action_mse_mean": float(np.mean(action_mses)) if action_mses else None,
        "action_mse_median": float(np.median(action_mses)) if action_mses else None,
        "action_step0_mse_mean": float(np.mean(action_step0_mses)) if action_step0_mses else None,
        "action_step0_mse_median": float(np.median(action_step0_mses)) if action_step0_mses else None,
        "action_train_style_mean": float(np.mean(action_train_style_scores)) if action_train_style_scores else None,
        "action_train_style_median": float(np.median(action_train_style_scores)) if action_train_style_scores else None,
        "bbox_iou_mean": float(np.mean(bbox_ious)) if bbox_ious else None,
        "bbox_iou_median": float(np.median(bbox_ious)) if bbox_ious else None,
        "bbox_eval_count": int(len(bbox_ious)),
        "signal_type_accuracy": float(np.mean(signal_type_matches)) if signal_type_matches else None,
        "signal_eval_count": int(len(signal_type_matches)),
        "signal_pred_action_count": signal_pred_action_count,
        "signal_pred_bbox_count": signal_pred_bbox_count,
        "signal_bbox_iou_mean": float(np.mean(signal_bbox_ious)) if signal_bbox_ious else None,
        "signal_bbox_iou_median": float(np.median(signal_bbox_ious)) if signal_bbox_ious else None,
        "signal_bbox_eval_count": int(len(signal_bbox_ious)),
        "max_action_vis_images": int(args.max_action_vis_images),
        "action_vis_saved_count": int(action_vis_saved),
        "max_bbox_vis_images": int(args.max_bbox_vis_images),
        "bbox_vis_saved_count": int(bbox_vis_saved),
        "records_json": str(records_path),
        "action_curve_png": str(action_curve_path),
        "action_chunk_curve_png": str(action_chunk_curve_path),
        "action_step0_curve_png": str(action_step0_curve_path),
        "action_train_style_curve_png": str(action_train_style_curve_path),
        "action_step0_dim_dir": str(action_step0_dim_dir),
        "action_step_vis_dir": str(action_vis_dir),
        "text_report_txt": str(text_report_path),
    }

    write_text_report(text_report_path, summary, records)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 120)
    print("[Done] Open-loop evaluation finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
