#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from centroid_estimation.data import build_records, load_split_records
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.training.trainer_utils.trainer_tools import resize_images


XYZ_PATTERNS = [
    re.compile(
        r"<centroid>\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*</centroid>",
        re.IGNORECASE,
    ),
    re.compile(
        r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]",
        re.IGNORECASE,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the VLM to directly predict object_body xyz centroids.")
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--bbox_file", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--base_vlm", default=None)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--split_json", default=None)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no_bbox_prompt", action="store_true")
    return parser.parse_args()


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_qwen_vl_interface(cfg: Any, checkpoint_path: str, device: str):
    qwen = get_vlm_model(config=cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    sub_state = {
        key[len("qwen_vl_interface.") :]: value
        for key, value in checkpoint.items()
        if str(key).startswith("qwen_vl_interface.")
    }
    if not sub_state:
        raise RuntimeError("No qwen_vl_interface.* weights found in checkpoint.")

    embed_key = "model.model.language_model.embed_tokens.weight"
    if embed_key in sub_state:
        target_rows = int(sub_state[embed_key].shape[0])
        current_rows = int(qwen.model.get_input_embeddings().weight.shape[0])
        if target_rows != current_rows:
            qwen.model.resize_token_embeddings(target_rows)
    missing, unexpected = qwen.load_state_dict(sub_state, strict=False)
    print(json.dumps({"missing": len(missing), "unexpected": len(unexpected)}, ensure_ascii=False))
    qwen.to(device)
    qwen.eval()
    return qwen


def parse_xyz(text: str) -> list[float] | None:
    for pattern in XYZ_PATTERNS:
        match = pattern.search(str(text))
        if match is not None:
            return [float(match.group(i)) for i in range(1, 4)]
    return None


def build_prompt(record: dict[str, Any], *, include_bbox: bool) -> str:
    object_name = str(record["object_name"])
    instruction = str(record.get("prompt") or "").strip()
    target = np.asarray(record["target_xyz"], dtype=np.float32)
    rough_range = (
        "In this dataset, x is usually forward distance around 0.4 to 0.8, "
        "y is lateral offset around -0.1 to 0.2, and z is vertical offset around -0.25 to -0.15."
    )
    bbox_text = ""
    if include_bbox:
        bbox = [round(float(v), 2) for v in record["bbox_xyxy"]]
        bbox_text = f"\nThe target bbox in the original 640x480 image is xyxy={bbox}."
    return (
        f"{instruction}\n\n"
        "You are estimating a robotics perception label from the current front-view RGB image.\n"
        f"The target object is: {object_name}."
        f"{bbox_text}\n"
        "Predict the target object's 3D centroid in the drone body frame.\n"
        "The coordinate order is [object_body_x, object_body_y, object_body_z].\n"
        "These are metric robot-frame coordinates, not image pixels and not bbox coordinates.\n"
        "Every output value must be a small decimal between -1.0 and 1.0.\n"
        "Never output image coordinates such as 80, 120, or 300.\n"
        f"{rough_range}\n"
        "A valid example is: <centroid>[0.55, 0.08, -0.20]</centroid>\n"
        "Output exactly one line in this format:\n"
        "<centroid>[x, y, z]</centroid>\n"
        "Use decimal numbers only. Do not explain."
    )


@torch.inference_mode()
def predict_one(qwen, image: Image.Image, prompt: str, *, image_size: tuple[int, int], max_new_tokens: int, temperature: float) -> str:
    model_image = resize_images(image, target_size=image_size)
    qwen_inputs = qwen.build_qwenvl_inputs(images=[[model_image]], instructions=[prompt])
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(temperature > 0.0),
    }
    if temperature > 0.0:
        generate_kwargs["temperature"] = float(temperature)
    generated = qwen.generate(**qwen_inputs, **generate_kwargs)
    gen_ids = generated.sequences if hasattr(generated, "sequences") else generated
    new_ids = gen_ids[:, int(qwen_inputs["input_ids"].shape[1]) :]
    return qwen.processor.batch_decode(new_ids, skip_special_tokens=False)[0].strip()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if r["parsed_xyz"] is not None]
    out: dict[str, Any] = {"total": len(records), "parsed": len(valid), "parse_rate": len(valid) / max(len(records), 1)}
    if not valid:
        return out
    gt = np.asarray([r["gt_xyz"] for r in valid], dtype=np.float32)
    pred = np.asarray([r["parsed_xyz"] for r in valid], dtype=np.float32)
    diff = pred - gt
    abs_diff = np.abs(diff)
    l2 = np.linalg.norm(diff, axis=1)
    out.update(
        {
            "mae_x": float(abs_diff[:, 0].mean()),
            "mae_y": float(abs_diff[:, 1].mean()),
            "mae_z": float(abs_diff[:, 2].mean()),
            "mean_l2_error": float(l2.mean()),
            "median_l2_error": float(np.median(l2)),
        }
    )
    return out


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config_yaml)
    if args.base_vlm:
        OmegaConf.update(cfg, "framework.qwenvl.base_vlm", args.base_vlm, merge=True)
    if args.attn_implementation:
        OmegaConf.update(cfg, "framework.qwenvl.attn_implementation", args.attn_implementation, merge=True)
    qwen = load_qwen_vl_interface(cfg, args.checkpoint, args.device)
    router_cfg = cfg.datasets.router_data
    image_size = tuple(int(v) for v in cfg_get(router_cfg, "image_size", [224, 224]))

    data_root = Path(args.data_root)
    records = build_records(data_root, args.metadata_csv, args.bbox_file, require_valid_bbox=True)
    if args.split_json:
        records = load_split_records(records, args.split_json, args.split)
    records = records[args.start : args.start + args.limit] if args.limit > 0 else records[args.start :]

    outputs: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        image = Image.open(record["image_path"]).convert("RGB")
        prompt = build_prompt(record, include_bbox=not args.no_bbox_prompt)
        raw_output = predict_one(
            qwen,
            image,
            prompt,
            image_size=image_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        parsed = parse_xyz(raw_output)
        gt = [float(v) for v in record["target_xyz"]]
        l2 = None
        abs_err = None
        if parsed is not None:
            diff = np.asarray(parsed, dtype=np.float32) - np.asarray(gt, dtype=np.float32)
            abs_err = np.abs(diff).astype(float).tolist()
            l2 = float(np.linalg.norm(diff))
        outputs.append(
            {
                "key": record["key"],
                "dataset": record["dataset"],
                "trajectory_id": str(record["trajectory_id"]),
                "frame": int(record["frame"]),
                "object_name": record["object_name"],
                "bbox_xyxy": [float(v) for v in record["bbox_xyxy"]],
                "gt_xyz": gt,
                "raw_output": raw_output,
                "parsed_xyz": parsed,
                "abs_err_xyz": abs_err,
                "l2_error": l2,
            }
        )
        print(json.dumps(outputs[-1], ensure_ascii=False))

    payload = {
        "settings": {
            "checkpoint": args.checkpoint,
            "split_json": args.split_json,
            "split": args.split,
            "start": args.start,
            "limit": args.limit,
            "include_bbox_prompt": not args.no_bbox_prompt,
        },
        "summary": summarize(outputs),
        "records": outputs,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
