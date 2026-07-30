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

from centroid_estimation.common import clamp_bbox_xyxy, save_json, valid_bbox_xyxy
from centroid_estimation.data import read_metadata
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.training.trainer_utils.trainer_tools import resize_images


POINT_PATTERN = re.compile(
    r"<point>\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*</point>",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate bboxes with StarVLA Qwen-VL only; no action/FAST path.")
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--base_vlm", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--route_mode", choices=("first_token", "generate"), default="first_token")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--force_bbox_prompt", action="store_true")
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
    if missing:
        print(json.dumps({"missing_first": missing[:20]}, ensure_ascii=False))
    if unexpected:
        print(json.dumps({"unexpected_first": unexpected[:20]}, ensure_ascii=False))
    qwen.to(device)
    qwen.eval()
    return qwen


def parse_bbox(text: str) -> list[float] | None:
    match = POINT_PATTERN.search(str(text))
    if match is None:
        match = re.search(
            r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]",
            str(text),
        )
    if match is None:
        return None
    return [float(match.group(i)) for i in range(1, 5)]


def build_prompt(row: dict[str, Any], pred_action_token: str, pred_bbox_token: str, force_bbox: bool) -> str:
    instruction = str(row.get("prompt") or "").strip()
    target_name = str(row["object_name"]).strip()
    if force_bbox:
        return (
            f"{instruction}\n"
            f"{instruction}\n"
            f"{instruction}\n\n"
            "GRASP PHASE: FIND AND GRASP THE OBJECT.\n"
            f"The current target is the object to be grasped: {target_name}.\n"
            f"{target_name}, {target_name}, {target_name} should be grasped.\n"
            f"Output {pred_bbox_token}<point>[x1, y1, x2, y2]</point> for the object's location in the current front view.\n"
            "Do not output anything else."
        )
    return (
        f"{instruction}\n"
        f"{instruction}\n"
        f"{instruction}\n\n"
        "GRASP PHASE: FIND AND GRASP THE OBJECT.\n"
        f"The current target is the object to be grasped: {target_name}.\n"
        f"{target_name}, {target_name}, {target_name} should be grasped.\n"
        f"If the object is still far away, output exactly {pred_action_token}.\n"
        f"If the object is close enough for grasping, output {pred_bbox_token}<point>[x1, y1, x2, y2]</point> "
        "for the object's location in the current front view.\n"
        "Do not output anything else."
    )


@torch.inference_mode()
def predict_one(
    qwen,
    image: Image.Image,
    prompt: str,
    *,
    image_size: tuple[int, int],
    pred_action_token: str,
    pred_bbox_token: str,
    route_mode: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    model_image = resize_images(image, target_size=image_size)
    qwen_inputs = qwen.build_qwenvl_inputs(images=[[model_image]], instructions=[prompt])
    token_ids = {
        "action": qwen.processor.tokenizer(pred_action_token, add_special_tokens=False).input_ids,
        "bbox": qwen.processor.tokenizer(pred_bbox_token, add_special_tokens=False).input_ids,
    }
    if route_mode == "first_token" and len(token_ids["action"]) == 1 and len(token_ids["bbox"]) == 1:
        out = qwen(**qwen_inputs, output_attentions=False, output_hidden_states=False, return_dict=True)
        first_logits = out.logits[:, -1, :]
        route_token_ids = torch.tensor(
            [int(token_ids["action"][0]), int(token_ids["bbox"][0])],
            device=first_logits.device,
            dtype=torch.long,
        )
        probs = torch.softmax(first_logits.index_select(-1, route_token_ids).float(), dim=-1)[0]
        choice = int(torch.argmax(probs).item())
        route = "action" if choice == 0 else "bbox"
        route_token = pred_action_token if route == "action" else pred_bbox_token
        generated_text = route_token
        if route == "bbox":
            forced_inputs = {key: value for key, value in qwen_inputs.items() if key != "labels"}
            input_ids = forced_inputs["input_ids"]
            route_id = torch.tensor([[int(token_ids["bbox"][0])]], device=input_ids.device, dtype=input_ids.dtype)
            forced_inputs["input_ids"] = torch.cat([input_ids, route_id], dim=1)
            if "attention_mask" in forced_inputs:
                forced_inputs["attention_mask"] = torch.cat(
                    [forced_inputs["attention_mask"], torch.ones_like(route_id)],
                    dim=1,
                )
            generated = qwen.generate(**forced_inputs, max_new_tokens=max_new_tokens, do_sample=False)
            gen_ids = generated.sequences if hasattr(generated, "sequences") else generated
            new_ids = gen_ids[:, int(qwen_inputs["input_ids"].shape[1]) :]
            generated_text = qwen.processor.batch_decode(new_ids, skip_special_tokens=False)[0].strip()
        return {
            "route": route,
            "route_token": route_token,
            "generated_text": generated_text,
            "route_action_prob": float(probs[0].item()),
            "route_bbox_prob": float(probs[1].item()),
            "route_confidence": float(torch.max(probs).item()),
        }

    generated = qwen.generate(**qwen_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = generated.sequences if hasattr(generated, "sequences") else generated
    new_ids = gen_ids[:, int(qwen_inputs["input_ids"].shape[1]) :]
    generated_text = qwen.processor.batch_decode(new_ids, skip_special_tokens=False)[0].strip()
    if generated_text.startswith(pred_bbox_token):
        route = "bbox"
        route_token = pred_bbox_token
    elif generated_text.startswith(pred_action_token):
        route = "action"
        route_token = pred_action_token
    else:
        route = "unknown"
        route_token = None
    return {"route": route, "route_token": route_token, "generated_text": generated_text}


def make_record(row: dict[str, Any], route_info: dict[str, Any], image: Image.Image, image_size: tuple[int, int]) -> dict[str, Any]:
    bbox_model = parse_bbox(route_info.get("generated_text", ""))
    bbox_image = None
    valid = False
    if bbox_model is not None:
        bbox_model = clamp_bbox_xyxy(bbox_model, image_size[0], image_size[1])
        sx = float(image.width) / float(image_size[0])
        sy = float(image.height) / float(image_size[1])
        bbox_image = clamp_bbox_xyxy(
            [bbox_model[0] * sx, bbox_model[1] * sy, bbox_model[2] * sx, bbox_model[3] * sy],
            image.width,
            image.height,
        )
        valid = valid_bbox_xyxy(bbox_image)
    return {
        "dataset": row["dataset"],
        "trajectory_id": str(row["trajectory_id"]),
        "frame": int(row["frame"]),
        "object_name": row["object_name"],
        "rgb_path": f'{row["dataset"]}/{row["trajectory_id"]}/{row["rgb_path"]}',
        "bbox_xyxy": bbox_image,
        "bbox_valid": bool(valid),
        "raw_output": str(route_info.get("generated_text", "")),
        "route": route_info.get("route"),
        "route_confidence": route_info.get("route_confidence"),
        "route_action_prob": route_info.get("route_action_prob"),
        "route_bbox_prob": route_info.get("route_bbox_prob"),
    }


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config_yaml)
    if args.base_vlm:
        OmegaConf.update(cfg, "framework.qwenvl.base_vlm", args.base_vlm, merge=True)
    qwen = load_qwen_vl_interface(cfg, args.checkpoint, args.device)
    router_cfg = cfg.datasets.router_data
    pred_action_token = str(cfg_get(router_cfg, "pred_action_token", "<|pred_action|>"))
    pred_bbox_token = str(cfg_get(router_cfg, "pred_bbox_token", "<|pred_bbox|>"))
    image_size = tuple(int(v) for v in cfg_get(router_cfg, "image_size", [224, 224]))

    data_root = Path(args.data_root)
    rows = read_metadata(args.metadata_csv)
    rows = rows[args.start : args.start + args.limit] if args.limit > 0 else rows[args.start :]
    output_json = Path(args.output_json)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        image_path = data_root / row["dataset"] / str(row["trajectory_id"]) / str(row["rgb_path"])
        image = Image.open(image_path).convert("RGB")
        prompt = build_prompt(row, pred_action_token, pred_bbox_token, args.force_bbox_prompt)
        route_info = predict_one(
            qwen,
            image,
            prompt,
            image_size=image_size,
            pred_action_token=pred_action_token,
            pred_bbox_token=pred_bbox_token,
            route_mode=args.route_mode,
            max_new_tokens=args.max_new_tokens,
        )
        records.append(make_record(row, route_info, image, image_size))
        if args.save_every > 0 and idx % args.save_every == 0:
            save_json(output_json, records)
            valid = sum(1 for r in records if r["bbox_valid"])
            print(f"saved {len(records)} records valid={valid} to {output_json}")
    save_json(output_json, records)
    valid = sum(1 for r in records if r["bbox_valid"])
    print(f"done: saved {len(records)} records valid={valid} to {output_json}")


if __name__ == "__main__":
    main()

