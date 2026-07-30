#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from centroid_estimation.common import bbox_norm_xyxy, clamp_bbox_xyxy
from centroid_estimation.data import image_to_tensor
from centroid_estimation.inference import load_checkpoint, predict_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one centroid from RGB + bbox + object_name.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--bbox_xyxy", type=float, nargs=4, required=True)
    parser.add_argument("--object_name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--crop_size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model, anchor, checkpoint = load_checkpoint(args.checkpoint, device)
    object_to_id = checkpoint["object_to_id"]
    if args.object_name not in object_to_id:
        raise ValueError(f"Unknown object_name={args.object_name!r}; known={sorted(object_to_id)}")

    image = Image.open(args.image).convert("RGB")
    bbox = clamp_bbox_xyxy(args.bbox_xyxy, image.width, image.height)
    x1, y1, x2, y2 = bbox
    crop = image.crop((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
    batch = {
        "full_image": image_to_tensor(image, args.image_size)[None],
        "crop_image": image_to_tensor(crop, args.crop_size)[None],
        "bbox_norm": torch.tensor([bbox_norm_xyxy(bbox)], dtype=torch.float32),
        "object_id": torch.tensor([object_to_id[args.object_name]], dtype=torch.long),
        "key": ["manual"],
        "object_name": [args.object_name],
    }
    pred_xyz, anchor_xyz = predict_batch(model, anchor, checkpoint, batch, device)
    print(
        json.dumps(
            {
                "image": args.image,
                "object_name": args.object_name,
                "bbox_xyxy": bbox,
                "pred_xyz": pred_xyz[0].astype(float).tolist(),
                "anchor_xyz": anchor_xyz[0].astype(float).tolist(),
                "target_mode": checkpoint["target_mode"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

