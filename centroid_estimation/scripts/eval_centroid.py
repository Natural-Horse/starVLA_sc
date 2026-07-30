#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from centroid_estimation.data import BBoxCentroidDataset, build_records, centroid_collate, load_split_records
from centroid_estimation.inference import load_checkpoint, predict_batch
from centroid_estimation.metrics import summarize_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a centroid checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--bbox_file", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument(
        "--disable_bbox_input",
        action="store_true",
        help="Ablate bbox information: use full image as crop and zero bbox features.",
    )
    return parser.parse_args()


def ablate_batch_bbox(batch: dict[str, object]) -> dict[str, object]:
    batch = dict(batch)
    batch["crop_image"] = batch["full_image"]
    batch["bbox_norm"] = torch.zeros_like(batch["bbox_norm"])
    return batch


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model, anchor, checkpoint = load_checkpoint(args.checkpoint, device)
    all_records = build_records(args.data_root, args.metadata_csv, args.bbox_file, require_valid_bbox=True)
    records = load_split_records(all_records, args.split_json, args.split)
    dataset = BBoxCentroidDataset(records, checkpoint["object_to_id"], image_size=args.image_size, crop_size=args.crop_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=centroid_collate)
    pred_records = []
    for batch in loader:
        if args.disable_bbox_input:
            batch = ablate_batch_bbox(batch)
        pred_xyz, anchor_xyz = predict_batch(model, anchor, checkpoint, batch, device)
        gt = batch["target_xyz"].detach().cpu().numpy()
        for idx in range(len(batch["key"])):
            pred_records.append(
                {
                    "key": batch["key"][idx],
                    "dataset": batch["dataset"][idx],
                    "trajectory_id": batch["trajectory_id"][idx],
                    "frame": int(batch["frame"][idx]),
                    "object_name": batch["object_name"][idx],
                    "gt_xyz": gt[idx].astype(float).tolist(),
                    "pred_xyz": pred_xyz[idx].astype(float).tolist(),
                    "anchor_xyz": anchor_xyz[idx].astype(float).tolist(),
                    "disable_bbox_input": bool(args.disable_bbox_input),
                }
            )
    metrics = summarize_errors(pred_records)
    metrics["disable_bbox_input"] = bool(args.disable_bbox_input)
    payload = {"metrics": metrics, "records": pred_records}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
