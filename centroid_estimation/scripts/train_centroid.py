#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from centroid_estimation.anchors import build_anchor
from centroid_estimation.data import (
    BBoxCentroidDataset,
    build_records,
    centroid_collate,
    load_split_records,
    make_object_to_id,
)
from centroid_estimation.metrics import summarize_errors
from centroid_estimation.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bbox-conditioned centroid regressor.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--bbox_file", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_mode", choices=("direct", "residual_class_mean", "residual_bbox_ridge"), default="residual_bbox_ridge")
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--backbone", choices=("resnet18", "resnet34"), default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--shared_backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
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


def compute_norm(values: np.ndarray) -> dict[str, list[float]]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist()}


def apply_norm(values: torch.Tensor, norm: dict[str, list[float]], device: torch.device) -> torch.Tensor:
    mean = torch.tensor(norm["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(norm["std"], dtype=torch.float32, device=device)
    return (values - mean) / std


def undo_norm(values: np.ndarray, norm: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    return values * std + mean


def anchor_batch(anchor: Any, batch: dict[str, Any]) -> np.ndarray:
    anchors = []
    for idx in range(len(batch["key"])):
        record = {
            "object_name": batch["object_name"][idx],
            "bbox_norm": batch["bbox_norm"][idx].detach().cpu().numpy(),
            "target_xyz": batch["target_xyz"][idx].detach().cpu().numpy(),
        }
        anchors.append(anchor.predict_record(record))
    return np.stack(anchors, axis=0).astype(np.float32)


def ablate_record_bbox(record: dict[str, Any]) -> dict[str, Any]:
    item = dict(record)
    item["bbox_norm"] = [0.0, 0.0, 0.0, 0.0]
    return item


def ablate_batch_bbox(batch: dict[str, Any]) -> dict[str, Any]:
    batch = dict(batch)
    batch["crop_image"] = batch["full_image"]
    batch["bbox_norm"] = torch.zeros_like(batch["bbox_norm"])
    return batch


def evaluate(
    model,
    loader,
    anchor,
    target_mode: str,
    norm: dict[str, list[float]],
    device: torch.device,
    *,
    disable_bbox_input: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    records = []
    with torch.inference_mode():
        for batch in loader:
            if disable_bbox_input:
                batch = ablate_batch_bbox(batch)
            full = batch["full_image"].to(device)
            crop = batch["crop_image"].to(device)
            bbox = batch["bbox_norm"].to(device)
            obj = batch["object_id"].to(device)
            out_norm = model(full, crop, bbox, obj).detach().cpu().numpy()
            pred_delta_or_xyz = undo_norm(out_norm, norm)
            anchors = anchor_batch(anchor, batch)
            if target_mode == "direct":
                pred_xyz = pred_delta_or_xyz
                anchor_xyz = np.zeros_like(pred_xyz)
            else:
                anchor_xyz = anchors
                pred_xyz = anchors + pred_delta_or_xyz
            gt = batch["target_xyz"].detach().cpu().numpy()
            for idx in range(len(batch["key"])):
                records.append(
                    {
                        "key": batch["key"][idx],
                        "dataset": batch["dataset"][idx],
                        "trajectory_id": batch["trajectory_id"][idx],
                        "frame": int(batch["frame"][idx]),
                        "object_name": batch["object_name"][idx],
                        "gt_xyz": gt[idx].astype(float).tolist(),
                        "pred_xyz": pred_xyz[idx].astype(float).tolist(),
                        "anchor_xyz": anchor_xyz[idx].astype(float).tolist(),
                    }
                )
    return summarize_errors(records), records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    all_records = build_records(args.data_root, args.metadata_csv, args.bbox_file, require_valid_bbox=True)
    train_records = load_split_records(all_records, args.split_json, "train")
    test_records = load_split_records(all_records, args.split_json, "test")
    if not train_records or not test_records:
        raise RuntimeError(f"Empty split: train={len(train_records)} test={len(test_records)}")

    object_to_id = make_object_to_id(all_records)
    anchor_train_records = [ablate_record_bbox(record) for record in train_records] if args.disable_bbox_input else train_records
    anchor = build_anchor(args.target_mode, anchor_train_records, object_to_id, alpha=args.ridge_alpha)
    train_targets = []
    for record in train_records:
        target = np.asarray(record["target_xyz"], dtype=np.float32)
        if args.target_mode == "direct":
            train_targets.append(target)
        else:
            anchor_record = ablate_record_bbox(record) if args.disable_bbox_input else record
            train_targets.append(target - anchor.predict_record(anchor_record))
    target_norm = compute_norm(np.stack(train_targets, axis=0))

    train_ds = BBoxCentroidDataset(train_records, object_to_id, image_size=args.image_size, crop_size=args.crop_size)
    test_ds = BBoxCentroidDataset(test_records, object_to_id, image_size=args.image_size, crop_size=args.crop_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=centroid_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=centroid_collate)

    model = build_model(
        args.target_mode,
        num_objects=len(object_to_id),
        backbone=args.backbone,
        pretrained=args.pretrained,
        shared_backbone=args.shared_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.SmoothL1Loss()

    best_l2 = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            if args.disable_bbox_input:
                batch = ablate_batch_bbox(batch)
            full = batch["full_image"].to(device)
            crop = batch["crop_image"].to(device)
            bbox = batch["bbox_norm"].to(device)
            obj = batch["object_id"].to(device)
            target = batch["target_xyz"].to(device)
            if args.target_mode != "direct":
                anchors = torch.from_numpy(anchor_batch(anchor, batch)).to(device)
                target = target - anchors
            target = apply_norm(target, target_norm, device)
            pred = model(full, crop, bbox, obj)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        metrics, pred_records = evaluate(
            model,
            test_loader,
            anchor,
            args.target_mode,
            target_norm,
            device,
            disable_bbox_input=args.disable_bbox_input,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(losses)) if losses else None
        metrics["disable_bbox_input"] = bool(args.disable_bbox_input)
        print(json.dumps(metrics, ensure_ascii=False))
        (output_dir / "metrics_latest.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if metrics.get("mean_l2_error", float("inf")) < best_l2:
            best_l2 = float(metrics["mean_l2_error"])
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "object_to_id": object_to_id,
                "target_mode": args.target_mode,
                "target_norm": target_norm,
                "anchor": {
                    "type": args.target_mode,
                    "state": anchor.state_dict(),
                },
                "model_args": {
                    "num_objects": len(object_to_id),
                    "backbone": args.backbone,
                    "pretrained": False,
                    "shared_backbone": args.shared_backbone,
                },
                "disable_bbox_input": bool(args.disable_bbox_input),
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            (output_dir / "predictions_best.json").write_text(json.dumps(pred_records, indent=2), encoding="utf-8")

    print(f"best mean_l2_error={best_l2:.6f}; checkpoint={output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
