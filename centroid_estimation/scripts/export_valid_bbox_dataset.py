#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from centroid_estimation.common import bbox_norm_xyxy, save_json
from centroid_estimation.data import build_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export bbox_valid frames into one consolidated centroid dataset.")
    parser.add_argument("--source_data_root", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--bbox_file", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--demo_count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy_images", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def flat_image_name(record: dict[str, Any]) -> str:
    return f"{record['dataset']}__traj{record['trajectory_id']}__frame{int(record['frame']):06d}.jpg"


def export_records(records: list[dict[str, Any]], output_root: Path, *, copy_images: bool) -> list[dict[str, Any]]:
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    for record in records:
        image_name = flat_image_name(record)
        dst = images_dir / image_name
        if copy_images:
            shutil.copy2(record["image_path"], dst)
        bbox = [float(v) for v in record["bbox_xyxy"]]
        bbox_norm = bbox_norm_xyxy(bbox)
        row = {
            "key": record["key"],
            "dataset": record["dataset"],
            "trajectory_id": str(record["trajectory_id"]),
            "frame": int(record["frame"]),
            "object_name": record["object_name"],
            "image": f"images/{image_name}",
            "source_image_path": record["image_path"],
            "source_rgb_path": record.get("rgb_path", ""),
            "bbox_x1": bbox[0],
            "bbox_y1": bbox[1],
            "bbox_x2": bbox[2],
            "bbox_y2": bbox[3],
            "bbox_norm_x1": bbox_norm[0],
            "bbox_norm_y1": bbox_norm[1],
            "bbox_norm_x2": bbox_norm[2],
            "bbox_norm_y2": bbox_norm[3],
            "object_body_x": float(record["target_xyz"][0]),
            "object_body_y": float(record["target_xyz"][1]),
            "object_body_z": float(record["target_xyz"][2]),
            "prompt": record.get("prompt", ""),
            "bbox_raw_output": record.get("bbox_raw_output", ""),
        }
        exported.append(row)
    return exported


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_intra(records: list[dict[str, Any]], train_even: bool = True) -> dict[str, Any]:
    train = [r for r in records if (int(r["frame"]) % 2 == 0) == train_even]
    test = [r for r in records if (int(r["frame"]) % 2 == 0) != train_even]
    return {
        "split_mode": "intra_trajectory_half",
        "train_frame_parity": "even" if train_even else "odd",
        "train_count": len(train),
        "test_count": len(test),
        "train": [{"key": r["key"], "image": r["image"]} for r in train],
        "test": [{"key": r["key"], "image": r["image"]} for r in test],
    }


def split_trajectory(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen = set()
    for record in records:
        stratum = (str(record["dataset"]), str(record["object_name"]))
        traj = str(record["trajectory_id"])
        key = (stratum, traj)
        if key not in seen:
            seen.add(key)
            strata[stratum].append(traj)
    train_trajs = set()
    for stratum, trajs in strata.items():
        trajs = list(trajs)
        rng.shuffle(trajs)
        train_trajs.update((stratum, traj) for traj in trajs[: len(trajs) // 2])
    train = [
        r
        for r in records
        if ((str(r["dataset"]), str(r["object_name"])), str(r["trajectory_id"])) in train_trajs
    ]
    test = [
        r
        for r in records
        if ((str(r["dataset"]), str(r["object_name"])), str(r["trajectory_id"])) not in train_trajs
    ]
    return {
        "split_mode": "trajectory_half",
        "seed": seed,
        "train_count": len(train),
        "test_count": len(test),
        "train": [{"key": r["key"], "image": r["image"]} for r in train],
        "test": [{"key": r["key"], "image": r["image"]} for r in test],
    }


def draw_demo(output_root: Path, rows: list[dict[str, Any]], count: int, seed: int) -> None:
    if count <= 0:
        return
    rng = random.Random(seed)
    chosen = rows if len(rows) <= count else rng.sample(rows, count)
    demo_dir = output_root / "demo_bbox"
    demo_dir.mkdir(parents=True, exist_ok=True)
    for row in chosen:
        image_path = output_root / row["image"]
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        bbox = [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]]
        draw.rectangle(bbox, outline=(255, 32, 32), width=3)
        label = f"{row['object_name']} {row['dataset']} traj{row['trajectory_id']} f{row['frame']}"
        draw.rectangle([4, 4, min(image.width - 1, 12 + 7 * len(label)), 24], fill=(255, 32, 32))
        draw.text((8, 7), label, fill=(255, 255, 255))
        out_name = Path(row["image"]).name
        image.save(demo_dir / out_name)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = build_records(args.source_data_root, args.metadata_csv, args.bbox_file, require_valid_bbox=True)
    exported = export_records(records, output_root, copy_images=args.copy_images)
    write_csv(output_root / "metadata_valid_bbox.csv", exported)
    save_json(output_root / "metadata_valid_bbox.json", exported)
    save_json(output_root / "splits_intra_traj_half_even.json", split_intra(exported, train_even=True))
    save_json(output_root / "splits_trajectory_half_seed42.json", split_trajectory(exported, seed=args.seed))
    draw_demo(output_root, exported, args.demo_count, args.seed)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "records": len(exported),
                "images_dir": str(output_root / "images"),
                "metadata_csv": str(output_root / "metadata_valid_bbox.csv"),
                "demo_dir": str(output_root / "demo_bbox"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

