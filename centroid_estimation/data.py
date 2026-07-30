from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .common import (
    IMAGE_H,
    IMAGE_W,
    bbox_norm_xyxy,
    clamp_bbox_xyxy,
    frame_key,
    load_json_or_csv,
    valid_bbox_xyxy,
)


def _parse_bbox_value(value: Any) -> list[float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return [float(v) for v in value]


def read_metadata(metadata_csv: str | Path) -> list[dict[str, Any]]:
    with Path(metadata_csv).open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    for row in rows:
        row["frame"] = int(row["frame"])
        row["trajectory_id"] = str(row["trajectory_id"])
        row["target_xyz"] = [
            float(row["object_body_x"]),
            float(row["object_body_y"]),
            float(row["object_body_z"]),
        ]
        row["key"] = frame_key(row["dataset"], row["trajectory_id"], row["frame"])
    return rows


def read_bbox_annotations(bbox_file: str | Path) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    for row in load_json_or_csv(bbox_file):
        key = frame_key(row["dataset"], str(row["trajectory_id"]), row["frame"])
        bbox = _parse_bbox_value(row.get("bbox_xyxy"))
        bbox_valid = bool(row.get("bbox_valid", False))
        if isinstance(row.get("bbox_valid"), str):
            bbox_valid = row["bbox_valid"].strip().lower() in {"1", "true", "yes", "y"}
        if bbox is not None:
            bbox = clamp_bbox_xyxy(bbox)
            bbox_valid = bbox_valid and valid_bbox_xyxy(bbox)
        else:
            bbox_valid = False
        item = dict(row)
        item["key"] = key
        item["bbox_xyxy"] = bbox
        item["bbox_valid"] = bbox_valid
        annotations[key] = item
    return annotations


def build_records(
    data_root: str | Path,
    metadata_csv: str | Path,
    bbox_file: str | Path,
    *,
    require_valid_bbox: bool = True,
) -> list[dict[str, Any]]:
    data_root = Path(data_root)
    metadata = read_metadata(metadata_csv)
    annotations = read_bbox_annotations(bbox_file)
    records: list[dict[str, Any]] = []
    for row in metadata:
        ann = annotations.get(row["key"])
        if ann is None:
            continue
        if require_valid_bbox and not ann.get("bbox_valid", False):
            continue
        bbox = ann.get("bbox_xyxy")
        if bbox is None:
            continue
        item = dict(row)
        item["bbox_xyxy"] = bbox
        item["bbox_norm"] = bbox_norm_xyxy(bbox)
        item["bbox_raw_output"] = ann.get("raw_output", "")
        item["bbox_valid"] = bool(ann.get("bbox_valid", False))
        item["image_path"] = resolve_image_path(data_root, item, ann)
        records.append(item)
    return records


def resolve_image_path(data_root: Path, metadata_row: dict[str, Any], annotation_row: dict[str, Any]) -> str:
    candidates = []
    ann_path = annotation_row.get("rgb_path")
    if ann_path:
        candidates.append(Path(str(ann_path)))
    meta_rgb = metadata_row.get("rgb_path")
    if meta_rgb:
        candidates.append(Path(metadata_row["dataset"]) / str(metadata_row["trajectory_id"]) / str(meta_rgb))
    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        rooted = data_root / candidate
        if rooted.exists():
            return str(rooted)
    fallback = data_root / metadata_row["dataset"] / str(metadata_row["trajectory_id"]) / str(metadata_row["rgb_path"])
    return str(fallback)


def make_object_to_id(records: list[dict[str, Any]]) -> dict[str, int]:
    names = sorted({str(r["object_name"]) for r in records})
    return {name: idx for idx, name in enumerate(names)}


def load_split_records(records: list[dict[str, Any]], split_json: str | Path, split: str) -> list[dict[str, Any]]:
    import json

    payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
    wanted = {item["key"] if isinstance(item, dict) else str(item) for item in payload[split]}
    return [record for record in records if record["key"] in wanted]


def image_to_tensor(image: Image.Image, size: int | tuple[int, int]) -> torch.Tensor:
    if isinstance(size, int):
        size = (size, size)
    image = image.convert("RGB").resize(tuple(size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


class BBoxCentroidDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        object_to_id: dict[str, int],
        *,
        image_size: int = 224,
        crop_size: int = 224,
    ) -> None:
        self.records = list(records)
        self.object_to_id = dict(object_to_id)
        self.image_size = int(image_size)
        self.crop_size = int(crop_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        bbox = clamp_bbox_xyxy(record["bbox_xyxy"], image.width, image.height)
        x1, y1, x2, y2 = bbox
        crop = image.crop((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
        if crop.width <= 0 or crop.height <= 0:
            crop = image
        object_name = str(record["object_name"])
        return {
            "full_image": image_to_tensor(image, self.image_size),
            "crop_image": image_to_tensor(crop, self.crop_size),
            "bbox_norm": torch.tensor(bbox_norm_xyxy(bbox, IMAGE_W, IMAGE_H), dtype=torch.float32),
            "bbox_xyxy": torch.tensor(bbox, dtype=torch.float32),
            "object_id": torch.tensor(self.object_to_id[object_name], dtype=torch.long),
            "target_xyz": torch.tensor(record["target_xyz"], dtype=torch.float32),
            "key": record["key"],
            "dataset": record["dataset"],
            "trajectory_id": str(record["trajectory_id"]),
            "frame": int(record["frame"]),
            "object_name": object_name,
            "image_path": record["image_path"],
        }


def centroid_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ["full_image", "crop_image", "bbox_norm", "bbox_xyxy", "object_id", "target_xyz"]
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    for key in ["key", "dataset", "trajectory_id", "frame", "object_name", "image_path"]:
        out[key] = [item[key] for item in batch]
    return out

