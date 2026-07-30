from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

IMAGE_W = 640
IMAGE_H = 480


def frame_key(dataset: str, trajectory_id: str | int, frame: str | int) -> str:
    return f"{dataset}|{trajectory_id}|{int(frame)}"


def clamp_bbox_xyxy(bbox: Iterable[float], width: int = IMAGE_W, height: int = IMAGE_H) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    return [x1, y1, x2, y2]


def valid_bbox_xyxy(bbox: Iterable[float], min_size: float = 1.0) -> bool:
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False
    if not all(v == v and abs(v) != float("inf") for v in (x1, y1, x2, y2)):
        return False
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def bbox_norm_xyxy(bbox: Iterable[float], width: int = IMAGE_W, height: int = IMAGE_H) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [x1 / float(width), y1 / float(height), x2 / float(width), y2 / float(height)]


def bbox_ridge_features(bbox_norm: Iterable[float], object_id: int, num_objects: int, eps: float = 1e-6) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_norm]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    features = [
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
        width,
        height,
        width * height,
        width / max(height, eps),
    ]
    onehot = [0.0] * int(num_objects)
    if 0 <= int(object_id) < int(num_objects):
        onehot[int(object_id)] = 1.0
    return features + onehot


def load_json_or_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "records" in payload:
            payload = payload["records"]
        if not isinstance(payload, list):
            raise ValueError(f"JSON annotation file must contain a list or records field: {path}")
        return [dict(item) for item in payload]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    raise ValueError(f"Unsupported file extension: {path}")


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

