from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def summarize_errors(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    gt = np.asarray([r["gt_xyz"] for r in records], dtype=np.float32)
    pred = np.asarray([r["pred_xyz"] for r in records], dtype=np.float32)
    diff = pred - gt
    abs_diff = np.abs(diff)
    l2 = np.linalg.norm(diff, axis=1)
    out: dict[str, Any] = {
        "mae_x": float(abs_diff[:, 0].mean()),
        "mae_y": float(abs_diff[:, 1].mean()),
        "mae_z": float(abs_diff[:, 2].mean()),
        "mean_l2_error": float(l2.mean()),
        "median_l2_error": float(np.median(l2)),
    }
    if all("anchor_xyz" in r and r["anchor_xyz"] is not None for r in records):
        anchor = np.asarray([r["anchor_xyz"] for r in records], dtype=np.float32)
        out["anchor_only_l2_error"] = float(np.linalg.norm(anchor - gt, axis=1).mean())
    out["model_l2_error"] = out["mean_l2_error"]
    out["per_object_mean_l2"] = _group_l2(records, "object_name")
    out["per_object_mae_xyz"] = _group_mae(records, "object_name")
    out["per_frame_index_mean_l2"] = _group_l2(records, "frame")
    return out


def _group_l2(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        gt = np.asarray(record["gt_xyz"], dtype=np.float32)
        pred = np.asarray(record["pred_xyz"], dtype=np.float32)
        groups[str(record[key])].append(float(np.linalg.norm(pred - gt)))
    return {name: float(np.mean(values)) for name, values in sorted(groups.items())}


def _group_mae(records: list[dict[str, Any]], key: str) -> dict[str, list[float]]:
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in records:
        gt = np.asarray(record["gt_xyz"], dtype=np.float32)
        pred = np.asarray(record["pred_xyz"], dtype=np.float32)
        groups[str(record[key])].append(np.abs(pred - gt))
    return {name: np.stack(values, axis=0).mean(axis=0).astype(float).tolist() for name, values in sorted(groups.items())}

