from __future__ import annotations

from typing import Any

import numpy as np

from .common import bbox_ridge_features


class ClassMeanAnchor:
    def __init__(self) -> None:
        self.means: dict[str, np.ndarray] = {}
        self.global_mean: np.ndarray | None = None

    def fit(self, records: list[dict[str, Any]]) -> "ClassMeanAnchor":
        grouped: dict[str, list[np.ndarray]] = {}
        all_targets = []
        for record in records:
            target = np.asarray(record["target_xyz"], dtype=np.float32)
            grouped.setdefault(str(record["object_name"]), []).append(target)
            all_targets.append(target)
        if not all_targets:
            raise ValueError("Cannot fit ClassMeanAnchor on empty records.")
        self.global_mean = np.stack(all_targets, axis=0).mean(axis=0)
        self.means = {key: np.stack(values, axis=0).mean(axis=0) for key, values in grouped.items()}
        return self

    def predict_record(self, record: dict[str, Any]) -> np.ndarray:
        if self.global_mean is None:
            raise RuntimeError("ClassMeanAnchor is not fitted.")
        return self.means.get(str(record["object_name"]), self.global_mean).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "means": {key: value.tolist() for key, value in self.means.items()},
            "global_mean": None if self.global_mean is None else self.global_mean.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ClassMeanAnchor":
        obj = cls()
        obj.means = {key: np.asarray(value, dtype=np.float32) for key, value in state.get("means", {}).items()}
        global_mean = state.get("global_mean")
        obj.global_mean = None if global_mean is None else np.asarray(global_mean, dtype=np.float32)
        return obj


class RidgeAnchor:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.weights: np.ndarray | None = None
        self.object_to_id: dict[str, int] = {}

    def fit(self, records: list[dict[str, Any]], object_to_id: dict[str, int]) -> "RidgeAnchor":
        if not records:
            raise ValueError("Cannot fit RidgeAnchor on empty records.")
        self.object_to_id = dict(object_to_id)
        num_objects = len(self.object_to_id)
        xs = []
        ys = []
        for record in records:
            object_id = self.object_to_id[str(record["object_name"])]
            xs.append(bbox_ridge_features(record["bbox_norm"], object_id, num_objects))
            ys.append(record["target_xyz"])
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        reg = self.alpha * np.eye(x_aug.shape[1], dtype=np.float64)
        reg[-1, -1] = 0.0
        self.weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y).astype(np.float32)
        return self

    def predict_record(self, record: dict[str, Any]) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("RidgeAnchor is not fitted.")
        num_objects = len(self.object_to_id)
        object_id = self.object_to_id.get(str(record["object_name"]), -1)
        feat = np.asarray(bbox_ridge_features(record["bbox_norm"], object_id, num_objects), dtype=np.float32)
        feat_aug = np.concatenate([feat, np.ones((1,), dtype=np.float32)], axis=0)
        return (feat_aug @ self.weights).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "weights": None if self.weights is None else self.weights.tolist(),
            "object_to_id": self.object_to_id,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "RidgeAnchor":
        obj = cls(alpha=float(state.get("alpha", 1.0)))
        obj.object_to_id = dict(state.get("object_to_id", {}))
        weights = state.get("weights")
        obj.weights = None if weights is None else np.asarray(weights, dtype=np.float32)
        return obj


class IdentityAnchor:
    def predict_record(self, record: dict[str, Any]) -> np.ndarray:
        return np.zeros((3,), dtype=np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {}


def build_anchor(target_mode: str, train_records: list[dict[str, Any]], object_to_id: dict[str, int], alpha: float = 1.0):
    if target_mode == "direct":
        return IdentityAnchor()
    if target_mode == "residual_class_mean":
        return ClassMeanAnchor().fit(train_records)
    if target_mode == "residual_bbox_ridge":
        return RidgeAnchor(alpha=alpha).fit(train_records, object_to_id)
    raise ValueError(f"Unsupported target_mode: {target_mode}")

