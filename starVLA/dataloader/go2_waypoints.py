"""Sparse SE(2) waypoint extraction for Go2 navigation demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NAV_STAGES = ("nav_to_pick", "nav_to_place")


@dataclass(frozen=True)
class WaypointExtractionConfig:
    rdp_epsilon_m: float = 0.12
    yaw_metric_scale_m_per_rad: float = 0.45
    max_translation_m: float = 0.50
    max_yaw_rad: float = np.deg2rad(25.0)
    duplicate_translation_m: float = 0.01
    duplicate_yaw_rad: float = np.deg2rad(1.0)


def wrap_to_pi(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def contiguous_stage_segments(stages: list[str] | np.ndarray) -> list[tuple[str, int, int]]:
    """Return half-open contiguous segments as ``(stage, start, stop)``."""
    values = np.asarray(stages, dtype=object)
    if values.size == 0:
        return []
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [values.size]))
    return [(str(values[start]), int(start), int(stop)) for start, stop in zip(starts, stops)]


def _rdp_indices(points: np.ndarray, epsilon: float) -> list[int]:
    if len(points) <= 2:
        return list(range(len(points)))

    start = points[0]
    end = points[-1]
    line = end - start
    line_norm_sq = float(np.dot(line, line))
    if line_norm_sq <= 1e-12:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        projection = np.clip(((points - start) @ line) / line_norm_sq, 0.0, 1.0)
        closest = start + projection[:, None] * line
        distances = np.linalg.norm(points - closest, axis=1)

    split = int(np.argmax(distances))
    if float(distances[split]) <= epsilon:
        return [0, len(points) - 1]

    left = _rdp_indices(points[: split + 1], epsilon)
    right = _rdp_indices(points[split:], epsilon)
    return left[:-1] + [split + idx for idx in right]


def _remove_consecutive_duplicates(poses: np.ndarray, cfg: WaypointExtractionConfig) -> np.ndarray:
    keep = [0]
    for idx in range(1, len(poses) - 1):
        previous = poses[keep[-1]]
        translation = float(np.linalg.norm(poses[idx, :2] - previous[:2]))
        yaw = abs(float(wrap_to_pi(poses[idx, 2] - previous[2])))
        if translation >= cfg.duplicate_translation_m or yaw >= cfg.duplicate_yaw_rad:
            keep.append(idx)
    if len(poses) > 1:
        keep.append(len(poses) - 1)
    return np.asarray(sorted(set(keep)), dtype=np.int64)


def extract_sparse_waypoints(
    poses_xyyaw: np.ndarray,
    cfg: WaypointExtractionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sparse source indices and corresponding world-frame ``[x,y,yaw]`` poses."""
    cfg = cfg or WaypointExtractionConfig()
    poses = np.asarray(poses_xyyaw, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError(f"poses_xyyaw must have shape [T,3], got {poses.shape}")
    if len(poses) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float64)
    if not np.isfinite(poses).all():
        raise ValueError("poses_xyyaw contains non-finite values")

    unwrapped = poses.copy()
    unwrapped[:, 2] = np.unwrap(unwrapped[:, 2])
    unique_indices = _remove_consecutive_duplicates(unwrapped, cfg)
    unique_poses = unwrapped[unique_indices]

    metric_points = unique_poses.copy()
    metric_points[:, 2] *= cfg.yaw_metric_scale_m_per_rad
    simplified_unique = _rdp_indices(metric_points, cfg.rdp_epsilon_m)
    mandatory = {int(unique_indices[idx]) for idx in simplified_unique}
    mandatory.update({0, len(poses) - 1})

    # Add observed poses when a simplified interval is too large for the controller.
    selected = set(mandatory)
    for left, right in zip(sorted(mandatory)[:-1], sorted(mandatory)[1:]):
        cursor = left
        while cursor < right:
            candidates = np.arange(cursor + 1, right + 1)
            translation = np.linalg.norm(unwrapped[candidates, :2] - unwrapped[cursor, :2], axis=1)
            yaw = np.abs(unwrapped[candidates, 2] - unwrapped[cursor, 2])
            exceeded = np.flatnonzero(
                (translation >= cfg.max_translation_m) | (yaw >= cfg.max_yaw_rad)
            )
            if exceeded.size == 0:
                break
            next_idx = int(candidates[exceeded[0]])
            if next_idx >= right:
                break
            selected.add(next_idx)
            cursor = next_idx

    indices = np.asarray(sorted(selected), dtype=np.int64)
    waypoints = poses[indices].copy()
    waypoints[:, 2] = wrap_to_pi(waypoints[:, 2])
    return indices, waypoints


def world_waypoints_to_body(current_xyyaw: np.ndarray, waypoint_xyyaw: np.ndarray) -> np.ndarray:
    current = np.asarray(current_xyyaw, dtype=np.float64).reshape(3)
    targets = np.asarray(waypoint_xyyaw, dtype=np.float64).reshape(-1, 3)
    delta = targets[:, :2] - current[:2]
    cosine = np.cos(current[2])
    sine = np.sin(current[2])
    body_x = cosine * delta[:, 0] + sine * delta[:, 1]
    body_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    body_yaw = wrap_to_pi(targets[:, 2] - current[2])
    return np.stack((body_x, body_y, body_yaw), axis=-1)


def future_waypoint_chunk(
    poses_xyyaw: np.ndarray,
    current_index: int,
    waypoint_indices: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed-size body-frame chunk of future sparse waypoints and its validity mask."""
    poses = np.asarray(poses_xyyaw, dtype=np.float64)
    sparse_indices = np.asarray(waypoint_indices, dtype=np.int64)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if len(poses) == 0 or len(sparse_indices) == 0:
        raise ValueError("poses and waypoint_indices must be non-empty")
    if not 0 <= current_index < len(poses):
        raise IndexError(f"current_index={current_index} outside trajectory length {len(poses)}")

    future = sparse_indices[sparse_indices > current_index]
    if future.size == 0:
        future = sparse_indices[-1:]
    chosen = future[:horizon]
    valid_count = len(chosen)
    if valid_count < horizon:
        chosen = np.concatenate((chosen, np.repeat(chosen[-1], horizon - valid_count)))

    body = world_waypoints_to_body(poses[current_index], poses[chosen]).astype(np.float32)
    mask = np.zeros((horizon,), dtype=np.float32)
    mask[:valid_count] = 1.0
    return body, mask
