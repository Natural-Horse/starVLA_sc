#!/usr/bin/env python3
"""Analyze pct_scene LeRobot trajectories and sparse Go2 waypoint labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from starVLA.dataloader.go2_waypoints import (
    NAV_STAGES,
    WaypointExtractionConfig,
    contiguous_stage_segments,
    extract_sparse_waypoints,
    wrap_to_pi,
)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "std": float(array.std()),
    }


def _pose_distribution(poses: list[list[float]]) -> dict:
    array = np.asarray(poses, dtype=np.float64)
    yaw = array[:, 2]
    mean_yaw = float(np.arctan2(np.sin(yaw).mean(), np.cos(yaw).mean()))
    resultant = float(np.hypot(np.sin(yaw).mean(), np.cos(yaw).mean()))
    circular_std = float(np.sqrt(max(0.0, -2.0 * np.log(max(resultant, 1e-12)))))
    return {
        "x": _summary(array[:, 0].tolist()),
        "y": _summary(array[:, 1].tolist()),
        "yaw_circular_mean_rad": mean_yaw,
        "yaw_circular_std_rad": circular_std,
    }


def analyze(dataset_root: Path, cfg: WaypointExtractionConfig) -> dict:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_root / 'data'}")

    records = []
    episode_durations = []
    episode_frames = []
    route_counts: dict[str, int] = {}
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=["episode_index", "observation.state", "task_stage"])
        episode_index = int(table["episode_index"][0].as_py())
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        stages = np.asarray(table["task_stage"].to_pylist(), dtype=object)
        poses = states[:, [0, 1, 3]]
        episode_frames.append(len(poses))
        episode_durations.append(len(poses) / fps)

        for stage in stages:
            route_counts[str(stage)] = route_counts.get(str(stage), 0) + 1

        for stage, start, stop in contiguous_stage_segments(stages):
            if stage not in NAV_STAGES:
                continue
            segment = poses[start:stop]
            waypoint_indices, _ = extract_sparse_waypoints(segment, cfg)
            xy_steps = np.linalg.norm(np.diff(segment[:, :2], axis=0), axis=1)
            yaw_steps = np.abs(wrap_to_pi(np.diff(segment[:, 2])))
            displacement = float(np.linalg.norm(segment[-1, :2] - segment[0, :2]))
            records.append(
                {
                    "episode_index": episode_index,
                    "stage": stage,
                    "start_frame": start,
                    "stop_frame_exclusive": stop,
                    "frames": stop - start,
                    "duration_s": (stop - start) / fps,
                    "trajectory_length_m": float(xy_steps.sum()),
                    "displacement_m": displacement,
                    "absolute_yaw_change_rad": float(yaw_steps.sum()),
                    "net_yaw_change_rad": float(wrap_to_pi(segment[-1, 2] - segment[0, 2])),
                    "start_xyyaw": segment[0].tolist(),
                    "end_xyyaw": segment[-1].tolist(),
                    "sparse_waypoint_count": int(len(waypoint_indices)),
                    "compression_ratio": float(len(waypoint_indices) / max(len(segment), 1)),
                }
            )

    by_stage = {}
    for stage in NAV_STAGES:
        stage_records = [record for record in records if record["stage"] == stage]
        by_stage[stage] = {
            "segments": len(stage_records),
            "duration_s": _summary([record["duration_s"] for record in stage_records]),
            "trajectory_length_m": _summary([record["trajectory_length_m"] for record in stage_records]),
            "displacement_m": _summary([record["displacement_m"] for record in stage_records]),
            "absolute_yaw_change_rad": _summary(
                [record["absolute_yaw_change_rad"] for record in stage_records]
            ),
            "sparse_waypoint_count": _summary(
                [record["sparse_waypoint_count"] for record in stage_records]
            ),
            "compression_ratio": _summary([record["compression_ratio"] for record in stage_records]),
            "start_pose_distribution": _pose_distribution(
                [record["start_xyyaw"] for record in stage_records]
            ),
            "end_pose_distribution": _pose_distribution(
                [record["end_xyyaw"] for record in stage_records]
            ),
        }

    return {
        "dataset_root": str(dataset_root),
        "episodes": len(parquet_files),
        "fps": fps,
        "total_frames": int(sum(episode_frames)),
        "episode_frames": _summary(episode_frames),
        "episode_duration_s": _summary(episode_durations),
        "task_stage_frame_counts": route_counts,
        "waypoint_config": {
            "rdp_epsilon_m": cfg.rdp_epsilon_m,
            "yaw_metric_scale_m_per_rad": cfg.yaw_metric_scale_m_per_rad,
            "max_translation_m": cfg.max_translation_m,
            "max_yaw_rad": cfg.max_yaw_rad,
        },
        "navigation": by_stage,
        "segments": records,
    }


def _fmt(summary: dict, unit: str = "") -> str:
    return (
        f"{summary['mean']:.3f}{unit} +/- {summary['std']:.3f}{unit} "
        f"(min {summary['min']:.3f}, median {summary['median']:.3f}, max {summary['max']:.3f})"
    )


def markdown_report(report: dict) -> str:
    lines = [
        "# Go2 Navigation Trajectory Analysis",
        "",
        f"- Dataset: `{report['dataset_root']}`",
        f"- Episodes: `{report['episodes']}`",
        f"- Frames: `{report['total_frames']}` at `{report['fps']}` FPS",
        f"- Episode duration: {_fmt(report['episode_duration_s'], ' s')}",
        "",
        "## Sparse Waypoint Parameters",
        "",
    ]
    for key, value in report["waypoint_config"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Navigation Stages", ""])
    for stage, values in report["navigation"].items():
        start = values["start_pose_distribution"]
        end = values["end_pose_distribution"]
        lines.extend(
            [
                f"### `{stage}`",
                "",
                f"- Segments: `{values['segments']}`",
                f"- Duration: {_fmt(values['duration_s'], ' s')}",
                f"- Trajectory length: {_fmt(values['trajectory_length_m'], ' m')}",
                f"- Start x/y mean: `{start['x']['mean']:.3f}`, `{start['y']['mean']:.3f}` m",
                f"- End x/y mean: `{end['x']['mean']:.3f}`, `{end['y']['mean']:.3f}` m",
                f"- Sparse waypoints: {_fmt(values['sparse_waypoint_count'])}",
                f"- Retained-frame ratio: {_fmt(values['compression_ratio'])}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--rdp-epsilon-m", type=float, default=0.12)
    parser.add_argument("--yaw-metric-scale", type=float, default=0.45)
    parser.add_argument("--max-translation-m", type=float, default=0.50)
    parser.add_argument("--max-yaw-deg", type=float, default=25.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = WaypointExtractionConfig(
        rdp_epsilon_m=args.rdp_epsilon_m,
        yaw_metric_scale_m_per_rad=args.yaw_metric_scale,
        max_translation_m=args.max_translation_m,
        max_yaw_rad=np.deg2rad(args.max_yaw_deg),
    )
    report = analyze(args.dataset_root.resolve(), cfg)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    rendered = markdown_report(report)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
