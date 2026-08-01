"""Go2 route and sparse-waypoint dataset backed by pct_scene LeRobot exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset

from starVLA.dataloader.go2_waypoints import (
    NAV_STAGES,
    WaypointExtractionConfig,
    contiguous_stage_segments,
    extract_sparse_waypoints,
    future_waypoint_chunk,
)


MAIN_ROUTES = ("nav", "grasp", "place", "done", "recover")
DEFAULT_ROUTE_TOKENS = {
    "nav": "<|nav|>",
    "grasp": "<|grasp|>",
    "place": "<|place|>",
    "done": "<|done|>",
    "recover": "<|recover|>",
}
DEFAULT_SUBTASKS = {
    "grasp": "Grasp the target object.",
    "place": "Place the held object.",
    "done": "Task completed.",
    "recover": "Recover and realign with the target.",
}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass
class _Episode:
    episode_index: int
    task_indices: np.ndarray
    poses: np.ndarray
    base_velocity: np.ndarray
    stages: np.ndarray
    subtasks: np.ndarray
    instructions: np.ndarray
    done: np.ndarray
    waypoint_indices_by_frame: dict[int, tuple[np.ndarray, int, int]]


def _load_task_instructions(tasks_path: Path) -> dict[int, str]:
    rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No task definitions found in {tasks_path}")

    task_instructions: dict[int, str] = {}
    for row_index, row in enumerate(rows):
        task_index = int(row.get("task_index", row_index))
        instruction = str(row.get("task", row.get("instruction", ""))).strip()
        if not instruction:
            raise ValueError(f"Task {task_index} has no task/instruction text in {tasks_path}")
        if task_index in task_instructions:
            raise ValueError(f"Duplicate task_index={task_index} in {tasks_path}")
        task_instructions[task_index] = instruction
    return task_instructions


class Go2WaypointRouterDataset(Dataset):
    """Five-route Go2 supervision with Flow Matching targets only on NAV frames."""

    def __init__(self, data_cfg: Any):
        super().__init__()
        self.data_cfg = data_cfg
        self.root = Path(str(_cfg_get(data_cfg, "root", ""))).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Go2 dataset root does not exist: {self.root}")

        info_path = self.root / "meta" / "info.json"
        self.info = json.loads(info_path.read_text())
        self.fps = float(self.info["fps"])
        self.image_size = tuple(int(v) for v in _cfg_get(data_cfg, "image_size", [224, 224]))
        if len(self.image_size) != 2:
            raise ValueError(f"image_size must be [width,height], got {self.image_size}")
        self.action_horizon = int(_cfg_get(data_cfg, "action_horizon", 4))
        self.include_state = bool(_cfg_get(data_cfg, "include_state", True))
        self.subtask_start_token = str(_cfg_get(data_cfg, "subtask_start_token", "<|subtask|>"))
        self.subtask_end_token = str(_cfg_get(data_cfg, "subtask_end_token", "<|end_subtask|>"))

        configured_tokens = _cfg_get(data_cfg, "route_tokens", {}) or {}
        self.route_tokens = {
            route: str(_cfg_get(configured_tokens, route, DEFAULT_ROUTE_TOKENS[route]))
            for route in MAIN_ROUTES
        }
        self.main_routes = tuple(str(route) for route in _cfg_get(data_cfg, "main_routes", MAIN_ROUTES))
        if set(self.main_routes) != set(MAIN_ROUTES):
            raise ValueError(f"main_routes must contain exactly {MAIN_ROUTES}, got {self.main_routes}")

        bbox_cfg = _cfg_get(data_cfg, "bbox", None)
        if bool(_cfg_get(bbox_cfg, "train_enabled", False)):
            raise ValueError("Go2WaypointRouterDataset has no bbox labels; set bbox.train_enabled=false")

        extraction_cfg = WaypointExtractionConfig(
            rdp_epsilon_m=float(_cfg_get(data_cfg, "rdp_epsilon_m", 0.12)),
            yaw_metric_scale_m_per_rad=float(_cfg_get(data_cfg, "yaw_metric_scale_m_per_rad", 0.45)),
            max_translation_m=float(_cfg_get(data_cfg, "max_translation_m", 0.50)),
            max_yaw_rad=np.deg2rad(float(_cfg_get(data_cfg, "max_yaw_deg", 25.0))),
        )
        self.waypoint_cfg = extraction_cfg

        tasks_path = self.root / "meta" / "tasks.jsonl"
        self.task_instructions = _load_task_instructions(tasks_path)
        self.router_prompt = str(
            _cfg_get(
                data_cfg,
                "router_prompt",
                "{instruction}\nChoose the current control route: NAV, GRASP, PLACE, DONE, or RECOVER. "
                "Output exactly one route token and one short subtask.",
            )
        )

        episode_start = int(_cfg_get(data_cfg, "episode_start", 0))
        num_episodes = _cfg_get(data_cfg, "num_episodes", None)
        parquet_paths = sorted((self.root / "data").glob("chunk-*/episode_*.parquet"))
        if num_episodes is None:
            selected_paths = parquet_paths[episode_start:]
        else:
            selected_paths = parquet_paths[episode_start : episode_start + int(num_episodes)]
        if not selected_paths:
            raise ValueError(
                f"No episodes selected from {len(parquet_paths)} files with "
                f"episode_start={episode_start}, num_episodes={num_episodes}"
            )

        route_strides_cfg = _cfg_get(data_cfg, "route_frame_stride", {}) or {}
        self.route_stride = {
            route: max(1, int(_cfg_get(route_strides_cfg, route, 1))) for route in MAIN_ROUTES
        }
        self.done_repeat = max(1, int(_cfg_get(data_cfg, "done_repeat", 1)))
        self.episodes: dict[int, _Episode] = {}
        self.samples: list[tuple[int, int]] = []
        self._video_cache: dict[tuple[int, str], VideoReader] = {}

        for parquet_path in selected_paths:
            episode = self._load_episode(parquet_path)
            self.episodes[episode.episode_index] = episode
            route_counters = {route: 0 for route in MAIN_ROUTES}
            for frame_index in range(len(episode.poses)):
                route = self._route_for_frame(episode, frame_index)
                counter = route_counters[route]
                route_counters[route] += 1
                if counter % self.route_stride[route] != 0:
                    continue
                repeat = self.done_repeat if route == "done" else 1
                self.samples.extend([(episode.episode_index, frame_index)] * repeat)

    def _load_episode(self, parquet_path: Path) -> _Episode:
        columns = [
            "episode_index",
            "task_index",
            "observation.state",
            "observation.base_velocity",
            "task_stage",
            "subtask",
            "instruction",
            "next.done",
        ]
        table = pq.read_table(parquet_path, columns=columns)
        episode_index = int(table["episode_index"][0].as_py())
        task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        unknown_task_indices = sorted(set(task_indices.tolist()) - set(self.task_instructions))
        if unknown_task_indices:
            raise ValueError(
                f"Episode {episode_index} references task_index values missing from meta/tasks.jsonl: "
                f"{unknown_task_indices}"
            )
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        poses = states[:, [0, 1, 3]]
        stages = np.asarray(table["task_stage"].to_pylist(), dtype=object)
        waypoint_by_frame: dict[int, tuple[np.ndarray, int, int]] = {}
        for stage, start, stop in contiguous_stage_segments(stages):
            if stage not in NAV_STAGES:
                continue
            local_indices, _ = extract_sparse_waypoints(poses[start:stop], self.waypoint_cfg)
            for frame_index in range(start, stop):
                waypoint_by_frame[frame_index] = (local_indices, start, stop)

        return _Episode(
            episode_index=episode_index,
            task_indices=task_indices,
            poses=poses,
            base_velocity=np.asarray(table["observation.base_velocity"].to_pylist(), dtype=np.float32),
            stages=stages,
            subtasks=np.asarray(table["subtask"].to_pylist(), dtype=object),
            instructions=np.asarray(table["instruction"].to_pylist(), dtype=object),
            done=np.asarray(table["next.done"].to_pylist(), dtype=bool),
            waypoint_indices_by_frame=waypoint_by_frame,
        )

    @staticmethod
    def _route_for_frame(episode: _Episode, frame_index: int) -> str:
        if bool(episode.done[frame_index]):
            return "done"
        stage = str(episode.stages[frame_index])
        if stage in NAV_STAGES:
            return "nav"
        if stage == "pick":
            return "grasp"
        if stage == "place":
            return "place"
        return "recover"

    def _video_path(self, episode_index: int, camera: str) -> Path:
        chunk = episode_index // int(self.info.get("chunks_size", 1000))
        template = str(
            self.info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
        )
        video_key = f"observation.images.{camera}"
        relative = template.format(
            episode_chunk=chunk,
            episode_index=episode_index,
            video_key=video_key,
        )
        return self.root / relative

    def _image(self, episode_index: int, frame_index: int, camera: str) -> Image.Image:
        cache_key = (episode_index, camera)
        reader = self._video_cache.get(cache_key)
        if reader is None:
            reader = VideoReader(str(self._video_path(episode_index, camera)), ctx=cpu(0), num_threads=1)
            # Keep memory bounded per DataLoader worker.
            if len(self._video_cache) >= 8:
                self._video_cache.pop(next(iter(self._video_cache)))
            self._video_cache[cache_key] = reader
        image = Image.fromarray(reader[frame_index].asnumpy())
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        return image.resize(self.image_size, resampling)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, frame_index = self.samples[index]
        episode = self.episodes[episode_index]
        route = self._route_for_frame(episode, frame_index)
        if route == "nav":
            subtask = str(episode.instructions[frame_index]).strip()
            if not subtask:
                subtask = "Approach and align with the target."
        else:
            subtask = DEFAULT_SUBTASKS[route]

        route_token = self.route_tokens[route]
        solution = f"{route_token}{self.subtask_start_token}{subtask}{self.subtask_end_token}"
        task_index = int(episode.task_indices[frame_index])
        task_instruction = self.task_instructions[task_index]
        output: dict[str, Any] = {
            "image": [
                self._image(episode_index, frame_index, "front"),
                self._image(episode_index, frame_index, "wrist"),
            ],
            "lang": self.router_prompt.format(instruction=task_instruction),
            "solution": solution,
            "route": route,
            "route_token": route_token,
            "subtask_text": subtask,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "task_index": task_index,
        }

        if route == "nav":
            sparse_indices, segment_start, stage_stop = episode.waypoint_indices_by_frame[frame_index]
            segment_poses = episode.poses[segment_start:stage_stop]
            local_frame = frame_index - segment_start
            waypoints, valid_mask = future_waypoint_chunk(
                segment_poses,
                local_frame,
                sparse_indices,
                self.action_horizon,
            )
            output["action"] = waypoints.astype(np.float16)
            output["action_mask"] = valid_mask.astype(np.float16)
            if self.include_state:
                output["state"] = episode.base_velocity[frame_index][None, :].astype(np.float16)
        return output


def collate_fn_go2(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def get_go2_waypoint_dataset(data_cfg: Any, **_: Any) -> Go2WaypointRouterDataset:
    return Go2WaypointRouterDataset(data_cfg)
