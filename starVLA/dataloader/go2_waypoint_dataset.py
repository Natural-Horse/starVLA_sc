"""Go2 route and sparse-waypoint dataset backed by pct_scene LeRobot exports."""

from __future__ import annotations

import json
import re
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
ACTION_DIM = 10
NAV_ACTION_SLICE = slice(0, 3)
ARM_ACTION_SLICE = slice(3, 10)
DEFAULT_ROUTE_TOKENS = {
    "nav": "<|nav|>",
    "grasp": "<|grasp|>",
    "place": "<|place|>",
    "done": "<|done|>",
    "recover": "<|recover|>",
}
EGOCENTRIC_DIRECTIONS = (
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
)
TURN_INSTRUCTION_PATTERN = re.compile(
    r"^Turn toward your (" + "|".join(EGOCENTRIC_DIRECTIONS) + r") to find the box\b"
)
GLOBAL_INSTRUCTION_PATTERN = re.compile(
    r"^(?P<base>.+?) Box1 is to the robot's "
    r"(?P<box1>" + "|".join(EGOCENTRIC_DIRECTIONS) + r") from its initial pose\. "
    r"Box2 is to the robot's "
    r"(?P<box2>" + "|".join(EGOCENTRIC_DIRECTIONS) + r") "
    r"from its first pose after grasping\.$"
)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _episode_offsets(root_infos: list[dict[str, Any]]) -> list[int]:
    """Return per-root episode index offsets so multiple datasets share one index space."""
    offsets: list[int] = []
    total = 0
    for root_info in root_infos:
        offsets.append(total)
        total += int(root_info.get("total_episodes", 0))
    return offsets


def _episode_offset(root_index: int, root_infos: list[dict[str, Any]]) -> int:
    return _episode_offsets(root_infos)[root_index]


def _episode_location(
    episode_index: int,
    root_infos: list[dict[str, Any]],
) -> tuple[int, int]:
    """Map a global episode index back to (root_index, local_episode_index)."""
    offsets = _episode_offsets(root_infos)
    for root_index in range(len(root_infos) - 1, -1, -1):
        if episode_index >= offsets[root_index]:
            return root_index, episode_index - offsets[root_index]
    raise ValueError(f"episode_index {episode_index} is out of range")


@dataclass
class _Episode:
    episode_index: int
    task_indices: np.ndarray
    poses: np.ndarray
    base_velocity: np.ndarray
    actions: np.ndarray
    stages: np.ndarray
    subtasks: np.ndarray
    instructions: np.ndarray
    done: np.ndarray
    waypoint_indices_by_frame: dict[int, tuple[np.ndarray, int, int]]
    global_instruction: str


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


def _first_turn_direction(
    stages: np.ndarray,
    instructions: np.ndarray,
    *,
    stage: str,
) -> str:
    for frame_stage, instruction in zip(stages, instructions):
        if str(frame_stage) != stage:
            continue
        match = TURN_INSTRUCTION_PATTERN.match(str(instruction).strip())
        if match is not None:
            return match.group(1)
    raise ValueError(f"stage={stage} has no canonical eight-direction turn instruction")


def _build_episode_instruction(
    task_instruction: str,
    stages: np.ndarray,
    instructions: np.ndarray,
) -> str:
    box1_direction = _first_turn_direction(
        stages, instructions, stage="nav_to_pick"
    )
    box2_direction = _first_turn_direction(
        stages, instructions, stage="nav_to_place"
    )
    normalized_task_instruction = str(task_instruction).strip()
    existing = GLOBAL_INSTRUCTION_PATTERN.match(normalized_task_instruction)
    if existing is not None:
        existing_directions = (existing.group("box1"), existing.group("box2"))
        expected_directions = (box1_direction, box2_direction)
        if existing_directions != expected_directions:
            raise ValueError(
                "global instruction directions disagree with local instructions: "
                f"global={existing_directions} local={expected_directions}"
            )
        return normalized_task_instruction
    return (
        f"{normalized_task_instruction} "
        f"Box1 is to the robot's {box1_direction} from its initial pose. "
        f"Box2 is to the robot's {box2_direction} from its first pose after grasping."
    )


class Go2WaypointRouterDataset(Dataset):
    """Five-route routing with masked NAV and Cartesian-arm Flow Matching targets."""

    def __init__(self, data_cfg: Any):
        super().__init__()
        self.data_cfg = data_cfg
        raw_roots = _cfg_get(data_cfg, "root", "")
        if not raw_roots:
            raise ValueError("Go2 dataset requires datasets.router_data.root")
        if isinstance(raw_roots, str):
            roots = [Path(raw_roots).expanduser().resolve()]
        else:
            roots = [Path(str(item)).expanduser().resolve() for item in raw_roots]
        for root in roots:
            if not root.exists():
                raise FileNotFoundError(f"Go2 dataset root does not exist: {root}")
        self.roots = roots
        self.root = self.roots[0]

        root_infos: list[dict[str, Any]] = []
        for root in self.roots:
            info_path = root / "meta" / "info.json"
            root_infos.append(json.loads(info_path.read_text()))
        self.root_infos = root_infos
        self.info = root_infos[0]
        for extra in root_infos[1:]:
            if float(extra.get("fps", -1.0)) != float(self.info.get("fps", -2.0)):
                raise ValueError("All Go2 dataset roots must share the same fps")
        self.fps = float(self.info["fps"])
        self.image_size = tuple(int(v) for v in _cfg_get(data_cfg, "image_size", [224, 224]))
        if len(self.image_size) != 2:
            raise ValueError(f"image_size must be [width,height], got {self.image_size}")
        self.action_horizon = int(_cfg_get(data_cfg, "action_horizon", 4))
        self.include_state = bool(_cfg_get(data_cfg, "include_state", False))
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

        task_instructions: dict[int, str] = {}
        for root in self.roots:
            tasks_path = root / "meta" / "tasks.jsonl"
            root_tasks = _load_task_instructions(tasks_path)
            for task_index, instruction in root_tasks.items():
                if task_index in task_instructions and task_instructions[task_index] != instruction:
                    raise ValueError(
                        f"Go2 dataset roots disagree on task_index={task_index}: "
                        f"{task_instructions[task_index]!r} vs {instruction!r}"
                    )
                task_instructions[task_index] = instruction
        self.task_instructions = task_instructions
        self.router_prompt = str(
            _cfg_get(
                data_cfg,
                "router_prompt",
                "{instruction}\nChoose the current control route: NAV, GRASP, PLACE, DONE, or RECOVER. "
                "Output exactly one route token and one local subtask instruction.",
            )
        )

        episode_start = int(_cfg_get(data_cfg, "episode_start", 0))
        num_episodes = _cfg_get(data_cfg, "num_episodes", None)
        parquet_paths: list[tuple[int, Path]] = []
        for root_index, root in enumerate(self.roots):
            for path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
                parquet_paths.append((root_index, path))
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
        include_routes_cfg = _cfg_get(data_cfg, "include_routes", MAIN_ROUTES)
        if isinstance(include_routes_cfg, str):
            include_routes = [route.strip() for route in include_routes_cfg.split(",") if route.strip()]
        else:
            include_routes = [str(route) for route in include_routes_cfg]
        unknown_routes = sorted(set(include_routes) - set(MAIN_ROUTES))
        if not include_routes or unknown_routes:
            raise ValueError(
                f"include_routes must be a non-empty subset of {MAIN_ROUTES}, got {include_routes}"
            )
        self.include_routes = frozenset(include_routes)
        self.done_repeat = max(1, int(_cfg_get(data_cfg, "done_repeat", 1)))
        self.episodes: dict[int, _Episode] = {}
        self.samples: list[tuple[int, int]] = []
        self._video_cache: dict[tuple[int, str], VideoReader] = {}

        for root_index, parquet_path in selected_paths:
            episode = self._load_episode(parquet_path, root_index=root_index)
            self.episodes[episode.episode_index] = episode
            route_counters = {route: 0 for route in MAIN_ROUTES}
            for frame_index in range(len(episode.poses)):
                route = self._route_for_frame(episode, frame_index)
                counter = route_counters[route]
                route_counters[route] += 1
                if route not in self.include_routes:
                    continue
                if counter % self.route_stride[route] != 0:
                    continue
                repeat = self.done_repeat if route == "done" else 1
                self.samples.extend([(episode.episode_index, frame_index)] * repeat)

    def save_dataset_statistics(self, out_path: Path | str) -> None:
        """Write the go2 dataset statistics file required by checkpoint loading.

        The go2 flow-matching pipeline keeps actions in physical units and does
        not use these statistics for un-normalization, but ``read_mode_config``
        asserts the file exists next to ``config.yaml``, so every training run
        writes it together with the run config.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        actions: list[np.ndarray] = []
        for episode in self.episodes.values():
            if episode.actions.size:
                actions.append(episode.actions)
        if not actions:
            raise ValueError("cannot build go2 dataset statistics without any actions")
        stacked = np.concatenate(actions, axis=0).astype(np.float64)
        q01 = np.quantile(stacked, 0.01, axis=0).tolist()
        q99 = np.quantile(stacked, 0.99, axis=0).tolist()
        mask = [True] * int(stacked.shape[1])
        stats = {
            "go2_waypoint_router_dataset": {
                "action": {"q01": q01, "q99": q99, "mask": mask},
                "num_trajectories": int(len(self.episodes)),
                "num_transitions": int(stacked.shape[0]),
            }
        }
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    def _load_episode(self, parquet_path: Path, *, root_index: int) -> _Episode:
        columns = [
            "episode_index",
            "task_index",
            "observation.state",
            "observation.base_velocity",
            "action",
            "task_stage",
            "subtask",
            "instruction",
            "next.done",
        ]
        table = pq.read_table(parquet_path, columns=columns)
        episode_index = int(table["episode_index"][0].as_py())
        episode_index = _episode_offset(root_index, self.root_infos) + episode_index
        task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        unknown_task_indices = sorted(set(task_indices.tolist()) - set(self.task_instructions))
        if unknown_task_indices:
            raise ValueError(
                f"Episode {episode_index} references task_index values missing from meta/tasks.jsonl: "
                f"{unknown_task_indices}"
            )
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        poses = states[:, [0, 1, 3]]
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(
                f"Episode {episode_index} action must have shape [T,{ACTION_DIM}], got {actions.shape}"
            )
        stages = np.asarray(table["task_stage"].to_pylist(), dtype=object)
        waypoint_by_frame: dict[int, tuple[np.ndarray, int, int]] = {}
        for stage, start, stop in contiguous_stage_segments(stages):
            if stage not in NAV_STAGES:
                continue
            local_indices, _ = extract_sparse_waypoints(poses[start:stop], self.waypoint_cfg)
            for frame_index in range(start, stop):
                waypoint_by_frame[frame_index] = (local_indices, start, stop)

        instructions = np.asarray(table["instruction"].to_pylist(), dtype=object)
        if any(not str(instruction).strip() for instruction in instructions):
            raise ValueError(f"Episode {episode_index} contains an empty local instruction")
        unique_task_indices = set(task_indices.tolist())
        if len(unique_task_indices) != 1:
            raise ValueError(
                f"Episode {episode_index} must use exactly one task_index, got "
                f"{sorted(unique_task_indices)}"
            )
        task_instruction = self.task_instructions[next(iter(unique_task_indices))]

        return _Episode(
            episode_index=episode_index,
            task_indices=task_indices,
            poses=poses,
            base_velocity=np.asarray(table["observation.base_velocity"].to_pylist(), dtype=np.float32),
            actions=actions,
            stages=stages,
            subtasks=np.asarray(table["subtask"].to_pylist(), dtype=object),
            instructions=instructions,
            done=np.asarray(table["next.done"].to_pylist(), dtype=bool),
            waypoint_indices_by_frame=waypoint_by_frame,
            global_instruction=_build_episode_instruction(
                task_instruction,
                stages,
                instructions,
            ),
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
        root_index, local_episode_index = _episode_location(episode_index, self.root_infos)
        root_info = self.root_infos[root_index]
        chunk = local_episode_index // int(root_info.get("chunks_size", 1000))
        template = str(
            root_info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
        )
        video_key = f"observation.images.{camera}"
        relative = template.format(
            episode_chunk=chunk,
            episode_index=local_episode_index,
            video_key=video_key,
        )
        return self.roots[root_index] / relative

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
        subtask = str(episode.instructions[frame_index]).strip()
        phase_label = str(episode.subtasks[frame_index]).strip()

        route_token = self.route_tokens[route]
        solution = f"{route_token}{self.subtask_start_token}{subtask}{self.subtask_end_token}"
        task_index = int(episode.task_indices[frame_index])
        output: dict[str, Any] = {
            "image": [
                self._image(episode_index, frame_index, "front"),
                self._image(episode_index, frame_index, "wrist"),
            ],
            "lang": self.router_prompt.format(instruction=episode.global_instruction),
            "solution": solution,
            "route": route,
            "route_token": route_token,
            "subtask_text": subtask,
            "phase_label": phase_label,
            "global_instruction": episode.global_instruction,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "task_index": task_index,
        }

        if route in {"nav", "grasp", "place"}:
            action = np.zeros((self.action_horizon, ACTION_DIM), dtype=np.float32)
            action_dim_mask = np.zeros((ACTION_DIM,), dtype=np.float32)
            state = np.zeros((1, ACTION_DIM), dtype=np.float32)

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
            action[:, NAV_ACTION_SLICE] = waypoints
            action_dim_mask[NAV_ACTION_SLICE] = 1.0
            if self.include_state:
                state[0, NAV_ACTION_SLICE] = episode.base_velocity[frame_index]
        elif route in {"grasp", "place"}:
            stage = str(episode.stages[frame_index])
            stage_stop = frame_index + 1
            while stage_stop < len(episode.stages) and str(episode.stages[stage_stop]) == stage:
                stage_stop += 1
            valid_count = min(self.action_horizon, stage_stop - frame_index)
            source = episode.actions[frame_index : frame_index + valid_count, ARM_ACTION_SLICE]
            action[:valid_count, ARM_ACTION_SLICE] = source
            action[valid_count:, ARM_ACTION_SLICE] = source[-1]
            valid_mask = np.arange(self.action_horizon) < valid_count
            action_dim_mask[ARM_ACTION_SLICE] = 1.0
            if self.include_state:
                previous_index = max(0, frame_index - 1)
                state[0, ARM_ACTION_SLICE] = episode.actions[previous_index, ARM_ACTION_SLICE]

        if route in {"nav", "grasp", "place"}:
            output["action"] = action.astype(np.float16)
            output["action_mask"] = valid_mask.astype(np.float16)
            output["action_dim_mask"] = action_dim_mask.astype(np.float16)
            if self.include_state:
                output["state"] = state.astype(np.float16)
        return output


def collate_fn_go2(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def get_go2_waypoint_dataset(data_cfg: Any, **_: Any) -> Go2WaypointRouterDataset:
    return Go2WaypointRouterDataset(data_cfg)
