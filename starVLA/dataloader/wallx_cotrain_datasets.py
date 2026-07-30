"""WallX-style cotrain datasets for StarVLA.

This module provides two lightweight LeRobot-based datasets:
1) VLA dataset for action generation training.
2) VLM dataset for bbox text supervision training.

Both datasets read from the same LeRobot source and avoid gr00t metadata
dependencies such as `meta/modality.json`.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

import lerobot.datasets.lerobot_dataset as _lerobot_dataset


def _install_wallx_v2_lerobot_compat() -> None:
    if getattr(_lerobot_dataset, "_starvla_wallx_v2_compat", False):
        return
    original_check = _lerobot_dataset.check_version_compatibility

    def check_version_compatibility(repo_id, version_to_check, current_version, enforce_breaking_major=True):
        repo = str(repo_id)
        version = str(version_to_check).lstrip("v")
        current = str(current_version).lstrip("v")
        if repo.startswith("dzb/lerobot_ego_") and version.startswith("2.") and current.startswith("3."):
            return
        return original_check(repo_id, version_to_check, current_version, enforce_breaking_major)

    _lerobot_dataset.check_version_compatibility = check_version_compatibility
    _lerobot_dataset._starvla_wallx_v2_compat = True


_install_wallx_v2_lerobot_compat()


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Support both dict-style and OmegaConf-style config access."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_scalar_int(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def _to_scalar_bool(x: Any) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(x.item())
    return bool(x)


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angles + torch.pi, 2 * torch.pi) - torch.pi


def _tensor_image_to_pil(image_chw: torch.Tensor, size_wh: tuple[int, int]) -> Image.Image:
    """Convert [C,H,W] float tensor in [0,1] to resized PIL image."""
    arr = image_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr_u8 = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr_u8).resize(size_wh, Image.BILINEAR)


def _parse_task(task_str: str) -> tuple[str, str, str]:
    """Parse '... Catch: xxx. Put: yyy' into (instruction, catch, put)."""
    m = re.search(r"(.*)Catch:\s*(.*)\.\s*Put:\s*(.*)", task_str)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return task_str.strip(), "", ""


def _normalize_target_name(target_name: str) -> str:
    """Normalize parsed target text to avoid prompt artifacts (e.g., trailing dots)."""
    text = str(target_name).strip()
    while text.endswith("."):
        text = text[:-1].rstrip()
    return text


def _is_valid_bbox(bbox: torch.Tensor) -> bool:
    """WallX-style bbox validity: length=4, strictly positive and ordered xyxy."""
    if bbox.numel() != 4:
        return False
    if torch.any(bbox <= 0):
        return False
    x1, y1, x2, y2 = bbox.tolist()
    if x2 <= x1 or y2 <= y1:
        return False
    return True


@dataclass
class _NormStats:
    min: torch.Tensor
    delta: torch.Tensor


@dataclass
class _WallXCommonConfig:
    repo_id: str
    root: str
    image_size: tuple[int, int]
    action_horizon: int
    episode_start: int
    num_episodes: int | None
    include_history_keyframes: bool
    max_history_keyframes: int | None
    history_keyframe_values: tuple[int, ...]
    snap_rotation_to_start: bool
    video_backend: str
    tolerance_s: float

    @classmethod
    def from_cfg(cls, data_cfg: Any) -> "_WallXCommonConfig":
        image_size_cfg = _cfg_get(data_cfg, "image_size", [224, 224])
        try:
            image_size_cfg = list(image_size_cfg)
        except TypeError as exc:
            raise ValueError(f"image_size must be [W,H], got {image_size_cfg}") from exc
        if len(image_size_cfg) != 2:
            raise ValueError(f"image_size must be [W,H], got {image_size_cfg}")
        image_size = (int(image_size_cfg[0]), int(image_size_cfg[1]))

        history_values_cfg = _cfg_get(data_cfg, "history_keyframe_values", [1, 2])
        history_values = tuple(int(v) for v in history_values_cfg)

        max_hist = _cfg_get(data_cfg, "max_history_keyframes", None)
        max_hist = int(max_hist) if max_hist is not None else None

        episode_start = int(_cfg_get(data_cfg, "episode_start", 0))
        num_episodes = _cfg_get(data_cfg, "num_episodes", None)
        num_episodes = int(num_episodes) if num_episodes is not None else None

        return cls(
            repo_id=str(_cfg_get(data_cfg, "repo_id", "dzb/lerobot_ego_data")),
            root=str(_cfg_get(data_cfg, "root", "")),
            image_size=image_size,
            action_horizon=int(_cfg_get(data_cfg, "action_horizon", 16)),
            episode_start=episode_start,
            num_episodes=num_episodes,
            include_history_keyframes=bool(_cfg_get(data_cfg, "include_history_keyframes", True)),
            max_history_keyframes=max_hist,
            history_keyframe_values=history_values,
            snap_rotation_to_start=bool(_cfg_get(data_cfg, "snap_rotation_to_start", False)),
            video_backend=str(_cfg_get(data_cfg, "video_backend", "pyav")),
            tolerance_s=float(_cfg_get(data_cfg, "tolerance_s", 1000.0)),
        )


@dataclass
class _PhotometricAugConfig:
    enabled: bool
    probability: float
    brightness: float
    contrast: float
    saturation: float
    hue: float
    sharpness: float

    @classmethod
    def from_cfg(cls, data_cfg: Any) -> "_PhotometricAugConfig":
        aug_cfg = _cfg_get(data_cfg, "photometric_augmentation", None)
        return cls(
            enabled=bool(_cfg_get(aug_cfg, "enabled", False)),
            probability=float(_cfg_get(aug_cfg, "probability", 1.0)),
            brightness=max(0.0, float(_cfg_get(aug_cfg, "brightness", 0.0))),
            contrast=max(0.0, float(_cfg_get(aug_cfg, "contrast", 0.0))),
            saturation=max(0.0, float(_cfg_get(aug_cfg, "saturation", 0.0))),
            hue=min(max(0.0, float(_cfg_get(aug_cfg, "hue", 0.0))), 0.5),
            sharpness=max(0.0, float(_cfg_get(aug_cfg, "sharpness", 0.0))),
        )


class _WallXLeRobotBase(Dataset):
    """Shared utilities for WallX-style datasets."""

    def __init__(self, data_cfg: Any):
        super().__init__()
        self.data_cfg = data_cfg
        self.common = _WallXCommonConfig.from_cfg(data_cfg)
        self.photometric_aug = _PhotometricAugConfig.from_cfg(data_cfg)

        if not self.common.root:
            raise ValueError("data_cfg.root is required for WallX cotrain datasets")

        meta = LeRobotDatasetMetadata(self.common.repo_id, root=self.common.root)
        fps = float(meta.fps)
        action_h = self.common.action_horizon

        delta_timestamps = {
            "action": [t / fps for t in range(action_h)],
            "keyframe": [t / fps for t in range(action_h)],
        }

        requested_episodes = None
        if self.common.num_episodes is not None:
            if self.common.num_episodes <= 0:
                raise ValueError(
                    f"num_episodes must be > 0 when provided, got {self.common.num_episodes}"
                )
            start = max(0, self.common.episode_start)
            end = min(meta.total_episodes, start + self.common.num_episodes)
            requested_episodes = list(range(start, end))
            if not requested_episodes:
                raise ValueError(
                    "No episodes selected. "
                    f"episode_start={self.common.episode_start}, "
                    f"num_episodes={self.common.num_episodes}, "
                    f"total_episodes={meta.total_episodes}"
                )

        # Pass explicit episode lists to avoid lexicographic parquet ordering when
        # chunk ids grow past three digits (for example chunk-1000 before chunk-101).
        use_direct_subset = requested_episodes is not None
        dataset_episodes = requested_episodes

        self.dataset = LeRobotDataset(
            self.common.repo_id,
            root=self.common.root,
            episodes=dataset_episodes,
            delta_timestamps=delta_timestamps,
            video_backend=self.common.video_backend,
            tolerance_s=self.common.tolerance_s,
        )

        self._selected_frame_indices: list[int] | None = None
        self._episode_start_local: dict[int, int] = {}
        self._source_to_local_index: dict[int, int] = {}
        self._rotation_snap_cache: dict[int, int] = {}

        if requested_episodes is not None and use_direct_subset:
            from_arr = self.dataset.episode_data_index["from"]
            to_arr = self.dataset.episode_data_index["to"]
            expanded_size = max(requested_episodes) + 1
            expanded_from = torch.full((expanded_size,), -1, dtype=from_arr.dtype)
            expanded_to = torch.full((expanded_size,), -1, dtype=to_arr.dtype)
            for ep_local, ep_global in enumerate(requested_episodes):
                ep_global = int(ep_global)
                self._episode_start_local[ep_global] = _to_scalar_int(from_arr[ep_local])
                expanded_from[ep_global] = from_arr[ep_local]
                expanded_to[ep_global] = to_arr[ep_local]
            self.dataset.episode_data_index = {"from": expanded_from, "to": expanded_to}

        if requested_episodes is not None and not use_direct_subset:
            from_arr = self.dataset.episode_data_index["from"]
            to_arr = self.dataset.episode_data_index["to"]
            selected_frame_indices: list[int] = []
            local_cursor = 0

            for ep_global in requested_episodes:
                ep_start = _to_scalar_int(from_arr[ep_global])
                ep_end = _to_scalar_int(to_arr[ep_global])
                self._episode_start_local[int(ep_global)] = local_cursor
                selected_frame_indices.extend(range(ep_start, ep_end))
                local_cursor += max(ep_end - ep_start, 0)

            self._selected_frame_indices = selected_frame_indices
            self._source_to_local_index = {
                int(source_idx): int(local_idx) for local_idx, source_idx in enumerate(selected_frame_indices)
            }

    @staticmethod
    def _sample_photometric_factor(delta: float) -> float:
        if delta <= 0:
            return 1.0
        return max(0.0, 1.0 + random.uniform(-delta, delta))

    def _sample_photometric_params(self) -> dict[str, float] | None:
        aug = self.photometric_aug
        if not aug.enabled:
            return None
        probability = min(max(aug.probability, 0.0), 1.0)
        if random.random() > probability:
            return None
        return {
            "brightness": self._sample_photometric_factor(aug.brightness),
            "contrast": self._sample_photometric_factor(aug.contrast),
            "saturation": self._sample_photometric_factor(aug.saturation),
            "sharpness": self._sample_photometric_factor(aug.sharpness),
            "hue": random.uniform(-aug.hue, aug.hue) if aug.hue > 0 else 0.0,
        }

    def _apply_photometric_augmentation(
        self,
        image: Image.Image,
        params: dict[str, float] | None,
    ) -> Image.Image:
        if params is None:
            return image
        image = TF.adjust_brightness(image, params["brightness"])
        image = TF.adjust_contrast(image, params["contrast"])
        image = TF.adjust_saturation(image, params["saturation"])
        image = TF.adjust_hue(image, params["hue"])
        image = TF.adjust_sharpness(image, params["sharpness"])
        return image

    def _image_from_tensor(
        self,
        image_chw: torch.Tensor,
        photometric_params: dict[str, float] | None = None,
    ) -> Image.Image:
        image = _tensor_image_to_pil(image_chw, self.common.image_size)
        return self._apply_photometric_augmentation(image, photometric_params)

    def __len__(self) -> int:
        if self._selected_frame_indices is not None:
            return len(self._selected_frame_indices)
        return len(self.dataset)

    def _resolve_source_index(self, index: int) -> int:
        if self._selected_frame_indices is None:
            return int(index)

        if index < 0 or index >= len(self._selected_frame_indices):
            raise IndexError(f"index {index} out of range for selected subset of size {len(self)}")
        return int(self._selected_frame_indices[index])

    def _resolve_local_index(self, source_index: int) -> int:
        if self._selected_frame_indices is None:
            return int(source_index)
        local_index = self._source_to_local_index.get(int(source_index))
        if local_index is None:
            raise KeyError(f"source index {source_index} is not in selected subset")
        return int(local_index)

    def _sample_scalar_field(self, sample: dict[str, Any], key: str, default: int = 0) -> int:
        if key not in sample:
            return int(default)
        return int(torch.as_tensor(sample[key]).reshape(-1)[0].item())

    def _snap_rotation_source_index(self, source_index: int) -> int:
        source_index = int(source_index)
        cached = self._rotation_snap_cache.get(source_index)
        if cached is not None:
            return cached

        sample = self.dataset[source_index]
        is_rotate = self._sample_scalar_field(sample, "is_rotate", 0)
        keyframe = self._sample_scalar_field(sample, "keyframe", 0)
        if is_rotate != 1 or keyframe == 1:
            self._rotation_snap_cache[source_index] = source_index
            return source_index

        ep_start = self._resolve_episode_start_index(sample)

        snapped_source = source_index
        for candidate_source in range(source_index, ep_start - 1, -1):
            candidate = self.dataset[candidate_source]
            candidate_is_rotate = self._sample_scalar_field(candidate, "is_rotate", 0)
            if candidate_source != source_index and candidate_is_rotate != 1:
                break
            candidate_keyframe = self._sample_scalar_field(candidate, "keyframe", 0)
            if candidate_is_rotate == 1 and candidate_keyframe == 1:
                snapped_source = candidate_source
                break

        self._rotation_snap_cache[source_index] = int(snapped_source)
        return int(snapped_source)

    def _resolve_training_sample(self, index: int) -> tuple[int, int, dict[str, Any]]:
        local_index = int(index)
        source_index = self._resolve_source_index(local_index)
        if self.common.snap_rotation_to_start:
            source_index = self._snap_rotation_source_index(source_index)
            local_index = self._resolve_local_index(source_index)
        sample = self.dataset[source_index]
        return local_index, source_index, sample

    def _resolve_episode_start_index(self, sample: dict[str, Any]) -> int:
        ep_global = _to_scalar_int(sample["episode_index"])

        ep_start_local = self._episode_start_local.get(ep_global)
        if ep_start_local is not None:
            return ep_start_local

        # No local frame remap: index space is the dataset index space.
        return _to_scalar_int(self.dataset.episode_data_index["from"][ep_global])

    def _collect_history_images(
        self,
        index: int,
        sample: dict[str, Any],
        photometric_params: dict[str, float] | None = None,
    ) -> list[Image.Image]:
        if not self.common.include_history_keyframes:
            return []

        ep_start = self._resolve_episode_start_index(sample)

        history_images: list[Image.Image] = []
        for local_j in range(ep_start, index):
            source_j = self._resolve_source_index(local_j)
            sj = self.dataset[source_j]
            if "keyframe" not in sj:
                continue
            keyframe_val = int(torch.as_tensor(sj["keyframe"]).reshape(-1)[0].item())
            if keyframe_val in self.common.history_keyframe_values:
                history_images.append(self._image_from_tensor(sj["video.front"], photometric_params))

        if self.common.max_history_keyframes is not None and len(history_images) > self.common.max_history_keyframes:
            history_images = history_images[-self.common.max_history_keyframes :]

        return history_images


class WallXVlaDataset(_WallXLeRobotBase):
    """VLA branch dataset for QwenPI cotraining."""

    def __init__(self, data_cfg: Any):
        super().__init__(data_cfg)
        self.action_in_ego = bool(_cfg_get(data_cfg, "action_in_ego", True))
        self.use_delta_action = bool(_cfg_get(data_cfg, "use_delta_action", False))
        self.truncate_keyframe_value = int(_cfg_get(data_cfg, "truncate_keyframe_value", 2))
        self.include_state = bool(_cfg_get(data_cfg, "include_state", False))
        self.normalize_action = bool(_cfg_get(data_cfg, "normalize_action", True))
        self.normalize_state = bool(_cfg_get(data_cfg, "normalize_state", True))

        default_action_prompt_grasp_true = (
            "{instruction}\n"
            "You are performing a drone navigation task. "
            "You now need to pull the {target_name}. Please predict the next action."
        )
        default_action_prompt_grasp_false = (
            "{instruction}\n"
            "You are performing a drone navigation task. "
            "You now need to grasp the {target_name}. Please predict the next action."
        )
        self.action_prompt_template_grasp_true = str(
            _cfg_get(data_cfg, "action_prompt_template_grasp_true", default_action_prompt_grasp_true)
        )
        self.action_prompt_template_grasp_false = str(
            _cfg_get(data_cfg, "action_prompt_template_grasp_false", default_action_prompt_grasp_false)
        )

        self.action_norm_stats, self.state_norm_stats = self._load_norm_stats()

    def _resolve_norm_stats_path(self) -> Path:
        dataset_root = Path(self.common.root)

        preferred = "norm_stats.json"
        if self.use_delta_action:
            preferred = "norm_stats_delta.json"
        if self.action_in_ego:
            preferred = "norm_stats_ego.json"

        candidates = [preferred, "norm_stats_ego.json", "norm_stats_delta.json", "norm_stats.json"]
        visited = set()
        for name in candidates:
            if name in visited:
                continue
            visited.add(name)
            file_path = dataset_root / name
            if file_path.exists():
                return file_path

        raise FileNotFoundError(
            f"No normalization stats file found under `{dataset_root}`. "
            f"Tried: {', '.join(candidates)}"
        )

    def _load_norm_stats(self) -> tuple[_NormStats | None, _NormStats | None]:
        stats_path = self._resolve_norm_stats_path()
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_json = json.load(f)

        norm_stats = stats_json.get("norm_stats", {})
        action_stats = norm_stats.get("action", {})
        state_stats = norm_stats.get("state", {})

        action_norm = None
        if "q01" in action_stats and "q99" in action_stats:
            action_q01 = torch.as_tensor(action_stats["q01"], dtype=torch.float32)
            action_q99 = torch.as_tensor(action_stats["q99"], dtype=torch.float32)
            action_norm = _NormStats(min=action_q01, delta=action_q99 - action_q01)

        state_norm = None
        if "q01" in state_stats and "q99" in state_stats:
            state_q01 = torch.as_tensor(state_stats["q01"], dtype=torch.float32)
            state_q99 = torch.as_tensor(state_stats["q99"], dtype=torch.float32)
            state_norm = _NormStats(min=state_q01, delta=state_q99 - state_q01)

        return action_norm, state_norm

    @staticmethod
    def _normalize_with_stats(x: torch.Tensor, stats: _NormStats) -> torch.Tensor:
        min_stat = stats.min.to(device=x.device, dtype=x.dtype)
        delta = stats.delta.to(device=x.device, dtype=x.dtype)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        x = (x - min_stat) / delta
        x = x * 2 - 1
        return torch.clamp(x, -1.0, 1.0)

    @staticmethod
    def _convert_action_to_ego(agent_pos: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Convert global action trajectory to ego frame."""
        pos = action[..., :3]
        rpy = action[..., 3:6]

        base_pos = agent_pos[:3]
        base_rpy = agent_pos[3:6]
        delta_pos = pos - base_pos

        roll, pitch, yaw = base_rpy
        cx = torch.cos(roll)
        sx = torch.sin(roll)
        cy = torch.cos(pitch)
        sy = torch.sin(pitch)
        cz = torch.cos(yaw)
        sz = torch.sin(yaw)

        rot = torch.stack(
            [
                torch.stack([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx]),
                torch.stack([sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx]),
                torch.stack([-sy, cy * sx, cy * cx]),
            ],
            dim=0,
        ).to(delta_pos)
        delta_pos_ego = delta_pos @ rot
        delta_rpy = _wrap_to_pi(rpy - base_rpy)
        return torch.cat((delta_pos_ego, delta_rpy), dim=-1)

    def _apply_action_mode(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.action_in_ego:
            action = self._convert_action_to_ego(state, action)

        if self.use_delta_action:
            delta_action = torch.zeros_like(action)
            if self.action_in_ego:
                delta_action[0] = action[0]
            else:
                delta_action[0] = action[0] - state
            if action.shape[0] > 1:
                delta_action[1:] = action[1:] - action[:-1]
            action = delta_action
        return action

    def _truncate_with_keyframe(self, action: torch.Tensor, keyframe_chunk: torch.Tensor | None) -> torch.Tensor:
        if keyframe_chunk is None:
            return action
        keyframe_chunk = torch.as_tensor(keyframe_chunk).reshape(-1)
        if keyframe_chunk.numel() != action.shape[0]:
            return action
        future = keyframe_chunk[1:]
        matches = (future == self.truncate_keyframe_value).nonzero(as_tuple=False)
        if matches.numel() == 0:
            return action
        last_rel = int(matches.max().item()) + 1
        if self.use_delta_action:
            action[last_rel:] = 0
        else:
            action[last_rel:] = action[last_rel].clone()
        return action

    def __getitem__(self, index: int) -> dict[str, Any]:
        effective_index, _, sample = self._resolve_training_sample(index)

        photometric_params = self._sample_photometric_params()
        current_image = self._image_from_tensor(sample["video.front"], photometric_params)
        history_images = self._collect_history_images(effective_index, sample, photometric_params)
        images = [current_image] + history_images

        state = torch.as_tensor(sample["state"], dtype=torch.float32)
        action = torch.as_tensor(sample["action"], dtype=torch.float32)
        action = self._apply_action_mode(action, state)
        action = self._truncate_with_keyframe(action, sample.get("keyframe", None))
        if self.normalize_action and self.action_norm_stats is not None:
            action = self._normalize_with_stats(action, self.action_norm_stats)

        task_str = str(sample.get("task", ""))
        instruction, catch_target, put_target = _parse_task(task_str)
        instruction_text = instruction if instruction else task_str
        grasp = _to_scalar_bool(sample.get("grasp", False))
        target_name = put_target if grasp else catch_target
        target_name = _normalize_target_name(target_name)
        if not target_name:
            target_name = "target object"

        action_prompt_template = (
            self.action_prompt_template_grasp_true
            if grasp
            else self.action_prompt_template_grasp_false
        )
        lang = action_prompt_template.format(
            instruction=instruction_text,
            target_name=target_name,
        )

        output = {
            "image": images,
            "lang": lang,
            "action": action.detach().cpu().numpy().astype(np.float16),
        }
        if self.include_state:
            out_state = state
            if self.normalize_state and self.state_norm_stats is not None:
                out_state = self._normalize_with_stats(out_state, self.state_norm_stats)
            output["state"] = out_state.detach().cpu().numpy()[None, :].astype(np.float16)
        return output


class WallXVlmBboxDataset(_WallXLeRobotBase):
    """VLM branch dataset for bbox text supervision."""

    def __init__(self, data_cfg: Any):
        super().__init__(data_cfg)
        default_bbox_prompt = (
            "Please identify the {target_name} in the front view and output its bounding box as "
            "<point>[x1, y1, x2, y2]</point>. "
            "The output must strictly follow the format <point>[x1, y1, x2, y2]</point> without any other text."
        )
        self.bbox_prompt_template = str(_cfg_get(data_cfg, "bbox_prompt_template", default_bbox_prompt))
        self.max_retry = int(_cfg_get(data_cfg, "max_retry", 32))
        self._random = random.Random(int(_cfg_get(data_cfg, "random_seed", 42)))

    def _bbox_solution(self, bbox_xyxy: torch.Tensor, src_hw: tuple[int, int]) -> str | None:
        src_h, src_w = src_hw
        dst_w, dst_h = self.common.image_size
        scale_x = dst_w / float(src_w)
        scale_y = dst_h / float(src_h)
        x1, y1, x2, y2 = bbox_xyxy.tolist()
        sx1 = int(np.clip(round(x1 * scale_x), 0, dst_w - 1))
        sy1 = int(np.clip(round(y1 * scale_y), 0, dst_h - 1))
        sx2 = int(np.clip(round(x2 * scale_x), 0, dst_w - 1))
        sy2 = int(np.clip(round(y2 * scale_y), 0, dst_h - 1))
        if sx1 <= 0 or sy1 <= 0 or sx2 <= 0 or sy2 <= 0:
            return None
        if sx2 <= sx1 or sy2 <= sy1:
            return None
        return f"<point>[{sx1}, {sy1}, {sx2}, {sy2}]</point>"

    def _make_vlm_sample(self, index: int) -> dict[str, Any] | None:
        effective_index, _, sample = self._resolve_training_sample(index)
        bbox = torch.as_tensor(sample.get("bbox", torch.tensor([])), dtype=torch.float32).reshape(-1)
        if not _is_valid_bbox(bbox):
            return None

        front = torch.as_tensor(sample["video.front"])
        src_h = int(front.shape[-2])
        src_w = int(front.shape[-1])
        photometric_params = self._sample_photometric_params()
        current_image = self._image_from_tensor(front, photometric_params)
        history_images = self._collect_history_images(effective_index, sample, photometric_params)
        images = [current_image] + history_images

        task_str = str(sample.get("task", ""))
        instruction, catch_target, put_target = _parse_task(task_str)
        grasp = _to_scalar_bool(sample.get("grasp", False))
        target_name = put_target if grasp else catch_target
        target_name = _normalize_target_name(target_name)
        if not target_name:
            target_name = "target object"

        lang = self.bbox_prompt_template.format(
            instruction=instruction if instruction else task_str,
            target_name=target_name,
        )
        solution = self._bbox_solution(bbox, src_hw=(src_h, src_w))
        if solution is None:
            return None

        return {
            "image": images,
            "lang": lang,
            "solution": solution,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        n = len(self)

        sample = self._make_vlm_sample(index % n)
        if sample is not None:
            return sample

        # Eval/train requirement: invalid bbox -> random resample until valid.
        max_attempts = max(self.max_retry, n)
        for _ in range(max_attempts):
            candidate = self._random.randrange(n)
            sample = self._make_vlm_sample(candidate)
            if sample is not None:
                return sample

        raise RuntimeError(
            f"Failed to sample a valid bbox example after {max_attempts} attempts in dataset `{self.common.repo_id}`."
        )



class WallXVlmSignalDataset(_WallXLeRobotBase):
    """VLM branch dataset for stop-vs-continue + conditional bbox supervision."""

    def __init__(self, data_cfg: Any):
        super().__init__(data_cfg)
        default_signal_prompt = (
            "You are performing a drone navigation task. "
            "Please determine whether you are currently close enough to {target_name} to execute the {operation} action. "
            "If you are not close enough, output only {pred_action_token}. "
            "If you are close enough, output the bounding box of {target_name} as <point>[x1, y1, x2, y2]</point>. "
            "The output must strictly follow the required format without any other text."
        )
        self.signal_prompt_template = str(_cfg_get(data_cfg, "signal_prompt_template", default_signal_prompt))
        self.max_retry = int(_cfg_get(data_cfg, "max_retry", 32))
        self.pred_action_token = str(_cfg_get(data_cfg, "pred_action_token", "<|pred_action|>"))
        self.pred_signal_pred_action = str(_cfg_get(data_cfg, "pred_signal_pred_action", "<pred_action>"))
        self.pred_signal_stop = str(_cfg_get(data_cfg, "pred_signal_stop", "<stop>"))
        self._random = random.Random(int(_cfg_get(data_cfg, "random_seed", 42)))

    def _bbox_solution(self, bbox_xyxy: torch.Tensor, src_hw: tuple[int, int]) -> str | None:
        src_h, src_w = src_hw
        dst_w, dst_h = self.common.image_size
        scale_x = dst_w / float(src_w)
        scale_y = dst_h / float(src_h)
        x1, y1, x2, y2 = bbox_xyxy.tolist()
        sx1 = int(np.clip(round(x1 * scale_x), 0, dst_w - 1))
        sy1 = int(np.clip(round(y1 * scale_y), 0, dst_h - 1))
        sx2 = int(np.clip(round(x2 * scale_x), 0, dst_w - 1))
        sy2 = int(np.clip(round(y2 * scale_y), 0, dst_h - 1))
        if sx1 <= 0 or sy1 <= 0 or sx2 <= 0 or sy2 <= 0:
            return None
        if sx2 <= sx1 or sy2 <= sy1:
            return None
        return f"<point>[{sx1}, {sy1}, {sx2}, {sy2}]</point>"

    def _make_vlm_signal_sample(self, index: int) -> dict[str, Any] | None:
        _, _, sample = self._resolve_training_sample(index)

        front = torch.as_tensor(sample["video.front"])
        src_h = int(front.shape[-2])
        src_w = int(front.shape[-1])
        photometric_params = self._sample_photometric_params()
        current_image = self._image_from_tensor(front, photometric_params)
        images = [current_image]

        task_str = str(sample.get("task", ""))
        _, catch_target, put_target = _parse_task(task_str)
        grasp = _to_scalar_bool(sample.get("grasp", False))
        target_name = put_target if grasp else catch_target
        target_name = _normalize_target_name(target_name)
        if not target_name:
            target_name = "target object"

        operation = "put" if grasp else "grasp"
        lang = self.signal_prompt_template.format(
            target_name=target_name,
            operation=operation,
            pred_action_token=self.pred_action_token,
            instruction="",
        )

        pred_signal_raw = str(sample.get("pred_signal", "")).strip()

        if pred_signal_raw in {self.pred_signal_pred_action, self.pred_action_token}:
            solution = self.pred_action_token
        elif pred_signal_raw == self.pred_signal_stop:
            bbox = torch.as_tensor(sample.get("bbox", torch.tensor([])), dtype=torch.float32).reshape(-1)
            if not _is_valid_bbox(bbox):
                # Requirement: stop samples with invalid bbox should be discarded and re-sampled.
                return None
            solution = self._bbox_solution(bbox, src_hw=(src_h, src_w))
            if solution is None:
                return None
        else:
            return None

        return {
            "image": images,
            "lang": lang,
            "solution": solution,
            "pred_signal": pred_signal_raw,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        n = len(self)

        sample = self._make_vlm_signal_sample(index % n)
        if sample is not None:
            return sample

        max_attempts = max(self.max_retry, n)
        for _ in range(max_attempts):
            candidate = self._random.randrange(n)
            sample = self._make_vlm_signal_sample(candidate)
            if sample is not None:
                return sample

        raise RuntimeError(
            f"Failed to sample a valid pred-signal example after {max_attempts} attempts in dataset `{self.common.repo_id}`."
        )


class WallXRouterDataset(WallXVlaDataset):
    """Unified router dataset for action-vs-bbox training.

    Each sample uses one prompt and one assistant answer:
    - far / continue samples: ``<|pred_action|>`` plus an action trajectory label
    - near / stop samples: ``<|pred_bbox|><point>[x1, y1, x2, y2]</point>``
    """

    def __init__(self, data_cfg: Any):
        super().__init__(data_cfg)
        self.max_retry = int(_cfg_get(data_cfg, "max_retry", 64))
        self.action_supervision = str(_cfg_get(data_cfg, "action_supervision", "flow_matching"))
        if self.action_supervision not in {"flow_matching", "fast_token_ce"}:
            raise ValueError(
                "datasets.router_data.action_supervision must be `flow_matching` or `fast_token_ce`, "
                f"got `{self.action_supervision}`"
            )
        self.pred_action_token = str(_cfg_get(data_cfg, "pred_action_token", "<|pred_action|>"))
        self.pred_bbox_token = str(_cfg_get(data_cfg, "pred_bbox_token", "<|pred_bbox|>"))
        self.pred_signal_pred_action = str(_cfg_get(data_cfg, "pred_signal_pred_action", "<pred_action>"))
        self.pred_signal_stop = str(_cfg_get(data_cfg, "pred_signal_stop", "<stop>"))
        self.operation_grasp_true = str(_cfg_get(data_cfg, "operation_grasp_true", "put"))
        self.operation_grasp_false = str(_cfg_get(data_cfg, "operation_grasp_false", "grasp"))
        self.bbox_solution_separator = str(_cfg_get(data_cfg, "bbox_solution_separator", ""))
        self.action_route_format = str(_cfg_get(data_cfg, "action_route_format", "token")).strip()
        if self.action_route_format not in {"token", "route_subtask"}:
            raise ValueError(
                "datasets.router_data.action_route_format must be `token` or `route_subtask`, "
                f"got `{self.action_route_format}`"
            )
        self.subtask_start_token = str(_cfg_get(data_cfg, "subtask_start_token", "<|subtask|>"))
        self.subtask_end_token = str(_cfg_get(data_cfg, "subtask_end_token", "<|end_subtask|>"))
        self.subtask_text_field = str(_cfg_get(data_cfg, "subtask_text_field", "subtask_text"))
        default_router_prompt_grasp = (
            "{instruction}\n"
            "{instruction}\n"
            "{instruction}\n\n"
            "GRASP PHASE: FIND AND GRASP THE OBJECT.\n"
            "The current target is the object to be grasped: {target_name}.\n"
            "{target_name}, {target_name}, {target_name} should be grasped.\n"
            "If the object is still far away, output exactly {pred_action_token}.\n"
            "If the object is close enough for grasping, output "
            "{pred_bbox_token}<point>[x1, y1, x2, y2]</point> for the object's location in the current front view.\n"
            "Do not output anything else."
        )
        default_router_prompt_put = (
            "{instruction}\n"
            "{instruction}\n"
            "{instruction}\n\n"
            "PLACE PHASE: GO TO THE PLACEMENT DESTINATION.\n"
            "The current target is the placement destination: {target_name}.\n"
            "Go to {target_name}.\n"
            "The placement destination is {target_name}, {target_name}, {target_name}.\n"
            "Always output exactly {pred_action_token}.\n"
            "Do not output a bounding box. Do not describe the scene. Do not output anything else."
        )
        default_fast_router_prompt_grasp = (
            "{instruction}\n"
            "{instruction}\n"
            "{instruction}\n\n"
            "GRASP PHASE: FIND AND GRASP THE OBJECT.\n"
            "The current target is the object to be grasped: {target_name}.\n"
            "{target_name}, {target_name}, {target_name} should be grasped.\n"
            "If the object is still far away, output {pred_action_token} followed immediately by the action token sequence.\n"
            "If the object is close enough for grasping, output "
            "{pred_bbox_token}<point>[x1, y1, x2, y2]</point> for the object's location in the current front view.\n"
            "Do not output anything else."
        )
        default_fast_router_prompt_put = (
            "{instruction}\n"
            "{instruction}\n"
            "{instruction}\n\n"
            "PLACE PHASE: GO TO THE PLACEMENT DESTINATION.\n"
            "The current target is the placement destination: {target_name}.\n"
            "Go to {target_name}.\n"
            "The placement destination is {target_name}, {target_name}, {target_name}.\n"
            "Always output {pred_action_token} followed immediately by the action token sequence.\n"
            "Do not output a bounding box. Do not describe the scene. Do not output anything else."
        )
        legacy_router_prompt = _cfg_get(data_cfg, "router_prompt_template", None)
        legacy_fast_router_prompt = _cfg_get(data_cfg, "router_prompt_template_fast_token_ce", None)
        self.router_prompt_template_grasp = str(
            _cfg_get(data_cfg, "router_prompt_template_grasp", legacy_router_prompt or default_router_prompt_grasp)
        )
        self.router_prompt_template_put = str(
            _cfg_get(data_cfg, "router_prompt_template_put", legacy_router_prompt or default_router_prompt_put)
        )
        self.router_prompt_template_fast_token_ce_grasp = str(
            _cfg_get(
                data_cfg,
                "router_prompt_template_fast_token_ce_grasp",
                legacy_fast_router_prompt or default_fast_router_prompt_grasp,
            )
        )
        self.router_prompt_template_fast_token_ce_put = str(
            _cfg_get(
                data_cfg,
                "router_prompt_template_fast_token_ce_put",
                legacy_fast_router_prompt or default_fast_router_prompt_put,
            )
        )
        self._random = random.Random(int(_cfg_get(data_cfg, "random_seed", 42)))
        self._fast_action_formatter = None

    def _select_router_prompt_template(self, grasp: bool) -> str:
        if self.action_supervision == "fast_token_ce":
            return (
                self.router_prompt_template_fast_token_ce_put
                if grasp
                else self.router_prompt_template_fast_token_ce_grasp
            )
        return self.router_prompt_template_put if grasp else self.router_prompt_template_grasp

    def _get_fast_action_formatter(self):
        if self._fast_action_formatter is not None:
            return self._fast_action_formatter

        tokenizer_cfg = _cfg_get(self.data_cfg, "action_tokenizer", None)
        tokenizer_path = str(_cfg_get(tokenizer_cfg, "path", "physical-intelligence/fast"))
        token_prefix = str(_cfg_get(tokenizer_cfg, "token_prefix", "<robot_action_"))
        token_count = int(_cfg_get(tokenizer_cfg, "token_count", 2048))

        from starVLA.model.modules.action_model.fast_ActionHeader import FastActionTokenFormatter

        self._fast_action_formatter = FastActionTokenFormatter(
            tokenizer_path=tokenizer_path,
            token_prefix=token_prefix,
            token_count=token_count,
        )
        return self._fast_action_formatter

    def _subtask_text(self, sample: dict[str, Any]) -> str:
        return str(sample.get(self.subtask_text_field, "")).strip()

    def _action_route_prefix(self, sample: dict[str, Any]) -> str | None:
        if self.action_route_format == "token":
            return self.pred_action_token

        subtask_text = self._subtask_text(sample)
        if not subtask_text:
            return None
        return f"{self.pred_action_token}{self.subtask_start_token}{subtask_text}{self.subtask_end_token}"

    def _action_solution(self, action: torch.Tensor, sample: dict[str, Any]) -> str | None:
        route_prefix = self._action_route_prefix(sample)
        if route_prefix is None:
            return None
        if self.action_supervision == "flow_matching":
            return route_prefix
        fast_tokens = self._get_fast_action_formatter().encode_string(
            action.detach().cpu().numpy().astype(np.float32)
        )
        return f"{route_prefix}{fast_tokens}"

    def _bbox_solution(self, bbox_xyxy: torch.Tensor, src_hw: tuple[int, int]) -> str | None:
        src_h, src_w = src_hw
        dst_w, dst_h = self.common.image_size
        scale_x = dst_w / float(src_w)
        scale_y = dst_h / float(src_h)
        x1, y1, x2, y2 = bbox_xyxy.tolist()
        sx1 = int(np.clip(round(x1 * scale_x), 0, dst_w - 1))
        sy1 = int(np.clip(round(y1 * scale_y), 0, dst_h - 1))
        sx2 = int(np.clip(round(x2 * scale_x), 0, dst_w - 1))
        sy2 = int(np.clip(round(y2 * scale_y), 0, dst_h - 1))
        if sx1 <= 0 or sy1 <= 0 or sx2 <= 0 or sy2 <= 0:
            return None
        if sx2 <= sx1 or sy2 <= sy1:
            return None
        return f"<point>[{sx1}, {sy1}, {sx2}, {sy2}]</point>"

    def _make_router_prompt(self, sample: dict[str, Any]) -> tuple[str, str, str, bool]:
        task_str = str(sample.get("task", ""))
        instruction, catch_target, put_target = _parse_task(task_str)
        instruction_text = instruction if instruction else task_str
        grasp = _to_scalar_bool(sample.get("grasp", False))
        target_name = put_target if grasp else catch_target
        target_name = _normalize_target_name(target_name)
        if not target_name:
            target_name = "target object"
        operation = self.operation_grasp_true if grasp else self.operation_grasp_false
        prompt_template = self._select_router_prompt_template(grasp)
        lang = prompt_template.format(
            instruction=instruction_text,
            target_name=target_name,
            operation=operation,
            pred_action_token=self.pred_action_token,
            pred_bbox_token=self.pred_bbox_token,
            subtask_start_token=self.subtask_start_token,
            subtask_end_token=self.subtask_end_token,
        )
        return lang, target_name, operation, grasp

    def _make_router_sample(self, index: int) -> dict[str, Any] | None:
        effective_index, _, sample = self._resolve_training_sample(index)

        front = torch.as_tensor(sample["video.front"])
        src_h = int(front.shape[-2])
        src_w = int(front.shape[-1])
        photometric_params = self._sample_photometric_params()
        current_image = self._image_from_tensor(front, photometric_params)
        history_images = self._collect_history_images(effective_index, sample, photometric_params)
        images = [current_image] + history_images

        lang, target_name, operation, _ = self._make_router_prompt(sample)
        pred_signal_raw = str(sample.get("pred_signal", "")).strip()

        output: dict[str, Any] = {
            "image": images,
            "lang": lang,
            "target_name": target_name,
            "operation": operation,
            "pred_signal": pred_signal_raw,
        }

        if pred_signal_raw in {self.pred_signal_pred_action, self.pred_action_token}:
            state = torch.as_tensor(sample["state"], dtype=torch.float32)
            action = torch.as_tensor(sample["action"], dtype=torch.float32)
            action = self._apply_action_mode(action, state)
            action = self._truncate_with_keyframe(action, sample.get("keyframe", None))
            if self.normalize_action and self.action_norm_stats is not None:
                action = self._normalize_with_stats(action, self.action_norm_stats)
            action_solution = self._action_solution(action, sample)
            if action_solution is None:
                return None

            output.update(
                {
                    "route": "action",
                    "route_token": self.pred_action_token,
                    "solution": action_solution,
                    "subtask_text": self._subtask_text(sample),
                    "action": action.detach().cpu().numpy().astype(np.float16),
                }
            )
            if self.include_state:
                out_state = state
                if self.normalize_state and self.state_norm_stats is not None:
                    out_state = self._normalize_with_stats(out_state, self.state_norm_stats)
                output["state"] = out_state.detach().cpu().numpy()[None, :].astype(np.float16)
            return output

        if pred_signal_raw == self.pred_signal_stop:
            bbox = torch.as_tensor(sample.get("bbox", torch.tensor([])), dtype=torch.float32).reshape(-1)
            if not _is_valid_bbox(bbox):
                return None
            bbox_solution = self._bbox_solution(bbox, src_hw=(src_h, src_w))
            if bbox_solution is None:
                return None

            output.update(
                {
                    "route": "bbox",
                    "route_token": self.pred_bbox_token,
                    "solution": f"{self.pred_bbox_token}{self.bbox_solution_separator}{bbox_solution}",
                    "bbox_solution": bbox_solution,
                }
            )
            return output

        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        n = len(self)
        sample = self._make_router_sample(index % n)
        if sample is not None:
            return sample

        max_attempts = max(self.max_retry, n)
        for _ in range(max_attempts):
            candidate = self._random.randrange(n)
            sample = self._make_router_sample(candidate)
            if sample is not None:
                return sample

        raise RuntimeError(
            f"Failed to sample a valid router example after {max_attempts} attempts in dataset `{self.common.repo_id}`."
        )


def collate_fn_vla(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def collate_fn_vlm(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def collate_fn_vlm_signal(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def collate_fn_router(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


class WallXSubtaskDataset(WallXRouterDataset):
    """Subtask-aware dataset: replaces router, all samples are action route.

    subtask_history is per-trajectory (cross-episode), built from subtask_configs.json.
    Only two primitives: Search and Fly to.
    """

    def __init__(self, data_cfg: Any):
        super().__init__(data_cfg)
        self.subtask_prompt_template = str(_cfg_get(
            data_cfg, "subtask_prompt_template", DEFAULT_SUBTASK_PROMPT,
        ))
        self._subtask_configs = self._load_subtask_configs()

    def _load_subtask_configs(self) -> dict:
        configs_path = Path(self.common.root) / "meta" / "subtask_configs.json"
        if not configs_path.exists():
            return {}
        with open(configs_path) as f:
            configs = json.load(f)
        return {c["trajectory_id"]: c for c in configs}

    def _get_subtask_history(self, sample: dict) -> str:
        ep_idx = int(sample.get("episode_index", 0))
        traj_id = ep_idx // 2
        current_subtask_id = int(sample.get("subtask_id", 1))

        config = self._subtask_configs.get(traj_id)
        if config is None or current_subtask_id <= 1:
            return ""

        completed = [st["subtask_text"] for st in config["subtasks"] if st["subtask_id"] < current_subtask_id]
        return ". ".join(completed)

    def _make_subtask_sample(self, index: int) -> dict[str, Any] | None:
        effective_index, _, sample = self._resolve_training_sample(index)

        front = torch.as_tensor(sample["video.front"])
        photometric_params = self._sample_photometric_params()
        current_image = self._image_from_tensor(front, photometric_params)
        history_images = self._collect_history_images(effective_index, sample, photometric_params)
        images = [current_image] + history_images

        subtask_text = str(sample.get("subtask_text", ""))
        if not subtask_text:
            return None

        subtask_history = self._get_subtask_history(sample)

        task_str = str(sample.get("task", ""))
        instruction, _, _ = _parse_task(task_str)
        lang = self.subtask_prompt_template.format(
            instruction=instruction if instruction else task_str,
            subtask_history=subtask_history,
        )

        state = torch.as_tensor(sample["state"], dtype=torch.float32)
        action = torch.as_tensor(sample["action"], dtype=torch.float32)
        action = self._apply_action_mode(action, state)
        action = self._truncate_with_keyframe(action, sample.get("keyframe", None))
        if self.normalize_action and self.action_norm_stats is not None:
            action = self._normalize_with_stats(action, self.action_norm_stats)

        output: dict[str, Any] = {
            "image": images,
            "lang": lang,
            "solution": subtask_text,
            "route": "action",
            "subtask_text": subtask_text,
            "action": action.detach().cpu().numpy().astype(np.float16),
        }
        if self.include_state:
            out_state = state
            if self.normalize_state and self.state_norm_stats is not None:
                out_state = self._normalize_with_stats(out_state, self.state_norm_stats)
            output["state"] = out_state.detach().cpu().numpy()[None, :].astype(np.float16)
        return output

    def __getitem__(self, index: int) -> dict[str, Any]:
        n = len(self)
        sample = self._make_subtask_sample(index % n)
        if sample is not None:
            return sample

        for _ in range(self.max_retry):
            candidate = self._random.randrange(n)
            sample = self._make_subtask_sample(candidate)
            if sample is not None:
                return sample
        raise RuntimeError(
            f"Failed to sample a valid subtask example after {self.max_retry} attempts in dataset `{self.common.repo_id}`."
        )


DEFAULT_SUBTASK_PROMPT = (
    "{instruction}\n"
    "You are performing a drone navigation task. The first image is the current front view; "
    "any following images are previous keyframes for context.\n"
    "Completed subtasks: {subtask_history}\n"
    "What subtask should you perform now? Output exactly one subtask."
)


def get_vla_dataset(data_cfg: Any, **_: Any) -> WallXVlaDataset:
    return WallXVlaDataset(data_cfg=data_cfg)


def get_vlm_dataset(data_cfg: Any, **_: Any) -> WallXVlmBboxDataset:
    return WallXVlmBboxDataset(data_cfg=data_cfg)


def get_vlm_signal_dataset(data_cfg: Any, **_: Any) -> WallXVlmSignalDataset:
    return WallXVlmSignalDataset(data_cfg=data_cfg)


def get_router_dataset(data_cfg: Any, **_: Any) -> WallXRouterDataset:
    return WallXRouterDataset(data_cfg=data_cfg)


def get_subtask_dataset(data_cfg: Any, **_: Any) -> WallXSubtaskDataset:
    return WallXSubtaskDataset(data_cfg=data_cfg)
