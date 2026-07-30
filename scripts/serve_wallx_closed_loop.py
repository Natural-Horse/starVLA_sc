#!/usr/bin/env python3
"""StarVLA websocket server compatible with wall-x closed-loop simulator clients.

This script is designed to be a drop-in server replacement for:
  - wall_x/serving/launch_serving.py
when the client side keeps using:
  - wall_x/serving/client_for_sim.py
  - wall_x/serving/batch_task_runner.py

Protocol compatibility:
1) On websocket connect, server sends metadata dict (msgpack).
2) Client sends a plain observation dict (msgpack, no envelope).
3) Server returns action dict (msgpack), with optional VQA payload:
   {
     "action": np.ndarray | None,          # [B, T, D], usually B=1
     "action_skipped": bool,
     "action_skip_reason": str (optional),
     "vqa": { ... } (optional),
     "server_timing": { "infer_ms": ..., "prev_total_ms": ... }
   }

Example:
CUDA_VISIBLE_DEVICES=0 \
/beijing-c/workspace/hxj/miniconda3/envs/starvla/bin/python \
/beijing-c/wallx_workspace/starVLA/scripts/serve_wallx_closed_loop.py \
  --config_yaml /beijing-c/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi.yaml \
  --checkpoint /beijing-c/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_cotrain_sort_vlm02_noGradClip_actionLR3e6_min_lr2e6_eval01/checkpoints/steps_10000_pytorch_model.pt \
  --host 0.0.0.0 \
  --port 8000 \
  --camera_keys face_view \
  --vqa_enabled \
  --frame_debug_root /beijing-c/wallx_workspace/starVLA/debug/1


CUDA_VISIBLE_DEVICES=0,1 \
/beijing-c/workspace/hxj/miniconda3/envs/starvla/bin/python \
/beijing-c/wallx_workspace/starVLA/scripts/serve_wallx_closed_loop.py \
  --config_yaml /beijing-c/wallx_workspace/starVLA/starVLA/config/training/starvla_cotrain_wallx_qwenpi.yaml \
  --checkpoint /beijing-c/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_cotrain_vlm_signal_data/checkpoints/steps_14000_pytorch_model.pt \
  --checkpoint_bbox /beijing-c/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_cotrain_vlm_signal_data/checkpoints/steps_14000_pytorch_model.pt \
  --checkpoint_policy /beijing-c/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_cotrain_vlm_signal_data/checkpoints/steps_14000_pytorch_model.pt \
  --device_bbox cuda:0 \
  --device_policy cuda:1 \
  --host 0.0.0.0 \
  --port 8000 \
  --vqa_enabled \
  --vqa_skip_action_on_bbox \
  --frame_debug_root /beijing-c/wallx_workspace/starVLA/debug/2 \
  --debug


"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import asyncio
import http
import json
import logging
import os
import re
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import PartialState
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

import websockets
import websockets.asyncio.server as _server
import websockets.frames

try:
    import msgpack
    import msgpack_numpy as m

    m.patch()
except ImportError:
    msgpack = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.model.framework import build_framework


logger = logging.getLogger(__name__)

POINT_PATTERN = re.compile(
    r"<point>\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*</point>",
    re.IGNORECASE,
)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_str(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if x is None:
        return ""
    return str(x)


def _normalize_obs_keys(obs: dict[Any, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for k, v in obs.items():
        if isinstance(k, bytes):
            key = k.decode("utf-8", errors="ignore")
        else:
            key = str(k)
        normalized[key] = v
    return normalized


def _sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text))
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"


def _is_history_key(key: str) -> bool:
    return re.fullmatch(r"history_(\d+)", key) is not None


def _extract_history_keys(obs: dict[str, Any]) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for key in obs.keys():
        match = re.fullmatch(r"history_(\d+)", key)
        if match is not None:
            indexed.append((int(match.group(1)), key))
    indexed.sort(key=lambda item: item[0])
    return [item[1] for item in indexed]


def _coerce_image_to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if isinstance(image, np.ndarray):
        arr = image
    else:
        arr = np.asarray(image)

    if arr.ndim > 3:
        arr = np.squeeze(arr)

    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def _parse_task_string(task_str: str) -> tuple[str, str, str]:
    match = re.search(r"(.*)Catch:\s*(.*)\.\s*Put:\s*(.*)", task_str)
    if match:
        return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
    return task_str.strip(), "", ""


def _normalize_target_name(target_name: str) -> str:
    text = str(target_name).strip()
    while text.endswith("."):
        text = text[:-1].rstrip()
    return text


def _to_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(x)
    s = _to_str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    return bool(default)


def _is_valid_bbox_xyxy(bbox: list[float] | tuple[float, float, float, float] | np.ndarray | None) -> bool:
    if bbox is None:
        return False
    b = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if b.size != 4:
        return False
    x1, y1, x2, y2 = b.tolist()
    if x1 <= 0 or y1 <= 0 or x2 <= 0 or y2 <= 0:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    return True


def _clip_bbox_xyxy(bbox: list[float], w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    return [x1, y1, x2, y2]


def _parse_bbox_from_text(text: str) -> list[float] | None:
    match = POINT_PATTERN.search(text)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]

    # Fallback: permissive plain list "[x1, y1, x2, y2]"
    simple = re.search(
        r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]",
        text,
    )
    if simple:
        return [float(simple.group(i)) for i in range(1, 5)]
    return None


def _resized_to_original_bbox(
    bbox_resized: list[float],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> list[float]:
    sx = float(src_w) / float(dst_w)
    sy = float(src_h) / float(dst_h)
    x1, y1, x2, y2 = bbox_resized
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def _decode_vlm_generation(
    qwen_vl_interface: Any,
    qwen_inputs: dict[str, torch.Tensor],
    generation_kwargs: dict[str, Any],
) -> str:
    generated = qwen_vl_interface.generate(**qwen_inputs, **generation_kwargs)
    gen_ids = generated.sequences if hasattr(generated, "sequences") else generated
    prompt_len = int(qwen_inputs["input_ids"].shape[1])
    new_tokens = gen_ids[:, prompt_len:]
    text = qwen_vl_interface.processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def _resolve_checkpoint_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    candidates: list[Path] = []
    for p in path.rglob("*"):
        if p.is_file() and (
            p.name.endswith("pytorch_model.pt")
            or p.name.endswith("model.safetensors")
            or p.suffix in {".pt", ".safetensors"}
        ):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found under directory: {path}")

    def _score(p: Path) -> tuple[int, float]:
        match = re.search(r"steps_(\d+)", p.name)
        step = int(match.group(1)) if match else -1
        return (step, p.stat().st_mtime)

    candidates.sort(key=_score)
    return candidates[-1]


def _resolve_config_path(config_yaml: str | None, checkpoint_path: Path) -> Path:
    if config_yaml:
        path = Path(config_yaml)
        if not path.exists():
            raise FileNotFoundError(f"config_yaml does not exist: {path}")
        return path

    run_dir = checkpoint_path.parents[1]
    auto_cfg = run_dir / "config.yaml"
    if auto_cfg.exists():
        return auto_cfg

    raise FileNotFoundError(
        "config_yaml is not provided and auto-detected config.yaml does not exist "
        f"next to checkpoint run dir: {auto_cfg}"
    )


def _load_model(cfg: Any, checkpoint: Path, device: str) -> torch.nn.Module:
    model = build_framework(cfg=cfg)

    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint))
    else:
        state_dict = torch.load(str(checkpoint), map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %d", len(missing))
        logger.warning("First missing keys: %s", missing[:10])
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %d", len(unexpected))
        logger.warning("First unexpected keys: %s", unexpected[:10])

    model = model.to(device)
    model.eval()
    return model


def _load_action_norm_stats(vla_cfg: Any) -> dict[str, np.ndarray] | None:
    root = Path(str(_cfg_get(vla_cfg, "root", "")))
    if not root.exists():
        logger.warning("Dataset root not found, skip action unnormalization stats: %s", root)
        return None

    use_delta_action = bool(_cfg_get(vla_cfg, "use_delta_action", False))
    action_in_ego = bool(_cfg_get(vla_cfg, "action_in_ego", True))

    preferred = "norm_stats.json"
    if use_delta_action:
        preferred = "norm_stats_delta.json"
    if action_in_ego:
        preferred = "norm_stats_ego.json"

    candidates = [preferred, "norm_stats_ego.json", "norm_stats_delta.json", "norm_stats.json"]
    chosen: Path | None = None
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        candidate = root / name
        if candidate.exists():
            chosen = candidate
            break

    if chosen is None:
        logger.warning("No norm_stats*.json found under dataset root: %s", root)
        return None

    with open(chosen, "r", encoding="utf-8") as f:
        stats_json = json.load(f)

    action_stats = stats_json.get("norm_stats", {}).get("action", {})
    if "q01" not in action_stats or "q99" not in action_stats:
        logger.warning("Invalid action stats format in %s (missing q01/q99).", chosen)
        return None

    q01 = np.asarray(action_stats["q01"], dtype=np.float32)
    q99 = np.asarray(action_stats["q99"], dtype=np.float32)
    logger.info("Loaded action norm stats from %s", chosen)
    return {"q01": q01, "q99": q99}


def _unnormalize_actions(normalized_actions: np.ndarray, action_stats: dict[str, np.ndarray]) -> np.ndarray:
    actions = np.asarray(normalized_actions, dtype=np.float32).copy()
    actions = np.clip(actions, -1.0, 1.0)

    q01 = np.asarray(action_stats["q01"], dtype=np.float32).reshape(-1)
    q99 = np.asarray(action_stats["q99"], dtype=np.float32).reshape(-1)
    dim = min(actions.shape[-1], q01.shape[0], q99.shape[0])
    if dim <= 0:
        return actions

    delta = q99[:dim] - q01[:dim]
    delta = np.where(np.abs(delta) < 1e-8, 1.0, delta)
    actions[..., :dim] = 0.5 * (actions[..., :dim] + 1.0) * delta + q01[:dim]
    return actions


class StarVLAWallXPolicy:
    """Dual-model policy adapter for closed-loop wall-x simulation.

    Thread A (bbox debug):
      - Runs bbox VLM generation only.
      - Saves local visualization/debug artifacts only.

    Thread B (signal->action):
      - Runs vlm_signal generation.
      - If output is <|pred_action|>, runs action generation.
      - If output is bbox, skips action and returns bbox to client.
    """

    def __init__(
        self,
        model_bbox: torch.nn.Module,
        model_policy: torch.nn.Module,
        cfg: Any,
        device_bbox: str,
        device_policy: str,
        camera_keys: list[str],
        pred_horizon: int,
        action_dim: int,
        action_stats: dict[str, np.ndarray] | None,
        unnormalize_actions: bool,
        default_prompt: str | None = None,
        vqa_enabled: bool = False,
        vqa_camera_key: str | None = None,
        vqa_skip_action_on_bbox: bool = True,
        vqa_max_new_tokens: int = 32,
        vqa_do_sample: bool = False,
        vqa_temperature: float = 0.2,
        vqa_top_p: float = 0.95,
        signal_max_new_tokens: int = 16,
        signal_do_sample: bool = False,
        signal_temperature: float = 0.2,
        signal_top_p: float = 0.95,
    ) -> None:
        self.model_bbox = model_bbox
        self.model_policy = model_policy
        self.cfg = cfg
        self.device_bbox = device_bbox
        self.device_policy = device_policy
        self.camera_keys = camera_keys
        self.pred_horizon = int(pred_horizon)
        self.action_dim = int(action_dim)
        self.action_stats = action_stats
        self.do_unnormalize_actions = bool(unnormalize_actions)
        self.default_prompt = default_prompt

        self.vqa_enabled = bool(vqa_enabled)
        self.vqa_camera_key = vqa_camera_key
        self.vqa_skip_action_on_bbox = bool(vqa_skip_action_on_bbox)

        # BBox generation config (thread A).
        self.vqa_max_new_tokens = int(vqa_max_new_tokens)
        self.vqa_do_sample = bool(vqa_do_sample)
        self.vqa_temperature = float(vqa_temperature)
        self.vqa_top_p = float(vqa_top_p)

        # Signal generation config (thread B), defaulting to conservative deterministic decoding.
        self.signal_max_new_tokens = int(signal_max_new_tokens)
        self.signal_do_sample = bool(signal_do_sample)
        self.signal_temperature = float(signal_temperature)
        self.signal_top_p = float(signal_top_p)

        # Keep one persistent thread pool to avoid per-frame thread churn.
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='starvla_dual')

        vla_cfg = cfg.datasets.vla_data
        vlm_bbox_cfg = cfg.datasets.vlm_data
        vlm_signal_cfg = _cfg_get(cfg.datasets, 'vlm_signal_data', None)

        self.include_state = bool(_cfg_get(vla_cfg, 'include_state', False))
        legacy_action_prompt_template = _cfg_get(vla_cfg, 'action_prompt_template', None)

        default_action_prompt_grasp_true = (
            '{instruction}\n'
            'You are performing a drone navigation task. '
            'You now need to pull the {target_name}. Please predict the next action.'
        )
        default_action_prompt_grasp_false = (
            '{instruction}\n'
            'You are performing a drone navigation task. '
            'You now need to grasp the {target_name}. Please predict the next action.'
        )

        self.action_prompt_template_grasp_true = str(
            _cfg_get(
                vla_cfg,
                'action_prompt_template_grasp_true',
                legacy_action_prompt_template or default_action_prompt_grasp_true,
            )
        )
        self.action_prompt_template_grasp_false = str(
            _cfg_get(
                vla_cfg,
                'action_prompt_template_grasp_false',
                legacy_action_prompt_template or default_action_prompt_grasp_false,
            )
        )

        self.bbox_prompt_template = str(
            _cfg_get(
                vlm_bbox_cfg,
                'bbox_prompt_template',
                (
                    'Please identify the {target_name} in the front view and output its bounding box as '
                    '<point>[x1, y1, x2, y2]</point>. '
                    'The output must strictly follow the format <point>[x1, y1, x2, y2]</point> without any other text.'
                ),
            )
        )

        default_signal_prompt = (
            'You are performing a drone navigation task. '
            'Please determine whether you are currently close enough to {target_name} to execute the {operation} action. '
            'If you are not close enough, output only {pred_action_token}. '
            'If you are close enough, output the bounding box of {target_name} as <point>[x1, y1, x2, y2]</point>. '
            'The output must strictly follow the required format without any other text.'
        )
        self.signal_prompt_template = str(
            _cfg_get(
                vlm_signal_cfg,
                'signal_prompt_template',
                default_signal_prompt,
            )
        )
        self.pred_action_token = str(_cfg_get(vlm_signal_cfg, 'pred_action_token', '<|pred_action|>'))

        vla_img_size = _cfg_get(vla_cfg, 'image_size', [224, 224])
        bbox_img_size = _cfg_get(vlm_bbox_cfg, 'image_size', [224, 224])
        signal_img_size = _cfg_get(vlm_signal_cfg, 'image_size', bbox_img_size)
        self.vla_image_size = (int(vla_img_size[0]), int(vla_img_size[1]))
        self.vlm_bbox_image_size = (int(bbox_img_size[0]), int(bbox_img_size[1]))
        self.vlm_signal_image_size = (int(signal_img_size[0]), int(signal_img_size[1]))

        self.qwen_vl_interface_bbox = getattr(model_bbox, 'qwen_vl_interface', None)
        self.qwen_vl_interface_policy = getattr(model_policy, 'qwen_vl_interface', None)
        if self.vqa_enabled and self.qwen_vl_interface_bbox is None:
            raise RuntimeError('BBox model has no qwen_vl_interface, cannot run bbox debug VQA.')
        if self.vqa_enabled and self.qwen_vl_interface_policy is None:
            raise RuntimeError('Policy model has no qwen_vl_interface, cannot run signal branch.')

    def close(self) -> None:
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            'action_dim': self.action_dim,
            'pred_horizon': self.pred_horizon,
            'device_bbox': self.device_bbox,
            'device_policy': self.device_policy,
            'predict_mode': 'starvla_qwenpi_dual',
            'camera_keys': self.camera_keys,
            'vqa_enabled': self.vqa_enabled,
            'vqa_camera_key': self.vqa_camera_key,
            'vqa_skip_action_on_bbox': self.vqa_skip_action_on_bbox,
            'pred_action_token': self.pred_action_token,
        }

    def _resolve_prompt(self, obs: dict[str, Any]) -> str:
        prompt = _to_str(obs.get('prompt', ''))
        if prompt:
            return prompt
        if self.default_prompt:
            return self.default_prompt
        return ''

    def _resolve_instruction(self, prompt: str) -> str:
        instruction, _, _ = _parse_task_string(prompt)
        return instruction if instruction else prompt

    def _resolve_target_info(self, obs: dict[str, Any], prompt: str) -> tuple[str, bool, str]:
        _, catch_target, put_target = _parse_task_string(prompt)

        if 'grasp' in obs:
            grasp_done = _to_bool(obs.get('grasp'), default=False)
        else:
            grasp_action_obs = _to_str(obs.get('grasp_action', 'grasp')).lower().strip()
            if grasp_action_obs in {'place', 'put', 'pull'}:
                grasp_done = True
            elif grasp_action_obs in {'grasp', 'catch'}:
                grasp_done = False
            else:
                grasp_done = False

        operation = 'put' if grasp_done else 'grasp'

        target_name = _normalize_target_name(_to_str(obs.get('target_name', '')))
        if not target_name:
            target_name = put_target if grasp_done else catch_target
        target_name = _normalize_target_name(target_name)
        if not target_name:
            target_name = 'target object'

        return target_name, grasp_done, operation

    def _collect_action_image_keys(self, obs: dict[str, Any]) -> list[str]:
        ordered_keys: list[str] = []
        seen: set[str] = set()

        for key in self.camera_keys:
            if key in obs and key not in seen and not _is_history_key(key):
                ordered_keys.append(key)
                seen.add(key)

        for key in _extract_history_keys(obs):
            if key not in seen:
                ordered_keys.append(key)
                seen.add(key)

        return ordered_keys

    def _collect_action_images(self, obs: dict[str, Any]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for key in self._collect_action_image_keys(obs):
            images.append(_coerce_image_to_pil(obs[key]))

        if images:
            return images

        for value in obs.values():
            if isinstance(value, (np.ndarray, Image.Image, torch.Tensor, list, tuple)):
                try:
                    img = _coerce_image_to_pil(value)
                    return [img]
                except Exception:
                    continue

        raise KeyError(
            f'No camera image found for keys {self.camera_keys}. Available obs keys: {list(obs.keys())}'
        )

    def _build_action_example(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt = self._resolve_prompt(obs)
        instruction = self._resolve_instruction(prompt)
        target_name, grasp_done, _ = self._resolve_target_info(obs, prompt)

        action_prompt_template = (
            self.action_prompt_template_grasp_true
            if grasp_done
            else self.action_prompt_template_grasp_false
        )
        lang = action_prompt_template.format(
            instruction=instruction,
            target_name=target_name,
        )

        images = self._collect_action_images(obs)
        example: dict[str, Any] = {'image': images, 'lang': lang}

        if self.include_state and ('state' in obs) and (obs['state'] is not None):
            state = np.asarray(obs['state'], dtype=np.float32)
            if state.ndim == 1:
                state = state[None, :]
            elif state.ndim > 2:
                state = state.reshape(-1, state.shape[-1])[:1]
            example['state'] = state.astype(np.float16)

        return example

    def _predict_action(self, obs: dict[str, Any], frame_tag: str | None = None) -> np.ndarray:
        example = self._build_action_example(obs)
        action_lang = _to_str(example.get('lang', ''))
        if frame_tag is not None:
            logger.info('%s [PROMPT] action_lang\n%s', frame_tag, action_lang if action_lang else '<empty>')
        else:
            logger.info('[PROMPT] action_lang\n%s', action_lang if action_lang else '<empty>')

        output = self.model_policy.predict_action(examples=[example])
        if 'normalized_actions' not in output:
            raise KeyError(f"Model output has no `normalized_actions`. Keys: {list(output.keys())}")

        actions = np.asarray(output['normalized_actions'], dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None, :, :]
        if actions.ndim != 3:
            raise ValueError(f'Unexpected action shape: {actions.shape}')

        actions = actions[:, : self.pred_horizon, : self.action_dim]

        if self.do_unnormalize_actions and self.action_stats is not None:
            actions = _unnormalize_actions(actions, self.action_stats)

        return actions

    def _resolve_vqa_image_key(self, obs: dict[str, Any]) -> str:
        candidates: list[str] = []
        obs_key = _to_str(obs.get('vqa_camera_key', '')).strip()
        if obs_key:
            candidates.append(obs_key)
        if self.vqa_camera_key:
            candidates.append(self.vqa_camera_key)
        candidates.extend(self.camera_keys)
        candidates.append('face_view')

        seen: set[str] = set()
        for key in candidates:
            if (not key) or (key in seen) or _is_history_key(key):
                continue
            seen.add(key)
            if key in obs:
                return key

        for key, value in obs.items():
            if _is_history_key(key):
                continue
            try:
                _coerce_image_to_pil(value)
                return key
            except Exception:
                continue

        raise KeyError(
            f'No VLM image found from current frame keys. Tried candidates={candidates}. Available keys={list(obs.keys())}'
        )

    @staticmethod
    def _build_generation_kwargs(max_new_tokens: int, do_sample: bool, temperature: float, top_p: float) -> dict[str, Any]:
        generation_kwargs: dict[str, Any] = {
            'max_new_tokens': int(max_new_tokens),
            'do_sample': bool(do_sample),
        }
        if do_sample:
            generation_kwargs['temperature'] = float(temperature)
            generation_kwargs['top_p'] = float(top_p)
        return generation_kwargs

    def _run_bbox_debug(self, obs: dict[str, Any], frame_tag: str | None = None) -> dict[str, Any]:
        if self.qwen_vl_interface_bbox is None:
            raise RuntimeError('qwen_vl_interface_bbox is missing, cannot run bbox debug.')

        image_key = self._resolve_vqa_image_key(obs)
        image_orig = _coerce_image_to_pil(obs[image_key]).convert('RGB')
        src_w, src_h = image_orig.size
        dst_w, dst_h = self.vlm_bbox_image_size
        image_resized = image_orig.resize((dst_w, dst_h), Image.BILINEAR)

        prompt = self._resolve_prompt(obs)
        instruction = self._resolve_instruction(prompt)
        target_name, grasp_done, operation = self._resolve_target_info(obs, prompt)
        bbox_lang = self.bbox_prompt_template.format(
            instruction=instruction,
            target_name=target_name,
        )
        if frame_tag is not None:
            logger.info('%s [PROMPT] bbox_lang(thread-A):\n%s', frame_tag, bbox_lang if bbox_lang else '<empty>')

        qwen_inputs = self.qwen_vl_interface_bbox.build_qwenvl_inputs(
            images=[[image_resized]],
            instructions=[bbox_lang],
        )
        pred_text = _decode_vlm_generation(
            qwen_vl_interface=self.qwen_vl_interface_bbox,
            qwen_inputs=qwen_inputs,
            generation_kwargs=self._build_generation_kwargs(
                self.vqa_max_new_tokens,
                self.vqa_do_sample,
                self.vqa_temperature,
                self.vqa_top_p,
            ),
        )

        pred_bbox_resized = _parse_bbox_from_text(pred_text)
        pred_bbox_orig = None
        response_type = 'unknown'
        if pred_bbox_resized is not None:
            pred_bbox_orig = _resized_to_original_bbox(
                pred_bbox_resized,
                src_w=src_w,
                src_h=src_h,
                dst_w=dst_w,
                dst_h=dst_h,
            )
            pred_bbox_orig = _clip_bbox_xyxy(pred_bbox_orig, w=src_w, h=src_h)
            if _is_valid_bbox_xyxy(pred_bbox_orig):
                response_type = 'bbox'
            else:
                pred_bbox_orig = None

        bbox_out = None
        if pred_bbox_orig is not None:
            bbox_out = [int(round(v)) for v in pred_bbox_orig]

        return {
            'response_type': response_type,
            'bbox': bbox_out,
            'bbox_resized': pred_bbox_resized,
            'answer': pred_text,
            'prompt': bbox_lang,
            'target_name': target_name,
            'operation': operation,
            'grasp': bool(grasp_done),
            'image_key': image_key,
        }

    def _run_signal(self, obs: dict[str, Any], frame_tag: str | None = None) -> dict[str, Any]:
        if self.qwen_vl_interface_policy is None:
            raise RuntimeError('qwen_vl_interface_policy is missing, cannot run signal branch.')

        image_key = self._resolve_vqa_image_key(obs)
        image_orig = _coerce_image_to_pil(obs[image_key]).convert('RGB')
        src_w, src_h = image_orig.size
        dst_w, dst_h = self.vlm_signal_image_size
        image_resized = image_orig.resize((dst_w, dst_h), Image.BILINEAR)

        prompt = self._resolve_prompt(obs)
        target_name, grasp_done, operation = self._resolve_target_info(obs, prompt)
        signal_lang = self.signal_prompt_template.format(
            target_name=target_name,
            operation=operation,
            pred_action_token=self.pred_action_token,
            instruction='',
        )
        if frame_tag is not None:
            logger.info('%s [PROMPT] signal_lang(thread-B):\n%s', frame_tag, signal_lang if signal_lang else '<empty>')

        qwen_inputs = self.qwen_vl_interface_policy.build_qwenvl_inputs(
            images=[[image_resized]],
            instructions=[signal_lang],
        )
        pred_text = _decode_vlm_generation(
            qwen_vl_interface=self.qwen_vl_interface_policy,
            qwen_inputs=qwen_inputs,
            generation_kwargs=self._build_generation_kwargs(
                self.signal_max_new_tokens,
                self.signal_do_sample,
                self.signal_temperature,
                self.signal_top_p,
            ),
        )
        pred_text_lower = pred_text.lower()

        pred_bbox_resized = _parse_bbox_from_text(pred_text)
        pred_bbox_orig = None
        response_type = 'unknown'
        if pred_bbox_resized is not None:
            pred_bbox_orig = _resized_to_original_bbox(
                pred_bbox_resized,
                src_w=src_w,
                src_h=src_h,
                dst_w=dst_w,
                dst_h=dst_h,
            )
            pred_bbox_orig = _clip_bbox_xyxy(pred_bbox_orig, w=src_w, h=src_h)
            if _is_valid_bbox_xyxy(pred_bbox_orig):
                response_type = 'bbox'
            else:
                pred_bbox_orig = None

        if response_type != 'bbox':
            if (self.pred_action_token in pred_text) or ('pred_action' in pred_text_lower):
                response_type = 'pred_action'
            else:
                response_type = 'pred_action'

        bbox_out = None
        if pred_bbox_orig is not None:
            bbox_out = [int(round(v)) for v in pred_bbox_orig]

        return {
            'response_type': response_type,
            'bbox': bbox_out,
            'bbox_resized': pred_bbox_resized,
            'answer': pred_text,
            'prompt': signal_lang,
            'target_name': target_name,
            'operation': operation,
            'grasp': bool(grasp_done),
            'image_key': image_key,
            'pred_action_token': self.pred_action_token,
        }

    def _run_signal_then_action(self, obs: dict[str, Any], frame_tag: str | None = None) -> dict[str, Any]:
        signal_payload = self._run_signal(obs, frame_tag=frame_tag)

        result: dict[str, Any] = {
            'signal': signal_payload,
            # Backward compatibility for existing clients that read response['vqa'].
            'vqa': signal_payload,
        }

        response_type = _to_str(signal_payload.get('response_type', '')).strip().lower()
        if self.vqa_skip_action_on_bbox and response_type == 'bbox':
            result['action'] = None
            result['action_skipped'] = True
            result['action_skip_reason'] = 'signal_predicted_bbox'
            return result

        actions = self._predict_action(obs, frame_tag=frame_tag)
        result['action'] = actions
        result['action_skipped'] = False
        return result

    def infer(self, obs: dict[str, Any], frame_id: int | None = None) -> dict[str, Any]:
        frame_tag = f'[FRAME {frame_id:06d}]' if frame_id is not None else '[FRAME ??????]'
        logger.info('%s [STAGE] policy.infer: 开始，准备规范化 obs', frame_tag)
        obs = _normalize_obs_keys(obs)

        prompt = self._resolve_prompt(obs)
        instruction = self._resolve_instruction(prompt)
        logger.info(
            '%s [STAGE] policy.infer: prompt/instruction 已解析，prompt_len=%d instruction_len=%d',
            frame_tag,
            len(prompt),
            len(instruction),
        )

        if not self.vqa_enabled:
            logger.info('%s [STAGE] policy.infer: 跳过 VLM 分支（vqa_enabled=False）', frame_tag)
            actions = self._predict_action(obs, frame_tag=frame_tag)
            return {
                'action': actions,
                'action_skipped': False,
                'signal': {'response_type': 'disabled', 'bbox': None, 'answer': ''},
                'vqa': {'response_type': 'disabled', 'bbox': None, 'answer': ''},
            }

        logger.info('%s [STAGE] policy.infer: 启动双线程（A:bbox-debug, B:signal->action）', frame_tag)
        t0 = time.monotonic()
        future_bbox = self._executor.submit(self._run_bbox_debug, obs, frame_tag)
        future_policy = self._executor.submit(self._run_signal_then_action, obs, frame_tag)

        bbox_payload: dict[str, Any]
        policy_result: dict[str, Any]

        try:
            bbox_payload = future_bbox.result()
        except Exception as exc:
            logger.exception('%s [STAGE] thread-A bbox debug 失败', frame_tag)
            bbox_payload = {
                'response_type': 'error',
                'bbox': None,
                'answer': '',
                'error': str(exc),
                'prompt': '',
                'image_key': '',
            }

        try:
            policy_result = future_policy.result()
        except Exception as exc:
            logger.exception('%s [STAGE] thread-B signal/action 失败', frame_tag)
            policy_result = {
                'action': None,
                'action_skipped': True,
                'action_skip_reason': f'signal_action_error: {exc}',
                'signal': {'response_type': 'error', 'bbox': None, 'answer': '', 'error': str(exc)},
                'vqa': {'response_type': 'error', 'bbox': None, 'answer': '', 'error': str(exc)},
            }

        policy_result['bbox_debug_local'] = bbox_payload
        policy_result.setdefault('debug_timing', {})
        policy_result['debug_timing']['dual_branch_ms'] = (time.monotonic() - t0) * 1000.0

        signal_payload = policy_result.get('signal', {}) if isinstance(policy_result, dict) else {}
        logger.info(
            '%s [STAGE] policy.infer: 完成 signal_type=%s signal_bbox=%s action_skipped=%s',
            frame_tag,
            _to_str(signal_payload.get('response_type', '')),
            signal_payload.get('bbox'),
            bool(policy_result.get('action_skipped', False)) if isinstance(policy_result, dict) else True,
        )
        return policy_result


class WebsocketPolicyServer:
    """Simple websocket policy server with wall-x compatible message flow."""

    def __init__(
        self,
        policy: StarVLAWallXPolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict[str, Any] | None = None,
        frame_debug_root: str | Path = "/beijing-c/wallx_workspace/starVLA/debug",
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._frame_debug_root = Path(frame_debug_root).expanduser().resolve()
        session_name = time.strftime("session_%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
        self._frame_debug_dir = self._frame_debug_root / session_name
        self._frame_debug_dir.mkdir(parents=True, exist_ok=True)
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        logger.info("Frame debug artifacts dir: %s", self._frame_debug_dir)

    def _build_connection_debug_dir(self, remote_address: Any) -> Path:
        if isinstance(remote_address, tuple) and len(remote_address) >= 2:
            peer = f"{remote_address[0]}_{remote_address[1]}"
        elif remote_address is not None:
            peer = str(remote_address)
        else:
            peer = "unknown"

        conn_name = f"conn_{_sanitize_filename(peer)}_{int(time.time() * 1000)}"
        conn_dir = self._frame_debug_dir / conn_name
        conn_dir.mkdir(parents=True, exist_ok=True)
        return conn_dir

    def _save_frame_debug(
        self,
        connection_dir: Path,
        frame_id: int,
        obs: dict[str, Any],
        prompt: str,
        instruction: str,
        history_keys: list[str],
        action_image_keys: list[str],
        response: dict[str, Any],
        infer_ms: float,
    ) -> None:
        frame_dir = connection_dir / f"frame_{frame_id:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        (frame_dir / "prompt.txt").write_text((prompt or "") + "\n", encoding="utf-8")
        (frame_dir / "instruction.txt").write_text((instruction or "") + "\n", encoding="utf-8")

        image_manifest: dict[str, str] = {}
        image_idx = 0
        for key, value in obs.items():
            try:
                image = _coerce_image_to_pil(value)
            except Exception:
                continue

            image_name = f"obs_{image_idx:02d}_{_sanitize_filename(key)}.jpg"
            image.save(frame_dir / image_name, quality=95)
            image_manifest[key] = image_name
            image_idx += 1

        bbox_debug_payload = response.get("bbox_debug_local") if isinstance(response, dict) else None
        signal_payload = response.get("signal") if isinstance(response, dict) else None
        if signal_payload is None and isinstance(response, dict):
            signal_payload = response.get("vqa")

        bbox_source_file: str | None = None
        bbox_overlay_file: str | None = None
        bbox_image_key = ""
        bbox_int: list[int] | None = None

        if isinstance(bbox_debug_payload, dict):
            bbox_image_key = _to_str(bbox_debug_payload.get("image_key", "")).strip()
            if (not bbox_image_key) or (bbox_image_key not in obs) or _is_history_key(bbox_image_key):
                for key in self._policy.camera_keys:
                    if (key in obs) and (not _is_history_key(key)):
                        bbox_image_key = key
                        break

            if bbox_image_key in obs:
                image_source = _coerce_image_to_pil(obs[bbox_image_key]).convert("RGB")
                bbox_source_file = "bbox_thread_source.jpg"
                image_source.save(frame_dir / bbox_source_file, quality=95)

                image_overlay = image_source.copy()
                draw = ImageDraw.Draw(image_overlay)
                raw_bbox = bbox_debug_payload.get("bbox")
                if _is_valid_bbox_xyxy(raw_bbox):
                    x1, y1, x2, y2 = [int(round(v)) for v in raw_bbox]
                    bbox_int = [x1, y1, x2, y2]
                    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
                    draw.text((x1, max(0, y1 - 18)), f"bbox: [{x1}, {y1}, {x2}, {y2}]", fill=(255, 0, 0))
                else:
                    draw.text((8, 8), "bbox: None", fill=(255, 0, 0))

                bbox_overlay_file = "bbox_thread_overlay.jpg"
                image_overlay.save(frame_dir / bbox_overlay_file, quality=95)

        action_file: str | None = None
        action_shape: list[int] | None = None
        first_action: list[float] | None = None
        action_mean: float | None = None

        if isinstance(response, dict) and response.get("action") is not None:
            action_arr = np.asarray(response["action"], dtype=np.float32)
            action_file = "action.npy"
            np.save(frame_dir / action_file, action_arr)
            action_shape = list(action_arr.shape)
            action_mean = float(action_arr.mean()) if action_arr.size > 0 else None

            if action_arr.ndim >= 3 and action_arr.shape[0] > 0 and action_arr.shape[1] > 0:
                first_action = [float(x) for x in action_arr[0, 0, :].tolist()]
            elif action_arr.size > 0:
                flat = action_arr.reshape(-1)
                first_action = [float(x) for x in flat[: min(16, flat.size)].tolist()]

        frame_info = {
            "frame_id": int(frame_id),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "instruction": instruction,
            "obs_keys": list(obs.keys()),
            "history_keys": history_keys,
            "action_image_keys": action_image_keys,
            "saved_images": image_manifest,
            "infer_ms": float(infer_ms),
            "action": {
                "file": action_file,
                "shape": action_shape,
                "first_action": first_action,
                "mean": action_mean,
                "action_skipped": bool(response.get("action_skipped", False)) if isinstance(response, dict) else False,
            },
            "bbox_debug": {
                "enabled": bool(self._policy.vqa_enabled),
                "image_key": bbox_image_key,
                "source_file": bbox_source_file,
                "overlay_file": bbox_overlay_file,
                "bbox": bbox_int,
                "response_type": _to_str(bbox_debug_payload.get("response_type", "")) if isinstance(bbox_debug_payload, dict) else "disabled",
                "answer": _to_str(bbox_debug_payload.get("answer", "")) if isinstance(bbox_debug_payload, dict) else "",
                "prompt": _to_str(bbox_debug_payload.get("prompt", "")) if isinstance(bbox_debug_payload, dict) else "",
            },
            "signal": {
                "response_type": _to_str(signal_payload.get("response_type", "")) if isinstance(signal_payload, dict) else "disabled",
                "bbox": signal_payload.get("bbox") if isinstance(signal_payload, dict) else None,
                "answer": _to_str(signal_payload.get("answer", "")) if isinstance(signal_payload, dict) else "",
                "prompt": _to_str(signal_payload.get("prompt", "")) if isinstance(signal_payload, dict) else "",
            },
        }

        with open(frame_dir / "frame_info.json", "w", encoding="utf-8") as f:
            json.dump(frame_info, f, ensure_ascii=False, indent=2)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            process_request=_health_check,
        ) as server:
            logger.info("Server started on %s:%d", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)

        if msgpack is None:
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="msgpack-numpy is not installed on server side",
            )
            return

        logger.info("Connection from %s [STAGE] 开始发送 metadata", websocket.remote_address)
        await websocket.send(msgpack.packb(self._metadata))
        logger.info("Connection from %s [STAGE] metadata 发送完成", websocket.remote_address)

        connection_dir = self._build_connection_debug_dir(websocket.remote_address)
        logger.info("Connection from %s [STAGE] 调试目录已创建: %s", websocket.remote_address, connection_dir)

        prev_total_time: float | None = None
        frame_id = 0
        while True:
            try:
                next_frame_id = frame_id + 1
                logger.info("[FRAME %06d][STAGE] 等待 client msgpack 消息...", next_frame_id)

                start_time = time.monotonic()
                raw_payload = await websocket.recv()
                frame_id = next_frame_id

                payload_bytes: int | str = "unknown"
                if isinstance(raw_payload, (bytes, bytearray, memoryview)):
                    payload_bytes = len(raw_payload)
                elif isinstance(raw_payload, str):
                    payload_bytes = len(raw_payload.encode("utf-8"))

                logger.info(
                    "[FRAME %06d][STAGE] 已接收 client 消息，payload_type=%s payload_bytes=%s，准备 msgpack 解包",
                    frame_id,
                    type(raw_payload).__name__,
                    payload_bytes,
                )

                raw_obs = msgpack.unpackb(raw_payload)
                logger.info(
                    "[FRAME %06d][STAGE] msgpack 解包完成，obs_type=%s",
                    frame_id,
                    type(raw_obs).__name__,
                )

                prompt = ""
                instruction = ""
                history_keys: list[str] = []
                action_image_keys: list[str] = []

                if isinstance(raw_obs, dict):
                    logger.info("[FRAME %06d][STAGE] 进入 obs 预处理阶段", frame_id)
                    obs = _normalize_obs_keys(raw_obs)
                    obs_keys = list(obs.keys())
                    history_keys = _extract_history_keys(obs)
                    action_image_keys = self._policy._collect_action_image_keys(obs)
                    prompt = self._policy._resolve_prompt(obs)
                    instruction = self._policy._resolve_instruction(prompt)

                    logger.info("[FRAME %06d] Received obs keys (%d): %s", frame_id, len(obs_keys), obs_keys)
                    logger.info(
                        "[FRAME %06d] Detected history frames: count=%d keys=%s",
                        frame_id,
                        len(history_keys),
                        history_keys,
                    )
                    logger.info("[FRAME %06d] Prompt: %s", frame_id, prompt if prompt else "<empty>")
                else:
                    obs = raw_obs
                    logger.warning(
                        "[FRAME %06d] Received non-dict obs payload: type=%s",
                        frame_id,
                        type(obs).__name__,
                    )
                    logger.info("[FRAME %06d] Prompt: <unavailable>", frame_id)

                logger.info("[FRAME %06d][STAGE] 进入 policy.infer", frame_id)
                infer_start = time.monotonic()
                response = self._policy.infer(obs, frame_id=frame_id)
                infer_ms = (time.monotonic() - infer_start) * 1000.0
                logger.info("[FRAME %06d][STAGE] policy.infer 返回，开始整理回包", frame_id)

                action_shape: list[int] | str | None = None
                if isinstance(response, dict) and response.get("action") is not None:
                    try:
                        action_shape = list(np.asarray(response["action"]).shape)
                    except Exception:
                        action_shape = type(response["action"]).__name__
                logger.info(
                    "[FRAME %06d] Inference done: action_shape=%s infer_ms=%.3f",
                    frame_id,
                    action_shape,
                    infer_ms,
                )

                if isinstance(obs, dict):
                    frame_debug_dir = connection_dir / f"frame_{frame_id:06d}"
                    logger.info("[FRAME %06d][STAGE] 开始保存调试产物到 %s", frame_id, frame_debug_dir)
                    try:
                        self._save_frame_debug(
                            connection_dir=connection_dir,
                            frame_id=frame_id,
                            obs=obs,
                            prompt=prompt,
                            instruction=instruction,
                            history_keys=history_keys,
                            action_image_keys=action_image_keys,
                            response=response,
                            infer_ms=infer_ms,
                        )
                        logger.info("[FRAME %06d][STAGE] 调试产物保存完成", frame_id)
                    except Exception:
                        logger.exception("[FRAME %06d][STAGE] 调试产物保存失败", frame_id)

                if isinstance(response, dict):
                    response.pop("bbox_debug_local", None)

                response["server_timing"] = {"infer_ms": infer_ms}
                if prev_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                logger.info("[FRAME %06d][STAGE] 开始回传 response 给 client", frame_id)
                await websocket.send(msgpack.packb(response))
                total_ms = (time.monotonic() - start_time) * 1000.0
                logger.info(
                    "[FRAME %06d][STAGE] 回传完成，总耗时=%.3fms infer_ms=%.3fms",
                    frame_id,
                    total_ms,
                    infer_ms,
                )
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                logger.exception("[FRAME %06d][STAGE] 请求处理失败，准备返回错误并关闭连接", frame_id)
                error_response = {"error": "Internal server error", "traceback": traceback.format_exc()}
                await websocket.send(msgpack.packb(error_response))
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Serve StarVLA checkpoint with wall-x websocket protocol for closed-loop simulation.'
    )
    parser.add_argument(
        '--config_yaml',
        type=str,
        default=None,
        help='Training config yaml used to build framework. If omitted, tries <run_dir>/config.yaml.',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Default checkpoint file (.pt/.safetensors) or checkpoint directory.',
    )
    parser.add_argument(
        '--checkpoint_bbox',
        type=str,
        default=None,
        help='Optional checkpoint for thread-A bbox model. Defaults to --checkpoint.',
    )
    parser.add_argument(
        '--checkpoint_policy',
        type=str,
        default=None,
        help='Optional checkpoint for thread-B signal/action model. Defaults to --checkpoint.',
    )
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Bind host.')
    parser.add_argument('--port', type=int, default=8000, help='Bind port.')
    parser.add_argument(
        '--frame_debug_root',
        type=str,
        default='/beijing-c/wallx_workspace/starVLA/debug',
        help='Directory for per-frame debug artifacts (images, prompt/instruction, bbox visualization, action outputs).',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Deprecated fallback device for both branches, e.g. cuda / cuda:0 / cpu.',
    )
    parser.add_argument(
        '--device_bbox',
        type=str,
        default='cuda:0',
        help='Device for thread-A bbox model.',
    )
    parser.add_argument(
        '--device_policy',
        type=str,
        default='cuda:1',
        help='Device for thread-B signal/action model.',
    )
    parser.add_argument(
        '--camera_keys',
        type=str,
        nargs='+',
        default=['face_view'],
        help='Observation image keys read from client obs and passed to model as image list (in order).',
    )
    parser.add_argument(
        '--pred_horizon',
        type=int,
        default=None,
        help='Prediction horizon to return. Default uses framework.action_model.future_action_window_size + 1.',
    )
    parser.add_argument(
        '--action_dim',
        type=int,
        default=None,
        help='Action dimension to return. Default uses framework.action_model.action_dim.',
    )
    parser.add_argument(
        '--default_prompt',
        type=str,
        default=None,
        help='Fallback prompt when client obs has no prompt.',
    )
    parser.add_argument(
        '--unnormalize_actions',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Whether to unnormalize model normalized_actions before sending to client.',
    )
    parser.add_argument(
        '--vqa_enabled',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable dual-branch VLM pipeline: thread-A bbox debug + thread-B signal/action.',
    )
    parser.add_argument(
        '--vqa_camera_key',
        type=str,
        default=None,
        help='Camera key used for VLM prompts, default tries obs.vqa_camera_key then first camera key.',
    )
    parser.add_argument(
        '--vqa_skip_action_on_bbox',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If true, when signal predicts bbox, skip action generation and return bbox to client directly.',
    )

    # Thread-A bbox generation args.
    parser.add_argument('--vqa_max_new_tokens', type=int, default=32, help='BBox branch generation max_new_tokens.')
    parser.add_argument(
        '--vqa_do_sample',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Use sampling in bbox branch generation.',
    )
    parser.add_argument('--vqa_temperature', type=float, default=0.2, help='BBox branch temperature when sampling.')
    parser.add_argument('--vqa_top_p', type=float, default=0.95, help='BBox branch top-p when sampling.')

    # Thread-B signal generation args (defaults are conservative/deterministic).
    parser.add_argument('--signal_max_new_tokens', type=int, default=16, help='Signal branch generation max_new_tokens.')
    parser.add_argument(
        '--signal_do_sample',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Use sampling in signal branch generation (default: deterministic greedy).',
    )
    parser.add_argument('--signal_temperature', type=float, default=0.2, help='Signal branch temperature when sampling.')
    parser.add_argument('--signal_top_p', type=float, default=0.95, help='Signal branch top-p when sampling.')

    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    return parser


def _normalize_runtime_device(requested: str, label: str) -> str:
    dev = str(requested)
    if dev.startswith('cuda'):
        if not torch.cuda.is_available():
            logger.warning('%s requested %s but CUDA is unavailable, fallback to cpu.', label, dev)
            return 'cpu'

        if dev == 'cuda':
            return 'cuda'

        match = re.fullmatch(r'cuda:(\d+)', dev)
        if match is not None:
            idx = int(match.group(1))
            count = int(torch.cuda.device_count())
            if idx >= count:
                fallback = 'cuda:0'
                logger.warning(
                    '%s requested %s but only %d CUDA devices are visible, fallback to %s.',
                    label,
                    dev,
                    count,
                    fallback,
                )
                return fallback
    return dev


def main() -> None:
    args = _build_argparser().parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    if msgpack is None:
        raise ImportError('msgpack-numpy is required. Please install: pip install msgpack msgpack-numpy')

    # accelerate.logging requires an initialized state even in single-process serving.
    PartialState()

    checkpoint_default = _resolve_checkpoint_path(args.checkpoint)
    checkpoint_bbox = _resolve_checkpoint_path(args.checkpoint_bbox) if args.checkpoint_bbox else checkpoint_default
    checkpoint_policy = _resolve_checkpoint_path(args.checkpoint_policy) if args.checkpoint_policy else checkpoint_default

    config_anchor_checkpoint = checkpoint_policy
    config_path = _resolve_config_path(args.config_yaml, config_anchor_checkpoint)
    cfg = OmegaConf.load(str(config_path))

    fallback_device = args.device if args.device is not None else 'cuda'
    device_bbox = args.device_bbox if args.device_bbox else fallback_device
    device_policy = args.device_policy if args.device_policy else fallback_device

    if args.device is not None:
        logger.warning('--device is deprecated; prefer --device_bbox/--device_policy for dual-branch serving.')

    device_bbox = _normalize_runtime_device(device_bbox, label='bbox_model')
    device_policy = _normalize_runtime_device(device_policy, label='policy_model')

    logger.info('Loading StarVLA dual-model serving stack...')
    logger.info('Config: %s', config_path)
    logger.info('Checkpoint (bbox): %s', checkpoint_bbox)
    logger.info('Checkpoint (policy): %s', checkpoint_policy)
    logger.info('Device (bbox): %s', device_bbox)
    logger.info('Device (policy): %s', device_policy)

    model_bbox = _load_model(cfg=cfg, checkpoint=checkpoint_bbox, device=device_bbox)
    model_policy = _load_model(cfg=cfg, checkpoint=checkpoint_policy, device=device_policy)

    action_dim = int(args.action_dim) if args.action_dim is not None else int(cfg.framework.action_model.action_dim)
    if args.pred_horizon is not None:
        pred_horizon = int(args.pred_horizon)
    else:
        future = int(_cfg_get(cfg.framework.action_model, 'future_action_window_size', 15))
        pred_horizon = future + 1

    action_stats = None
    if args.unnormalize_actions:
        action_stats = _load_action_norm_stats(cfg.datasets.vla_data)
        if action_stats is None:
            logger.warning(
                'Action unnormalization is enabled but no norm stats were loaded. '
                'The server will return normalized actions.'
            )

    policy = StarVLAWallXPolicy(
        model_bbox=model_bbox,
        model_policy=model_policy,
        cfg=cfg,
        device_bbox=device_bbox,
        device_policy=device_policy,
        camera_keys=list(args.camera_keys),
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        action_stats=action_stats,
        unnormalize_actions=bool(args.unnormalize_actions),
        default_prompt=args.default_prompt,
        vqa_enabled=bool(args.vqa_enabled),
        vqa_camera_key=args.vqa_camera_key,
        vqa_skip_action_on_bbox=bool(args.vqa_skip_action_on_bbox),
        vqa_max_new_tokens=int(args.vqa_max_new_tokens),
        vqa_do_sample=bool(args.vqa_do_sample),
        vqa_temperature=float(args.vqa_temperature),
        vqa_top_p=float(args.vqa_top_p),
        signal_max_new_tokens=int(args.signal_max_new_tokens),
        signal_do_sample=bool(args.signal_do_sample),
        signal_temperature=float(args.signal_temperature),
        signal_top_p=float(args.signal_top_p),
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = 'unknown'

    metadata = policy.metadata.copy()
    metadata['framework'] = str(_cfg_get(cfg.framework, 'name', 'unknown'))
    metadata['config_yaml'] = str(config_path)
    metadata['checkpoint_bbox'] = str(checkpoint_bbox)
    metadata['checkpoint_policy'] = str(checkpoint_policy)

    logger.info('Server hostname: %s', hostname)
    logger.info('Server IP: %s', local_ip)
    logger.info('Server endpoint: ws://%s:%d', args.host, args.port)
    logger.info('Health check endpoint: http://%s:%d/healthz', args.host, args.port)
    logger.info('Frame debug root: %s', args.frame_debug_root)
    logger.info(
        'Serving config: action_dim=%d pred_horizon=%d camera_keys=%s vqa_enabled=%s unnormalize_actions=%s',
        action_dim,
        pred_horizon,
        list(args.camera_keys),
        bool(args.vqa_enabled),
        bool(args.unnormalize_actions),
    )

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
        frame_debug_root=args.frame_debug_root,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
