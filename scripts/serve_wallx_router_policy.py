#!/usr/bin/env python3
"""WebSocket server for StarVLA unified router checkpoints.

This server intentionally uses a small protocol for IsaacDataCollect instead of
the older wall-x serving protocol. Requests are msgpack dictionaries carrying a
current RGB image, optional history images, task fields, and optional state.
Responses carry the router decision plus either an ego-frame action horizon or a
bbox prediction.
"""

from __future__ import annotations

import argparse
import asyncio
import http
import json
import logging
import re
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

import websockets
import websockets.frames

try:
    import websockets.asyncio.server as ws_server
except Exception:  # pragma: no cover - depends on websockets version.
    ws_server = None

try:
    import msgpack
    import msgpack_numpy as msgpack_numpy

    msgpack_numpy.patch()
except ImportError:  # pragma: no cover - deployment dependency check happens at runtime.
    msgpack = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.model.framework import build_framework
from starVLA.model.modules.action_model.fast_ActionHeader import load_fast_action_processor
from starVLA.training.trainer_utils.trainer_tools import resize_images


PROTOCOL = "starvla_router_v1"
LOGGER = logging.getLogger("starvla_router_server")
POINT_PATTERN = re.compile(
    r"<point>\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*</point>",
    re.IGNORECASE,
)


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def cfg_set(cfg: Any, key: str, value: Any) -> None:
    OmegaConf.update(cfg, key, value, merge=True)


def normalize_keys(payload: dict[Any, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(key, bytes):
            out[key.decode("utf-8", errors="ignore")] = value
        else:
            out[str(key)] = value
    return out


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def coerce_image_to_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim > 3:
        arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return Image.fromarray(arr, mode="RGB")


def parse_bbox(text: str) -> Optional[list[float]]:
    match = POINT_PATTERN.search(str(text))
    if match is None:
        match = re.search(
            r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]",
            str(text),
        )
    if match is None:
        return None
    return [float(match.group(i)) for i in range(1, 5)]


def clip_bbox_xyxy(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        float(np.clip(x1, 0, width - 1)),
        float(np.clip(y1, 0, height - 1)),
        float(np.clip(x2, 0, width - 1)),
        float(np.clip(y2, 0, height - 1)),
    ]


def resolve_checkpoint_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    candidates: list[Path] = []
    for candidate in path.rglob("*"):
        if candidate.is_file() and (
            candidate.name.endswith("pytorch_model.pt")
            or candidate.name.endswith("model.safetensors")
            or candidate.suffix in {".pt", ".safetensors"}
        ):
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found under directory: {path}")

    def score(candidate: Path) -> tuple[int, float]:
        match = re.search(r"steps_(\d+)", candidate.name)
        step = int(match.group(1)) if match else -1
        return step, candidate.stat().st_mtime

    candidates.sort(key=score)
    return candidates[-1]


def load_model(
    cfg: Any,
    checkpoint: Path,
    device: str,
    *,
    checkpoint_modules: Optional[str] = None,
) -> torch.nn.Module:
    ensure_accelerate_state()
    model = build_framework(cfg=cfg)
    qwen = getattr(model, "qwen_vl_interface", None)
    if qwen is not None:
        qwenvl_cfg = cfg_get(cfg_get(cfg, "framework", None), "qwenvl", None)
        configured_base = str(cfg_get(qwenvl_cfg, "base_vlm", ""))
        tokenizer_len = len(qwen.processor.tokenizer)
        embed_rows = int(qwen.model.get_input_embeddings().weight.shape[0])
        model_type = getattr(qwen.model.config, "model_type", None)
        LOGGER.info(
            "Effective VLM: base_vlm=%s interface=%s model_type=%s tokenizer_len=%d embed_rows=%d",
            configured_base,
            type(qwen).__name__,
            model_type,
            tokenizer_len,
            embed_rows,
        )
        if "Qwen3.5" in configured_base and tokenizer_len < 200000:
            raise RuntimeError(
                "Configured Qwen3.5 base_vlm but loaded a non-Qwen3.5 tokenizer/model. "
                f"base_vlm={configured_base!r}, interface={type(qwen).__name__}, "
                f"model_type={model_type!r}, tokenizer_len={tokenizer_len}, embed_rows={embed_rows}"
            )
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint))
    else:
        state_dict = torch.load(str(checkpoint), map_location="cpu")

    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "module"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    def maybe_resize_qwen_embeddings(module: torch.nn.Module, weights: dict[str, torch.Tensor]) -> None:
        qwen = getattr(module, "qwen_vl_interface", None)
        if qwen is None and hasattr(module, "processor") and hasattr(module, "model"):
            qwen = module
        if qwen is None:
            return

        embed_key = None
        for candidate in (
            "qwen_vl_interface.model.model.language_model.embed_tokens.weight",
            "model.model.language_model.embed_tokens.weight",
        ):
            if candidate in weights:
                embed_key = candidate
                break
        if embed_key is None:
            return

        target_rows = int(weights[embed_key].shape[0])
        current_rows = int(qwen.model.get_input_embeddings().weight.shape[0])
        if target_rows == current_rows:
            return

        tokenizer_len = len(qwen.processor.tokenizer)
        if target_rows < tokenizer_len:
            raise RuntimeError(
                "Checkpoint Qwen embedding rows are smaller than tokenizer length: "
                f"checkpoint_rows={target_rows}, tokenizer_len={tokenizer_len}"
            )
        LOGGER.info(
            "Resizing Qwen token embeddings to match checkpoint: current_rows=%d checkpoint_rows=%d tokenizer_len=%d",
            current_rows,
            target_rows,
            tokenizer_len,
        )
        qwen.model.resize_token_embeddings(target_rows)

    if checkpoint_modules:
        loaded = []
        for module_path in [part.strip() for part in checkpoint_modules.split(",") if part.strip()]:
            module = model
            for attr in module_path.split("."):
                module = getattr(module, attr)
            prefix = module_path + "."
            sub_state_dict = {
                key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
            }
            if not sub_state_dict:
                raise RuntimeError(f"No checkpoint parameters found for module `{module_path}`")
            maybe_resize_qwen_embeddings(module, sub_state_dict)
            missing, unexpected = module.load_state_dict(sub_state_dict, strict=False)
            LOGGER.info(
                "Loaded module `%s` from checkpoint; missing=%d unexpected=%d",
                module_path,
                len(missing),
                len(unexpected),
            )
            if missing:
                LOGGER.warning("Missing `%s` keys: first=%s", module_path, missing[:10])
            if unexpected:
                LOGGER.warning("Unexpected `%s` keys: first=%s", module_path, unexpected[:10])
            loaded.append(module_path)
        LOGGER.info("Loaded checkpoint modules: %s", loaded)
    else:
        maybe_resize_qwen_embeddings(model, state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            LOGGER.warning("Missing checkpoint keys: %d; first=%s", len(missing), missing[:10])
        if unexpected:
            LOGGER.warning("Unexpected checkpoint keys: %d; first=%s", len(unexpected), unexpected[:10])

    model = model.to(device)
    model.eval()
    return model


def ensure_accelerate_state() -> None:
    """Initialize accelerate state for modules that use accelerate logging."""
    try:
        from accelerate import PartialState

        PartialState()
    except Exception:
        LOGGER.exception("Failed to initialize accelerate PartialState.")
        raise


def load_action_norm_stats(router_cfg: Any) -> Optional[dict[str, np.ndarray]]:
    root = Path(str(cfg_get(router_cfg, "root", ""))).expanduser()
    if not root.exists():
        LOGGER.warning("Dataset root not found; action unnormalization disabled: %s", root)
        return None

    preferred = "norm_stats.json"
    if to_bool(cfg_get(router_cfg, "use_delta_action", False)):
        preferred = "norm_stats_delta.json"
    if to_bool(cfg_get(router_cfg, "action_in_ego", True), True):
        preferred = "norm_stats_ego.json"

    for name in dict.fromkeys([preferred, "norm_stats_ego.json", "norm_stats_delta.json", "norm_stats.json"]):
        path = root / name
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        stats = payload.get("norm_stats", {}).get("action", {})
        if "q01" not in stats or "q99" not in stats:
            LOGGER.warning("Ignoring invalid action stats file: %s", path)
            continue
        LOGGER.info("Loaded action stats from %s", path)
        return {
            "q01": np.asarray(stats["q01"], dtype=np.float32),
            "q99": np.asarray(stats["q99"], dtype=np.float32),
        }

    LOGGER.warning("No usable norm_stats*.json found under %s", root)
    return None


def unnormalize_actions(actions: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    out = np.asarray(actions, dtype=np.float32).copy()
    out = np.clip(out, -1.0, 1.0)
    q01 = np.asarray(stats["q01"], dtype=np.float32).reshape(-1)
    q99 = np.asarray(stats["q99"], dtype=np.float32).reshape(-1)
    dim = min(out.shape[-1], q01.shape[0], q99.shape[0])
    if dim <= 0:
        return out
    delta = q99[:dim] - q01[:dim]
    delta = np.where(np.abs(delta) < 1e-8, 1.0, delta)
    out[..., :dim] = 0.5 * (out[..., :dim] + 1.0) * delta + q01[:dim]
    return out


class FastActionDecodeError(RuntimeError):
    def __init__(self, message: str, debug_payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.debug_payload = debug_payload


class FastActionCoeffStoppingCriteria(StoppingCriteria):
    def __init__(self, decoder: "FastActionDecoder", prompt_len: int) -> None:
        self.decoder = decoder
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        generated_ids = input_ids[:, self.prompt_len :]
        batch_fast_ids, _ = self.decoder._extract_fast_token_ids(generated_ids)
        if any(len(ids) == 0 for ids in batch_fast_ids):
            return False
        coeff_counts = [len(self.decoder._decode_bpe_text(ids)) for ids in batch_fast_ids]
        return all(count >= self.decoder.expected_coeff_count for count in coeff_counts)


class FastActionCoeffLogitsProcessor(LogitsProcessor):
    def __init__(self, decoder: "FastActionDecoder", prompt_len: int) -> None:
        self.decoder = decoder
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated_ids = input_ids[:, self.prompt_len :]
        batch_fast_ids, _ = self.decoder._extract_fast_token_ids(generated_ids)
        constrained = torch.full_like(scores, -torch.inf)
        action_min = self.decoder.action_token_min
        for row_idx, fast_ids in enumerate(batch_fast_ids):
            coeff_count = len(self.decoder._decode_bpe_text(fast_ids)) if fast_ids else 0
            remaining = self.decoder.expected_coeff_count - coeff_count
            if remaining <= 0:
                for token_id in self.decoder.stop_token_ids:
                    if 0 <= token_id < scores.shape[-1]:
                        constrained[row_idx, token_id] = scores[row_idx, token_id]
                continue

            allowed_fast_ids = self.decoder.fast_ids_by_max_coeff_len[min(remaining, self.decoder.max_fast_coeff_len)]
            if not allowed_fast_ids:
                continue
            allowed_token_ids = torch.as_tensor(
                [action_min + fast_id for fast_id in allowed_fast_ids],
                device=scores.device,
                dtype=torch.long,
            )
            constrained[row_idx, allowed_token_ids] = scores[row_idx, allowed_token_ids]
        return constrained


class TokenSequenceStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_sequences: list[list[int]]) -> None:
        self.stop_sequences = [list(seq) for seq in stop_sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if not self.stop_sequences:
            return False
        for row in input_ids:
            row_ids = [int(token_id) for token_id in row.detach().cpu().tolist()]
            if not any(
                len(row_ids) >= len(stop_seq) and row_ids[-len(stop_seq) :] == stop_seq
                for stop_seq in self.stop_sequences
            ):
                return False
        return True


class FastActionDecoder:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: Any,
        *,
        pred_action_token: str,
        tokenizer_path: Optional[str],
        token_prefix: Optional[str],
        token_count: Optional[int],
        action_horizon: int,
        action_dim: int,
        max_new_tokens: int,
        decode_attempts: int,
        retry_do_sample: bool,
        retry_temperature: float,
        retry_top_p: float,
        debug_dir: Optional[str] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.pred_action_token = pred_action_token
        self.max_new_tokens = int(max_new_tokens)
        if self.max_new_tokens <= 0:
            raise ValueError(f"fast_max_new_tokens must be positive, got {self.max_new_tokens}")
        self.decode_attempts = int(decode_attempts)
        if self.decode_attempts <= 0:
            raise ValueError(f"fast_decode_attempts must be positive, got {self.decode_attempts}")
        self.retry_do_sample = bool(retry_do_sample)
        self.retry_temperature = float(retry_temperature)
        self.retry_top_p = float(retry_top_p)

        action_tokenizer_cfg = cfg_get(cfg_get(cfg.framework, "action_tokenizer", None), "path", None)
        path = tokenizer_path or action_tokenizer_cfg or "physical-intelligence/fast"
        self.processor = load_fast_action_processor(str(path))
        self.tokenizer_path = str(path)

        framework_tokenizer_cfg = cfg_get(cfg.framework, "action_tokenizer", None)
        self.token_prefix = str(token_prefix or cfg_get(framework_tokenizer_cfg, "token_prefix", "<robot_action_"))
        self.token_count = int(token_count or cfg_get(framework_tokenizer_cfg, "token_count", 2048))
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.expected_coeff_count = self.action_horizon * self.action_dim
        self.debug_dir = Path(debug_dir).expanduser() if debug_dir else None
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.processor, "time_horizon"):
            self.processor.time_horizon = self.action_horizon
        if hasattr(self.processor, "action_dim"):
            self.processor.action_dim = self.action_dim

        self.action_token_min = getattr(self.model.qwen_vl_interface, "_ACTION_TOKEN_MIN", None)
        self.action_token_max = getattr(self.model.qwen_vl_interface, "_ACTION_TOKEN_MAX", None)
        if self.action_token_min is None or self.action_token_max is None:
            raise ValueError(
                "FAST action tokens are not available in qwen_vl_interface. "
                "Use an ActionRouter base VLM and set special_tokens.require_fast_action_tokens=true."
            )
        self.action_token_min = int(self.action_token_min)
        self.action_token_max = int(self.action_token_max)
        expected_max = self.action_token_min + self.token_count - 1
        if self.action_token_max < expected_max:
            raise ValueError(
                "Configured FAST token count exceeds VLM action-token range: "
                f"token_count={self.token_count}, range=[{self.action_token_min}, {self.action_token_max}]"
            )

        token_ids = self.model.qwen_vl_interface.processor.tokenizer(
            self.pred_action_token,
            add_special_tokens=False,
        ).input_ids
        if len(token_ids) != 1:
            raise ValueError(f"`{self.pred_action_token}` must be a single tokenizer id, got {token_ids}")
        self.pred_action_token_id = int(token_ids[0])
        self.stop_token_ids = self._stop_token_ids()
        self.fast_token_coeff_lengths = tuple(
            len(self._decode_bpe_text([fast_id])) for fast_id in range(self.token_count)
        )
        self.max_fast_coeff_len = self.expected_coeff_count
        self.fast_ids_by_max_coeff_len = self._fast_ids_by_max_coeff_len(self.max_fast_coeff_len)

    def metadata(self) -> dict[str, Any]:
        return {
            "tokenizer_path": self.tokenizer_path,
            "token_prefix": self.token_prefix,
            "token_count": self.token_count,
            "action_token_range": [self.action_token_min, self.action_token_max],
            "max_new_tokens": self.max_new_tokens,
            "decode_attempts": self.decode_attempts,
            "retry_do_sample": self.retry_do_sample,
            "retry_temperature": self.retry_temperature,
            "retry_top_p": self.retry_top_p,
            "debug_dir": str(self.debug_dir) if self.debug_dir is not None else None,
            "coeff_constrained": True,
        }

    def _stop_token_ids(self) -> list[int]:
        tokenizer = self.model.qwen_vl_interface.processor.tokenizer
        token_ids: list[int] = []
        for value in (
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
            getattr(self.model.qwen_vl_interface.model.generation_config, "eos_token_id", None),
            getattr(self.model.qwen_vl_interface.model.generation_config, "pad_token_id", None),
        ):
            values = value if isinstance(value, (list, tuple)) else [value]
            for token_id in values:
                if token_id is not None and int(token_id) not in token_ids:
                    token_ids.append(int(token_id))
        return token_ids

    def _fast_ids_by_max_coeff_len(self, max_len: int) -> list[list[int]]:
        buckets: list[list[int]] = [[] for _ in range(max_len + 1)]
        for remaining in range(1, max_len + 1):
            buckets[remaining] = [
                fast_id
                for fast_id, coeff_len in enumerate(self.fast_token_coeff_lengths)
                if 0 < coeff_len <= remaining
            ]
        return buckets

    def _extract_fast_token_ids(self, generated_ids: torch.Tensor) -> tuple[list[list[int]], list[list[int]]]:
        batches: list[list[int]] = []
        raw_batches: list[list[int]] = []
        for row_idx in range(generated_ids.shape[0]):
            row_ids = [int(token_id) for token_id in generated_ids[row_idx].detach().cpu().tolist()]
            raw_fast_ids: list[int] = []
            span_fast_ids: list[int] = []
            in_span = False
            for token_id in row_ids:
                is_action_token = self.action_token_min <= token_id <= self.action_token_max
                if is_action_token:
                    fast_id = token_id - self.action_token_min
                    raw_fast_ids.append(fast_id)
                    if not in_span:
                        in_span = True
                    span_fast_ids.append(fast_id)
                elif in_span:
                    break
            raw_batches.append(raw_fast_ids)
            batches.append(span_fast_ids)
        return batches, raw_batches

    def _decode_bpe_text(self, fast_ids: list[int]) -> str:
        try:
            return self.processor.bpe_tokenizer.decode(
                fast_ids,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.processor.bpe_tokenizer.decode(fast_ids)

    def _trim_fast_ids_to_expected_coeffs(
        self,
        fast_ids: list[int],
        *,
        allow_trim: bool,
    ) -> tuple[list[int], dict[str, Any]]:
        coeff_count = len(self._decode_bpe_text(fast_ids))
        info: dict[str, Any] = {
            "input_token_count": len(fast_ids),
            "input_coeff_count": coeff_count,
            "expected_coeff_count": self.expected_coeff_count,
            "trimmed": False,
        }
        if coeff_count == self.expected_coeff_count:
            info["decode_token_count"] = len(fast_ids)
            info["decode_coeff_count"] = coeff_count
            return fast_ids, info

        first_over: Optional[dict[str, int]] = None
        for end in range(1, len(fast_ids) + 1):
            prefix = fast_ids[:end]
            prefix_coeff_count = len(self._decode_bpe_text(prefix))
            if prefix_coeff_count == self.expected_coeff_count:
                if not allow_trim:
                    first_over = {
                        "token_count": end,
                        "coeff_count": prefix_coeff_count,
                    }
                    break
                info.update(
                    {
                        "decode_token_count": end,
                        "decode_coeff_count": prefix_coeff_count,
                        "trimmed": end < len(fast_ids),
                    }
                )
                return prefix, info
            if prefix_coeff_count > self.expected_coeff_count:
                first_over = {
                    "token_count": end,
                    "coeff_count": prefix_coeff_count,
                }
                break

        info["first_over_expected"] = first_over
        raise ValueError(
            "FAST generated token span cannot be decoded to the configured action shape: "
            f"tokens={len(fast_ids)}, coeffs={coeff_count}, expected_coeffs={self.expected_coeff_count}, "
            f"first_over_expected={first_over}"
        )

    def _decode_fast_ids(
        self,
        batch_fast_ids: list[list[int]],
        *,
        allow_trim: bool,
    ) -> tuple[np.ndarray, list[list[int]], list[dict[str, Any]]]:
        if any(len(ids) == 0 for ids in batch_fast_ids):
            raise ValueError(f"FAST generation produced no action tokens: {batch_fast_ids}")
        invalid = [
            token_id
            for ids in batch_fast_ids
            for token_id in ids
            if token_id < 0 or token_id >= self.token_count
        ]
        if invalid:
            raise ValueError(f"FAST generation produced token ids outside configured range: {invalid[:8]}")

        decode_fast_ids: list[list[int]] = []
        decode_infos: list[dict[str, Any]] = []
        for ids in batch_fast_ids:
            trimmed_ids, info = self._trim_fast_ids_to_expected_coeffs(ids, allow_trim=allow_trim)
            decode_fast_ids.append(trimmed_ids)
            decode_infos.append(info)

        if any(bool(info.get("trimmed")) for info in decode_infos):
            LOGGER.warning("Trimmed FAST action tokens before decode: %s", decode_infos)

        decoded = self.processor.decode(
            decode_fast_ids,
            time_horizon=self.action_horizon,
            action_dim=self.action_dim,
        )
        actions = np.asarray(decoded, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None, ...]
        if actions.ndim != 3:
            raise ValueError(f"FAST decoded actions must have shape [B,T,D], got {actions.shape}")
        if actions.shape[1] > self.action_horizon:
            actions = actions[:, : self.action_horizon, :]
        if actions.shape[1] < self.action_horizon:
            pad_len = self.action_horizon - actions.shape[1]
            pad = np.repeat(actions[:, -1:, :], pad_len, axis=1)
            actions = np.concatenate([actions, pad], axis=1)
        if actions.shape[2] > self.action_dim:
            actions = actions[:, :, : self.action_dim]
        if not np.isfinite(actions).all():
            raise ValueError("FAST decoded actions contain non-finite values")
        return actions, decode_fast_ids, decode_infos

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): FastActionDecoder._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [FastActionDecoder._json_safe(v) for v in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    def _dump_debug(self, payload: dict[str, Any]) -> None:
        if self.debug_dir is None:
            return
        context = payload.get("context") or {}
        request_id = context.get("request_id", "unknown") if isinstance(context, dict) else "unknown"
        path = self.debug_dir / f"fast_request_{request_id}_{int(time.time() * 1000)}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(payload), f, ensure_ascii=False, indent=2)
        LOGGER.info("Wrote FAST debug payload: %s", path)

    @torch.inference_mode()
    def predict(
        self,
        examples: list[dict[str, Any]],
        *,
        route_prefix: Optional[str] = None,
        debug_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        train_obs_image_size = self.model._train_image_size() if hasattr(self.model, "_train_image_size") else None
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.model.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        route_prefix = str(route_prefix or self.pred_action_token)
        route_prefix_ids = self.model.qwen_vl_interface.processor.tokenizer(
            route_prefix,
            add_special_tokens=False,
        ).input_ids
        if not route_prefix_ids:
            raise ValueError(f"route_prefix tokenized to an empty sequence: {route_prefix!r}")
        if int(route_prefix_ids[0]) != self.pred_action_token_id:
            raise ValueError(
                "FAST route_prefix must start with the pred action token. "
                f"route_prefix={route_prefix!r}"
            )

        forced_inputs = {key: value for key, value in qwen_inputs.items() if key != "labels"}
        input_ids = forced_inputs["input_ids"]
        route_ids = torch.tensor(
            route_prefix_ids,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )[None, :].expand(input_ids.shape[0], -1)
        forced_inputs["input_ids"] = torch.cat([input_ids, route_ids], dim=1)
        if "attention_mask" in forced_inputs:
            attention_mask = forced_inputs["attention_mask"]
            forced_inputs["attention_mask"] = torch.cat(
                [attention_mask, torch.ones_like(route_ids)],
                dim=1,
            )

        prompt_len = int(qwen_inputs["input_ids"].shape[1])
        forced_prompt_len = int(forced_inputs["input_ids"].shape[1])
        last_debug_payload: dict[str, Any] | None = None
        last_error: Exception | None = None

        for attempt_idx in range(self.decode_attempts):
            attempt_number = attempt_idx + 1
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "logits_processor": LogitsProcessorList(
                    [FastActionCoeffLogitsProcessor(self, prompt_len=forced_prompt_len)]
                ),
                "stopping_criteria": StoppingCriteriaList(
                    [FastActionCoeffStoppingCriteria(self, prompt_len=forced_prompt_len)]
                ),
            }
            if attempt_idx > 0 and self.retry_do_sample:
                generation_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": self.retry_temperature,
                        "top_p": self.retry_top_p,
                    }
                )

            generated = self.model.qwen_vl_interface.generate(
                **forced_inputs,
                **generation_kwargs,
            )
            gen_ids = generated.sequences if hasattr(generated, "sequences") else generated
            new_ids = gen_ids[:, prompt_len:]
            generated_text = self.model.qwen_vl_interface.processor.batch_decode(
                new_ids,
                skip_special_tokens=False,
            )
            batch_fast_ids, raw_batch_fast_ids = self._extract_fast_token_ids(new_ids)
            span_coeff_counts = [len(self._decode_bpe_text(ids)) if ids else 0 for ids in batch_fast_ids]
            raw_coeff_counts = [len(self._decode_bpe_text(ids)) if ids else 0 for ids in raw_batch_fast_ids]
            debug_payload = {
                "context": debug_context or {},
                "attempt": attempt_number,
                "decode_attempts": self.decode_attempts,
                "retry_do_sample": bool(attempt_idx > 0 and self.retry_do_sample),
                "retry_temperature": self.retry_temperature,
                "retry_top_p": self.retry_top_p,
                "coeff_constrained": True,
                "max_new_tokens": self.max_new_tokens,
                "action_horizon": self.action_horizon,
                "action_dim": self.action_dim,
                "expected_coeff_count": self.expected_coeff_count,
                "generated_token_ids": new_ids.detach().cpu().tolist(),
                "raw_action_fast_token_ids": raw_batch_fast_ids,
                "raw_action_coeff_counts": raw_coeff_counts,
                "span_action_fast_token_ids": batch_fast_ids,
                "span_action_coeff_counts": span_coeff_counts,
                "generated_text": [text.strip() for text in generated_text],
            }
            try:
                normalized_actions, decode_fast_ids, decode_infos = self._decode_fast_ids(
                    batch_fast_ids,
                    allow_trim=False,
                )
            except Exception as exc:
                debug_payload["error"] = repr(exc)
                self._dump_debug(debug_payload)
                last_debug_payload = debug_payload
                last_error = exc
                continue

            debug_payload["decode_fast_token_ids"] = decode_fast_ids
            debug_payload["decode_infos"] = decode_infos
            self._dump_debug(debug_payload)
            return {
                "normalized_actions": normalized_actions,
                "fast_token_ids": decode_fast_ids,
                "raw_fast_token_ids": raw_batch_fast_ids,
                "decode_infos": decode_infos,
                "generated_text": [text.strip() for text in generated_text],
                "attempt": attempt_number,
            }

        message = f"FAST generation failed after {self.decode_attempts} attempts"
        if last_error is not None:
            message = f"{message}: {last_error}"
        raise FastActionDecodeError(message, last_debug_payload or {"context": debug_context or {}})


class RouterPolicy:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: Any,
        *,
        action_stats: Optional[dict[str, np.ndarray]],
        unnormalize: bool,
        action_decode_mode: str,
        fast_decoder: Optional[FastActionDecoder],
        route_mode: str,
        route_max_new_tokens: int,
        bbox_max_new_tokens: int,
        route_confidence_threshold: Optional[float],
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.router_cfg = cfg.datasets.router_data
        self.action_stats = action_stats
        self.unnormalize = bool(unnormalize)
        self.action_decode_mode = str(action_decode_mode)
        if self.action_decode_mode not in {"flow", "fast"}:
            raise ValueError(f"Unsupported action_decode_mode={self.action_decode_mode!r}")
        self.fast_decoder = fast_decoder
        if self.action_decode_mode == "fast" and self.fast_decoder is None:
            raise ValueError("FAST action_decode_mode requires a FastActionDecoder")
        self.route_mode = route_mode
        self.route_max_new_tokens = int(route_max_new_tokens)
        self.bbox_max_new_tokens = int(bbox_max_new_tokens)
        self.route_confidence_threshold = route_confidence_threshold

        self.pred_action_token = str(cfg_get(self.router_cfg, "pred_action_token", "<|pred_action|>"))
        self.pred_bbox_token = str(cfg_get(self.router_cfg, "pred_bbox_token", "<|pred_bbox|>"))
        router_framework_cfg = cfg_get(cfg_get(cfg, "framework", None), "router", None)
        self.action_route_format = str(
            cfg_get(self.router_cfg, "action_route_format", cfg_get(router_framework_cfg, "action_route_format", "token"))
        ).strip()
        if self.action_route_format not in {"token", "route_subtask"}:
            raise ValueError(f"Unsupported action_route_format={self.action_route_format!r}")
        self.subtask_start_token = str(
            cfg_get(self.router_cfg, "subtask_start_token", cfg_get(router_framework_cfg, "subtask_start_token", "<|subtask|>"))
        )
        self.subtask_end_token = str(
            cfg_get(
                self.router_cfg,
                "subtask_end_token",
                cfg_get(router_framework_cfg, "subtask_end_token", "<|end_subtask|>"),
            )
        )
        self.include_state = to_bool(cfg_get(self.router_cfg, "include_state", False))
        self.image_size = tuple(int(x) for x in cfg_get(self.router_cfg, "image_size", [224, 224]))
        framework_action_cfg = cfg_get(cfg.framework, "action_model", None)
        default_action_horizon = cfg_get(framework_action_cfg, "action_horizon", None)
        if default_action_horizon is None:
            future_window = cfg_get(framework_action_cfg, "future_action_window_size", None)
            default_action_horizon = int(future_window) + 1 if future_window is not None else 16
        self.action_horizon = int(cfg_get(self.router_cfg, "action_horizon", default_action_horizon))
        self.action_dim = int(cfg_get(cfg.framework.action_model, "action_dim", 6))
        self.action_in_ego = to_bool(cfg_get(self.router_cfg, "action_in_ego", True), True)
        self.use_delta_action = to_bool(cfg_get(self.router_cfg, "use_delta_action", False))
        default_flow_prompt_grasp = (
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
        default_flow_prompt_put = (
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
        default_fast_prompt_grasp = (
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
        default_fast_prompt_put = (
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
        legacy_flow_prompt = cfg_get(self.router_cfg, "router_prompt_template", None)
        legacy_fast_prompt = cfg_get(self.router_cfg, "router_prompt_template_fast_token_ce", None)
        self.prompt_template_grasp = str(
            cfg_get(self.router_cfg, "router_prompt_template_grasp", legacy_flow_prompt or default_flow_prompt_grasp)
        )
        self.prompt_template_put = str(
            cfg_get(self.router_cfg, "router_prompt_template_put", legacy_flow_prompt or default_flow_prompt_put)
        )
        self.prompt_template_fast_grasp = str(
            cfg_get(
                self.router_cfg,
                "router_prompt_template_fast_token_ce_grasp",
                legacy_fast_prompt or default_fast_prompt_grasp,
            )
        )
        self.prompt_template_fast_put = str(
            cfg_get(
                self.router_cfg,
                "router_prompt_template_fast_token_ce_put",
                legacy_fast_prompt or default_fast_prompt_put,
            )
        )

    def _select_prompt_template(self, operation: str) -> str:
        normalized_operation = str(operation or "").strip().lower()
        is_put_phase = normalized_operation in {"put", "place", "placement", "placing"}
        if self.action_decode_mode == "fast":
            return self.prompt_template_fast_put if is_put_phase else self.prompt_template_fast_grasp
        return self.prompt_template_put if is_put_phase else self.prompt_template_grasp

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "action": {
                "frame": "ego_delta" if self.action_in_ego else "world",
                "horizon": self.action_horizon,
                "dim": self.action_dim,
                "unnormalized": bool(self.unnormalize and self.action_stats is not None),
                "use_delta_action": self.use_delta_action,
                "decode_mode": self.action_decode_mode,
            },
            "router": {
                "pred_action_token": self.pred_action_token,
                "pred_bbox_token": self.pred_bbox_token,
                "route_mode": self.route_mode,
                "action_route_format": self.action_route_format,
                "subtask_start_token": self.subtask_start_token,
                "subtask_end_token": self.subtask_end_token,
            },
            "image_size": list(self.image_size),
            "fast_action": self.fast_decoder.metadata() if self.fast_decoder is not None else None,
        }

    def _build_prompt(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        task = normalize_keys(request.get("task", {}) or {})
        prompt = task.get("prompt") or task.get("lang") or request.get("prompt") or request.get("lang")
        if prompt:
            return str(prompt), task

        instruction = str(task.get("instruction") or request.get("instruction") or "").strip()
        target_name = str(task.get("target_name") or request.get("target_name") or "target object").strip()
        operation = str(task.get("operation") or request.get("operation") or "").strip()
        if not operation:
            operation = "put" if to_bool(task.get("grasp_done", request.get("grasp_done")), False) else "grasp"
        prompt_template = self._select_prompt_template(operation)
        prompt = prompt_template.format(
            instruction=instruction,
            target_name=target_name,
            operation=operation,
            pred_action_token=self.pred_action_token,
            pred_bbox_token=self.pred_bbox_token,
            subtask_start_token=self.subtask_start_token,
            subtask_end_token=self.subtask_end_token,
        )
        task.setdefault("target_name", target_name)
        task.setdefault("operation", operation)
        return prompt, task

    def _parse_action_route_solution(self, generated_text: str) -> tuple[str, Optional[str]]:
        text = str(generated_text or "").strip()
        if not text.startswith(self.pred_action_token):
            text = f"{self.pred_action_token}{text}"

        if self.action_route_format != "route_subtask":
            return self.pred_action_token, None

        start_idx = text.find(self.subtask_start_token)
        if start_idx < 0:
            return self.pred_action_token, None
        subtask_start = start_idx + len(self.subtask_start_token)
        end_idx = text.find(self.subtask_end_token, subtask_start)
        if end_idx < 0:
            return self.pred_action_token, None

        subtask_text = text[subtask_start:end_idx].strip()
        route_solution = (
            f"{self.pred_action_token}"
            f"{self.subtask_start_token}"
            f"{subtask_text}"
            f"{self.subtask_end_token}"
        )
        return route_solution, subtask_text or None

    def _route_stopping_criteria(self) -> Optional[StoppingCriteriaList]:
        if self.action_route_format != "route_subtask":
            return None
        stop_ids = self.model.qwen_vl_interface.processor.tokenizer(
            self.subtask_end_token,
            add_special_tokens=False,
        ).input_ids
        if not stop_ids:
            return None
        return StoppingCriteriaList([TokenSequenceStoppingCriteria([stop_ids])])

    def _build_example(self, raw_request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        request = normalize_keys(raw_request)
        obs = normalize_keys(request.get("obs", {}) or {})
        image = obs.get("image", obs.get("rgb", obs.get("front_rgb")))
        if image is None:
            image = request.get("image", request.get("rgb", request.get("front_rgb")))
        if image is None:
            raise ValueError("Request is missing obs.image / obs.rgb")

        current = coerce_image_to_rgb(image)
        history_raw = obs.get("history", request.get("history", [])) or []
        if isinstance(history_raw, dict):
            history_raw = [history_raw[key] for key in sorted(history_raw.keys())]
        history = [coerce_image_to_rgb(item) for item in list(history_raw)]

        prompt, task = self._build_prompt(request)
        example: dict[str, Any] = {
            "image": [current] + history,
            "lang": prompt,
        }

        state = obs.get("state", request.get("state"))
        if self.include_state and state is not None:
            state_arr = np.asarray(state, dtype=np.float32).reshape(1, -1)
            example["state"] = state_arr.astype(np.float16)

        source = {
            "request_id": request.get("request_id"),
            "timestamp": request.get("timestamp"),
            "task": task,
            "image_width": int(current.width),
            "image_height": int(current.height),
            "history_count": len(history),
            "prompt": prompt,
        }
        return example, source

    def infer(self, raw_request: dict[str, Any]) -> dict[str, Any]:
        example, source = self._build_example(raw_request)
        task = source.get("task", {}) or {}
        LOGGER.info(
            "Received policy request request_id=%s image=%sx%s history=%d "
            "instruction=%r target_name=%r operation=%r\nFull model prompt:\n%s",
            source.get("request_id"),
            source.get("image_width"),
            source.get("image_height"),
            source.get("history_count"),
            task.get("instruction") or task.get("prompt") or task.get("lang"),
            task.get("target_name"),
            task.get("operation"),
            source.get("prompt"),
        )
        route_generation_tokens = self.route_max_new_tokens
        if self.route_mode == "first_token" and self.action_route_format == "token":
            route_generation_tokens = self.bbox_max_new_tokens
        route_output = self.model.predict_route(
            examples=[example],
            max_new_tokens=route_generation_tokens,
            route_mode=self.route_mode,
            continue_bbox=True,
            continue_action=self.action_route_format == "route_subtask",
            route_confidence_threshold=self.route_confidence_threshold,
            stopping_criteria=self._route_stopping_criteria(),
        )
        route_info = dict(route_output["routes"][0])
        route_type = str(route_info.get("route", "unknown"))
        action_route_solution = None
        if route_type == "action":
            action_route_solution, subtask_text = self._parse_action_route_solution(
                str(route_info.get("generated_text", ""))
            )
            route_info["action_route_solution"] = action_route_solution
            if subtask_text is not None:
                route_info["subtask_text"] = subtask_text
                LOGGER.info(
                    "Predicted action subtask request_id=%s subtask=%r route_solution=%r",
                    source.get("request_id"),
                    subtask_text,
                    action_route_solution,
                )
            elif self.action_route_format == "route_subtask":
                LOGGER.warning(
                    "Action route did not contain a parseable subtask request_id=%s generated_text=%r",
                    source.get("request_id"),
                    route_info.get("generated_text"),
                )

        response: dict[str, Any] = {
            "protocol": PROTOCOL,
            "request_id": source.get("request_id"),
            "route": route_info,
            "action": None,
            "bbox": None,
            "source": {
                "image_width": source["image_width"],
                "image_height": source["image_height"],
                "history_count": source["history_count"],
                "task": source["task"],
            },
        }

        if route_type == "action":
            if self.action_decode_mode == "fast":
                assert self.fast_decoder is not None
                action_output = self.fast_decoder.predict(
                    examples=[example],
                    route_prefix=action_route_solution,
                    debug_context={
                        "request_id": source.get("request_id"),
                        "route": route_info,
                        "source": response["source"],
                    },
                )
            else:
                action_output = self.model.predict_action_with_route_token(
                    examples=[example],
                    route_token=action_route_solution or self.pred_action_token,
                )
            normalized = np.asarray(action_output["normalized_actions"][0], dtype=np.float32)
            action = normalized
            is_unnormalized = False
            if self.unnormalize and self.action_stats is not None:
                action = unnormalize_actions(normalized, self.action_stats)
                is_unnormalized = True

            response["action"] = {
                "frame": "ego_delta" if self.action_in_ego else "world",
                "horizon": action.astype(np.float32),
                "first": action[0].astype(np.float32) if action.size else None,
                "normalized": not is_unnormalized,
            }
            if self.action_decode_mode == "fast":
                response["action"]["fast_token_ids"] = np.asarray(
                    action_output.get("fast_token_ids", [[]])[0],
                    dtype=np.int32,
                )
                response["action"]["raw_fast_token_ids"] = np.asarray(
                    action_output.get("raw_fast_token_ids", [[]])[0],
                    dtype=np.int32,
                )
                response["action"]["decode_info"] = action_output.get("decode_infos", [{}])[0]
                response["action"]["fast_decode_attempt"] = int(action_output.get("attempt", 1))
                response["action"]["generated_text"] = str(action_output.get("generated_text", [""])[0])
            response["action_skipped"] = False
            return response

        if route_type == "bbox":
            text = str(route_info.get("generated_text", ""))
            bbox_model = parse_bbox(text)
            if bbox_model is not None:
                model_w, model_h = self.image_size[0], self.image_size[1]
                image_w = int(source["image_width"])
                image_h = int(source["image_height"])
                bbox_model = clip_bbox_xyxy(bbox_model, model_w, model_h)
                sx = float(image_w) / float(model_w)
                sy = float(image_h) / float(model_h)
                bbox_image = [
                    bbox_model[0] * sx,
                    bbox_model[1] * sy,
                    bbox_model[2] * sx,
                    bbox_model[3] * sy,
                ]
                bbox_image = clip_bbox_xyxy(bbox_image, image_w, image_h)
                response["bbox"] = {
                    "xyxy_model": np.asarray(bbox_model, dtype=np.float32),
                    "xyxy_image": np.asarray(bbox_image, dtype=np.float32),
                    "model_image_size": [model_w, model_h],
                    "image_size": [image_w, image_h],
                    "text": text,
                }
            response["action_skipped"] = True
            response["action_skip_reason"] = "router_predicted_bbox"
            return response

        response["action_skipped"] = True
        response["action_skip_reason"] = f"router_predicted_{route_type}"
        return response


class RouterWebSocketServer:
    def __init__(self, policy: RouterPolicy, host: str, port: int) -> None:
        self.policy = policy
        self.host = host
        self.port = int(port)
        self.metadata = self.policy.metadata()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        serve_kwargs = {
            "compression": None,
            "max_size": None,
            "ping_interval": None,
            "ping_timeout": None,
            "process_request": health_check,
        }
        if ws_server is not None:
            async with ws_server.serve(self.handler, self.host, self.port, **serve_kwargs) as server:
                LOGGER.info("Server listening on ws://%s:%d", self.host, self.port)
                await server.serve_forever()
        else:
            async with websockets.serve(self.handler, self.host, self.port, **serve_kwargs):
                LOGGER.info("Server listening on ws://%s:%d", self.host, self.port)
                await asyncio.Future()

    async def handler(self, websocket: Any, path: Optional[str] = None) -> None:
        if msgpack is None:
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="msgpack-numpy is not installed",
            )
            return

        LOGGER.info("Connection opened from %s", getattr(websocket, "remote_address", "unknown"))
        await websocket.send(msgpack.packb(self.metadata, use_bin_type=True))
        frame_id = 0
        while True:
            try:
                raw_payload = await websocket.recv()
                frame_id += 1
                request = msgpack.unpackb(raw_payload, raw=False)
                if not isinstance(request, dict):
                    raise ValueError(f"Request must be a dict, got {type(request).__name__}")

                t0 = time.monotonic()
                response = self.policy.infer(request)
                response.setdefault("server_timing", {})
                response["server_timing"]["infer_ms"] = (time.monotonic() - t0) * 1000.0
                LOGGER.info(
                    "frame=%06d request_id=%s route=%s infer_ms=%.1f",
                    frame_id,
                    response.get("request_id"),
                    response.get("route", {}).get("route"),
                    response["server_timing"]["infer_ms"],
                )
                await websocket.send(msgpack.packb(response, use_bin_type=True))
            except websockets.ConnectionClosed:
                LOGGER.info("Connection closed from %s", getattr(websocket, "remote_address", "unknown"))
                break
            except Exception:
                LOGGER.exception("Request failed")
                error_response = {
                    "protocol": PROTOCOL,
                    "error": "internal_server_error",
                    "traceback": traceback.format_exc(),
                }
                await websocket.send(msgpack.packb(error_response, use_bin_type=True))


def health_check(connection: Any, request: Any = None) -> Any:
    if isinstance(connection, str):
        path = connection
    elif request is not None:
        path = getattr(request, "path", None)
    else:
        path = connection
    if path != "/healthz":
        return None
    if hasattr(connection, "respond"):
        return connection.respond(http.HTTPStatus.OK, "ok\n")
    body = b"ok\n"
    return http.HTTPStatus.OK, [("Content-Type", "text/plain"), ("Content-Length", str(len(body)))], body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve StarVLA router policy for IsaacDataCollect.")
    parser.add_argument("--config_yaml", required=True, help="Router training config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file or run/checkpoints directory.")
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--route_mode", choices=("first_token", "generate"), default="first_token")
    parser.add_argument("--route_max_new_tokens", type=int, default=32)
    parser.add_argument("--bbox_max_new_tokens", type=int, default=32)
    parser.add_argument("--route_confidence_threshold", type=float, default=None)
    parser.add_argument(
        "--action_route_format",
        choices=("token", "route_subtask"),
        default=None,
        help="Override framework.router.action_route_format for action-route parsing.",
    )
    parser.add_argument("--subtask_start_token", default=None)
    parser.add_argument("--subtask_end_token", default=None)
    parser.add_argument(
        "--action_decode_mode",
        choices=("flow", "fast"),
        default="flow",
        help="Use the flow action expert or autoregressive FAST action tokens for action-route outputs.",
    )
    parser.add_argument(
        "--checkpoint_modules",
        default=None,
        help=(
            "Comma-separated module paths to load from the checkpoint. "
            "Defaults to qwen_vl_interface in FAST mode and full-model loading in flow mode."
        ),
    )
    parser.add_argument("--base_vlm", default=None, help="Override framework.qwenvl.base_vlm before model load.")
    parser.add_argument(
        "--special_tokens_policy",
        choices=("auto_add", "strict", "none"),
        default=None,
        help="Override framework.qwenvl.special_tokens.policy before model load.",
    )
    parser.add_argument(
        "--require_fast_action_tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override framework.qwenvl.special_tokens.require_fast_action_tokens before model load.",
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=None,
        help="Override router/action-model action horizon for metadata and FAST decode shape.",
    )
    parser.add_argument("--fast_tokenizer_path", default=None, help="Override framework.action_tokenizer.path.")
    parser.add_argument("--fast_token_prefix", default=None, help="Override FAST VLM token prefix.")
    parser.add_argument("--fast_token_count", type=int, default=None, help="Override FAST VLM token count.")
    parser.add_argument("--fast_max_new_tokens", type=int, default=256, help="FAST action generation budget.")
    parser.add_argument(
        "--fast_decode_attempts",
        type=int,
        default=1,
        help="Maximum FAST action generation attempts before returning an error.",
    )
    parser.add_argument(
        "--fast_retry_do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sampling for FAST retries after the first greedy attempt.",
    )
    parser.add_argument("--fast_retry_temperature", type=float, default=0.7, help="FAST retry sampling temperature.")
    parser.add_argument("--fast_retry_top_p", type=float, default=0.95, help="FAST retry sampling top-p.")
    parser.add_argument(
        "--fast_debug_dir",
        default=None,
        help="Optional directory for FAST generation/decode debug JSON files.",
    )
    parser.add_argument(
        "--unnormalize_actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return unnormalized actions in the trained action frame.",
    )
    parser.add_argument("--log_level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def apply_cli_overrides(cfg: Any, args: argparse.Namespace) -> None:
    if args.base_vlm:
        cfg_set(cfg, "framework.qwenvl.base_vlm", args.base_vlm)
    if args.special_tokens_policy:
        cfg_set(cfg, "framework.qwenvl.special_tokens.policy", args.special_tokens_policy)
    if args.require_fast_action_tokens is not None:
        cfg_set(
            cfg,
            "framework.qwenvl.special_tokens.require_fast_action_tokens",
            bool(args.require_fast_action_tokens),
        )
    if args.action_route_format:
        cfg_set(cfg, "framework.router.action_route_format", args.action_route_format)
        cfg_set(cfg, "datasets.router_data.action_route_format", args.action_route_format)
    if args.subtask_start_token:
        cfg_set(cfg, "framework.router.subtask_start_token", args.subtask_start_token)
        cfg_set(cfg, "datasets.router_data.subtask_start_token", args.subtask_start_token)
    if args.subtask_end_token:
        cfg_set(cfg, "framework.router.subtask_end_token", args.subtask_end_token)
        cfg_set(cfg, "datasets.router_data.subtask_end_token", args.subtask_end_token)
    if args.action_horizon is not None:
        if args.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {args.action_horizon}")
        cfg_set(cfg, "datasets.router_data.action_horizon", int(args.action_horizon))
        cfg_set(cfg, "framework.action_model.action_horizon", int(args.action_horizon))
        cfg_set(cfg, "framework.action_model.future_action_window_size", int(args.action_horizon) - 1)
    if args.fast_tokenizer_path:
        cfg_set(cfg, "framework.action_tokenizer.path", args.fast_tokenizer_path)
        cfg_set(cfg, "datasets.router_data.action_tokenizer.path", args.fast_tokenizer_path)
    if args.fast_token_prefix:
        cfg_set(cfg, "framework.action_tokenizer.token_prefix", args.fast_token_prefix)
        cfg_set(cfg, "datasets.router_data.action_tokenizer.token_prefix", args.fast_token_prefix)
        cfg_set(cfg, "framework.qwenvl.special_tokens.action_token_prefix", args.fast_token_prefix)
    if args.fast_token_count is not None:
        if args.fast_token_count <= 0:
            raise ValueError(f"fast_token_count must be positive, got {args.fast_token_count}")
        cfg_set(cfg, "framework.action_tokenizer.token_count", int(args.fast_token_count))
        cfg_set(cfg, "datasets.router_data.action_tokenizer.token_count", int(args.fast_token_count))
        cfg_set(cfg, "framework.qwenvl.special_tokens.action_token_count", int(args.fast_token_count))


def release_unused_flow_action_model(model: torch.nn.Module) -> None:
    action_model = getattr(model, "action_model", None)
    if action_model is None:
        return
    action_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    LOGGER.info("Moved unused flow action_model to CPU for FAST serving.")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if msgpack is None:
        raise RuntimeError("msgpack and msgpack_numpy are required")

    host_name = socket.gethostname()
    LOGGER.info("Starting StarVLA router server on host=%s", host_name)
    cfg = OmegaConf.load(args.config_yaml)
    apply_cli_overrides(cfg, args)
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    LOGGER.info("Resolved checkpoint: %s", checkpoint)
    checkpoint_modules = args.checkpoint_modules
    if checkpoint_modules is None and args.action_decode_mode == "fast":
        checkpoint_modules = "qwen_vl_interface"
    model = load_model(cfg, checkpoint, args.device, checkpoint_modules=checkpoint_modules)
    if args.action_decode_mode == "fast":
        release_unused_flow_action_model(model)
    action_stats = load_action_norm_stats(cfg.datasets.router_data)
    if args.unnormalize_actions and action_stats is None:
        LOGGER.warning("Action unnormalization requested but stats are missing; returning normalized actions.")

    framework_action_cfg = cfg_get(cfg.framework, "action_model", None)
    action_horizon = cfg_get(cfg.datasets.router_data, "action_horizon", None)
    if action_horizon is None:
        action_horizon = cfg_get(framework_action_cfg, "action_horizon", None)
    if action_horizon is None:
        future_window = cfg_get(framework_action_cfg, "future_action_window_size", 15)
        action_horizon = int(future_window) + 1
    action_dim = int(cfg_get(framework_action_cfg, "action_dim", 6))

    fast_decoder = None
    if args.action_decode_mode == "fast":
        fast_decoder = FastActionDecoder(
            model,
            cfg,
            pred_action_token=str(cfg_get(cfg.datasets.router_data, "pred_action_token", "<|pred_action|>")),
            tokenizer_path=args.fast_tokenizer_path,
            token_prefix=args.fast_token_prefix,
            token_count=args.fast_token_count,
            action_horizon=int(action_horizon),
            action_dim=action_dim,
            max_new_tokens=args.fast_max_new_tokens,
            decode_attempts=args.fast_decode_attempts,
            retry_do_sample=args.fast_retry_do_sample,
            retry_temperature=args.fast_retry_temperature,
            retry_top_p=args.fast_retry_top_p,
            debug_dir=args.fast_debug_dir,
        )

    policy = RouterPolicy(
        model,
        cfg,
        action_stats=action_stats,
        unnormalize=args.unnormalize_actions,
        action_decode_mode=args.action_decode_mode,
        fast_decoder=fast_decoder,
        route_mode=args.route_mode,
        route_max_new_tokens=args.route_max_new_tokens,
        bbox_max_new_tokens=args.bbox_max_new_tokens,
        route_confidence_threshold=args.route_confidence_threshold,
    )
    RouterWebSocketServer(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
