"""StarVLA and mock backends behind the Go2 evaluation protocol."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Protocol

from .protocol import ProtocolError, normalize_decision

DEFAULT_ROUTER_PROMPT = (
    "{instruction}\nChoose the current control route: NAV, GRASP, PLACE, DONE, "
    "or RECOVER. Output exactly one route token and one local subtask instruction."
)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class EvaluationBackend(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def reset(self, episode_id: str | None) -> dict[str, Any]: ...


def _decode_jpeg(image_payload: Any):
    from PIL import Image

    if not isinstance(image_payload, dict):
        raise ProtocolError("image payload must be an object")
    if image_payload.get("encoding") != "jpeg_base64":
        raise ProtocolError("only jpeg_base64 images are supported")
    encoded = image_payload.get("data")
    if not isinstance(encoded, str) or not encoded:
        raise ProtocolError("image data must be a non-empty base64 string")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ProtocolError("image data is not valid base64") from exc
    if len(raw) > 8 * 1024 * 1024:
        raise ProtocolError("one encoded image exceeds 8 MiB")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB").copy()
    except Exception as exc:
        raise ProtocolError("image data is not a valid JPEG") from exc


class StarVLABackend:
    """Load a trained QwenPI checkpoint and expose typed Go2 inference."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        use_bf16: bool = True,
    ):
        import torch

        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.device = device
        self.model = self._load_model(self.checkpoint)
        if use_bf16:
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(device).eval()

        datasets_cfg = _cfg_get(self.model.config, "datasets", None)
        router_cfg = _cfg_get(datasets_cfg, "router_data", None)
        self.router_prompt = str(
            _cfg_get(router_cfg, "router_prompt", DEFAULT_ROUTER_PROMPT)
        )
        self.include_state = bool(_cfg_get(router_cfg, "include_state", False))
        self.rtc_enabled = bool(
            getattr(getattr(self.model, "action_model", None), "rtc_enabled", False)
        )
        # 按 route 分别保存上一帧 chunk：episode_id -> {nav|grasp: chunk}
        self._rtc_prev: dict[str, dict[str, Any]] = {}
        self._rtc_last_route: dict[str, str] = {}

    @staticmethod
    def _load_model(checkpoint: str | Path):
        """加载完整 QwenPI checkpoint，兼容旧词表 padding 差异。

        ``baseframework.from_pretrained`` 使用严格 ``load_state_dict``，遇到旧
        checkpoint（embed_tokens/lm_head 151677 vs 配置 151936）会直接失败。
        这里复用评测脚本的 ``adapt_padded_vocab_state_dict`` 路径扩展 padding 行。
        """
        import torch
        from accelerate import PartialState
        from omegaconf import OmegaConf

        from starVLA.model.framework.__init__ import build_framework
        from starVLA.model.framework.share_tools import dict_to_namespace
        from starVLA.training.trainer_utils.trainer_tools import (
            adapt_padded_vocab_state_dict,
        )

        # 模型构建过程（prepare_qwen_special_tokens 等）会使用 accelerate 的
        # logging，必须先初始化分布式状态，否则抛出 RuntimeError。
        PartialState()

        checkpoint = Path(checkpoint).expanduser().resolve()
        config_path = checkpoint.parents[1] / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"checkpoint run 缺少 config.yaml: {config_path}")
        config = dict_to_namespace(
            OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        )
        config.trainer.pretrained_checkpoint = None
        model = build_framework(cfg=config)
        stats_path = checkpoint.parents[1] / "dataset_statistics.json"
        if stats_path.is_file():
            import json

            with open(stats_path, encoding="utf-8") as stream:
                model.norm_stats = json.load(stream)
        if checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(checkpoint))
        else:
            state_dict = torch.load(checkpoint, map_location="cpu", mmap=True)
        state_dict, expanded_keys = adapt_padded_vocab_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=True)
        if expanded_keys:
            print(
                "[vla] 已兼容扩展旧 checkpoint 的词表 padding 行："
                + ", ".join(expanded_keys)
            )
        return model

    def _denormalize_decision(self, raw: dict[str, Any]) -> dict[str, Any]:
        """把模型输出的归一化 action（≈[-1,1]）还原为物理单位。

        训练侧 go2 dataloader 用 q01/q99 把目标归一化到 [-1,1]，推理输出同样
        在该空间，必须用同一份统计量还原；NAV 用 0:3，机械臂用 3:10。
        """
        import numpy as np

        norm_stats = getattr(self.model, "norm_stats", None)
        if not norm_stats:
            return raw
        unnorm_key = next(iter(norm_stats.keys()))
        stats = norm_stats[unnorm_key]["action"]
        q01 = np.asarray(stats["q01"], dtype=np.float64)
        q99 = np.asarray(stats["q99"], dtype=np.float64)
        per_step = q01.ndim == 2

        waypoints = raw.get("nav_waypoints")
        if waypoints is not None:
            arr = np.clip(np.asarray(waypoints, dtype=np.float64), -1.0, 1.0)
            if per_step:
                t = min(arr.shape[0], q01.shape[0])
                q01_slice = np.broadcast_to(q01[:t, 0:3], arr.shape)
                q99_slice = np.broadcast_to(q99[:t, 0:3], arr.shape)
            else:
                q01_slice = np.broadcast_to(q01[0:3], arr.shape)
                q99_slice = np.broadcast_to(q99[0:3], arr.shape)
            raw["nav_waypoints"] = 0.5 * (arr + 1.0) * (q99_slice - q01_slice) + q01_slice
        arm_targets = raw.get("arm_targets_base")
        if arm_targets is not None:
            arr = np.clip(np.asarray(arm_targets, dtype=np.float64), -1.0, 1.0)
            if per_step:
                t = min(arr.shape[0], q01.shape[0])
                q01_slice = np.broadcast_to(q01[:t, 3:10], arr.shape)
                q99_slice = np.broadcast_to(q99[:t, 3:10], arr.shape)
            else:
                q01_slice = np.broadcast_to(q01[3:10], arr.shape)
                q99_slice = np.broadcast_to(q99[3:10], arr.shape)
            raw["arm_targets_base"] = (
                0.5 * (arr + 1.0) * (q99_slice - q01_slice) + q01_slice
            )
        return raw

    def _remember_rtc_chunk(
        self, episode_id: str, raw: dict[str, Any]
    ) -> None:
        """按 route 保存归一化空间的上一帧 action chunk。

        nav chunk 只填 0:3，grasp/place chunk 只填 3:10，其余维度为 0，
        与训练目标中 inactive dims=0 的分布一致；跨 route 不互相借用。
        """
        import numpy as np

        waypoints = raw.get("nav_waypoints")
        arm_targets = raw.get("arm_targets_base")
        route = str(raw.get("route", "")).strip().lower()
        if route not in {"nav", "grasp", "place"}:
            return
        if waypoints is not None:
            chunk = np.zeros((len(waypoints), 10), dtype=np.float32)
            chunk[:, 0:3] = np.asarray(waypoints, dtype=np.float32)
        elif arm_targets is not None:
            chunk = np.zeros((len(arm_targets), 10), dtype=np.float32)
            chunk[:, 3:10] = np.asarray(arm_targets, dtype=np.float32)
        else:
            return
        self._rtc_prev.setdefault(episode_id, {})[route] = chunk

    def _rtc_kwargs(self, episode_id: str, route: str) -> dict[str, Any]:
        """取当前 route 自己的上一帧 chunk 作为 RTC 条件。

        首次进入某 route（没有同 route 历史）时不传 prev，避免把
        上一 route 的 chunk 或全 0 值钉进 prefix。
        """
        if not self.rtc_enabled or not episode_id:
            return {}
        prev = self._rtc_prev.get(episode_id, {}).get(route)
        if prev is None:
            return {"prev_action_chunk": None, "inference_delay": 0}
        return {"prev_action_chunk": prev, "inference_delay": 1}

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "starvla_qwenpi",
            "checkpoint": self.checkpoint,
            "device": self.device,
            "capabilities": ["typed_route", "sparse_body_waypoints", "base_frame_arm_targets"],
        }

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        episode_id = str(payload.get("episode_id", "") or "")
        images = payload["images"]
        front = _decode_jpeg(images["front"])
        ordered_images = [front]
        if images.get("wrist") is not None:
            ordered_images.append(_decode_jpeg(images["wrist"]))

        state = None
        if self.include_state:
            body_velocity = (
                (payload.get("state") or {}).get("base_velocity_body", [0.0, 0.0, 0.0])
            )
            if not isinstance(body_velocity, (list, tuple)) or len(body_velocity) != 3:
                raise ProtocolError("state.base_velocity_body must contain [vx,vy,wz]")
            arm_state = (payload.get("state") or {}).get(
                "arm_tcp_base", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
            if not isinstance(arm_state, (list, tuple)) or len(arm_state) != 7:
                raise ProtocolError(
                    "state.arm_tcp_base must contain [x,y,z,roll,pitch,yaw,gripper]"
                )
            state = [
                [
                    *(float(value) for value in body_velocity),
                    *(float(value) for value in arm_state),
                ]
            ]
        prompt = self.router_prompt.format(
            instruction=str(payload["instruction"]).strip()
        )
        example: dict[str, Any] = {"image": ordered_images, "lang": prompt}
        if state is not None:
            example["state"] = state
        locked_route = payload.get("locked_route")
        locked_subtask = payload.get("locked_subtask")
        if locked_route:
            if not isinstance(locked_route, str) or locked_route not in {"nav", "grasp", "place"}:
                raise ProtocolError("locked_route must be nav/grasp/place")
            if locked_subtask is not None and not isinstance(locked_subtask, str):
                raise ProtocolError("locked_subtask must be a string")
            # 先做一次轻量 first-token route 检测（不生成 subtask，~85ms）。
            # route 与锁定一致 -> 沿用 locked subtask，只跑 action（~180ms）；
            # route 变化（如 nav->grasp）-> 完整推理重新生成 subtask + action。
            detection = self.model.predict_route(
                examples=[example],
                max_new_tokens=0,
                do_sample=False,
                route_mode="first_token",
                continue_action=False,
                continue_bbox=False,
                allow_bbox=False,
            )
            detected_route = str(detection["routes"][0]["route"]).strip().lower()
            if episode_id:
                # 跨 route 切换（nav<->grasp/place）时清空该 episode 的 RTC
                # 历史，避免复用上一阶段的旧 chunk 作 prefix。
                last_route = self._rtc_last_route.get(episode_id)
                if last_route != detected_route:
                    self._rtc_prev.pop(episode_id, None)
                    if detected_route in {"nav", "grasp", "place"}:
                        self._rtc_last_route[episode_id] = detected_route
            # RTC 条件只取当前 route 自己的历史 chunk；route 变化时通常还没有
            # 同 route 历史，此时不传 prev（delay=0）。
            rtc_kwargs = self._rtc_kwargs(episode_id, detected_route)
            if detected_route == locked_route:
                raw = self.model.predict_locked_action(
                    examples=[example],
                    locked_route=locked_route,
                    locked_subtask=locked_subtask,
                    **rtc_kwargs,
                )
            else:
                raw = self.model.predict_typed_action(
                    examples=[example],
                    allow_bbox=False,
                    **rtc_kwargs,
                )
        else:
            # 非锁定路径不携带 prev：route 在推理后才确定，且首次/切换
            # route 时本来就不该用上一 route 的 chunk 作条件。
            raw = self.model.predict_typed_action(
                examples=[example],
                allow_bbox=False,
            )
        if episode_id:
            route = str(raw.get("route", "")).strip().lower()
            if route in {"nav", "grasp", "place"}:
                self._rtc_last_route[episode_id] = route
            self._remember_rtc_chunk(episode_id, raw)
        raw = self._denormalize_decision(raw)
        return normalize_decision(raw)

    def reset(self, episode_id: str | None) -> dict[str, Any]:
        if episode_id:
            self._rtc_prev.pop(str(episode_id), None)
        return {"reset": True, "episode_id": episode_id}


class MockBackend:
    """Dependency-light backend for protocol and tunnel smoke tests."""

    def __init__(self, route: str = "nav"):
        self.route = route

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "mock",
            "capabilities": ["typed_route", "sparse_body_waypoints", "base_frame_arm_targets"],
        }

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        decision = {
            "route": self.route,
            "subtask": "Mock evaluation decision.",
            "nav_waypoints": (
                [[0.25, 0.0, 0.0], [0.50, 0.0, 0.0]]
                if self.route == "nav"
                else None
            ),
            "arm_targets_base": (
                [[0.35, 0.0, 0.20, 0.0, 0.0, 0.0, 1.0]]
                if self.route in {"grasp", "place"}
                else None
            ),
            "raw_text": f"<|{self.route}|>",
            "route_confidence": 1.0,
        }
        return normalize_decision(decision)

    def reset(self, episode_id: str | None) -> dict[str, Any]:
        return {"reset": True, "episode_id": episode_id}
