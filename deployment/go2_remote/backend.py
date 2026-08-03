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

        from starVLA.model.framework.base_framework import baseframework

        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.device = device
        self.model = baseframework.from_pretrained(self.checkpoint)
        if use_bf16:
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(device).eval()

        datasets_cfg = _cfg_get(self.model.config, "datasets", None)
        router_cfg = _cfg_get(datasets_cfg, "router_data", None)
        self.router_prompt = str(
            _cfg_get(router_cfg, "router_prompt", DEFAULT_ROUTER_PROMPT)
        )
        self.include_state = bool(_cfg_get(router_cfg, "include_state", False))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "starvla_qwenpi",
            "checkpoint": self.checkpoint,
            "device": self.device,
            "capabilities": ["typed_route", "sparse_body_waypoints", "base_frame_arm_targets"],
        }

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
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
        raw = self.model.predict_typed_action(
            examples=[example],
            allow_bbox=False,
        )
        return normalize_decision(raw)

    def reset(self, episode_id: str | None) -> dict[str, Any]:
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
