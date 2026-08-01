"""Versioned JSON contract shared by the Go2 evaluation server and client."""

from __future__ import annotations

import math
from typing import Any

PROTOCOL_VERSION = "starvla-go2-eval/v2"
ROUTES = frozenset({"nav", "grasp", "place", "done", "recover"})
REQUEST_TYPES = frozenset({"health", "infer", "reset"})


class ProtocolError(ValueError):
    """A request or model result violates the evaluation contract."""


def validate_request(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ProtocolError("request must be a JSON object")
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol_version={message.get('protocol_version')!r}"
        )
    request_type = str(message.get("type", ""))
    if request_type not in REQUEST_TYPES:
        raise ProtocolError(f"unsupported request type={request_type!r}")
    request_id = str(message.get("request_id", "")).strip()
    if not request_id:
        raise ProtocolError("request_id is required")
    if request_type == "infer":
        payload = message.get("payload")
        if not isinstance(payload, dict):
            raise ProtocolError("infer payload must be an object")
        instruction = str(payload.get("instruction", "")).strip()
        if not instruction:
            raise ProtocolError("payload.instruction is required")
        images = payload.get("images")
        if not isinstance(images, dict) or "front" not in images:
            raise ProtocolError("payload.images.front is required")
    return message


def normalize_decision(raw: Any) -> dict[str, Any]:
    """Convert one model result to finite JSON-safe typed fields."""

    if not isinstance(raw, dict):
        raise ProtocolError("model decision must be an object")
    route = str(raw.get("route", "")).strip().lower()
    if route not in ROUTES:
        raise ProtocolError(f"model returned unsupported route={route!r}")

    waypoints = raw.get("nav_waypoints")
    normalized_waypoints: list[list[float]] | None = None
    if route == "nav":
        if hasattr(waypoints, "tolist"):
            waypoints = waypoints.tolist()
        if not isinstance(waypoints, (list, tuple)) or not waypoints:
            raise ProtocolError("NAV decision requires non-empty nav_waypoints")
        if len(waypoints) > 32:
            raise ProtocolError("NAV decision exceeds 32 waypoints")
        normalized_waypoints = []
        for index, point in enumerate(waypoints):
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise ProtocolError(f"nav_waypoints[{index}] must contain [dx,dy,dyaw]")
            values = [float(value) for value in point]
            if not all(math.isfinite(value) for value in values):
                raise ProtocolError(f"nav_waypoints[{index}] contains non-finite values")
            normalized_waypoints.append(values)

    arm_targets = raw.get("arm_targets_base")
    normalized_arm_targets: list[list[float]] | None = None
    if arm_targets is not None:
        if hasattr(arm_targets, "tolist"):
            arm_targets = arm_targets.tolist()
        if not isinstance(arm_targets, (list, tuple)) or not arm_targets:
            raise ProtocolError("arm_targets_base must be a non-empty action chunk")
        if len(arm_targets) > 32:
            raise ProtocolError("arm_targets_base exceeds 32 targets")
        normalized_arm_targets = []
        for index, target in enumerate(arm_targets):
            if not isinstance(target, (list, tuple)) or len(target) != 7:
                raise ProtocolError(
                    f"arm_targets_base[{index}] must contain [x,y,z,roll,pitch,yaw,gripper]"
                )
            values = [float(value) for value in target]
            if not all(math.isfinite(value) for value in values):
                raise ProtocolError(f"arm_targets_base[{index}] contains non-finite values")
            if not 0.0 <= values[-1] <= 1.0:
                raise ProtocolError(f"arm_targets_base[{index}] gripper must be in [0,1]")
            normalized_arm_targets.append(values)

    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        result = float(value)
        if not math.isfinite(result):
            raise ProtocolError("decision contains a non-finite scalar")
        return result

    route_probs = raw.get("route_probs")
    if route_probs is not None:
        if not isinstance(route_probs, dict):
            raise ProtocolError("route_probs must be an object")
        route_probs = {
            str(key): _optional_float(value) for key, value in route_probs.items()
        }

    return {
        "route": route,
        "subtask": None if raw.get("subtask") is None else str(raw["subtask"]),
        "nav_waypoints": normalized_waypoints,
        "arm_targets_base": normalized_arm_targets,
        "stop_probability": _optional_float(raw.get("stop_probability")),
        "target_name": (
            None if raw.get("target_name") is None else str(raw["target_name"])
        ),
        "grasp_primitive": (
            None
            if raw.get("grasp_primitive") is None
            else str(raw["grasp_primitive"])
        ),
        "raw_text": str(raw.get("raw_text", "")),
        "route_confidence": _optional_float(raw.get("route_confidence")),
        "route_probs": route_probs,
    }


def response(
    *,
    request_id: str,
    response_type: str,
    ok: bool,
    data: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "type": response_type,
        "request_id": request_id,
        "ok": bool(ok),
    }
    if data is not None:
        result["data"] = data
    if error is not None:
        result["error"] = error
    return result
