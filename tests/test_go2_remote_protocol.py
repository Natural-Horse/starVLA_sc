from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from deployment.go2_remote.backend import MockBackend, _decode_jpeg
from deployment.go2_remote.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    normalize_decision,
    validate_request,
)


def _request() -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "infer",
        "request_id": "request-1",
        "payload": {
            "instruction": "Move the cola from box one to box two.",
            "images": {"front": {"encoding": "jpeg_base64", "data": "unused"}},
        },
    }


def test_validate_inference_envelope() -> None:
    assert validate_request(_request())["request_id"] == "request-1"


def test_reject_unknown_protocol_version() -> None:
    request = _request()
    request["protocol_version"] = "old"
    with pytest.raises(ProtocolError, match="unsupported protocol_version"):
        validate_request(request)


def test_nav_decision_is_json_safe() -> None:
    decision = normalize_decision(
        {
            "route": "NAV",
            "nav_waypoints": ((0.2, 0, 0), (0.4, 0.1, 0.2)),
            "route_confidence": 0.8,
        }
    )
    assert decision["route"] == "nav"
    assert decision["nav_waypoints"] == [[0.2, 0.0, 0.0], [0.4, 0.1, 0.2]]


def test_non_nav_decision_drops_waypoints() -> None:
    decision = normalize_decision({"route": "grasp", "nav_waypoints": [[1, 2, 3]]})
    assert decision["nav_waypoints"] is None


def test_arm_decision_is_json_safe() -> None:
    decision = normalize_decision(
        {
            "route": "grasp",
            "arm_targets_base": ((0.35, 0.0, 0.2, 0.0, 0.1, 0.0, 1.0),),
        }
    )
    assert decision["arm_targets_base"] == [[0.35, 0.0, 0.2, 0.0, 0.1, 0.0, 1.0]]


def test_mock_backend_matches_typed_contract() -> None:
    result = MockBackend("nav").infer(_request()["payload"])
    assert result["route"] == "nav"
    assert len(result["nav_waypoints"]) == 2


def test_jpeg_payload_decodes_to_rgb() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(buffer, format="JPEG")
    decoded = _decode_jpeg(
        {
            "encoding": "jpeg_base64",
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
    )
    assert decoded.mode == "RGB"
    assert decoded.size == (4, 3)
