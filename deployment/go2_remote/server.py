"""Loopback-only WebSocket service for remote Go2 simulation evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import traceback
from typing import Any

from .backend import EvaluationBackend, MockBackend, StarVLABackend
from .protocol import PROTOCOL_VERSION, ProtocolError, response, validate_request

LOGGER = logging.getLogger(__name__)


class Go2EvaluationServer:
    def __init__(
        self,
        backend: EvaluationBackend,
        *,
        host: str = "127.0.0.1",
        port: int = 10093,
        max_message_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        self.backend = backend
        self.host = host
        self.port = int(port)
        self.max_message_bytes = int(max_message_bytes)

    def serve_forever(self) -> None:
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            LOGGER.info("Go2 evaluation server stopped by user")

    async def run(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            from websockets import serve

        LOGGER.info(
            "Go2 evaluation server listening on ws://%s:%d protocol=%s",
            self.host,
            self.port,
            PROTOCOL_VERSION,
        )
        async with serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=self.max_message_bytes,
            max_queue=2,
            ping_interval=None,
        ):
            await asyncio.Future()

    async def _handler(self, websocket, *_: Any) -> None:
        LOGGER.info("evaluation client connected: %s", websocket.remote_address)
        try:
            async for raw_message in websocket:
                started = time.perf_counter()
                request_id = "unknown"
                try:
                    if not isinstance(raw_message, str):
                        raise ProtocolError("binary WebSocket frames are not supported")
                    message = validate_request(json.loads(raw_message))
                    request_id = str(message["request_id"])
                    result = self._dispatch(message)
                    result.setdefault("timing", {})["server_total_ms"] = round(
                        (time.perf_counter() - started) * 1000.0,
                        3,
                    )
                except Exception as exc:
                    LOGGER.exception("evaluation request failed request_id=%s", request_id)
                    result = response(
                        request_id=request_id,
                        response_type="error",
                        ok=False,
                        error={
                            "code": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                await websocket.send(json.dumps(result, ensure_ascii=True))
        except Exception:
            LOGGER.debug("evaluation client disconnected:\n%s", traceback.format_exc())

    def _dispatch(self, message: dict[str, Any]) -> dict[str, Any]:
        request_id = str(message["request_id"])
        request_type = str(message["type"])
        if request_type == "health":
            return response(
                request_id=request_id,
                response_type="health_result",
                ok=True,
                data={
                    "protocol_version": PROTOCOL_VERSION,
                    "backend": self.backend.metadata,
                },
            )
        if request_type == "reset":
            payload = message.get("payload") or {}
            return response(
                request_id=request_id,
                response_type="reset_result",
                ok=True,
                data=self.backend.reset(payload.get("episode_id")),
            )
        inference_started = time.perf_counter()
        decision = self.backend.infer(message["payload"])
        return response(
            request_id=request_id,
            response_type="inference_result",
            ok=True,
            data=decision,
        ) | {
            "timing": {
                "inference_ms": round(
                    (time.perf_counter() - inference_started) * 1000.0,
                    3,
                )
            }
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 Go2 StarVLA 远程评测推理服务。")
    parser.add_argument("--checkpoint", help="训练产出的完整 QwenPI checkpoint。")
    parser.add_argument(
        "--host",
        choices=("127.0.0.1", "localhost", "::1"),
        default="127.0.0.1",
        help="仅允许监听回环地址。",
    )
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument(
        "--mock-route",
        choices=("nav", "grasp", "place", "done", "recover"),
        help="不加载模型，仅用于协议和 SSH 隧道 smoke test。",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.mock_route:
        backend: EvaluationBackend = MockBackend(args.mock_route)
    else:
        if not args.checkpoint:
            raise SystemExit("真实推理必须传 --checkpoint；协议测试可使用 --mock-route。")
        backend = StarVLABackend(
            args.checkpoint,
            device=args.device,
            use_bf16=not args.no_bf16,
        )
    Go2EvaluationServer(backend, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
