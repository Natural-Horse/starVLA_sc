#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from centroid_estimation.common import clamp_bbox_xyxy, save_json, valid_bbox_xyxy
from centroid_estimation.data import read_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate previous_data_sample50 bbox via StarVLA router server.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--server", default="ws://127.0.0.1:8000")
    parser.add_argument("--operation", default="grasp")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force_bbox_prompt", action="store_true")
    parser.add_argument("--save_every", type=int, default=50)
    return parser.parse_args()


def build_request(row: dict[str, Any], image: np.ndarray, operation: str, force_bbox_prompt: bool) -> dict[str, Any]:
    instruction = str(row.get("prompt") or "")
    target_name = str(row["object_name"])
    if force_bbox_prompt:
        prompt = (
            f"{instruction}\n\n"
            "GRASP PHASE: FIND AND GRASP THE OBJECT.\n"
            f"The current target is the object to be grasped: {target_name}.\n"
            f"{target_name}, {target_name}, {target_name} should be grasped.\n"
            "Output exactly <|pred_bbox|><point>[x1, y1, x2, y2]</point> for the object's location "
            "in the current front view. Do not output anything else."
        )
        return {
            "request_id": row["key"],
            "obs": {"image": image, "history": []},
            "prompt": prompt,
        }
    return {
        "request_id": row["key"],
        "obs": {"image": image, "history": []},
        "task": {
            "instruction": instruction,
            "target_name": target_name,
            "operation": operation,
        },
    }


def make_record(row: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    bbox_payload = response.get("bbox") or {}
    route_payload = response.get("route") or {}
    bbox = bbox_payload.get("xyxy_image")
    raw_output = bbox_payload.get("text") or route_payload.get("generated_text") or ""
    valid = False
    bbox_list = None
    if bbox is not None:
        bbox_list = clamp_bbox_xyxy(np.asarray(bbox, dtype=np.float32).tolist())
        valid = valid_bbox_xyxy(bbox_list)
    return {
        "dataset": row["dataset"],
        "trajectory_id": str(row["trajectory_id"]),
        "frame": int(row["frame"]),
        "object_name": row["object_name"],
        "rgb_path": f'{row["dataset"]}/{row["trajectory_id"]}/{row["rgb_path"]}',
        "bbox_xyxy": bbox_list,
        "bbox_valid": bool(valid),
        "raw_output": str(raw_output),
        "route": route_payload.get("route"),
        "route_confidence": route_payload.get("route_confidence"),
        "route_action_prob": route_payload.get("route_action_prob"),
        "route_bbox_prob": route_payload.get("route_bbox_prob"),
    }


async def run() -> None:
    args = parse_args()
    try:
        import msgpack
        import msgpack_numpy as msgpack_numpy
        import websockets
    except ImportError as exc:
        raise RuntimeError("annotate_bbox_with_router.py requires websockets, msgpack, and msgpack_numpy.") from exc
    msgpack_numpy.patch()

    data_root = Path(args.data_root)
    rows = read_metadata(args.metadata_csv)
    if args.limit > 0:
        rows = rows[args.start : args.start + args.limit]
    else:
        rows = rows[args.start :]

    records: list[dict[str, Any]] = []
    output_json = Path(args.output_json)
    async with websockets.connect(args.server, max_size=None, ping_interval=None) as websocket:
        metadata_raw = await websocket.recv()
        server_metadata = msgpack.unpackb(metadata_raw, raw=False)
        print(json.dumps({"server_metadata": server_metadata}, ensure_ascii=False))
        for idx, row in enumerate(rows, start=1):
            image_path = data_root / row["dataset"] / str(row["trajectory_id"]) / str(row["rgb_path"])
            image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            request = build_request(row, image, args.operation, args.force_bbox_prompt)
            await websocket.send(msgpack.packb(request, use_bin_type=True))
            response_raw = await websocket.recv()
            response = msgpack.unpackb(response_raw, raw=False)
            if response.get("error"):
                raise RuntimeError(f"Server returned error for {row['key']}: {response.get('traceback')}")
            records.append(make_record(row, response))
            if args.save_every > 0 and idx % args.save_every == 0:
                save_json(output_json, records)
                print(f"saved {len(records)} records to {output_json}")
    save_json(output_json, records)
    print(f"done: saved {len(records)} records to {output_json}")


if __name__ == "__main__":
    asyncio.run(run())

