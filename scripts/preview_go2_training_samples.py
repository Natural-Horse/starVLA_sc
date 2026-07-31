#!/usr/bin/env python3
"""Export representative Go2 router samples as a dual-camera contact sheet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.go2_waypoint_dataset import Go2WaypointRouterDataset


DEFAULT_STAGES = ("nav_to_pick", "pick", "nav_to_place", "place", "done")


def _pick_sample(dataset, wanted_stage: str, preferred_episode: int):
    by_episode: dict[int, list[int]] = {}
    for index, (episode_index, frame_index) in enumerate(dataset.samples):
        episode = dataset.episodes[episode_index]
        route = dataset._route_for_frame(episode, frame_index)
        stage = str(episode.stages[frame_index])
        matches = route == "done" if wanted_stage == "done" else stage == wanted_stage
        if matches:
            by_episode.setdefault(episode_index, []).append(index)
    if not by_episode:
        raise ValueError(f"No sample found for stage {wanted_stage!r}")
    episode_index = min(by_episode, key=lambda value: abs(value - preferred_episode))
    candidates = by_episode[episode_index]
    return dataset[candidates[len(candidates) // 2]]


def _sample_record(dataset, sample: dict) -> dict:
    episode_index = int(sample["episode_index"])
    frame_index = int(sample["frame_index"])
    return {
        "episode_index": episode_index,
        "frame_index": frame_index,
        "task_stage": str(dataset.episodes[episode_index].stages[frame_index]),
        "route": sample["route"],
        "prompt": sample["lang"],
        "solution": sample["solution"],
        "state": np.asarray(sample.get("state", [])).astype(float).tolist(),
        "action": np.asarray(sample.get("action", [])).astype(float).tolist(),
        "action_mask": np.asarray(sample.get("action_mask", [])).astype(float).tolist(),
    }


def export_contact_sheet(dataset, samples: list[dict], output_image: Path) -> None:
    cell_width, cell_height = samples[0]["image"][0].size
    label_height = 66
    canvas = Image.new("RGB", (cell_width * 2, (cell_height + label_height) * len(samples)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, sample in enumerate(samples):
        y = row * (cell_height + label_height)
        for column, image in enumerate(sample["image"]):
            canvas.paste(image.convert("RGB"), (column * cell_width, y))
        record = _sample_record(dataset, sample)
        draw.rectangle((0, y + cell_height, cell_width * 2, y + cell_height + label_height), fill=(245, 245, 245))
        draw.text(
            (8, y + cell_height + 6),
            f"ep={record['episode_index']:03d} frame={record['frame_index']:04d} "
            f"stage={record['task_stage']} route={record['route']} | front / wrist",
            fill="black",
            font=font,
        )
        draw.text((8, y + cell_height + 27), f"target: {record['solution']}", fill="black", font=font)
        if record["action"]:
            action = np.asarray(record["action"])
            state = np.asarray(record["state"])
            draw.text(
                (8, y + cell_height + 48),
                f"state={np.round(state, 3).tolist()} wp0={np.round(action[0], 3).tolist()} "
                f"mask={[int(value) for value in record['action_mask']]}",
                fill="black",
                font=font,
            )
    output_image.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    parser.add_argument("--preferred-episodes", nargs="+", type=int, default=[0, 25, 50, 75, 99])
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.stages) != len(args.preferred_episodes):
        raise ValueError("--stages and --preferred-episodes must have the same length")
    cfg = OmegaConf.load(args.config)
    data_cfg = cfg.datasets.router_data
    data_cfg.root = str(args.dataset_root.resolve())
    data_cfg.image_size = [args.image_width, args.image_height]
    data_cfg.episode_start = 0
    data_cfg.num_episodes = None
    dataset = Go2WaypointRouterDataset(data_cfg)
    samples = [
        _pick_sample(dataset, stage, episode)
        for stage, episode in zip(args.stages, args.preferred_episodes)
    ]
    export_contact_sheet(dataset, samples, args.output_image)
    records = [_sample_record(dataset, sample) for sample in samples]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"dataset_samples": len(dataset), "exported_samples": len(records)}))


if __name__ == "__main__":
    main()
