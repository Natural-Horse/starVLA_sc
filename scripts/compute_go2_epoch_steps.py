#!/usr/bin/env python3
"""Count filtered Go2 training samples and distributed optimizer steps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.dataloader.go2_waypoint_dataset import Go2WaypointRouterDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        help="LeRobot dataset root; comma-separated list to train on multiple roots.",
    )
    parser.add_argument("--routes", nargs="+", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("world_size", "batch_size", "gradient_accumulation_steps", "epochs"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.datasets.router_data, resolve=True))
    if args.dataset_root is not None:
        roots = [item.strip() for item in str(args.dataset_root).split(",") if item.strip()]
        data_cfg.root = roots if len(roots) > 1 else roots[0]
    data_cfg.include_routes = list(args.routes)

    split_cfg = cfg.datasets.get("split")
    if split_cfg and bool(split_cfg.get("enable", False)):
        if str(split_cfg.get("mode", "contiguous")) != "contiguous":
            raise ValueError("Only contiguous dataset splits are supported")
        raw_roots = data_cfg.root
        roots = raw_roots if isinstance(raw_roots, (list, tuple)) else [raw_roots]
        total_episodes = 0
        for root in roots:
            info = json.loads((Path(str(root)) / "meta" / "info.json").read_text())
            total_episodes += int(info["total_episodes"])
        train_episodes = int(math.floor(total_episodes * float(split_cfg.train_ratio)))
        train_episodes = max(1, min(train_episodes, total_episodes - 1))
        data_cfg.episode_start = 0
        data_cfg.num_episodes = train_episodes

    dataset = Go2WaypointRouterDataset(data_cfg)
    sample_count = len(dataset)
    if sample_count == 0:
        raise ValueError(f"No samples found for routes={args.routes}")

    global_batches = math.ceil(sample_count / args.batch_size)
    batches_per_process = math.ceil(global_batches / args.world_size)
    optimizer_steps = math.ceil(batches_per_process / args.gradient_accumulation_steps) * args.epochs
    print(sample_count, optimizer_steps)


if __name__ == "__main__":
    main()
