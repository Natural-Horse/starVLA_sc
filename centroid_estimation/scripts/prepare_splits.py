#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from centroid_estimation.common import save_json
from centroid_estimation.data import build_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare centroid train/test splits.")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--bbox_file", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--split_mode",
        choices=("intra_trajectory_half", "trajectory_half", "trajectory_percent"),
        required=True,
    )
    parser.add_argument("--train_frame_parity", choices=("even", "odd"), default="even")
    parser.add_argument(
        "--train_percent",
        type=float,
        default=0.5,
        help="For trajectory_percent, fraction of valid frames per trajectory used for training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evenly_spaced_indices(count: int, wanted: int) -> set[int]:
    if count <= 0 or wanted <= 0:
        return set()
    wanted = min(count, wanted)
    if wanted == 1:
        return {count // 2}
    return {round(i * (count - 1) / (wanted - 1)) for i in range(wanted)}


def main() -> None:
    args = parse_args()
    data_root = args.data_root or str(Path(args.metadata_csv).resolve().parent)
    records = build_records(data_root, args.metadata_csv, args.bbox_file, require_valid_bbox=True)
    if not records:
        raise RuntimeError("No valid bbox records found.")

    if args.split_mode == "intra_trajectory_half":
        train_even = args.train_frame_parity == "even"
        train = [r for r in records if (int(r["frame"]) % 2 == 0) == train_even]
        test = [r for r in records if (int(r["frame"]) % 2 == 0) != train_even]
    elif args.split_mode == "trajectory_half":
        rng = random.Random(args.seed)
        strata: dict[tuple[str, str], list[str]] = defaultdict(list)
        seen = set()
        for record in records:
            stratum = (str(record["dataset"]), str(record["object_name"]))
            traj = str(record["trajectory_id"])
            key = (stratum, traj)
            if key not in seen:
                seen.add(key)
                strata[stratum].append(traj)
        train_trajs = set()
        for stratum, trajs in strata.items():
            trajs = list(trajs)
            rng.shuffle(trajs)
            n_train = len(trajs) // 2
            train_trajs.update((stratum, traj) for traj in trajs[:n_train])
        train = [
            r
            for r in records
            if ((str(r["dataset"]), str(r["object_name"])), str(r["trajectory_id"])) in train_trajs
        ]
        test = [
            r
            for r in records
            if ((str(r["dataset"]), str(r["object_name"])), str(r["trajectory_id"])) not in train_trajs
        ]
    else:
        if not (0.0 < args.train_percent < 1.0):
            raise ValueError("--train_percent must be in (0, 1) for trajectory_percent.")
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for record in records:
            grouped[(str(record["dataset"]), str(record["trajectory_id"]))].append(record)
        train_keys = set()
        for _, group in grouped.items():
            group = sorted(group, key=lambda r: int(r["frame"]))
            n_train = max(1, int(round(len(group) * args.train_percent)))
            if len(group) > 1:
                n_train = min(n_train, len(group) - 1)
            selected_indices = evenly_spaced_indices(len(group), n_train)
            for idx in selected_indices:
                train_keys.add(group[idx]["key"])
        train = [r for r in records if r["key"] in train_keys]
        test = [r for r in records if r["key"] not in train_keys]

    payload = {
        "split_mode": args.split_mode,
        "train_frame_parity": args.train_frame_parity if args.split_mode == "intra_trajectory_half" else None,
        "train_percent": args.train_percent if args.split_mode == "trajectory_percent" else None,
        "seed": args.seed,
        "train_count": len(train),
        "test_count": len(test),
        "train": [{"key": r["key"]} for r in train],
        "test": [{"key": r["key"]} for r in test],
    }
    save_json(args.output_json, payload)
    print(f"wrote {args.output_json}: train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
