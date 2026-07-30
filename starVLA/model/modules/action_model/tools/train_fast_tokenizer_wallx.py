from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def _load_universal_action_processor(source_tokenizer_dir: str):
    processor_py = Path(source_tokenizer_dir) / "processing_action_tokenizer.py"
    if not processor_py.exists():
        raise FileNotFoundError(f"Missing processing_action_tokenizer.py under `{source_tokenizer_dir}`")

    spec = importlib.util.spec_from_file_location("starvla_fast_processor_train", processor_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import `{processor_py}`")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UniversalActionProcessor, processor_py


def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    return np.remainder(angles + np.pi, 2 * np.pi) - np.pi


def _convert_action_to_ego(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    pos = action[..., :3]
    rpy = action[..., 3:6]

    base_pos = state[:3]
    base_rpy = state[3:6]
    delta_pos = pos - base_pos

    roll, pitch, yaw = base_rpy
    cx = np.cos(roll)
    sx = np.sin(roll)
    cy = np.cos(pitch)
    sy = np.sin(pitch)
    cz = np.cos(yaw)
    sz = np.sin(yaw)

    rot = np.asarray(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=action.dtype,
    )
    delta_pos_ego = delta_pos @ rot
    delta_rpy = _wrap_to_pi(rpy - base_rpy)
    return np.concatenate((delta_pos_ego, delta_rpy), axis=-1)


def _normalize_with_stats(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    delta = q99 - q01
    delta = np.where(delta == 0, 1.0, delta)
    x = (x - q01) / delta
    x = x * 2 - 1
    return np.clip(x, -1.0, 1.0)


def _load_action_stats(dataset_root: Path) -> dict[str, Any]:
    stats_path = dataset_root / "norm_stats_ego.json"
    if not stats_path.exists():
        stats_path = dataset_root / "norm_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"No norm_stats_ego.json or norm_stats.json found under `{dataset_root}`")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_json = json.load(f)
    return stats_json["norm_stats"]["action"]


def _episode_files(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))


def _episode_index_from_path(path: Path) -> int:
    stem = path.stem
    return int(stem.split("_")[-1])


def _read_meta(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return {}
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pad_or_skip_tail(chunk: np.ndarray, horizon: int, pad_tail: str) -> np.ndarray | None:
    if chunk.shape[0] >= horizon:
        return chunk[:horizon]
    if pad_tail == "skip":
        return None
    if chunk.shape[0] == 0:
        return None
    pad = np.repeat(chunk[-1:], horizon - chunk.shape[0], axis=0)
    return np.concatenate([chunk, pad], axis=0)


def _apply_action_mode(
    *,
    state: np.ndarray,
    action: np.ndarray,
    keyframe: np.ndarray | None,
    action_in_ego: bool,
    use_delta_action: bool,
    truncate_keyframe_value: int,
) -> np.ndarray:
    if action_in_ego:
        action = _convert_action_to_ego(state, action)

    if use_delta_action:
        delta_action = np.zeros_like(action)
        if action_in_ego:
            delta_action[0] = action[0]
        else:
            delta_action[0] = action[0] - state
        if action.shape[0] > 1:
            delta_action[1:] = action[1:] - action[:-1]
        action = delta_action

    if keyframe is not None and keyframe.shape[0] == action.shape[0]:
        future = keyframe.reshape(-1)[1:]
        matches = np.nonzero(future == truncate_keyframe_value)[0]
        if matches.size > 0:
            last_rel = int(matches.max()) + 1
            if use_delta_action:
                action[last_rel:] = 0
            else:
                action[last_rel:] = action[last_rel].copy()
    return action


def collect_action_chunks(args: argparse.Namespace) -> list[np.ndarray]:
    dataset_root = Path(args.dataset_root)
    info = _read_meta(dataset_root)
    total_episodes = int(info.get("total_episodes", len(_episode_files(dataset_root))))

    if args.num_episodes is not None:
        episode_start = int(args.episode_start)
        episode_end = min(total_episodes, episode_start + int(args.num_episodes))
    elif args.train_ratio is not None:
        episode_start = 0
        episode_end = max(1, min(total_episodes, int(math.floor(total_episodes * float(args.train_ratio)))))
    else:
        episode_start = int(args.episode_start)
        episode_end = total_episodes

    selected_episodes = set(range(episode_start, episode_end))
    pred_action_values = {item.strip() for item in args.pred_action_values.split(",") if item.strip()}
    action_stats = _load_action_stats(dataset_root) if args.normalize_action else None

    chunks: list[np.ndarray] = []
    for parquet_path in _episode_files(dataset_root):
        episode_index = _episode_index_from_path(parquet_path)
        if episode_index not in selected_episodes:
            continue

        table = pq.read_table(parquet_path, columns=["state", "action", "keyframe", "pred_signal"])
        data = table.to_pydict()
        states = np.asarray(data["state"], dtype=np.float32)
        actions = np.asarray(data["action"], dtype=np.float32)
        keyframes = np.asarray(data["keyframe"], dtype=np.int64)
        pred_signals = [str(x).strip() for x in data["pred_signal"]]

        for frame_idx, pred_signal in enumerate(pred_signals):
            if args.action_routes_only and pred_signal not in pred_action_values:
                continue

            action_chunk = _pad_or_skip_tail(actions[frame_idx : frame_idx + args.action_horizon], args.action_horizon, args.pad_tail)
            if action_chunk is None:
                continue
            keyframe_chunk = _pad_or_skip_tail(
                keyframes[frame_idx : frame_idx + args.action_horizon],
                args.action_horizon,
                args.pad_tail,
            )

            action_chunk = _apply_action_mode(
                state=states[frame_idx],
                action=action_chunk.astype(np.float32),
                keyframe=keyframe_chunk,
                action_in_ego=args.action_in_ego,
                use_delta_action=args.use_delta_action,
                truncate_keyframe_value=args.truncate_keyframe_value,
            )
            if action_stats is not None:
                action_chunk = _normalize_with_stats(action_chunk, action_stats)
            chunks.append(action_chunk.astype(np.float32))

            if args.max_samples is not None and len(chunks) >= int(args.max_samples):
                return chunks

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a WallX FAST action tokenizer from LeRobot parquet actions.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-tokenizer-dir", default="/diff/wallx_workspace/wall-x/fast-tokenizer")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--pad-tail", choices=["repeat", "skip"], default="repeat")
    parser.add_argument("--pred-action-values", default="<pred_action>,<|pred_action|>")
    parser.add_argument("--action-routes-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-in-ego", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-delta-action", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--normalize-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncate-keyframe-value", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    if save_dir.exists() and any(save_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"`{save_dir}` already exists and is not empty. Pass --overwrite to replace files.")
    save_dir.mkdir(parents=True, exist_ok=True)

    chunks = collect_action_chunks(args)
    if not chunks:
        raise RuntimeError("No action chunks collected. Check dataset path and pred_signal filters.")

    processor_cls, processor_py = _load_universal_action_processor(args.source_tokenizer_dir)
    processor = processor_cls.fit(
        chunks,
        scale=float(args.scale),
        vocab_size=int(args.vocab_size),
        time_horizon=int(args.action_horizon),
        action_dim=int(args.action_dim),
    )
    processor.save_pretrained(save_dir)
    shutil.copy2(processor_py, save_dir / "processing_action_tokenizer.py")
    nested_tokenizer_json = save_dir / "bpe_tokenizer" / "tokenizer.json"
    nested_tokenizer_config = save_dir / "bpe_tokenizer" / "tokenizer_config.json"
    if nested_tokenizer_json.exists():
        shutil.copy2(nested_tokenizer_json, save_dir / "tokenizer.json")
    if nested_tokenizer_config.exists():
        shutil.copy2(nested_tokenizer_config, save_dir / "tokenizer_config.json")

    metadata = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "num_action_chunks": len(chunks),
        "action_horizon": int(args.action_horizon),
        "action_dim": int(args.action_dim),
        "vocab_size": int(args.vocab_size),
        "scale": float(args.scale),
        "train_ratio": args.train_ratio,
        "episode_start": args.episode_start,
        "num_episodes": args.num_episodes,
        "pad_tail": args.pad_tail,
        "action_routes_only": bool(args.action_routes_only),
        "pred_action_values": sorted({item.strip() for item in args.pred_action_values.split(",") if item.strip()}),
        "action_in_ego": bool(args.action_in_ego),
        "use_delta_action": bool(args.use_delta_action),
        "normalize_action": bool(args.normalize_action),
        "truncate_keyframe_value": int(args.truncate_keyframe_value),
    }
    with open(save_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved FAST tokenizer to: {save_dir}")
    print(f"[OK] Action chunks: {len(chunks)}")
    print(f"[OK] Horizon/action_dim/vocab: {args.action_horizon}/{args.action_dim}/{args.vocab_size}")


if __name__ == "__main__":
    main()
