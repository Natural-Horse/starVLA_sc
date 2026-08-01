#!/usr/bin/env python3
"""在 held-out Go2 episode 上生成 route/subtask 并统计严格准确率。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.go2_waypoint_dataset import Go2WaypointRouterDataset


ACTION_ROUTES = frozenset({"nav", "grasp", "place"})
SUBTASK_PATTERN = re.compile(r"<\|subtask\|>(.*?)<\|end_subtask\|>", re.DOTALL)


def _normalize_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def parse_generated_subtask(text: str) -> str | None:
    match = SUBTASK_PATTERN.search(str(text))
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _accuracy(records: list[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return float(np.mean([bool(record[key]) for record in records]))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subtask[str(record["target_subtask"])].append(record)
        by_route[str(record["target_route"])].append(record)

    action_records = [record for record in records if record["target_route"] in ACTION_ROUTES]
    per_subtask = {
        label: {
            "count": len(items),
            "route_accuracy": _accuracy(items, "route_correct"),
            "subtask_exact_accuracy": _accuracy(items, "subtask_correct"),
            "joint_accuracy": _accuracy(items, "joint_correct"),
            "top_predictions": Counter(
                str(item.get("predicted_subtask") or "<missing>") for item in items
            ).most_common(8),
        }
        for label, items in sorted(by_subtask.items())
    }
    per_route = {
        route: {
            "count": len(items),
            "route_accuracy": _accuracy(items, "route_correct"),
            "subtask_exact_accuracy": _accuracy(items, "subtask_correct"),
            "joint_accuracy": _accuracy(items, "joint_correct"),
        }
        for route, items in sorted(by_route.items())
    }
    subtask_accuracies = [
        float(values["subtask_exact_accuracy"])
        for values in per_subtask.values()
        if values["subtask_exact_accuracy"] is not None
    ]
    return {
        "samples": len(records),
        "route_accuracy": _accuracy(records, "route_correct"),
        "subtask_exact_accuracy": _accuracy(records, "subtask_correct"),
        "joint_route_subtask_accuracy": _accuracy(records, "joint_correct"),
        "macro_subtask_exact_accuracy": (
            float(np.mean(subtask_accuracies)) if subtask_accuracies else None
        ),
        "action_route_samples": len(action_records),
        "action_route_accuracy": _accuracy(action_records, "route_correct"),
        "action_subtask_exact_accuracy": _accuracy(action_records, "subtask_correct"),
        "action_joint_accuracy": _accuracy(action_records, "joint_correct"),
        "per_route": per_route,
        "per_subtask": per_subtask,
    }


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_model_for_vlm_evaluation(checkpoint: Path):
    """加载完整模型，但不要求与 VLM 评测无关的 action normalization stats。"""

    from accelerate import PartialState
    from starVLA.model.framework.__init__ import build_framework
    from starVLA.model.framework.share_tools import dict_to_namespace
    from starVLA.training.trainer_utils.trainer_tools import adapt_padded_vocab_state_dict

    PartialState()
    checkpoint = checkpoint.resolve()
    config_path = checkpoint.parents[1] / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint run 缺少 config.yaml: {config_path}")
    config = dict_to_namespace(
        OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    )
    config.trainer.pretrained_checkpoint = None
    model = build_framework(cfg=config)
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint))
    else:
        state_dict = torch.load(checkpoint, map_location="cpu", mmap=True)
    state_dict, expanded_keys = adapt_padded_vocab_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=True)
    if expanded_keys:
        print("已兼容扩展旧 checkpoint 的词表 padding 行：" + ", ".join(expanded_keys))
    return model


def _build_eval_dataset(config_path: Path, dataset_root: Path) -> tuple[Go2WaypointRouterDataset, dict[str, int]]:
    cfg = OmegaConf.load(config_path)
    data_cfg = cfg.datasets.router_data
    data_cfg.root = str(dataset_root.resolve())
    data_cfg.shuffle = False
    data_cfg.done_repeat = 1
    total_episodes = int(json.loads((dataset_root / "meta" / "info.json").read_text())["total_episodes"])
    split_cfg = cfg.datasets.split
    if bool(split_cfg.enable):
        train_episodes = int(np.floor(total_episodes * float(split_cfg.train_ratio)))
        train_episodes = max(1, min(train_episodes, total_episodes - 1))
        data_cfg.episode_start = train_episodes
        data_cfg.num_episodes = total_episodes - train_episodes
    else:
        train_episodes = 0
        data_cfg.episode_start = 0
        data_cfg.num_episodes = total_episodes
    return Go2WaypointRouterDataset(data_cfg), {
        "total_episodes": total_episodes,
        "eval_episode_start": int(data_cfg.episode_start),
        "eval_num_episodes": int(data_cfg.num_episodes),
    }


def _batches(values: list[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成式评测 Go2 VLM route/subtask。")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "starVLA/config/training/starvla_go2_qwen3vl_waypoint_router.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="完整生成 route 与局部 instruction 的 token 上限。",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--routes", nargs="+", choices=["nav", "grasp", "place", "done", "recover"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("该评测需要 CUDA GPU")

    dataset, split = _build_eval_dataset(args.config.resolve(), args.dataset_root.resolve())
    selected_indices = list(range(len(dataset)))
    if args.routes:
        selected_routes = frozenset(args.routes)
        selected_indices = [
            index
            for index, (episode_index, frame_index) in enumerate(dataset.samples)
            if dataset._route_for_frame(dataset.episodes[episode_index], frame_index)
            in selected_routes
        ]
    if args.max_samples is not None:
        selected_indices = selected_indices[: max(0, int(args.max_samples))]
    if not selected_indices:
        raise ValueError("没有选中评测样本")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = _load_model_for_vlm_evaluation(args.checkpoint)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    records: list[dict[str, Any]] = []
    progress = tqdm(total=len(selected_indices), desc="VLM route/subtask eval")
    for batch_indices in _batches(selected_indices, args.batch_size):
        samples = [dataset[index] for index in batch_indices]
        prediction = model.predict_route(
            examples=samples,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            route_mode="first_token",
            continue_action=True,
            continue_bbox=False,
            allow_bbox=False,
        )
        for sample, predicted in zip(samples, prediction["routes"]):
            target_route = str(sample["route"])
            target_subtask = str(sample["subtask_text"])
            predicted_route = str(predicted.get("route") or "unknown")
            raw_text = str(predicted.get("generated_text") or "")
            predicted_subtask = parse_generated_subtask(raw_text)
            route_correct = predicted_route == target_route
            subtask_correct = _normalize_label(predicted_subtask) == _normalize_label(target_subtask)
            records.append(
                {
                    "episode_index": int(sample["episode_index"]),
                    "frame_index": int(sample["frame_index"]),
                    "target_route": target_route,
                    "predicted_route": predicted_route,
                    "target_subtask": target_subtask,
                    "predicted_subtask": predicted_subtask,
                    "route_correct": route_correct,
                    "subtask_correct": subtask_correct,
                    "joint_correct": route_correct and subtask_correct,
                    "route_confidence": predicted.get("route_confidence"),
                    "raw_text": raw_text,
                }
            )
        progress.update(len(batch_indices))
    progress.close()

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "config": str(args.config.resolve()),
        "code_commit": _git_commit(),
        "split": split,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        **summarize_records(records),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
