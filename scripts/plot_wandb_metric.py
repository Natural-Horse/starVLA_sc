#!/usr/bin/env python3
"""Plot one metric from a local W&B run directory."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a single metric from a local wandb run-*.wandb file.",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="W&B run directory, e.g. .../wandb/latest-run or .../wandb/run-YYYYMMDD_HHMMSS-id.",
    )
    parser.add_argument(
        "metric",
        help='Metric name to plot, e.g. "action_dit_loss" or "vlm_loss/source=dzb".',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <run_dir>/files/<metric>_curve.png.",
    )
    parser.add_argument(
        "--log-scale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Manually enable or disable log scale on the y axis. If omitted, the script chooses automatically.",
    )
    parser.add_argument(
        "--log-y",
        choices=("auto", "true", "false"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output figure DPI.",
    )
    return parser.parse_args()


def find_wandb_file(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    if run_dir.is_file() and run_dir.suffix == ".wandb":
        return run_dir

    candidates = sorted(run_dir.glob("run-*.wandb"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = "\n".join(str(path) for path in candidates)
        raise SystemExit(f"Found multiple run-*.wandb files. Please pass one directly:\n{names}")

    raise SystemExit(f"No run-*.wandb file found under: {run_dir}")


def parse_value(value_json: str) -> Any:
    try:
        return json.loads(value_json)
    except json.JSONDecodeError:
        return value_json


def item_key(item: Any) -> str:
    return ".".join(item.nested_key)


def read_metric(wandb_file: Path, metric: str) -> tuple[list[float], list[float], set[str]]:
    ds = DataStore()
    ds.open_for_scan(str(wandb_file))

    steps: list[float] = []
    values: list[float] = []
    available: set[str] = set()
    fallback_step = 0

    while True:
        data = ds.scan_data()
        if data is None:
            break

        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue

        fallback_step += 1
        row: dict[str, Any] = {}
        for item in record.history.item:
            key = item_key(item)
            available.add(key)
            row[key] = parse_value(item.value_json)

        if metric not in row:
            continue

        raw_value = row[metric]
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue

        raw_step = row.get("_step", fallback_step)
        try:
            step = float(raw_step)
        except (TypeError, ValueError):
            step = float(fallback_step)

        steps.append(step)
        values.append(value)

    return steps, values, available


def sanitize_filename(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._=-]+", "_", name).strip("_")
    return clean or "metric"


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def auto_use_log_y(values: list[float]) -> bool:
    positives = sorted(value for value in values if value > 0 and math.isfinite(value))
    if len(positives) != len(values) or len(positives) < 2:
        return False

    low = percentile(positives, 0.05)
    high = percentile(positives, 0.95)
    if low <= 0 or high <= 0:
        return False

    robust_range = high / low
    full_range = positives[-1] / positives[0]
    return robust_range >= 100.0 or full_range >= 1000.0


def resolve_log_y(values: list[float], log_scale: bool | None, log_y: str | None) -> bool:
    if log_scale is not None:
        return log_scale
    if log_y == "true":
        return True
    if log_y == "false":
        return False
    return auto_use_log_y(values)


def default_output_path(run_dir: Path, metric: str) -> Path:
    run_dir = run_dir.expanduser().resolve()
    output_dir = run_dir / "files" if (run_dir / "files").is_dir() else run_dir
    return output_dir / f"{sanitize_filename(metric)}_curve.png"


def plot_metric(
    steps: list[float],
    values: list[float],
    metric: str,
    output: Path,
    use_log_y: bool,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 36,
            "xtick.labelsize": 32,
            "ytick.labelsize": 32,
        }
    )

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    ax.plot(steps, values, linewidth=1.2, color="tab:blue")

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.02)

    x_formatter = ScalarFormatter(useMathText=True)
    x_formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    if use_log_y:
        ax.set_yscale("log")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    wandb_file = find_wandb_file(args.run_dir)
    steps, values, available = read_metric(wandb_file, args.metric)
    if not values:
        maybe = sorted(name for name in available if args.metric.lower() in name.lower() or "loss" in name.lower())
        hint = "\n".join(maybe[:80])
        raise SystemExit(
            f'Metric "{args.metric}" was not found as a numeric history metric in {wandb_file}.'
            + (f"\nAvailable related/loss metrics:\n{hint}" if hint else "")
        )

    output = args.output or default_output_path(args.run_dir, args.metric)
    use_log_y = resolve_log_y(values, args.log_scale, args.log_y)
    plot_metric(
        steps=steps,
        values=values,
        metric=args.metric,
        output=output,
        use_log_y=use_log_y,
        dpi=args.dpi,
    )

    y_scale = "log" if use_log_y else "linear"
    print(f"wandb_file: {wandb_file}")
    print(f"metric: {args.metric}")
    print(f"points: {len(values)}")
    print(f"step_range: {steps[0]:g}..{steps[-1]:g}")
    print(f"y_range: {min(values):g}..{max(values):g}")
    print(f"y_scale: {y_scale}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
