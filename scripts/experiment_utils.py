from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

from transfer_learning.config import Config


def make_run_config(
    base_cfg: Config,
    output_dir: Path,
    dataset_name: str,
    data_root: str,
    strategy: str,
    metadata_csv: str = "",
    train_samples: int = 0,
    val_samples: int = 0,
    test_samples: int = 0,
    download: bool = False,
    epochs: int = 0,
) -> Config:
    raw = deepcopy(base_cfg.raw)
    raw["output_dir"] = str(output_dir).replace("\\", "/")
    raw["training"]["strategy"] = strategy
    if epochs > 0:
        raw["training"]["epochs"] = epochs
    raw["dataset"]["name"] = dataset_name
    raw["dataset"]["root"] = data_root
    raw["dataset"]["download"] = download
    if metadata_csv:
        raw["dataset"]["metadata_csv"] = metadata_csv
    for key, value in {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
    }.items():
        raw["dataset"][key] = int(value)
    return Config(raw=raw)


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_bar_chart(path: Path, rows: list[dict], group_key: str, label_key: str, value_key: str) -> None:
    import matplotlib.pyplot as plt

    groups = list(dict.fromkeys(str(row[group_key]) for row in rows))
    labels = list(dict.fromkeys(str(row[label_key]) for row in rows))
    values = {
        (str(row[group_key]), str(row[label_key])): float(row[value_key])
        for row in rows
    }

    width = 0.8 / max(len(labels), 1)
    x_positions = list(range(len(groups)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, label in enumerate(labels):
        xs = [x + idx * width for x in x_positions]
        ys = [values.get((group, label), 0.0) for group in groups]
        ax.bar(xs, ys, width=width, label=label)

    offset = width * (len(labels) - 1) / 2
    ax.set_xticks([x + offset for x in x_positions])
    ax.set_xticklabels(groups)
    ax.set_ylabel(value_key)
    all_values = [float(row[value_key]) for row in rows]
    low = min(0.0, min(all_values))
    high = max(1.0 if "acc" in value_key or "f1" in value_key else 0.0, max(all_values))
    ax.set_ylim(low, high * 1.05 if high > 0 else high)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_line_chart(path: Path, rows: list[dict], x_key: str, label_key: str, value_key: str) -> None:
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(str(row[label_key]) for row in rows))
    fig, ax = plt.subplots(figsize=(9, 5))
    for label in labels:
        points = [
            (float(row[x_key]), float(row[value_key]))
            for row in rows
            if str(row[label_key]) == label
        ]
        points.sort(key=lambda item: item[0])
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", label=label)

    ax.set_xlabel(x_key)
    ax.set_ylabel(value_key)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_training_metric_curve(path: Path, curves: list[dict], metric_key: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for curve in curves:
        metrics = curve["metrics"]
        points = [
            (int(row["epoch"]), float(row[metric_key]))
            for row in metrics
            if metric_key in row
        ]
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", label=str(curve["label"]))

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_key)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)

