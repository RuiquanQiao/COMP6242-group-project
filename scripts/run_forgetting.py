from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiment_utils import read_summary, save_bar_chart, write_csv
from transfer_learning.config import Config, load_config
from transfer_learning.data import build_dataloader, dataset_cfg_from_raw
from transfer_learning.device import resolve_device
from transfer_learning.evaluate import evaluate
from transfer_learning.model import build_resnet18


@dataclass(frozen=True)
class DownstreamRun:
    run_dir: Path
    checkpoint: Path
    summary_json: Path
    label: str
    group: str
    dataset: str
    strategy: str
    fraction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure ImageNet previous-task forgetting for completed downstream "
            "runs. Each input run directory must contain best.pt and summary.json."
        )
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--imagenet_root", type=str, default="data/imagenet_official")
    parser.add_argument("--output_dir", type=str, default="outputs/forgetting")
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help=(
            "Completed run directories or parent directories. A completed run "
            "directory contains best.pt and summary.json; parent directories "
            "are searched recursively."
        ),
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Optional comma-separated labels matching --run_dirs.",
    )
    parser.add_argument("--source_test_samples", type=int, default=50000)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    runs = _collect_runs(args)
    if not runs:
        raise FileNotFoundError("No completed run directories were found.")

    out_dir = Path(args.output_dir)
    source_before = _evaluate_imagenet(base_cfg, args, backbone_ckpt=None)

    rows: list[dict] = []
    for run in runs:
        print(f"evaluating ImageNet forgetting: {run.label}", flush=True)
        source_after = _evaluate_imagenet(base_cfg, args, backbone_ckpt=run.checkpoint)
        target_summary = read_summary(run.summary_json)
        rows.append(_result_row(run, args, source_before, source_after, target_summary))

    write_csv(out_dir / "forgetting_results.csv", rows)
    save_bar_chart(out_dir / "forgetting_top1.png", rows, "group", "strategy", "forgetting_top1")
    save_bar_chart(out_dir / "forgetting_macro_f1.png", rows, "group", "strategy", "forgetting_macro_f1")
    print(f"saved: {out_dir / 'forgetting_results.csv'}")
    print(f"saved: {out_dir / 'forgetting_top1.png'}")
    print(f"saved: {out_dir / 'forgetting_macro_f1.png'}")


def _collect_runs(args: argparse.Namespace) -> list[DownstreamRun]:
    run_dirs = _expand_run_dirs(_parse_run_dirs(args.run_dirs), args.skip_missing)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if labels and len(labels) != len(run_dirs):
        raise ValueError("--labels must have the same number of entries as --run_dirs.")

    runs: list[DownstreamRun] = []
    for index, run_dir in enumerate(run_dirs):
        checkpoint = run_dir / "best.pt"
        summary_json = run_dir / "summary.json"
        if not checkpoint.exists() or not summary_json.exists():
            if args.skip_missing:
                print(f"skipping missing run directory: {run_dir}", flush=True)
                continue
            raise FileNotFoundError(f"Expected best.pt and summary.json in: {run_dir}")
        summary = read_summary(summary_json)
        dataset = str(summary.get("dataset", _infer_dataset(run_dir)))
        strategy = str(summary.get("strategy", run_dir.name))
        if strategy == "scratch":
            print(
                f"skipping scratch run because it was not fine-tuned from ImageNet pretrained weights: {run_dir}",
                flush=True,
            )
            continue
        fraction = _infer_fraction(run_dir)
        group = _infer_group(run_dir, dataset, fraction)
        label = labels[index] if labels else _default_label(group, strategy)
        runs.append(
            DownstreamRun(
                run_dir=run_dir,
                checkpoint=checkpoint,
                summary_json=summary_json,
                label=label,
                group=group,
                dataset=dataset,
                strategy=strategy,
                fraction=fraction,
            )
        )
    return runs


def _evaluate_imagenet(
    base_cfg: Config,
    args: argparse.Namespace,
    backbone_ckpt: Path | None,
) -> dict[str, float]:
    cfg = _imagenet_config(base_cfg, args)
    device = resolve_device(cfg.raw)
    dataset_cfg = dataset_cfg_from_raw(cfg.raw)
    batch_size = int(cfg.raw["training"]["batch_size"])
    loader = build_dataloader(dataset_cfg, "test", batch_size, cfg.seed, args.dummy)

    model = build_resnet18(dataset_cfg.num_classes, pretrained=True).to(device)
    if backbone_ckpt is not None:
        target_state = torch.load(backbone_ckpt, map_location="cpu")["model"]
        backbone_state = {
            key: value
            for key, value in target_state.items()
            if not key.startswith("fc.")
        }
        model.load_state_dict(backbone_state, strict=False)
    return evaluate(model, loader, device)


def _imagenet_config(base_cfg: Config, args: argparse.Namespace) -> Config:
    raw = deepcopy(base_cfg.raw)
    raw["dataset"]["name"] = "imagenet"
    raw["dataset"]["root"] = args.imagenet_root
    raw["dataset"]["metadata_csv"] = ""
    raw["dataset"]["num_classes"] = 1000
    raw["dataset"]["download"] = False
    raw["dataset"]["train_samples"] = 0
    raw["dataset"]["val_samples"] = 0
    raw["dataset"]["test_samples"] = int(args.source_test_samples)
    return Config(raw=raw)


def _result_row(
    run: DownstreamRun,
    args: argparse.Namespace,
    source_before: dict[str, float],
    source_after: dict[str, float],
    target_summary: dict,
) -> dict:
    return {
        "label": run.label,
        "group": run.group,
        "run_dir": str(run.run_dir).replace("\\", "/"),
        "source_dataset": "imagenet",
        "downstream_dataset": run.dataset,
        "fraction": run.fraction,
        "strategy": run.strategy,
        "checkpoint": str(run.checkpoint).replace("\\", "/"),
        "source_checkpoint": "torchvision::ResNet18_Weights.IMAGENET1K_V1",
        "source_test_samples": args.source_test_samples,
        "source_before_top1": source_before["top1_acc"],
        "source_after_top1": source_after["top1_acc"],
        "forgetting_top1": source_before["top1_acc"] - source_after["top1_acc"],
        "source_before_macro_f1": source_before["macro_f1"],
        "source_after_macro_f1": source_after["macro_f1"],
        "forgetting_macro_f1": source_before["macro_f1"] - source_after["macro_f1"],
        "downstream_test_top1": target_summary["test_top1_acc"],
        "downstream_test_macro_f1": target_summary["test_macro_f1"],
    }


def _parse_run_dirs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                paths.append(Path(item))
    return paths


def _expand_run_dirs(paths: list[Path], skip_missing: bool) -> list[Path]:
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        matches = _find_completed_runs(path)
        if not matches:
            if skip_missing:
                print(f"skipping directory with no completed runs: {path}", flush=True)
                continue
            raise FileNotFoundError(
                f"No completed runs found under {path}. "
                "Expected a directory containing best.pt and summary.json, "
                "or a parent directory containing such run folders."
            )
        for match in matches:
            key = match.resolve()
            if key not in seen:
                seen.add(key)
                run_dirs.append(match)
    return run_dirs


def _find_completed_runs(path: Path) -> list[Path]:
    if (path / "best.pt").exists() and (path / "summary.json").exists():
        return [path]
    if not path.exists() or not path.is_dir():
        return []
    return sorted(
        summary.parent
        for summary in path.rglob("summary.json")
        if (summary.parent / "best.pt").exists()
    )


def _infer_dataset(run_dir: Path) -> str:
    for part in reversed(run_dir.parts):
        lowered = part.lower()
        if lowered in {"eurosat", "cifar10", "cifar-10"}:
            return "cifar10" if lowered == "cifar-10" else lowered
    return ""


def _infer_fraction(run_dir: Path) -> str:
    for part in run_dir.parts:
        if part.startswith("frac_"):
            return part.removeprefix("frac_")
    return ""


def _infer_group(run_dir: Path, dataset: str, fraction: str) -> str:
    if fraction:
        return f"frac_{fraction}"
    if dataset:
        return dataset
    return run_dir.parent.name


def _default_label(group: str, strategy: str) -> str:
    return f"{group}/{strategy}" if group else strategy


if __name__ == "__main__":
    main()
