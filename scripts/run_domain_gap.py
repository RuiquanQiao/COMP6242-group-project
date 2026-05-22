from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiment_utils import (
    ArtifactPaths,
    existing_artifacts,
    make_run_config,
    read_metrics,
    read_summary,
    save_bar_chart,
    save_training_metric_curve,
    write_csv,
)
from transfer_learning.config import load_config

STRATEGIES = ["scratch", "partial_ft", "full_ft"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 4.2: same-size cross-domain transfer study.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/domain_gap")
    parser.add_argument("--eurosat_root", type=str, default="data/eurosat/2750")
    parser.add_argument("--eurosat_metadata", type=str, default="data/metadata.csv")
    parser.add_argument("--cifar_root", type=str, default="data")
    parser.add_argument("--download_cifar", action="store_true")
    parser.add_argument("--train_samples", type=int, default=18900)
    parser.add_argument("--val_samples", type=int, default=4050)
    parser.add_argument("--test_samples", type=int, default=4050)
    parser.add_argument(
        "--strategies",
        type=str,
        default="",
        help="Comma-separated strategies. With --aggregate_only, empty means discover existing strategy folders.",
    )
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Regenerate results.csv and plots from existing run artifacts without training.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse runs that already have summary.json and metrics.json.",
    )
    parser.add_argument("--dummy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    rows: list[dict] = []
    curves: list[dict] = []
    out_dir = Path(args.output_dir)

    datasets = [
        ("eurosat", args.eurosat_root, args.eurosat_metadata, False),
        ("cifar10", args.cifar_root, "", args.download_cifar),
    ]
    for dataset_name, root, metadata_csv, download in datasets:
        strategies = _selected_strategies(args, out_dir / dataset_name)
        for strategy in strategies:
            run_dir = out_dir / dataset_name / strategy
            cfg = make_run_config(
                base_cfg,
                output_dir=run_dir,
                dataset_name=dataset_name,
                data_root=root,
                metadata_csv=metadata_csv,
                strategy=strategy,
                train_samples=args.train_samples,
                val_samples=args.val_samples,
                test_samples=args.test_samples,
                download=download,
                epochs=args.epochs,
            )

            artifacts = existing_artifacts(run_dir)
            if args.aggregate_only:
                if artifacts is None:
                    raise FileNotFoundError(
                        f"Missing existing artifacts for {dataset_name}/{strategy}: {run_dir}"
                    )
                print(f"using existing: {run_dir}", flush=True)
            elif args.skip_existing and artifacts is not None:
                print(f"skipping existing: {run_dir}", flush=True)
            else:
                from transfer_learning.train import train_main

                artifacts = train_main(cfg, dummy=args.dummy)

            _append_outputs(rows, curves, dataset_name, strategy, artifacts)

    if not rows:
        raise FileNotFoundError(f"No completed dataset/strategy folders found in {out_dir}.")
    _write_outputs(out_dir, rows, curves)
    print(f"saved: {out_dir / 'results.csv'}")

def _append_outputs(
    rows: list[dict],
    curves: list[dict],
    dataset_name: str,
    strategy: str,
    artifacts: ArtifactPaths,
) -> None:
    summary = read_summary(artifacts.summary_json)
    rows.append(_row(summary))
    curves.append(
        {
            "label": f"{dataset_name}/{strategy}",
            "dataset": dataset_name,
            "strategy": strategy,
            "metrics": read_metrics(artifacts.metrics_json),
        }
    )


def _write_outputs(out_dir: Path, rows: list[dict], curves: list[dict]) -> None:
    write_csv(out_dir / "results.csv", rows)
    save_bar_chart(out_dir / "test_top1_acc.png", rows, "dataset", "strategy", "test_top1_acc")
    save_bar_chart(out_dir / "test_macro_f1.png", rows, "dataset", "strategy", "test_macro_f1")

    save_training_metric_curve(out_dir / "train_loss_curve.png", curves, "train_loss")
    save_training_metric_curve(out_dir / "val_loss_curve.png", curves, "val_loss")
    save_training_metric_curve(out_dir / "val_top1_acc_curve.png", curves, "val_top1_acc")
    save_training_metric_curve(out_dir / "val_macro_f1_curve.png", curves, "val_macro_f1")

    for dataset in _ordered_values(curves, "dataset"):
        dataset_curves = [
            {**curve, "label": curve["strategy"]}
            for curve in curves
            if curve["dataset"] == dataset
        ]
        save_training_metric_curve(out_dir / f"{dataset}_train_loss_curve.png", dataset_curves, "train_loss")
        save_training_metric_curve(out_dir / f"{dataset}_val_loss_curve.png", dataset_curves, "val_loss")
        save_training_metric_curve(out_dir / f"{dataset}_val_top1_acc_curve.png", dataset_curves, "val_top1_acc")
        save_training_metric_curve(out_dir / f"{dataset}_val_macro_f1_curve.png", dataset_curves, "val_macro_f1")


def _ordered_values(items: list[dict], key: str) -> list[str]:
    return list(dict.fromkeys(str(item[key]) for item in items if key in item))


def _selected_strategies(args: argparse.Namespace, dataset_dir: Path) -> list[str]:
    if args.strategies:
        return [s.strip() for s in args.strategies.split(",") if s.strip()]
    if args.aggregate_only:
        if not dataset_dir.exists():
            return []
        return [
            path.name
            for path in sorted(dataset_dir.iterdir())
            if path.is_dir() and (path / "summary.json").exists() and (path / "metrics.json").exists()
        ]
    return STRATEGIES


def _row(summary: dict) -> dict:
    return {
        "dataset": summary["dataset"],
        "model": summary["model"],
        "strategy": summary["strategy"],
        "best_val_top1": summary["best_val_top1"],
        "test_top1_acc": summary["test_top1_acc"],
        "test_macro_f1": summary["test_macro_f1"],
        "train_seconds": summary["train_seconds"],
        "trainable_params": summary["trainable_params"],
        "total_params": summary["total_params"],
    }


if __name__ == "__main__":
    main()

