from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiment_utils import make_run_config, read_summary, save_bar_chart, write_csv
from transfer_learning.config import load_config
from transfer_learning.train import train_main

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
    parser.add_argument("--strategies", type=str, default=",".join(STRATEGIES))
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--dummy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    rows: list[dict] = []

    datasets = [
        ("eurosat", args.eurosat_root, args.eurosat_metadata, False),
        ("cifar10", args.cifar_root, "", args.download_cifar),
    ]
    for dataset_name, root, metadata_csv, download in datasets:
        for strategy in strategies:
            cfg = make_run_config(
                base_cfg,
                output_dir=Path(args.output_dir) / dataset_name / strategy,
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
            artifacts = train_main(cfg, dummy=args.dummy)
            summary = read_summary(artifacts.summary_json)
            rows.append(_row(summary))

    out_dir = Path(args.output_dir)
    write_csv(out_dir / "results.csv", rows)
    save_bar_chart(out_dir / "test_top1_acc.png", rows, "dataset", "strategy", "test_top1_acc")
    save_bar_chart(out_dir / "test_macro_f1.png", rows, "dataset", "strategy", "test_macro_f1")
    print(f"saved: {out_dir / 'results.csv'}")


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

