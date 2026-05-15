from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiment_utils import make_run_config, read_summary, save_bar_chart, write_csv
from transfer_learning.config import load_config
from transfer_learning.train import evaluate_checkpoint, train_main

STRATEGIES = ["linear_probe", "partial_ft", "full_ft"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forgetting analysis: train EuroSAT, then train CIFAR-10.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/forgetting")
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

    for strategy in strategies:
        task_a_cfg = make_run_config(
            base_cfg,
            output_dir=Path(args.output_dir) / strategy / "task_a_eurosat",
            dataset_name="eurosat",
            data_root=args.eurosat_root,
            metadata_csv=args.eurosat_metadata,
            strategy=strategy,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            epochs=args.epochs,
        )
        task_a = train_main(task_a_cfg, dummy=args.dummy)
        a_before = evaluate_checkpoint(task_a_cfg, task_a.best_ckpt, split="test", dummy=args.dummy)

        task_b_cfg = make_run_config(
            base_cfg,
            output_dir=Path(args.output_dir) / strategy / "task_b_cifar10",
            dataset_name="cifar10",
            data_root=args.cifar_root,
            strategy=strategy,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            download=args.download_cifar,
            epochs=args.epochs,
        )
        task_b = train_main(task_b_cfg, dummy=args.dummy, init_ckpt=task_a.best_ckpt)
        b_summary = read_summary(task_b.summary_json)
        a_after = evaluate_checkpoint(task_a_cfg, task_b.best_ckpt, split="test", dummy=args.dummy)

        rows.append(
            {
                "task_order": "eurosat_to_cifar10",
                "strategy": strategy,
                "a_before_top1": a_before["top1_acc"],
                "a_after_top1": a_after["top1_acc"],
                "forgetting_top1": a_before["top1_acc"] - a_after["top1_acc"],
                "a_before_macro_f1": a_before["macro_f1"],
                "a_after_macro_f1": a_after["macro_f1"],
                "forgetting_macro_f1": a_before["macro_f1"] - a_after["macro_f1"],
                "b_test_top1": b_summary["test_top1_acc"],
                "b_test_macro_f1": b_summary["test_macro_f1"],
            }
        )

    out_dir = Path(args.output_dir)
    write_csv(out_dir / "forgetting_results.csv", rows)
    save_bar_chart(out_dir / "forgetting_top1.png", rows, "task_order", "strategy", "forgetting_top1")
    print(f"saved: {out_dir / 'forgetting_results.csv'}")


if __name__ == "__main__":
    main()

