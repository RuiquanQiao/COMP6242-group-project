from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiment_utils import make_run_config, read_summary, save_bar_chart, save_line_chart, write_csv
from transfer_learning.config import load_config
from transfer_learning.train import train_main

STRATEGIES = ["scratch", "linear_probe", "full_ft"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 4.3: transfer benefit under different data sizes.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, default="data/metadata.csv")
    parser.add_argument("--data_root", type=str, default="data/eurosat/2750")
    parser.add_argument("--output_dir", type=str, default="outputs/data_fraction")
    parser.add_argument("--fractions", type=str, default="0.1,0.3,0.6,1.0")
    parser.add_argument("--base_train_samples", type=int, default=18900)
    parser.add_argument("--val_samples", type=int, default=4050)
    parser.add_argument("--test_samples", type=int, default=4050)
    parser.add_argument("--strategies", type=str, default=",".join(STRATEGIES))
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--dummy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    rows: list[dict] = []

    for fraction in fractions:
        train_samples = max(1, int(args.base_train_samples * fraction))
        for strategy in strategies:
            cfg = make_run_config(
                base_cfg,
                output_dir=Path(args.output_dir) / f"frac_{fraction:g}" / strategy,
                dataset_name="eurosat",
                data_root=args.data_root,
                metadata_csv=args.metadata_csv,
                strategy=strategy,
                train_samples=train_samples,
                val_samples=args.val_samples,
                test_samples=args.test_samples,
                epochs=args.epochs,
            )
            artifacts = train_main(cfg, dummy=args.dummy)
            summary = read_summary(artifacts.summary_json)
            rows.append(_row(summary, fraction, train_samples))

    gains = _transfer_gains(rows)
    out_dir = Path(args.output_dir)
    write_csv(out_dir / "results.csv", rows)
    write_csv(out_dir / "transfer_gain.csv", gains)
    save_bar_chart(out_dir / "test_top1_acc.png", rows, "fraction", "strategy", "test_top1_acc")
    save_line_chart(out_dir / "transfer_gain_top1.png", gains, "fraction", "strategy", "transfer_gain_top1")
    print(f"saved: {out_dir / 'results.csv'}")
    print(f"saved: {out_dir / 'transfer_gain.csv'}")
    print(f"saved: {out_dir / 'transfer_gain_top1.png'}")


def _row(summary: dict, fraction: float, train_samples: int) -> dict:
    return {
        "fraction": f"{fraction:g}",
        "train_samples": train_samples,
        "dataset": summary["dataset"],
        "model": summary["model"],
        "strategy": summary["strategy"],
        "test_top1_acc": summary["test_top1_acc"],
        "test_macro_f1": summary["test_macro_f1"],
        "train_seconds": summary["train_seconds"],
    }


def _transfer_gains(rows: list[dict]) -> list[dict]:
    by_fraction = {}
    for row in rows:
        by_fraction.setdefault(row["fraction"], {})[row["strategy"]] = row

    gains = []
    for fraction, values in by_fraction.items():
        scratch = values.get("scratch")
        if not scratch:
            continue
        for strategy in ["linear_probe", "partial_ft", "full_ft"]:
            if strategy in values:
                gains.append(
                    {
                        "fraction": fraction,
                        "strategy": strategy,
                        "transfer_gain_top1": values[strategy]["test_top1_acc"] - scratch["test_top1_acc"],
                        "transfer_gain_macro_f1": values[strategy]["test_macro_f1"] - scratch["test_macro_f1"],
                    }
                )
    return gains


if __name__ == "__main__":
    main()

