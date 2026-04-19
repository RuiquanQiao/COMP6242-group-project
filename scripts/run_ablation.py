from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_baseline.config import Config, load_config
from eurosat_baseline.train import train_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transfer-learning ablation on EuroSAT.")
    parser.add_argument("--config", type=str, required=True, help="Path to base YAML config.")
    parser.add_argument("--dummy", action="store_true", help="Use dummy synthetic data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    strategies = ["zero_shot", "linear_probe", "partial_unfreeze", "full_finetune"]
    rows: list[dict[str, str | float | int]] = []
    base_out = Path(base_cfg.raw["output_dir"])

    for strategy in strategies:
        cfg = with_strategy(base_cfg, strategy=strategy, output_dir=base_out / strategy)
        print(f"\n=== Running strategy: {strategy} ===")
        artifacts = train_main(cfg, dummy=args.dummy)
        summary = json.loads(Path(artifacts.summary_json).read_text(encoding="utf-8"))
        rows.append(
            {
                "strategy": strategy,
                "best_val_top1": summary["best_val_top1"],
                "test_top1_acc": summary["test_top1_acc"],
                "test_macro_f1": summary["test_macro_f1"],
                "train_seconds": summary["train_seconds"],
                "trainable_params": summary["trainable_params"],
                "total_params": summary["total_params"],
            }
        )

    out_csv = base_out / "ablation_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "best_val_top1",
                "test_top1_acc",
                "test_macro_f1",
                "train_seconds",
                "trainable_params",
                "total_params",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAblation results saved: {out_csv}")


def with_strategy(base_cfg: Config, strategy: str, output_dir: Path) -> Config:
    raw = deepcopy(base_cfg.raw)
    raw["output_dir"] = str(output_dir).replace("\\", "/")
    raw["training"]["strategy"] = strategy
    if strategy == "zero_shot":
        raw["training"]["epochs"] = 1
    return Config(raw=raw)


if __name__ == "__main__":
    main()
