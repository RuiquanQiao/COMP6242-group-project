from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_baseline.config import Config, load_config
from eurosat_baseline.train import train_main

STRATEGIES = ["zero_shot", "from_scratch", "linear_probe", "partial_unfreeze", "full_finetune"]
UNSET_STRATEGY = "__CLI__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--dummy", action="store_true", help="Use dummy synthetic data.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="",
        choices=STRATEGIES,
        help="Override training.strategy from config.",
    )
    parser.add_argument("--epochs", type=int, default=0, help="Override training.epochs.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Override output_dir for this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    artifacts = train_main(cfg, dummy=args.dummy)
    print(f"best checkpoint: {artifacts.best_ckpt}")
    print(f"metrics log: {artifacts.metrics_json}")
    print(f"summary: {artifacts.summary_json}")


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    raw = deepcopy(cfg.raw)
    if args.strategy:
        raw["training"]["strategy"] = args.strategy
    current = str(raw["training"].get("strategy", UNSET_STRATEGY))
    if current == UNSET_STRATEGY:
        raise ValueError(
            "training.strategy is unset. Please pass --strategy "
            "(zero_shot|from_scratch|linear_probe|partial_unfreeze|full_finetune)."
        )
    if args.epochs and args.epochs > 0:
        raw["training"]["epochs"] = int(args.epochs)
    if args.output_dir:
        raw["output_dir"] = args.output_dir
    return Config(raw=raw)


if __name__ == "__main__":
    main()
