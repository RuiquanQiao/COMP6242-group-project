from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transfer_learning.config import Config, load_config
from transfer_learning.model import STRATEGIES
from transfer_learning.train import train_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one transfer-learning run.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, choices=["eurosat", "cifar10"], default="")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--metadata_csv", type=str, default="")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--strategy", type=str, choices=STRATEGIES, required=True)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--train_samples", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=0)
    parser.add_argument("--test_samples", type=int, default=0)
    parser.add_argument("--init_ckpt", type=str, default="")
    parser.add_argument("--dummy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    artifacts = train_main(cfg, dummy=args.dummy, init_ckpt=args.init_ckpt or None)
    print(f"best checkpoint: {artifacts.best_ckpt}")
    print(f"metrics log: {artifacts.metrics_json}")
    print(f"summary: {artifacts.summary_json}")


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    raw = deepcopy(cfg.raw)
    raw.setdefault("model", {})
    raw.setdefault("dataset", {})
    raw.setdefault("training", {})

    if args.output_dir:
        raw["output_dir"] = args.output_dir
    if args.dataset:
        raw["dataset"]["name"] = args.dataset
    if args.data_root:
        raw["dataset"]["root"] = args.data_root
    if args.metadata_csv:
        raw["dataset"]["metadata_csv"] = args.metadata_csv
    if args.download:
        raw["dataset"]["download"] = True
    raw["training"]["strategy"] = args.strategy
    if args.epochs > 0:
        raw["training"]["epochs"] = args.epochs
    if args.batch_size > 0:
        raw["training"]["batch_size"] = args.batch_size
    for key in ["train_samples", "val_samples", "test_samples"]:
        value = getattr(args, key)
        if value > 0:
            raw["dataset"][key] = value
    return Config(raw=raw)


if __name__ == "__main__":
    main()

