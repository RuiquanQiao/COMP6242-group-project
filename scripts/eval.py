from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transfer_learning.config import Config, load_config
from transfer_learning.model import STRATEGIES
from transfer_learning.train import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset", type=str, choices=["eurosat", "cifar10", "imagenet"], default="")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--metadata_csv", type=str, default="")
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--strategy", type=str, choices=STRATEGIES, required=True)
    parser.add_argument("--dummy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    metrics = evaluate_checkpoint(cfg, args.ckpt, split=args.split, dummy=args.dummy)
    print(
        f"split={args.split} loss={metrics['loss']:.4f} "
        f"top1={metrics['top1_acc']:.4f} macro_f1={metrics['macro_f1']:.4f}"
    )


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    raw = deepcopy(cfg.raw)
    if args.dataset:
        raw["dataset"]["name"] = args.dataset
    if args.data_root:
        raw["dataset"]["root"] = args.data_root
    if args.metadata_csv:
        raw["dataset"]["metadata_csv"] = args.metadata_csv
    if args.num_classes > 0:
        raw["dataset"]["num_classes"] = args.num_classes
    raw["training"]["strategy"] = args.strategy
    return Config(raw=raw)


if __name__ == "__main__":
    main()

