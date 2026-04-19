from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_baseline.config import load_config
from eurosat_baseline.train import train_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--dummy", action="store_true", help="Use dummy synthetic data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    artifacts = train_main(cfg, dummy=args.dummy)
    print(f"best checkpoint: {artifacts.best_ckpt}")
    print(f"metrics log: {artifacts.metrics_json}")
    print(f"summary: {artifacts.summary_json}")


if __name__ == "__main__":
    main()
