from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from egtea_baseline.config import load_config
from egtea_baseline.data import DatasetConfig, build_dataloader
from egtea_baseline.evaluate import evaluate
from egtea_baseline.model import build_mobilenetv2


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint. Optional.")
    parser.add_argument("--split", type=str, default="", help="Override split: val/test.")
    parser.add_argument("--dummy", action="store_true", help="Use dummy synthetic data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raw = cfg.raw
    ds = raw["dataset"]
    split = args.split or raw["evaluation"]["split"]
    device = _resolve_device(cfg.device)

    ds_cfg = DatasetConfig(
        root=Path(ds["root"]),
        metadata_csv=Path(ds["metadata_csv"]),
        num_classes=int(ds["num_classes"]),
        image_size=int(ds["image_size"]),
        num_workers=int(ds["num_workers"]),
    )
    loader = build_dataloader(
        dataset_cfg=ds_cfg,
        split=split,
        batch_size=int(raw["training"]["batch_size"]),
        dummy=args.dummy,
    )
    model = build_mobilenetv2(
        num_classes=ds_cfg.num_classes,
        train_backbone=bool(raw["training"]["train_backbone"]),
    ).to(device)

    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)

    metrics = evaluate(model, loader, device)
    print(f"split={split} loss={metrics['loss']:.4f} top1={metrics['top1_acc']:.4f}")


if __name__ == "__main__":
    main()
