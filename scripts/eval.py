from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_baseline.config import Config, load_config
from eurosat_baseline.data import DatasetConfig, build_dataloader
from eurosat_baseline.device import device_summary, resolve_device
from eurosat_baseline.evaluate import evaluate
from eurosat_baseline.model import build_mobilenetv2, configure_trainable_layers

STRATEGIES = ["zero_shot", "from_scratch", "linear_probe", "partial_unfreeze", "full_finetune"]
UNSET_STRATEGY = "__CLI__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint. Optional.")
    parser.add_argument("--split", type=str, default="", help="Override split: val/test.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="",
        choices=STRATEGIES,
        help="Override training.strategy from config.",
    )
    parser.add_argument("--dummy", action="store_true", help="Use dummy synthetic data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    raw = cfg.raw
    ds = raw["dataset"]
    split = args.split or raw["evaluation"]["split"]
    device = resolve_device(raw)
    print(f"runtime device: {device_summary(device)}")

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
    strategy = str(raw["training"].get("strategy", "linear_probe"))
    model = build_mobilenetv2(
        num_classes=ds_cfg.num_classes,
        pretrained=(strategy != "from_scratch"),
    ).to(device)
    configure_trainable_layers(
        model=model,
        strategy=strategy,
        partial_blocks=int(raw["training"].get("partial_blocks", 2)),
    )

    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)

    metrics = evaluate(model, loader, device)
    print(
        f"split={split} loss={metrics['loss']:.4f} "
        f"top1={metrics['top1_acc']:.4f} macro_f1={metrics['macro_f1']:.4f}"
    )


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
    return Config(raw=raw)


if __name__ == "__main__":
    main()
