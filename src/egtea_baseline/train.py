from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from tqdm import tqdm

from .config import Config
from .data import DatasetConfig, build_dataloader
from .evaluate import evaluate
from .model import build_mobilenetv2


@dataclass
class RunArtifacts:
    best_ckpt: Path
    metrics_json: Path


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dataset_cfg_from_raw(raw: Dict) -> DatasetConfig:
    ds = raw["dataset"]
    return DatasetConfig(
        root=Path(ds["root"]),
        metadata_csv=Path(ds["metadata_csv"]),
        num_classes=int(ds["num_classes"]),
        image_size=int(ds["image_size"]),
        num_workers=int(ds["num_workers"]),
    )


def train_main(cfg: Config, dummy: bool = False) -> RunArtifacts:
    _set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cfg = _dataset_cfg_from_raw(cfg.raw)
    train_cfg = cfg.raw["training"]

    train_loader = build_dataloader(
        dataset_cfg=ds_cfg,
        split="train",
        batch_size=int(train_cfg["batch_size"]),
        dummy=dummy,
    )
    val_loader = build_dataloader(
        dataset_cfg=ds_cfg,
        split="val",
        batch_size=int(train_cfg["batch_size"]),
        dummy=dummy,
    )

    model = build_mobilenetv2(
        num_classes=ds_cfg.num_classes,
        train_backbone=bool(train_cfg["train_backbone"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_val_acc = -1.0
    best_ckpt = out_dir / "best.pt"
    history = []

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0
        for images, labels in tqdm(train_loader, desc=f"train epoch {epoch}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        train_loss = total_loss / max(total_steps, 1)
        val_metrics = evaluate(model, val_loader, device=device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_top1_acc": val_metrics["top1_acc"],
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1_acc']:.4f}"
        )

        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_ckpt)

    metrics_json = out_dir / "metrics.json"
    metrics_json.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return RunArtifacts(best_ckpt=best_ckpt, metrics_json=metrics_json)
