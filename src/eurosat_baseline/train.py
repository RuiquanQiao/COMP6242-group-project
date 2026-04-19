from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from tqdm import tqdm

from .config import Config
from .data import DatasetConfig, build_dataloader
from .evaluate import evaluate
from .model import build_mobilenetv2, configure_trainable_layers


@dataclass
class RunArtifacts:
    best_ckpt: Path
    metrics_json: Path
    summary_json: Path


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
    strategy = str(train_cfg.get("strategy", "linear_probe"))
    partial_blocks = int(train_cfg.get("partial_blocks", 2))

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
    test_loader = build_dataloader(
        dataset_cfg=ds_cfg,
        split="test",
        batch_size=int(train_cfg["batch_size"]),
        dummy=dummy,
    )

    model = build_mobilenetv2(num_classes=ds_cfg.num_classes).to(device)
    configure_trainable_layers(model, strategy=strategy, partial_blocks=partial_blocks)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = None
    if trainable_params > 0:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )

    best_val_acc = -1.0
    best_ckpt = out_dir / "best.pt"
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        if optimizer is None:
            model.eval()
        else:
            model.train()
        total_loss = 0.0
        total_steps = 0
        for images, labels in tqdm(train_loader, desc=f"train epoch {epoch}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
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
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1_acc']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "strategy": strategy},
                best_ckpt,
            )

    metrics_json = out_dir / "metrics.json"
    metrics_json.write_text(json.dumps(history, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - start_time

    state = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    test_metrics = evaluate(model, test_loader, device=device)
    summary = {
        "strategy": strategy,
        "epochs": int(train_cfg["epochs"]),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_seconds": elapsed,
        "best_val_top1": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_top1_acc": test_metrics["top1_acc"],
        "test_macro_f1": test_metrics["macro_f1"],
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"strategy={strategy} test_top1={test_metrics['top1_acc']:.4f} "
        f"test_f1={test_metrics['macro_f1']:.4f} time={elapsed:.1f}s"
    )
    return RunArtifacts(best_ckpt=best_ckpt, metrics_json=metrics_json, summary_json=summary_json)
