from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from .config import Config
from .data import build_dataloader, dataset_cfg_from_raw
from .device import device_summary, resolve_device
from .evaluate import evaluate
from .model import build_resnet18, configure_trainable_layers, should_use_pretrained


@dataclass
class RunArtifacts:
    best_ckpt: Path
    metrics_json: Path
    summary_json: Path


def train_main(cfg: Config, dummy: bool = False, init_ckpt: str | Path | None = None) -> RunArtifacts:
    _set_seed(cfg.seed)
    raw = cfg.raw
    device = resolve_device(raw)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"runtime device: {device_summary(device)}", flush=True)

    dataset_cfg = dataset_cfg_from_raw(raw)
    training = raw["training"]
    strategy = str(training["strategy"])
    batch_size = int(training["batch_size"])

    train_loader = build_dataloader(dataset_cfg, "train", batch_size, cfg.seed, dummy)
    val_loader = build_dataloader(dataset_cfg, "val", batch_size, cfg.seed, dummy)
    test_loader = build_dataloader(dataset_cfg, "test", batch_size, cfg.seed, dummy)

    model = build_resnet18(
        num_classes=dataset_cfg.num_classes,
        pretrained=should_use_pretrained(strategy),
    ).to(device)
    if init_ckpt:
        state = torch.load(init_ckpt, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
    configure_trainable_layers(model, strategy)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    best_val_acc = -1.0
    convergence_threshold = float(training.get("convergence_threshold", 0.9))
    convergence_epoch = None
    best_ckpt = out_dir / "best.pt"
    start_time = time.perf_counter()

    for epoch in range(1, int(training["epochs"]) + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, raw)
        val_metrics = evaluate(model, val_loader, device)
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
            f"val_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_ckpt)
        if convergence_epoch is None and val_metrics["top1_acc"] >= convergence_threshold:
            convergence_epoch = epoch

    elapsed = time.perf_counter() - start_time
    model.load_state_dict(torch.load(best_ckpt, map_location="cpu")["model"], strict=True)
    test_metrics = evaluate(model, test_loader, device)

    metrics_json = out_dir / "metrics.json"
    metrics_json.write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "dataset": dataset_cfg.name,
        "model": "resnet18",
        "strategy": strategy,
        "epochs": int(training["epochs"]),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_seconds": elapsed,
        "best_val_top1": best_val_acc,
        "convergence_threshold": convergence_threshold,
        "convergence_epoch": convergence_epoch,
        "test_loss": test_metrics["loss"],
        "test_top1_acc": test_metrics["top1_acc"],
        "test_macro_f1": test_metrics["macro_f1"],
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"strategy={strategy} test_top1={test_metrics['top1_acc']:.4f} "
        f"test_f1={test_metrics['macro_f1']:.4f} time={elapsed:.1f}s",
        flush=True,
    )
    return RunArtifacts(best_ckpt=best_ckpt, metrics_json=metrics_json, summary_json=summary_json)


def evaluate_checkpoint(cfg: Config, ckpt: str | Path, split: str = "test", dummy: bool = False) -> dict[str, float]:
    raw = cfg.raw
    device = resolve_device(raw)
    dataset_cfg = dataset_cfg_from_raw(raw)
    strategy = str(raw["training"]["strategy"])
    loader = build_dataloader(dataset_cfg, split, int(raw["training"]["batch_size"]), cfg.seed, dummy)
    model = build_resnet18(dataset_cfg.num_classes, should_use_pretrained(strategy)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"], strict=True)
    return evaluate(model, loader, device)


def _train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    raw: dict,
) -> float:
    model.train()
    _set_frozen_batchnorm_eval(model)
    total_loss = 0.0
    total_samples = 0
    show_progress = sys.stdout.isatty() or os.environ.get("EUROSAT_FORCE_PROGRESS", "0") == "1"
    show_progress = show_progress or bool(raw.get("runtime", {}).get("force_progress", False))

    for images, labels in tqdm(train_loader, desc=f"train epoch {epoch}", leave=False, disable=not show_progress):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    return total_loss / max(total_samples, 1)


def _set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Keep frozen BatchNorm layers from drifting during fine-tuning.

    For strategies such as linear probing, the backbone parameters are frozen,
    but calling model.train() would still update BatchNorm running statistics.
    That silently changes the pretrained representation and corrupts forgetting
    comparisons, so frozen BatchNorm modules are forced back to eval mode.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            params = list(module.parameters(recurse=False))
            if params and all(not param.requires_grad for param in params):
                module.eval()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
