from __future__ import annotations

import os
import sys
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model import compute_top1_accuracy


@torch.inference_mode()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_steps = 0
    criterion = nn.CrossEntropyLoss()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    force_progress = os.environ.get("EUROSAT_FORCE_PROGRESS", "0") == "1"
    show_progress = sys.stdout.isatty() or force_progress

    for images, labels in tqdm(dataloader, desc="eval", leave=False, disable=not show_progress):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        acc = compute_top1_accuracy(logits, labels)
        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_targets.append(labels.detach().cpu())
        total_loss += loss.item()
        total_acc += acc
        total_steps += 1

    if total_steps == 0:
        raise RuntimeError("Empty dataloader in evaluation.")
    macro_f1 = _macro_f1(torch.cat(all_preds), torch.cat(all_targets))
    return {
        "loss": total_loss / total_steps,
        "top1_acc": total_acc / total_steps,
        "macro_f1": macro_f1,
    }


def _macro_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    classes = torch.unique(targets)
    f1s: list[float] = []
    for cls in classes:
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fp = ((preds == cls) & (targets != cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return float(sum(f1s) / max(len(f1s), 1))
