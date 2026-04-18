from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import compute_top1_accuracy


@torch.inference_mode()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_steps = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in tqdm(dataloader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        acc = compute_top1_accuracy(logits, labels)
        total_loss += loss.item()
        total_acc += acc
        total_steps += 1

    if total_steps == 0:
        raise RuntimeError("Empty dataloader in evaluation.")
    return {
        "loss": total_loss / total_steps,
        "top1_acc": total_acc / total_steps,
    }
