from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_mobilenetv2(num_classes: int) -> nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def configure_trainable_layers(model: nn.Module, strategy: str, partial_blocks: int = 2) -> None:
    strategy = strategy.lower()
    # Freeze everything first; then selectively unfreeze based on strategy.
    for param in model.parameters():
        param.requires_grad = False

    if strategy in {"zero_shot", "zero-shot"}:
        return

    # Always train classification head except zero-shot.
    for param in model.classifier.parameters():
        param.requires_grad = True

    if strategy == "linear_probe":
        return
    if strategy == "full_finetune":
        for param in model.features.parameters():
            param.requires_grad = True
        return
    if strategy == "partial_unfreeze":
        blocks = list(model.features.children())
        for block in blocks[-max(1, int(partial_blocks)) :]:
            for param in block.parameters():
                param.requires_grad = True
        return
    raise ValueError(f"Unknown training strategy: {strategy}")


@torch.inference_mode()
def compute_top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()
