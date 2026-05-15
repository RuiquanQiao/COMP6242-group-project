from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


STRATEGIES = ["scratch", "linear_probe", "partial_ft", "full_ft"]


def build_resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    if model.fc.out_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def configure_trainable_layers(model: nn.Module, strategy: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if strategy == "scratch":
        for param in model.parameters():
            param.requires_grad = True
        return

    for param in model.fc.parameters():
        param.requires_grad = True

    if strategy == "linear_probe":
        return
    if strategy == "partial_ft":
        for param in model.layer4.parameters():
            param.requires_grad = True
        return
    if strategy == "full_ft":
        for param in model.parameters():
            param.requires_grad = True
        return
    raise ValueError(f"Unknown training strategy: {strategy}")


def should_use_pretrained(strategy: str) -> bool:
    return strategy != "scratch"
