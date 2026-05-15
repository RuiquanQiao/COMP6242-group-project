from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder


@dataclass
class DatasetConfig:
    name: str
    root: Path
    num_classes: int = 10
    image_size: int = 224
    num_workers: int = 4
    metadata_csv: Path | None = None
    download: bool = False
    val_ratio: float = 0.15
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0


class EuroSatDataset(Dataset):
    def __init__(self, rows: list[tuple[str, int]], root: Path, transform: transforms.Compose):
        self.rows = rows
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path_str, label = self.rows[index]
        image_path = Path(image_path_str)
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), int(label)


class DummyDataset(Dataset):
    def __init__(self, num_samples: int, num_classes: int, image_size: int):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.rand(3, self.image_size, self.image_size), index % self.num_classes


def dataset_cfg_from_raw(raw: dict) -> DatasetConfig:
    ds = raw["dataset"]
    return DatasetConfig(
        name=str(ds.get("name", "eurosat")),
        root=Path(ds["root"]),
        metadata_csv=Path(ds["metadata_csv"]) if ds.get("metadata_csv") else None,
        num_classes=int(ds.get("num_classes", 10)),
        image_size=int(ds.get("image_size", 224)),
        num_workers=int(ds.get("num_workers", 4)),
        download=bool(ds.get("download", False)),
        val_ratio=float(ds.get("val_ratio", 0.15)),
        train_samples=int(ds.get("train_samples", 0)),
        val_samples=int(ds.get("val_samples", 0)),
        test_samples=int(ds.get("test_samples", 0)),
    )


def build_dataloader(
    dataset_cfg: DatasetConfig,
    split: str,
    batch_size: int,
    seed: int = 42,
    dummy: bool = False,
) -> DataLoader:
    if dummy:
        size = _split_limit(dataset_cfg, split) or (256 if split == "train" else 64)
        dataset: Dataset = DummyDataset(size, dataset_cfg.num_classes, dataset_cfg.image_size)
    elif dataset_cfg.name.lower() == "eurosat":
        dataset = _build_eurosat(dataset_cfg, split, seed)
    elif dataset_cfg.name.lower() in {"cifar10", "cifar-10"}:
        dataset = _build_cifar10(dataset_cfg, split, seed)
    elif dataset_cfg.name.lower() in {"imagenet", "imagenet1k", "imagenet-1k"}:
        dataset = _build_imagenet(dataset_cfg, split, seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_cfg.name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=dataset_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _build_transforms(image_size: int, is_train: bool) -> transforms.Compose:
    steps: list = [transforms.Resize((image_size, image_size), antialias=True)]
    if is_train:
        steps += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=15)]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(steps)


def _build_eurosat(dataset_cfg: DatasetConfig, split: str, seed: int) -> Dataset:
    if dataset_cfg.metadata_csv is None:
        raise ValueError("EuroSAT requires dataset.metadata_csv.")
    rows = _read_eurosat_rows(dataset_cfg.metadata_csv, split)
    rows = _balanced_sample(rows, _split_limit(dataset_cfg, split), seed)
    return EuroSatDataset(rows, dataset_cfg.root, _build_transforms(dataset_cfg.image_size, split == "train"))


def _build_cifar10(dataset_cfg: DatasetConfig, split: str, seed: int) -> Dataset:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split for CIFAR-10: {split}")

    train = split != "test"
    dataset = CIFAR10(
        root=str(dataset_cfg.root),
        train=train,
        transform=_build_transforms(dataset_cfg.image_size, split == "train"),
        download=dataset_cfg.download,
    )
    indices = list(range(len(dataset)))
    if train:
        rng = random.Random(seed)
        rng.shuffle(indices)
        val_size = int(len(indices) * dataset_cfg.val_ratio)
        indices = indices[:val_size] if split == "val" else indices[val_size:]

    labels = [int(dataset.targets[i]) for i in indices]
    pairs = list(zip(indices, labels))
    selected = [idx for idx, _ in _balanced_sample(pairs, _split_limit(dataset_cfg, split), seed)]
    return Subset(dataset, selected)


def _build_imagenet(dataset_cfg: DatasetConfig, split: str, seed: int) -> Dataset:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split for ImageNet: {split}")

    split_dir = "val" if split == "test" else split
    root = dataset_cfg.root / split_dir
    if not root.exists():
        raise FileNotFoundError(
            f"ImageNet split directory not found: {root}. "
            "Expected an ImageFolder layout such as root/val/n01440764/*.JPEG."
        )

    dataset = ImageFolder(
        root=str(root),
        transform=_build_transforms(dataset_cfg.image_size, is_train=False),
    )
    pairs = [(idx, int(label)) for idx, (_, label) in enumerate(dataset.samples)]
    selected = [idx for idx, _ in _balanced_sample(pairs, _split_limit(dataset_cfg, split), seed)]
    return Subset(dataset, selected) if selected else dataset


def _read_eurosat_rows(metadata_csv: Path, split: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "label", "split"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("metadata.csv must include image_path,label,split columns.")
        for row in reader:
            if row["split"] == split:
                rows.append((row["image_path"], int(row["label"])))
    if not rows:
        raise ValueError(f"No rows found for split={split} in {metadata_csv}")
    return rows


def _balanced_sample(items: list[tuple], limit: int, seed: int) -> list[tuple]:
    if limit <= 0 or len(items) <= limit:
        return items
    by_label: dict[int, list[tuple]] = {}
    for item in items:
        by_label.setdefault(int(item[1]), []).append(item)

    rng = random.Random(seed)
    for values in by_label.values():
        rng.shuffle(values)

    labels = sorted(by_label)
    selected: list[tuple] = []
    while len(selected) < limit and labels:
        for label in list(labels):
            if by_label[label]:
                selected.append(by_label[label].pop())
                if len(selected) == limit:
                    break
            else:
                labels.remove(label)
    rng.shuffle(selected)
    return selected


def _split_limit(dataset_cfg: DatasetConfig, split: str) -> int:
    return {
        "train": dataset_cfg.train_samples,
        "val": dataset_cfg.val_samples,
        "test": dataset_cfg.test_samples,
    }.get(split, 0)
