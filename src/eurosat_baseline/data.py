from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class DatasetConfig:
    root: Path
    metadata_csv: Path
    num_classes: int
    image_size: int
    num_workers: int


class EuroSatDataset(Dataset):
    def __init__(self, rows: list[tuple[str, int]], root: Path, transform: transforms.Compose):
        self.rows = rows
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        x = torch.rand(3, self.image_size, self.image_size)
        y = torch.randint(0, self.num_classes, size=(1,)).item()
        return x, y


def _build_transforms(image_size: int, is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _read_split_rows(metadata_csv: Path, split: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "label", "split"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("metadata.csv must include columns: image_path,label,split")
        for row in reader:
            if row["split"] == split:
                rows.append((row["image_path"], int(row["label"])))
    if not rows:
        raise ValueError(f"No rows found for split={split} in {metadata_csv}")
    return rows


def build_dataloader(
    dataset_cfg: DatasetConfig,
    split: str,
    batch_size: int,
    dummy: bool = False,
) -> DataLoader:
    is_train = split == "train"
    if dummy:
        dataset = DummyDataset(
            num_samples=256 if is_train else 64,
            num_classes=dataset_cfg.num_classes,
            image_size=dataset_cfg.image_size,
        )
    else:
        rows = _read_split_rows(dataset_cfg.metadata_csv, split)
        dataset = EuroSatDataset(
            rows=rows,
            root=dataset_cfg.root,
            transform=_build_transforms(dataset_cfg.image_size, is_train=is_train),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=dataset_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def read_label_names(metadata_csv: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "label_name" in row and row["label_name"]:
                labels[int(row["label"])] = row["label_name"]
    return labels
