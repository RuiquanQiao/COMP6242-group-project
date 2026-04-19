from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from torchvision.datasets import EuroSAT

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EuroSAT metadata and split files.")
    parser.add_argument("--root", type=str, default="data", help="Data root directory.")
    parser.add_argument(
        "--images_root",
        type=str,
        default="",
        help="Optional path to EuroSAT class folders. Auto-discovered if empty.",
    )
    parser.add_argument("--download", action="store_true", help="Download EuroSAT via torchvision.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out_csv",
        type=str,
        default="data/metadata.csv",
        help="Output metadata CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.download:
        EuroSAT(root=str(root), download=True)
        print("EuroSAT downloaded.")

    if args.images_root:
        images_root = Path(args.images_root)
    else:
        images_root = auto_find_images_root(root)

    if not images_root.exists():
        raise FileNotFoundError(f"images root not found: {images_root}")

    class_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found under: {images_root}")

    rng = random.Random(args.seed)
    rows: list[dict[str, str | int]] = []
    for label, class_dir in enumerate(class_dirs):
        images = sorted(
            [
                p
                for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ]
        )
        if not images:
            continue
        rng.shuffle(images)
        n = len(images)
        n_test = int(n * args.test_ratio)
        n_val = int(n * args.val_ratio)
        n_train = n - n_val - n_test
        for idx, image_path in enumerate(images):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            rows.append(
                {
                    "image_path": str(image_path.relative_to(images_root)).replace("\\", "/"),
                    "label": label,
                    "label_name": class_dir.name,
                    "split": split,
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "label_name", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"images_root={images_root}")
    print(f"metadata={out_csv} rows={len(rows)} classes={len(class_dirs)}")


def auto_find_images_root(root: Path) -> Path:
    candidates = [
        root / "eurosat" / "2750",
        root / "EuroSAT" / "2750",
        root / "2750",
    ]
    for c in candidates:
        if c.exists():
            return c
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() == "2750":
            return path
    raise FileNotFoundError("Cannot auto-detect EuroSAT image folder. Please provide --images_root.")


if __name__ == "__main__":
    main()
