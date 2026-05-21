from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps
from scipy.io import loadmat

VAL_ARCHIVE_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
DEVKIT_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
OFFICIAL_HOST = "image-net.org"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the official ImageNet-1K validation set and optionally create a compact derivative."
    )
    parser.add_argument("--root", type=str, default="data/imagenet_official")
    parser.add_argument("--compact_root", type=str, default="data/imagenet_official_resized")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--make_compact", action="store_true")
    parser.add_argument("--compact_size", type=int, default=256)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--keep_archives", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    archives_dir = root / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    val_archive = archives_dir / Path(VAL_ARCHIVE_URL).name
    devkit_archive = archives_dir / Path(DEVKIT_URL).name

    if not any([args.download, args.extract, args.make_compact]):
        args.download = True
        args.extract = True
        args.make_compact = True

    if args.download:
        download_file(VAL_ARCHIVE_URL, val_archive)
        download_file(DEVKIT_URL, devkit_archive)

    val_dir = root / "val"
    if args.extract:
        prepare_official_val_layout(val_archive, devkit_archive, val_dir)
        write_manifest(
            root / "official_manifest.json",
            {
                "source": "official_imagenet",
                "variant": "ILSVRC2012 validation set",
                "source_urls": [VAL_ARCHIVE_URL, DEVKIT_URL],
                "prepared_at_unix": int(time.time()),
                "val_dir": str(val_dir).replace("\\", "/"),
                "num_classes": count_class_dirs(val_dir),
                "num_images": count_images(val_dir),
            },
        )

    if args.make_compact:
        if not val_dir.exists():
            raise FileNotFoundError(f"Official validation directory not found: {val_dir}")
        compact_root = Path(args.compact_root)
        create_compact_copy(
            src_root=val_dir,
            dst_root=compact_root / "val",
            shorter_side=args.compact_size,
            jpeg_quality=args.jpeg_quality,
        )
        write_manifest(
            compact_root / "compact_manifest.json",
            {
                "source": "official_imagenet_derivative",
                "derived_from": str(val_dir).replace("\\", "/"),
                "prepared_at_unix": int(time.time()),
                "shorter_side": args.compact_size,
                "jpeg_quality": args.jpeg_quality,
                "num_classes": count_class_dirs(compact_root / "val"),
                "num_images": count_images(compact_root / "val"),
            },
        )

    if not args.keep_archives and args.download:
        for archive_path in [val_archive, devkit_archive]:
            if archive_path.exists():
                archive_path.unlink()


def download_file(url: str, destination: Path) -> None:
    if OFFICIAL_HOST not in url:
        raise ValueError(f"Refusing non-official URL: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", "0"))
        if destination.exists() and total > 0 and destination.stat().st_size == total:
            print(f"skip existing download: {destination}")
            return
        print(f"downloading: {url}")
        downloaded = 0
        chunk_size = 8 * 1024 * 1024
        with destination.open("wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(
                        f"\r{destination.name}: {downloaded / (1024 ** 3):.2f} / {total / (1024 ** 3):.2f} GB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
        if total > 0:
            print()


def prepare_official_val_layout(val_archive: Path, devkit_archive: Path, val_dir: Path) -> None:
    if val_dir.exists() and count_images(val_dir) == 50000 and count_class_dirs(val_dir) == 1000:
        print(f"skip existing prepared validation directory: {val_dir}")
        return
    if not val_archive.exists():
        raise FileNotFoundError(f"Missing official validation archive: {val_archive}")
    if not devkit_archive.exists():
        raise FileNotFoundError(f"Missing official devkit archive: {devkit_archive}")

    with tempfile.TemporaryDirectory(prefix="imagenet_val_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        flat_dir = tmp_dir / "flat_val"
        flat_dir.mkdir(parents=True, exist_ok=True)
        devkit_dir = tmp_dir / "devkit"
        devkit_dir.mkdir(parents=True, exist_ok=True)

        print(f"extracting validation archive to temporary directory: {flat_dir}")
        with tarfile.open(val_archive, "r") as tar:
            tar.extractall(flat_dir)

        print(f"extracting devkit archive to temporary directory: {devkit_dir}")
        with tarfile.open(devkit_archive, "r:gz") as tar:
            tar.extractall(devkit_dir)

        labels = read_val_wnids(devkit_dir)
        image_paths = sorted(flat_dir.glob("*.JPEG"))
        if len(image_paths) != len(labels):
            raise RuntimeError(
                f"Validation archive/devkit mismatch: {len(image_paths)} images vs {len(labels)} labels."
            )

        if val_dir.exists():
            shutil.rmtree(val_dir)
        val_dir.mkdir(parents=True, exist_ok=True)
        for image_path, wnid in zip(image_paths, labels):
            class_dir = val_dir / wnid
            class_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(class_dir / image_path.name))
        print(f"prepared official ImageFolder layout: {val_dir}")


def read_val_wnids(devkit_root: Path) -> list[str]:
    meta_path = next(devkit_root.glob("**/data/meta.mat"))
    gt_path = next(devkit_root.glob("**/data/ILSVRC2012_validation_ground_truth.txt"))
    synsets = loadmat(meta_path, squeeze_me=True)["synsets"]
    idx_to_wnid: dict[int, str] = {}
    for entry in synsets:
        ilsvrc_id = int(entry[0])
        wnid = str(entry[1])
        num_children = int(entry[4])
        if num_children == 0:
            idx_to_wnid[ilsvrc_id] = wnid
    with gt_path.open("r", encoding="utf-8") as f:
        return [idx_to_wnid[int(line.strip())] for line in f if line.strip()]


def create_compact_copy(src_root: Path, dst_root: Path, shorter_side: int, jpeg_quality: int) -> None:
    if dst_root.exists() and count_images(dst_root) == count_images(src_root) and count_class_dirs(dst_root) == count_class_dirs(src_root):
        print(f"skip existing compact validation directory: {dst_root}")
        return

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(src_root.glob("*/*.JPEG"))
    total = len(image_paths)
    for idx, src_path in enumerate(image_paths, start=1):
        rel_path = src_path.relative_to(src_root)
        dst_path = dst_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = resize_shorter_side(image, shorter_side)
            image.save(dst_path, format="JPEG", quality=jpeg_quality, optimize=True)
        if idx % 500 == 0 or idx == total:
            print(f"compact copy: {idx}/{total}")


def resize_shorter_side(image: Image.Image, shorter_side: int) -> Image.Image:
    width, height = image.size
    if min(width, height) == shorter_side:
        return image
    scale = shorter_side / min(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.glob("*/*.JPEG"))


def count_class_dirs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.iterdir() if path.is_dir())


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote manifest: {path}")


if __name__ == "__main__":
    main()
