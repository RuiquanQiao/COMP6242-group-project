from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiment_utils import ArtifactPaths, make_run_config, read_summary, save_bar_chart, write_csv
from transfer_learning.config import Config, load_config
from transfer_learning.data import build_dataloader, dataset_cfg_from_raw
from transfer_learning.device import resolve_device
from transfer_learning.evaluate import evaluate
from transfer_learning.model import build_resnet18
from transfer_learning.train import train_main

STRATEGIES = ["linear_probe", "partial_ft", "full_ft"]
DATASET_NUM_CLASSES = {"cifar10": 10, "eurosat": 10, "imagenet": 1000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure previous-task forgetting after target fine-tuning."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/forgetting")
    parser.add_argument("--source_ckpt", type=str, default="")
    parser.add_argument("--source_dataset", type=str, default="imagenet", choices=["cifar10", "eurosat", "imagenet"])
    parser.add_argument("--target_dataset", type=str, default="eurosat", choices=["cifar10", "eurosat"])
    parser.add_argument("--imagenet_root", type=str, default="data/imagenet_official")
    parser.add_argument("--eurosat_root", type=str, default="data/eurosat/2750")
    parser.add_argument("--eurosat_metadata", type=str, default="data/metadata.csv")
    parser.add_argument("--cifar_root", type=str, default="data")
    parser.add_argument("--download_cifar", action="store_true")
    parser.add_argument("--source_train_samples", type=int, default=0)
    parser.add_argument("--source_val_samples", type=int, default=0)
    parser.add_argument("--source_test_samples", type=int, default=50000)
    parser.add_argument("--target_train_samples", type=int, default=18900)
    parser.add_argument("--target_val_samples", type=int, default=4050)
    parser.add_argument("--target_test_samples", type=int, default=4050)
    parser.add_argument("--strategies", type=str, default=",".join(STRATEGIES))
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--dummy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_dataset == args.target_dataset:
        raise ValueError("source_dataset and target_dataset must be different.")

    base_cfg = load_config(args.config)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if any(strategy == "scratch" for strategy in strategies):
        raise ValueError("Forgetting starts from a trained source model; use linear_probe, partial_ft, or full_ft.")

    source_artifacts = _source_artifacts(args, base_cfg)
    source_ckpt = source_artifacts.best_ckpt if source_artifacts is not None else None
    source_before = _evaluate_source(args, base_cfg, source_ckpt)

    rows: list[dict] = []
    out_dir = Path(args.output_dir)
    for strategy in strategies:
        target_artifacts = _train_target(args, base_cfg, out_dir, strategy, source_ckpt)
        target_summary = read_summary(target_artifacts.summary_json)
        source_after = _evaluate_source(
            args,
            base_cfg,
            source_ckpt,
            backbone_ckpt=target_artifacts.best_ckpt,
        )
        rows.append(_result_row(args, strategy, source_before, source_after, target_summary))

    write_csv(out_dir / "forgetting_results.csv", rows)
    save_bar_chart(out_dir / "forgetting_top1.png", rows, "target_dataset", "strategy", "forgetting_top1")
    save_bar_chart(out_dir / "forgetting_macro_f1.png", rows, "target_dataset", "strategy", "forgetting_macro_f1")
    print(f"saved: {out_dir / 'forgetting_results.csv'}")
    print(f"saved: {out_dir / 'forgetting_top1.png'}")
    print(f"saved: {out_dir / 'forgetting_macro_f1.png'}")


def _source_artifacts(args: argparse.Namespace, base_cfg: Config) -> ArtifactPaths | None:
    if args.source_dataset == "imagenet":
        return None

    source_ckpt = Path(args.source_ckpt or f"outputs/domain_gap/{args.source_dataset}/full_ft/best.pt")
    if source_ckpt.exists():
        return ArtifactPaths(
            best_ckpt=source_ckpt,
            metrics_json=source_ckpt.parent / "metrics.json",
            summary_json=source_ckpt.parent / "summary.json",
        )
    if args.dummy:
        cfg = _dataset_run_config(
            args,
            base_cfg,
            output_dir=Path(args.output_dir) / f"source_{args.source_dataset}" / "scratch",
            dataset_name=args.source_dataset,
            strategy="scratch",
            train_samples=args.source_train_samples,
            val_samples=args.source_val_samples,
            test_samples=args.source_test_samples,
        )
        return train_main(cfg, dummy=True)
    raise FileNotFoundError(
        f"Source checkpoint not found: {source_ckpt}. "
        "Run the source-task training first or pass --source_ckpt explicitly."
    )


def _train_target(
    args: argparse.Namespace,
    base_cfg: Config,
    out_dir: Path,
    strategy: str,
    source_ckpt: Path | None,
) -> ArtifactPaths:
    cfg = _dataset_run_config(
        args,
        base_cfg,
        output_dir=out_dir / f"{args.source_dataset}_to_{args.target_dataset}" / strategy,
        dataset_name=args.target_dataset,
        strategy=strategy,
        train_samples=args.target_train_samples,
        val_samples=args.target_val_samples,
        test_samples=args.target_test_samples,
    )
    init_ckpt = None if args.source_dataset == "imagenet" else source_ckpt
    return train_main(cfg, dummy=args.dummy, init_ckpt=init_ckpt)


def _evaluate_source(
    args: argparse.Namespace,
    base_cfg: Config,
    source_ckpt: Path | None,
    backbone_ckpt: Path | None = None,
) -> dict[str, float]:
    cfg = _dataset_run_config(
        args,
        base_cfg,
        output_dir=Path(args.output_dir) / "source_eval",
        dataset_name=args.source_dataset,
        strategy="scratch",
        train_samples=args.source_train_samples,
        val_samples=args.source_val_samples,
        test_samples=args.source_test_samples,
    )
    device = resolve_device(cfg.raw)
    dataset_cfg = dataset_cfg_from_raw(cfg.raw)
    batch_size = int(cfg.raw["training"]["batch_size"])
    loader = build_dataloader(dataset_cfg, "test", batch_size, cfg.seed, args.dummy)

    model = build_resnet18(
        dataset_cfg.num_classes,
        pretrained=args.source_dataset == "imagenet",
    ).to(device)
    if source_ckpt is not None:
        source_state = torch.load(source_ckpt, map_location="cpu")["model"]
        model.load_state_dict(source_state, strict=True)

    if backbone_ckpt is not None:
        target_state = torch.load(backbone_ckpt, map_location="cpu")["model"]
        backbone_state = {
            key: value
            for key, value in target_state.items()
            if not key.startswith("fc.")
        }
        model.load_state_dict(backbone_state, strict=False)
    return evaluate(model, loader, device)


def _dataset_run_config(
    args: argparse.Namespace,
    base_cfg: Config,
    output_dir: Path,
    dataset_name: str,
    strategy: str,
    train_samples: int,
    val_samples: int,
    test_samples: int,
) -> Config:
    root, metadata_csv, download = _dataset_args(args, dataset_name)
    cfg = make_run_config(
        base_cfg,
        output_dir=output_dir,
        dataset_name=dataset_name,
        data_root=root,
        metadata_csv=metadata_csv,
        strategy=strategy,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        download=download,
        epochs=args.epochs,
    )
    cfg.raw["dataset"]["num_classes"] = DATASET_NUM_CLASSES[dataset_name]
    if dataset_name == "imagenet":
        cfg.raw["dataset"]["download"] = False
    return cfg


def _dataset_args(args: argparse.Namespace, dataset_name: str) -> tuple[str, str, bool]:
    if dataset_name == "cifar10":
        return args.cifar_root, "", args.download_cifar
    if dataset_name == "eurosat":
        return args.eurosat_root, args.eurosat_metadata, False
    if dataset_name == "imagenet":
        return args.imagenet_root, "", False
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _result_row(
    args: argparse.Namespace,
    strategy: str,
    source_before: dict[str, float],
    source_after: dict[str, float],
    target_summary: dict,
) -> dict:
    return {
        "scenario": "previous_task_forgetting",
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "strategy": strategy,
        "source_checkpoint": args.source_ckpt or ("torchvision::ResNet18_Weights.IMAGENET1K_V1" if args.source_dataset == "imagenet" else ""),
        "source_test_samples": args.source_test_samples,
        "target_train_samples": args.target_train_samples,
        "target_val_samples": args.target_val_samples,
        "target_test_samples": args.target_test_samples,
        "source_before_top1": source_before["top1_acc"],
        "source_after_top1": source_after["top1_acc"],
        "forgetting_top1": source_before["top1_acc"] - source_after["top1_acc"],
        "source_before_macro_f1": source_before["macro_f1"],
        "source_after_macro_f1": source_after["macro_f1"],
        "forgetting_macro_f1": source_before["macro_f1"] - source_after["macro_f1"],
        "target_test_top1": target_summary["test_top1_acc"],
        "target_test_macro_f1": target_summary["test_macro_f1"],
    }


if __name__ == "__main__":
    main()
