from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiment_utils import make_run_config, read_summary, save_bar_chart, save_line_chart, write_csv
from transfer_learning.config import Config, load_config
from transfer_learning.data import build_dataloader, dataset_cfg_from_raw
from transfer_learning.device import resolve_device
from transfer_learning.evaluate import evaluate
from transfer_learning.model import build_resnet18
from transfer_learning.train import train_main

STRATEGIES = ["linear_probe", "partial_ft", "full_ft"]
SCENARIOS = ["eurosat_ablation", "domain_gap", "data_fraction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure ImageNet catastrophic forgetting after downstream fine-tuning."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/forgetting")
    parser.add_argument("--eurosat_root", type=str, default="data/eurosat/2750")
    parser.add_argument("--eurosat_metadata", type=str, default="data/metadata.csv")
    parser.add_argument("--cifar_root", type=str, default="data")
    parser.add_argument("--download_cifar", action="store_true")
    parser.add_argument("--imagenet_root", type=str, default="data/imagenet")
    parser.add_argument("--imagenet_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--imagenet_samples", type=int, default=0)
    parser.add_argument("--train_samples", type=int, default=18900)
    parser.add_argument("--val_samples", type=int, default=4050)
    parser.add_argument("--test_samples", type=int, default=4050)
    parser.add_argument("--base_train_samples", type=int, default=18900)
    parser.add_argument("--fractions", type=str, default="0.1,0.3,0.6,1.0")
    parser.add_argument("--strategies", type=str, default=",".join(STRATEGIES))
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIOS))
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--dummy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    skipped = [s for s in strategies if s == "scratch"]
    strategies = [s for s in strategies if s != "scratch"]
    for strategy in skipped:
        print(f"skipping strategy={strategy}: ImageNet forgetting requires ImageNet pretraining")
    if not strategies:
        raise ValueError("No ImageNet-pretrained strategies selected for forgetting analysis.")
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]

    rows: list[dict] = []
    imagenet_before = _evaluate_imagenet(args, base_cfg)

    for scenario in scenarios:
        if scenario == "eurosat_ablation":
            rows.extend(_run_eurosat_ablation(args, base_cfg, strategies, imagenet_before))
        elif scenario == "domain_gap":
            rows.extend(_run_domain_gap(args, base_cfg, strategies, imagenet_before))
        elif scenario == "data_fraction":
            rows.extend(_run_data_fraction(args, base_cfg, strategies, fractions, imagenet_before))
        else:
            raise ValueError(f"Unknown forgetting scenario: {scenario}")

    out_dir = Path(args.output_dir)
    write_csv(out_dir / "forgetting_results.csv", rows)
    save_bar_chart(out_dir / "forgetting_top1.png", rows, "scenario_group", "strategy", "forgetting_top1")
    save_bar_chart(out_dir / "forgetting_macro_f1.png", rows, "scenario_group", "strategy", "forgetting_macro_f1")

    fraction_rows = [row for row in rows if row["scenario"] == "data_fraction"]
    if fraction_rows:
        save_line_chart(
            out_dir / "forgetting_by_fraction_top1.png",
            fraction_rows,
            "finetune_fraction",
            "strategy",
            "forgetting_top1",
        )
        save_line_chart(
            out_dir / "forgetting_by_fraction_macro_f1.png",
            fraction_rows,
            "finetune_fraction",
            "strategy",
            "forgetting_macro_f1",
        )

    print(f"saved: {out_dir / 'forgetting_results.csv'}")
    print(f"saved: {out_dir / 'forgetting_top1.png'}")
    print(f"saved: {out_dir / 'forgetting_macro_f1.png'}")


def _run_eurosat_ablation(
    args: argparse.Namespace,
    base_cfg: Config,
    strategies: list[str],
    imagenet_before: dict[str, float],
) -> list[dict]:
    rows = []
    for strategy in strategies:
        rows.append(
            _run_finetune_then_imagenet_eval(
                args,
                base_cfg,
                scenario="eurosat_ablation",
                scenario_group="4.1_eurosat",
                dataset_name="eurosat",
                data_root=args.eurosat_root,
                metadata_csv=args.eurosat_metadata,
                download=False,
                strategy=strategy,
                train_samples=0,
                val_samples=0,
                test_samples=0,
                fraction="1",
                imagenet_before=imagenet_before,
            )
        )
    return rows


def _run_domain_gap(
    args: argparse.Namespace,
    base_cfg: Config,
    strategies: list[str],
    imagenet_before: dict[str, float],
) -> list[dict]:
    rows = []
    datasets = [
        ("eurosat", args.eurosat_root, args.eurosat_metadata, False),
        ("cifar10", args.cifar_root, "", args.download_cifar),
    ]
    for dataset_name, root, metadata_csv, download in datasets:
        for strategy in strategies:
            rows.append(
                _run_finetune_then_imagenet_eval(
                    args,
                    base_cfg,
                    scenario="domain_gap",
                    scenario_group=f"4.2_{dataset_name}",
                    dataset_name=dataset_name,
                    data_root=root,
                    metadata_csv=metadata_csv,
                    download=download,
                    strategy=strategy,
                    train_samples=args.train_samples,
                    val_samples=args.val_samples,
                    test_samples=args.test_samples,
                    fraction="same_size",
                    imagenet_before=imagenet_before,
                )
            )
    return rows


def _run_data_fraction(
    args: argparse.Namespace,
    base_cfg: Config,
    strategies: list[str],
    fractions: list[float],
    imagenet_before: dict[str, float],
) -> list[dict]:
    rows = []
    for fraction in fractions:
        train_samples = max(1, int(args.base_train_samples * fraction))
        for strategy in strategies:
            rows.append(
                _run_finetune_then_imagenet_eval(
                    args,
                    base_cfg,
                    scenario="data_fraction",
                    scenario_group=f"4.3_eurosat_{fraction:g}",
                    dataset_name="eurosat",
                    data_root=args.eurosat_root,
                    metadata_csv=args.eurosat_metadata,
                    download=False,
                    strategy=strategy,
                    train_samples=train_samples,
                    val_samples=args.val_samples,
                    test_samples=args.test_samples,
                    fraction=f"{fraction:g}",
                    imagenet_before=imagenet_before,
                )
            )
    return rows


def _run_finetune_then_imagenet_eval(
    args: argparse.Namespace,
    base_cfg: Config,
    scenario: str,
    scenario_group: str,
    dataset_name: str,
    data_root: str,
    metadata_csv: str,
    download: bool,
    strategy: str,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    fraction: str,
    imagenet_before: dict[str, float],
) -> dict:
    run_name = _run_name(scenario, dataset_name, fraction, strategy)
    finetune_cfg = make_run_config(
        base_cfg,
        output_dir=Path(args.output_dir) / run_name / dataset_name,
        dataset_name=dataset_name,
        data_root=data_root,
        metadata_csv=metadata_csv,
        strategy=strategy,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        download=download,
        epochs=args.epochs,
    )
    trained = train_main(finetune_cfg, dummy=args.dummy)
    finetune_summary = read_summary(trained.summary_json)
    imagenet_after = _evaluate_imagenet(args, base_cfg, backbone_ckpt=trained.best_ckpt)

    return {
        "scenario": scenario,
        "scenario_group": scenario_group,
        "strategy": strategy,
        "pretrain_dataset": "imagenet",
        "finetune_dataset": dataset_name,
        "finetune_fraction": fraction,
        "finetune_train_samples": train_samples,
        "finetune_val_samples": val_samples,
        "finetune_test_samples": test_samples,
        "imagenet_before_top1": imagenet_before["top1_acc"],
        "imagenet_after_top1": imagenet_after["top1_acc"],
        "forgetting_top1": imagenet_before["top1_acc"] - imagenet_after["top1_acc"],
        "imagenet_before_macro_f1": imagenet_before["macro_f1"],
        "imagenet_after_macro_f1": imagenet_after["macro_f1"],
        "forgetting_macro_f1": imagenet_before["macro_f1"] - imagenet_after["macro_f1"],
        "finetune_test_top1": finetune_summary["test_top1_acc"],
        "finetune_test_macro_f1": finetune_summary["test_macro_f1"],
    }


def _evaluate_imagenet(
    args: argparse.Namespace,
    base_cfg: Config,
    backbone_ckpt: str | Path | None = None,
) -> dict[str, float]:
    cfg = _imagenet_config(args, base_cfg)
    device = resolve_device(cfg.raw)
    dataset_cfg = dataset_cfg_from_raw(cfg.raw)
    batch_size = int(cfg.raw["training"]["batch_size"])
    loader = build_dataloader(dataset_cfg, args.imagenet_split, batch_size, cfg.seed, args.dummy)

    model = build_resnet18(num_classes=1000, pretrained=True).to(device)
    if backbone_ckpt:
        state = torch.load(backbone_ckpt, map_location="cpu")["model"]
        backbone_state = {
            key: value
            for key, value in state.items()
            if not key.startswith("fc.")
        }
        model.load_state_dict(backbone_state, strict=False)
    return evaluate(model, loader, device)


def _imagenet_config(args: argparse.Namespace, base_cfg: Config) -> Config:
    raw = deepcopy(base_cfg.raw)
    raw["dataset"]["name"] = "imagenet"
    raw["dataset"]["root"] = args.imagenet_root
    raw["dataset"]["metadata_csv"] = ""
    raw["dataset"]["num_classes"] = 1000
    raw["dataset"]["train_samples"] = 0
    raw["dataset"]["val_samples"] = args.imagenet_samples
    raw["dataset"]["test_samples"] = args.imagenet_samples
    return Config(raw=raw)


def _run_name(scenario: str, dataset_name: str, fraction: str, strategy: str) -> str:
    parts = [scenario, dataset_name]
    if fraction not in {"1", "same_size"}:
        parts.append(f"frac_{fraction}")
    elif fraction == "same_size":
        parts.append("same_size")
    parts.append(strategy)
    return "_".join(parts)


if __name__ == "__main__":
    main()
