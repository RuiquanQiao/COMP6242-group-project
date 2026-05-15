from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiment_utils import make_run_config, read_summary, save_bar_chart, save_line_chart, write_csv
from transfer_learning.config import Config, load_config
from transfer_learning.train import evaluate_checkpoint, train_main

STRATEGIES = ["linear_probe", "partial_ft", "full_ft"]
SCENARIOS = ["domain_gap", "reverse_domain_gap", "data_fraction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forgetting analysis for domain-gap and data-size settings.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/forgetting")
    parser.add_argument("--eurosat_root", type=str, default="data/eurosat/2750")
    parser.add_argument("--eurosat_metadata", type=str, default="data/metadata.csv")
    parser.add_argument("--cifar_root", type=str, default="data")
    parser.add_argument("--download_cifar", action="store_true")
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
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
    rows: list[dict] = []

    for scenario in scenarios:
        if scenario == "domain_gap":
            rows.extend(_run_domain_gap(args, base_cfg, strategies, reverse=False))
        elif scenario == "reverse_domain_gap":
            rows.extend(_run_domain_gap(args, base_cfg, strategies, reverse=True))
        elif scenario == "data_fraction":
            rows.extend(_run_data_fraction(args, base_cfg, strategies, fractions))
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
            "a_fraction",
            "strategy",
            "forgetting_top1",
        )
        save_line_chart(
            out_dir / "forgetting_by_fraction_macro_f1.png",
            fraction_rows,
            "a_fraction",
            "strategy",
            "forgetting_macro_f1",
        )
    print(f"saved: {out_dir / 'forgetting_results.csv'}")
    print(f"saved: {out_dir / 'forgetting_top1.png'}")
    print(f"saved: {out_dir / 'forgetting_macro_f1.png'}")


def _run_domain_gap(args: argparse.Namespace, base_cfg: Config, strategies: list[str], reverse: bool) -> list[dict]:
    scenario = "reverse_domain_gap" if reverse else "domain_gap"
    eurosat = _dataset_spec(
        name="eurosat",
        root=args.eurosat_root,
        metadata_csv=args.eurosat_metadata,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        download=False,
    )
    cifar = _dataset_spec(
        name="cifar10",
        root=args.cifar_root,
        metadata_csv="",
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        download=args.download_cifar,
    )
    task_a, task_b = (cifar, eurosat) if reverse else (eurosat, cifar)
    rows = []
    for strategy in strategies:
        rows.append(_run_sequence(args, base_cfg, scenario, strategy, task_a, task_b))
    return rows


def _run_data_fraction(
    args: argparse.Namespace,
    base_cfg: Config,
    strategies: list[str],
    fractions: list[float],
) -> list[dict]:
    rows = []
    for fraction in fractions:
        train_samples = max(1, int(args.base_train_samples * fraction))
        task_a = _dataset_spec(
            name="eurosat",
            root=args.eurosat_root,
            metadata_csv=args.eurosat_metadata,
            train_samples=train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            download=False,
            fraction=fraction,
        )
        task_b = _dataset_spec(
            name="cifar10",
            root=args.cifar_root,
            metadata_csv="",
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            download=args.download_cifar,
        )
        for strategy in strategies:
            rows.append(_run_sequence(args, base_cfg, "data_fraction", strategy, task_a, task_b))
    return rows


def _run_sequence(
    args: argparse.Namespace,
    base_cfg: Config,
    scenario: str,
    strategy: str,
    task_a: dict,
    task_b: dict,
) -> dict:
    run_name = _run_name(scenario, strategy, task_a, task_b)
    task_a_cfg = make_run_config(
        base_cfg,
        output_dir=Path(args.output_dir) / run_name / "task_a",
        dataset_name=task_a["name"],
        data_root=task_a["root"],
        metadata_csv=task_a["metadata_csv"],
        strategy=strategy,
        train_samples=task_a["train_samples"],
        val_samples=task_a["val_samples"],
        test_samples=task_a["test_samples"],
        download=task_a["download"],
        epochs=args.epochs,
    )
    trained_a = train_main(task_a_cfg, dummy=args.dummy)
    a_before = evaluate_checkpoint(task_a_cfg, trained_a.best_ckpt, split="test", dummy=args.dummy)

    task_b_cfg = make_run_config(
        base_cfg,
        output_dir=Path(args.output_dir) / run_name / "task_b",
        dataset_name=task_b["name"],
        data_root=task_b["root"],
        metadata_csv=task_b["metadata_csv"],
        strategy=strategy,
        train_samples=task_b["train_samples"],
        val_samples=task_b["val_samples"],
        test_samples=task_b["test_samples"],
        download=task_b["download"],
        epochs=args.epochs,
    )
    trained_b = train_main(task_b_cfg, dummy=args.dummy, init_ckpt=trained_a.best_ckpt)
    b_summary = read_summary(trained_b.summary_json)
    a_after = evaluate_checkpoint(task_a_cfg, trained_b.best_ckpt, split="test", dummy=args.dummy)

    return {
        "scenario": scenario,
        "scenario_group": _scenario_group(scenario, task_a),
        "task_order": f"{task_a['name']}_to_{task_b['name']}",
        "strategy": strategy,
        "a_dataset": task_a["name"],
        "b_dataset": task_b["name"],
        "a_fraction": task_a["fraction"],
        "a_train_samples": task_a["train_samples"],
        "b_train_samples": task_b["train_samples"],
        "a_before_top1": a_before["top1_acc"],
        "a_after_top1": a_after["top1_acc"],
        "forgetting_top1": a_before["top1_acc"] - a_after["top1_acc"],
        "a_before_macro_f1": a_before["macro_f1"],
        "a_after_macro_f1": a_after["macro_f1"],
        "forgetting_macro_f1": a_before["macro_f1"] - a_after["macro_f1"],
        "b_test_top1": b_summary["test_top1_acc"],
        "b_test_macro_f1": b_summary["test_macro_f1"],
    }


def _dataset_spec(
    name: str,
    root: str,
    metadata_csv: str,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    download: bool,
    fraction: float = 1.0,
) -> dict:
    return {
        "name": name,
        "root": root,
        "metadata_csv": metadata_csv,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "download": download,
        "fraction": f"{fraction:g}",
    }


def _run_name(scenario: str, strategy: str, task_a: dict, task_b: dict) -> str:
    fraction = f"_frac_{task_a['fraction']}" if scenario == "data_fraction" else ""
    return f"{scenario}{fraction}_{task_a['name']}_to_{task_b['name']}_{strategy}"


def _scenario_group(scenario: str, task_a: dict) -> str:
    if scenario == "data_fraction":
        return f"data_fraction_{task_a['fraction']}"
    return scenario


if __name__ == "__main__":
    main()
