from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STYLE = {
    "font": "DejaVu Sans",
    "palette": {
        "Scratch": "#737373",
        "Linear probe": "#2563eb",
        "Partial unfreeze": "#16a34a",
        "Full fine-tune": "#ef4444",
        "Baseline": "#111111",
    },
    "grid": "#e6e8ef",
    "text": "#111827",
    "muted": "#64748b",
    "line_width": 1.9,
    "alpha": 0.92,
    "marker_size": 58,
    "linestyles": {"Scratch": "-", "Linear probe": "--", "Partial unfreeze": "-.", "Full fine-tune": ":"},
    "markers": {"Scratch": "D", "Linear probe": "D", "Partial unfreeze": "D", "Full fine-tune": "D"},
}
STRATEGY_LABELS = {
    "scratch": "Scratch",
    "linear_probe": "Linear probe",
    "partial_ft": "Partial unfreeze",
    "full_ft": "Full fine-tune",
}
STRATEGY_ORDER = ["Scratch", "Linear probe", "Partial unfreeze", "Full fine-tune"]
NO_SCRATCH_ORDER = ["Linear probe", "Partial unfreeze", "Full fine-tune"]


def apply_style() -> None:
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "font.family": STYLE["font"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#b8bdc7",
            "axes.labelcolor": STYLE["text"],
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": STYLE["text"],
            "ytick.color": STYLE["text"],
            "text.color": STYLE["text"],
            "axes.grid": True,
            "grid.color": STYLE["grid"],
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": STYLE["line_width"],
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8bdc7")
    ax.spines["bottom"].set_color("#b8bdc7")


def percent_axis(ax: plt.Axes, axis: str = "y") -> None:
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    if axis in {"x", "both"}:
        ax.xaxis.set_major_formatter(fmt)
    if axis in {"y", "both"}:
        ax.yaxis.set_major_formatter(fmt)


def title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", weight="bold")


def pct(series: pd.Series) -> pd.Series:
    return series.astype(float) * 100.0


def marker(strategy: str) -> str:
    return STYLE["markers"].get(strategy, "D")


def color(strategy: str) -> str:
    return STYLE["palette"][strategy]


def save(fig: plt.Figure, out_dir: Path, rel_png: str) -> list[Path]:
    out_png = out_dir / rel_png
    out_png.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = out_png.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def ablation_results(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "outputs/eurosat_ablation/results.csv")
    df["Strategy"] = df["strategy"].map(STRATEGY_LABELS)
    df["Top1"] = pct(df["test_top1_acc"])
    df["MacroF1"] = pct(df["test_macro_f1"])
    df["Params"] = df["trainable_params"].astype(float)
    return df


def domain_results(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "outputs/domain_gap/results.csv")
    df["Dataset"] = df["dataset"].map({"eurosat": "EuroSAT", "cifar10": "CIFAR-10"})
    df["Strategy"] = df["strategy"].map(STRATEGY_LABELS)
    df["Top1"] = pct(df["test_top1_acc"])
    df["MacroF1"] = pct(df["test_macro_f1"])
    scratch = df[df["Strategy"] == "Scratch"].set_index("Dataset")["Top1"]
    df["Gain"] = df.apply(lambda r: r["Top1"] - scratch.loc[r["Dataset"]], axis=1)
    return df


def main_forgetting(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "outputs/forgetting_main/forgetting_results.csv")
    df["Strategy"] = df["strategy"].map(STRATEGY_LABELS)
    df["Forgetting"] = pct(df["forgetting_top1"])
    df["EuroSATTop1"] = pct(df["downstream_test_top1"])
    return df


def domain_forgetting(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "outputs/forgetting_domain_gap/forgetting_results.csv")
    df["Dataset"] = df["downstream_dataset"].map({"eurosat": "EuroSAT", "cifar10": "CIFAR-10"})
    df["Strategy"] = df["strategy"].map(STRATEGY_LABELS)
    df["Forgetting"] = pct(df["forgetting_top1"])
    df["ForgettingF1"] = pct(df["forgetting_macro_f1"])
    df["DownstreamTop1"] = pct(df["downstream_test_top1"])
    return df


def data_fraction_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [10, 68.02, 76.20, 93.58, 25.56],
            [30, 82.83, 85.10, 96.65, 13.82],
            [60, 89.46, 86.99, 96.88, 7.41],
            [100, 89.39, 87.93, 97.15, 7.75],
        ],
        columns=["TrainPct", "Scratch", "Linear probe", "Full fine-tune", "Full fine-tune gain"],
    )


def data_fraction_forgetting() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [10, "Linear probe", 37.26],
            [10, "Full fine-tune", 68.52],
            [30, "Linear probe", 37.65],
            [30, "Full fine-tune", 69.51],
            [60, "Linear probe", 38.14],
            [60, "Full fine-tune", 69.51],
            [100, "Linear probe", 37.63],
            [100, "Full fine-tune", 69.56],
        ],
        columns=["TrainPct", "Strategy", "Forgetting"],
    )


def fig_eurosat_top1(root: Path) -> plt.Figure:
    df = ablation_results(root).sort_values("Top1", ascending=False)
    fig, ax = plt.subplots(figsize=(4.7, 3.1), constrained_layout=True)
    y = np.arange(len(df))
    ax.hlines(y, df["Top1"].min() - 1.0, df["Top1"], color=STYLE["grid"], lw=2)
    for yi, (_, row) in zip(y, df.iterrows()):
        ax.scatter(row["Top1"], yi, s=68 if row["Strategy"] == "Partial unfreeze" else 54, color=color(row["Strategy"]), marker=marker(row["Strategy"]), zorder=3)
        ax.annotate(f"{row['Top1']:.2f}%", (row["Top1"], yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5)
    ax.set_yticks(y, df["Strategy"])
    ax.invert_yaxis()
    ax.set_xlabel("EuroSAT test top-1 accuracy")
    title(ax, "EuroSAT accuracy by strategy")
    ax.set_xlim(86.5, 99.4)
    percent_axis(ax, "x")
    despine(ax)
    return fig


def fig_eurosat_val_curve(root: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.9, 3.2), constrained_layout=True)
    for raw in ["scratch", "linear_probe", "partial_ft", "full_ft"]:
        label = STRATEGY_LABELS[raw]
        metrics = json.loads((root / "outputs/eurosat_ablation" / raw / "metrics.json").read_text())
        epochs = [row["epoch"] for row in metrics]
        vals = [row["val_top1_acc"] * 100 for row in metrics]
        ax.plot(epochs, vals, label=label, color=color(label), linestyle=STYLE["linestyles"][label], marker=marker(label), lw=STYLE["line_width"])
    ax.axhline(90, color=STYLE["muted"], lw=0.9)
    ax.annotate("90% threshold", (1.15, 90), xytext=(0, 5), textcoords="offset points", fontsize=8.2, color=STYLE["muted"])
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation top-1 accuracy")
    title(ax, "Validation accuracy across epochs")
    ax.set_ylim(70, 100)
    percent_axis(ax)
    ax.legend(loc="lower right")
    despine(ax)
    return fig


def fig_domain_top1(root: Path) -> plt.Figure:
    df = domain_results(root)
    fig, ax = plt.subplots(figsize=(4.9, 3.2), constrained_layout=True)
    datasets = ["EuroSAT", "CIFAR-10"]
    x = np.arange(len(datasets))
    width = 0.18
    for offset, strategy in zip(np.linspace(-1.5 * width, 1.5 * width, 4), STRATEGY_ORDER):
        vals = [df[(df.Dataset == ds) & (df.Strategy == strategy)]["Top1"].iloc[0] for ds in datasets]
        ax.bar(x + offset, vals, width=width, label=strategy, color=color(strategy), alpha=STYLE["alpha"], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x, datasets)
    ax.set_ylabel("Downstream top-1 accuracy")
    title(ax, "Same-size downstream accuracy")
    ax.set_ylim(72, 101)
    percent_axis(ax)
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=1, borderaxespad=0.2)
    despine(ax)
    return fig


def fig_domain_gain(root: Path) -> plt.Figure:
    df = domain_results(root)
    fig, ax = plt.subplots(figsize=(4.9, 3.2), constrained_layout=True)
    datasets = ["EuroSAT", "CIFAR-10"]
    x = np.arange(len(datasets))
    width = 0.22
    ax.axhline(0, color=STYLE["muted"], lw=0.95)
    for offset, strategy in zip(np.linspace(-width, width, 3), NO_SCRATCH_ORDER):
        vals = [df[(df.Dataset == ds) & (df.Strategy == strategy)]["Gain"].iloc[0] for ds in datasets]
        ax.bar(x + offset, vals, width=width, label=strategy, color=color(strategy), alpha=STYLE["alpha"], edgecolor="white", linewidth=0.5)
        ax.scatter(x + offset, vals, color=color(strategy), marker=marker(strategy), s=26, zorder=3, edgecolor="white", linewidth=0.35)
        for xi, val in zip(x + offset, vals):
            label = f"{val:+.2f}"
            if abs(val) < 0.5:
                # Tiny gains are statistically meaningful but visually vanish on
                # the same scale as the 13-point CIFAR-10 gains. Use an offset
                # label and a marker at the true bar tip rather than inflating it.
                ax.annotate(
                    label,
                    (xi, val),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="bold",
                    color=color(strategy),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=color(strategy), shrinkA=1, shrinkB=2),
                )
            else:
                ax.annotate(
                    label,
                    (xi, val),
                    xytext=(0, 4 if val >= 0 else -8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8.0,
                    fontweight="bold" if abs(val) > 10 else "normal",
                )
    ax.set_xticks(x, datasets)
    ax.set_ylabel("Gain over scratch")
    title(ax, "Transfer gain by domain")
    ax.set_ylim(-6, 15.5)
    percent_axis(ax)
    ax.legend(loc="upper left")
    despine(ax)
    return fig


def fig_data_top1(_: Path) -> plt.Figure:
    df = data_fraction_results()
    fig, ax = plt.subplots(figsize=(4.9, 3.2), constrained_layout=True)
    x = df["TrainPct"].to_numpy()
    for strategy in ["Scratch", "Linear probe", "Full fine-tune"]:
        ax.plot(x, df[strategy], label=strategy, color=color(strategy), linestyle=STYLE["linestyles"][strategy], marker=marker(strategy), lw=STYLE["line_width"])
    ax.fill_between(x, df["Scratch"], df["Full fine-tune"], color=color("Full fine-tune"), alpha=0.12, label="Full fine-tune gain")
    ax.set_xticks(x, [f"{v}%" for v in x])
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("EuroSAT top-1 accuracy")
    title(ax, "Accuracy under data scarcity")
    ax.set_ylim(64, 100)
    percent_axis(ax)
    ax.legend(loc="lower right")
    despine(ax)
    return fig


def fig_data_gain(_: Path) -> plt.Figure:
    df = data_fraction_results()
    fig, ax = plt.subplots(figsize=(4.7, 3.2), constrained_layout=True)
    y = np.arange(len(df))
    ax.hlines(y, 0, df["Full fine-tune gain"], color=STYLE["grid"], lw=2)
    ax.scatter(df["Full fine-tune gain"], y, color=color("Full fine-tune"), s=58, marker=marker("Full fine-tune"))
    for yi, (_, row) in zip(y, df.iterrows()):
        ax.annotate(f"+{row['Full fine-tune gain']:.2f}", (row["Full fine-tune gain"], yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, weight="bold" if row["TrainPct"] == 10 else "normal")
    ax.set_yticks(y, [f"{v}%" for v in df["TrainPct"]])
    ax.invert_yaxis()
    ax.set_xlabel("Full fine-tune gain over scratch")
    ax.set_ylabel("Training data fraction")
    title(ax, "Transfer gain by data size")
    ax.set_xlim(0, 29)
    percent_axis(ax, "x")
    despine(ax)
    return fig


def fig_forgetting_main(root: Path) -> plt.Figure:
    df = main_forgetting(root).set_index("Strategy").loc[NO_SCRATCH_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(4.7, 3.2), constrained_layout=True)
    y = np.arange(len(df))
    ax.hlines(y, 0, df["Forgetting"], color=STYLE["grid"], lw=2)
    for yi, (_, row) in zip(y, df.iterrows()):
        ax.scatter(row["Forgetting"], yi, color=color(row["Strategy"]), s=58, marker=marker(row["Strategy"]))
        ax.annotate(f"{row['Forgetting']:.2f}", (row["Forgetting"], yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5)
    ax.set_yticks(y, df["Strategy"])
    ax.invert_yaxis()
    ax.set_xlabel("ImageNet top-1 forgetting")
    title(ax, "ImageNet forgetting after EuroSAT")
    ax.set_xlim(0, 76)
    percent_axis(ax, "x")
    despine(ax)
    return fig


def fig_forgetting_main_tradeoff(root: Path) -> plt.Figure:
    df = main_forgetting(root)
    fig, ax = plt.subplots(figsize=(4.9, 3.2), constrained_layout=True)

    label_offsets = {
        "Linear probe": (6, 4, "left", "bottom"),
        "Partial unfreeze": (-8, 18, "right", "bottom"),
        "Full fine-tune": (8, -18, "left", "top"),
    }

    for _, row in df.iterrows():
        ax.scatter(
            row["Forgetting"],
            row["EuroSATTop1"],
            color=color(row["Strategy"]),
            marker=marker(row["Strategy"]),
            s=62,
            zorder=3,
        )

        dx, dy, ha, va = label_offsets[row["Strategy"]]
        ax.annotate(
            row["Strategy"],
            xy=(row["Forgetting"], row["EuroSATTop1"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.0,
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
            arrowprops=dict(arrowstyle="-", lw=0.55, color=STYLE["muted"]),
        )

    ax.annotate("preferred\nregion", (38, 98.8), ha="left", va="top", fontsize=8.4, color=STYLE["muted"])
    ax.set_xlabel("ImageNet top-1 forgetting")
    ax.set_ylabel("EuroSAT top-1 accuracy")
    title(ax, "Accuracy-forgetting trade-off")
    ax.set_xlim(34, 72)
    ax.set_ylim(87, 99.4)
    percent_axis(ax, "both")
    despine(ax)
    return fig


def fig_forgetting_domain(root: Path) -> plt.Figure:
    df = domain_forgetting(root)
    fig, ax = plt.subplots(figsize=(4.9, 3.2), constrained_layout=True)
    datasets = ["CIFAR-10", "EuroSAT"]
    x = np.arange(len(datasets))
    width = 0.22
    for offset, strategy in zip(np.linspace(-width, width, 3), NO_SCRATCH_ORDER):
        vals = [df[(df.Dataset == ds) & (df.Strategy == strategy)]["Forgetting"].iloc[0] for ds in datasets]
        ax.bar(x + offset, vals, width=width, label=strategy, color=color(strategy), alpha=STYLE["alpha"], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x, datasets)
    ax.set_ylabel("ImageNet top-1 forgetting")
    title(ax, "ImageNet top-1 forgetting")
    ax.set_ylim(0, 76)
    percent_axis(ax)
    ax.legend(loc="upper left")
    despine(ax)
    return fig


def fig_forgetting_domain_tradeoff(root: Path) -> plt.Figure:
    df = domain_forgetting(root)
    fig, ax = plt.subplots(figsize=(4.9, 3.2), constrained_layout=True)

    label_offsets = {
        ("EuroSAT", "Linear probe"): (6, 4, "left", "bottom"),
        ("CIFAR-10", "Linear probe"): (6, 4, "left", "bottom"),
        ("EuroSAT", "Partial unfreeze"): (-8, 18, "right", "bottom"),
        ("EuroSAT", "Full fine-tune"): (8, -18, "left", "top"),
        ("CIFAR-10", "Partial unfreeze"): (-8, 16, "right", "bottom"),
        ("CIFAR-10", "Full fine-tune"): (8, -16, "left", "top"),
    }

    for _, row in df.iterrows():
        label = f"{row['Dataset']} {row['Strategy']}"
        ax.scatter(
            row["Forgetting"],
            row["DownstreamTop1"],
            color=color(row["Strategy"]),
            marker=marker(row["Strategy"]),
            s=58,
            zorder=3,
        )

        dx, dy, ha, va = label_offsets[(row["Dataset"], row["Strategy"])]
        ax.annotate(
            label,
            xy=(row["Forgetting"], row["DownstreamTop1"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.1,
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.85),
            arrowprops=dict(arrowstyle="-", lw=0.5, color=STYLE["muted"]),
        )

    ax.set_xlabel("ImageNet top-1 forgetting")
    ax.set_ylabel("Downstream top-1 accuracy")
    title(ax, "Domain trade-off")
    ax.set_xlim(34, 72)
    ax.set_ylim(76, 100)
    percent_axis(ax, "both")
    despine(ax)
    return fig


def fig_forgetting_data_methods(_: Path) -> plt.Figure:
    df = data_fraction_forgetting()
    fig, ax = plt.subplots(figsize=(4.9, 3.2), constrained_layout=True)
    for strategy in ["Linear probe", "Full fine-tune"]:
        sub = df[df.Strategy == strategy]
        ax.plot(sub["TrainPct"], sub["Forgetting"], label=strategy, color=color(strategy), linestyle=STYLE["linestyles"][strategy], marker=marker(strategy), lw=STYLE["line_width"])
    ax.set_xticks([10, 30, 60, 100], ["10%", "30%", "60%", "100%"])
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("ImageNet top-1 forgetting")
    title(ax, "Forgetting remains strategy-driven")
    ax.set_ylim(32, 73)
    percent_axis(ax)
    ax.legend(loc="center right")
    despine(ax)
    return fig


def fig_forgetting_data_tradeoff(_: Path) -> plt.Figure:
    gains = data_fraction_results()[["TrainPct", "Full fine-tune gain"]]
    forgetting = data_fraction_forgetting()
    full = forgetting[forgetting.Strategy == "Full fine-tune"].merge(gains, on="TrainPct")
    fig, ax = plt.subplots(figsize=(4.9, 3.2), constrained_layout=True)
    ax.plot(full["TrainPct"], full["Full fine-tune gain"], label="Full fine-tune gain", color=color("Partial unfreeze"), linestyle="-", marker=marker("Partial unfreeze"), lw=STYLE["line_width"])
    ax.plot(full["TrainPct"], full["Forgetting"], label="Full fine-tune forgetting", color=color("Full fine-tune"), linestyle=STYLE["linestyles"]["Full fine-tune"], marker=marker("Full fine-tune"), lw=STYLE["line_width"])
    ax.set_xticks([10, 30, 60, 100], ["10%", "30%", "60%", "100%"])
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Percentage points")
    title(ax, "Gain fades, forgetting persists")
    ax.set_ylim(0, 76)
    percent_axis(ax)
    ax.legend(loc="center right")
    despine(ax)
    return fig


FIGURES = {
    "eurosat_ablation/test_top1_acc.png": fig_eurosat_top1,
    "eurosat_ablation/val_top1_acc_curve.png": fig_eurosat_val_curve,
    "domain_gap/test_top1_acc.png": fig_domain_top1,
    "domain_gap/transfer_gain_top1.png": fig_domain_gain,
    "data_fraction/test_top1_acc.png": fig_data_top1,
    "data_fraction/transfer_gain_top1.png": fig_data_gain,
    "forgetting_main/forgetting_top1.png": fig_forgetting_main,
    "forgetting_main/transfer_forgetting_tradeoff.png": fig_forgetting_main_tradeoff,
    "forgetting_domain_gap/forgetting_top1.png": fig_forgetting_domain,
    "forgetting_domain_gap/transfer_forgetting_tradeoff.png": fig_forgetting_domain_tradeoff,
    "forgetting_data_fraction/forgetting_top1_transfer_methods.png": fig_forgetting_data_methods,
    "forgetting_data_fraction/transfer_forgetting_tradeoff.png": fig_forgetting_data_tradeoff,
}


def write_readme(out_dir: Path) -> None:
    text = """# New Style Picture

This directory contains regenerated publication-style report figures.

Data source:

- `eurosat_ablation/*`: generated from `outputs/eurosat_ablation/results.csv` and per-run `metrics.json`.
- `domain_gap/*`: generated from `outputs/domain_gap/results.csv`.
- `forgetting_main/*`: generated from `outputs/forgetting_main/forgetting_results.csv`.
- `forgetting_domain_gap/*`: generated from `outputs/forgetting_domain_gap/forgetting_results.csv`.
- `data_fraction/*` and `forgetting_data_fraction/*`: generated from the paper table values because the repository does not include `outputs/data_fraction/results.csv`, `transfer_gain.csv`, or the original `frac_*/*/summary.json` artifacts.

All percentage values are plotted directly as percentages, not 0-1 decimals.
Each figure is exported as PNG, PDF, and SVG.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/new_style_picture")
    args = parser.parse_args()
    out_dir = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out

    apply_style()
    saved = []
    for rel, builder in FIGURES.items():
        saved.extend(save(builder(PROJECT_ROOT), out_dir, rel))
    write_readme(out_dir)
    print(f"Output folder: {out_dir}")
    print(f"Figures: {len(FIGURES)}")
    print(f"Files: {len(saved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
