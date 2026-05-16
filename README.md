# COMP6242 Transfer Learning Experiments

Terminal-first code for the ResNet18 transfer-learning experiments in `report.md`.

Experiments covered:

- 4.1 EuroSAT strategy ablation
- 4.2 EuroSAT vs CIFAR-10 same-size domain comparison
- 4.3 EuroSAT data-size comparison
- ImageNet forgetting analysis after every ImageNet-pretrained downstream run

For detailed code explanation, see `interpretation.md`.

## Install

Choose one environment file.

CPU:

```bash
pip install -r requirements-cpu.txt
```

CUDA 12.4:

```bash
pip install -r requirements-cu124.txt
```

Check PyTorch:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Prepare EuroSAT

Generate the metadata file used by all EuroSAT experiments:

```bash
python scripts/prepare_eurosat.py --root data --download --out_csv data/metadata.csv
```

If EuroSAT is already downloaded:

```bash
python scripts/prepare_eurosat.py --images_root "data/eurosat/2750" --out_csv data/metadata.csv
```

CIFAR-10 is loaded directly through torchvision. It does not use this preparation script.


## Run One Model

```bash
python scripts/train.py --config configs/base.yaml --strategy full_ft --output_dir outputs/single_full_ft
```

Evaluate a saved checkpoint:

```bash
python scripts/eval.py --config configs/base.yaml --strategy full_ft --ckpt outputs/single_full_ft/best.pt
```

Each run saves:

```text
best.pt        best validation checkpoint
metrics.json   epoch-by-epoch train/validation metrics
summary.json   final test metrics, convergence epoch, and run summary
```

## Run Experiments

### 4.1 EuroSAT Strategy Ablation

```bash
python scripts/run_ablation.py --config configs/base.yaml
```

Outputs:

```text
outputs/eurosat_ablation/results.csv
outputs/eurosat_ablation/test_top1_acc.png
outputs/eurosat_ablation/val_top1_acc_curve.png
outputs/eurosat_ablation/val_macro_f1_curve.png
```

If the run was interrupted, reuse completed strategy folders and train only
missing ones with:

```bash
python scripts/run_ablation.py --config configs/base.yaml --skip_existing
```

If every selected strategy folder already has `summary.json` and `metrics.json`,
refresh only the top-level CSV and plots with:

```bash
python scripts/run_ablation.py --config configs/base.yaml --aggregate_only
```

### 4.2 EuroSAT vs CIFAR-10 Same-Size Comparison

```bash
python scripts/run_domain_gap.py --config configs/base.yaml --download_cifar
```

Use `--download_cifar` the first time you run this experiment. CIFAR-10 is
downloaded by torchvision to `data/` by default. If CIFAR-10 already exists
there, torchvision reuses the local files. After the first successful download,
you can omit `--download_cifar`.

Use another CIFAR-10 location with:

```bash
python scripts/run_domain_gap.py --config configs/base.yaml --cifar_root "PATH/TO/CIFAR_ROOT"
```

If training was interrupted after some runs completed, reuse completed
subdirectories and train only missing runs with:

```bash
python scripts/run_domain_gap.py --config configs/base.yaml --download_cifar --skip_existing
```

If the per-run folders already contain `summary.json` and `metrics.json`, but
the top-level CSV or plots are stale, regenerate only the aggregate outputs with:

```bash
python scripts/run_domain_gap.py --config configs/base.yaml --aggregate_only
```

`--aggregate_only` uses the same strategy selection as the normal run. With
default arguments, it expects EuroSAT and CIFAR-10 folders for `scratch`,
`partial_ft`, and `full_ft`. If the original run used a strategy subset or a
custom output directory, pass the same options again:

```bash
python scripts/run_domain_gap.py --config configs/base.yaml --aggregate_only --strategies scratch,full_ft
python scripts/run_domain_gap.py --config configs/base.yaml --aggregate_only --output_dir outputs/my_domain_gap
```

This rewrites:

```text
outputs/domain_gap/results.csv
outputs/domain_gap/test_top1_acc.png
outputs/domain_gap/test_macro_f1.png
outputs/domain_gap/val_top1_acc_curve.png
outputs/domain_gap/val_macro_f1_curve.png
```

It does not retrain models or overwrite the per-run checkpoints and JSON files.

### 4.3 EuroSAT Data-Size Comparison

```bash
python scripts/run_data_fraction.py --config configs/base.yaml
```

If the run was interrupted, reuse completed fraction/strategy folders and train
only missing ones with:

```bash
python scripts/run_data_fraction.py --config configs/base.yaml --skip_existing
```

If every selected fraction/strategy folder already has `summary.json` and
`metrics.json`, refresh only the top-level CSV and plots with:

```bash
python scripts/run_data_fraction.py --config configs/base.yaml --aggregate_only
```

Use the same `--fractions`, `--strategies`, and `--output_dir` values that were
used for the original run when aggregating a subset or custom output folder.


### ImageNet Forgetting Analysis

If ImageNet, EuroSAT, and CIFAR-10 are already available locally:

```bash
python scripts/run_forgetting.py --config configs/base.yaml --imagenet_root "PATH/TO/IMAGENET"
```

For the first full run, add `--download_cifar` because the default scenarios
include `domain_gap`, which uses CIFAR-10:

```bash
python scripts/run_forgetting.py --config configs/base.yaml --imagenet_root "PATH/TO/IMAGENET" --download_cifar
```

This script measures how much ImageNet validation performance drops after an
ImageNet-pretrained ResNet18 is fine-tuned on each downstream transfer setting.
ImageNet is always the retained pretraining task. EuroSAT and CIFAR-10 are
downstream fine-tuning tasks.

The forgetting script re-runs the selected downstream settings, saves their
checkpoints under `outputs/forgetting/`, then evaluates each fine-tuned
backbone on ImageNet. It does not read checkpoints from `outputs/eurosat_ablation/`,
`outputs/domain_gap/`, or `outputs/data_fraction/`.

`--download_cifar` is only for the CIFAR-10 part of this script. Use it when
the selected scenarios include `domain_gap` and CIFAR-10 has not already been
downloaded under `--cifar_root` (`data/` by default). It does not download
EuroSAT or ImageNet. If CIFAR-10 already exists locally, or if you run only
EuroSAT scenarios such as `eurosat_ablation` or `data_fraction`, omit
`--download_cifar`.

ImageNet is not downloaded automatically; provide a local ImageFolder-style
validation split with standard ImageNet WNID class folders:

```text
PATH/TO/IMAGENET/val/n01440764/*.JPEG
PATH/TO/IMAGENET/val/n01443537/*.JPEG
...
```

The default run covers the transfer experiments in the report:

```text
eurosat_ablation  4.1 EuroSAT strategy ablation
domain_gap        4.2 EuroSAT and CIFAR-10 same-size comparison
data_fraction     4.3 EuroSAT 10% / 30% / 60% / 100% comparison
```

Only ImageNet-pretrained strategies are included by default:

```text
linear_probe
partial_ft
full_ft
```

The `scratch` strategy is excluded from ImageNet forgetting because it has no
ImageNet pretraining to forget. If `scratch` is passed in `--strategies`, the
script skips it.

For EuroSAT-only quick checks or limited compute, evaluate on a balanced
ImageNet subset and skip the CIFAR-10 scenario:

```bash
python scripts/run_forgetting.py --config configs/base.yaml --scenarios eurosat_ablation,data_fraction --imagenet_root "PATH/TO/IMAGENET" --imagenet_samples 5000
```

Run a subset of scenarios with:

```bash
python scripts/run_forgetting.py --config configs/base.yaml --scenarios data_fraction --imagenet_root "PATH/TO/IMAGENET"
python scripts/run_forgetting.py --config configs/base.yaml --scenarios domain_gap --imagenet_root "PATH/TO/IMAGENET" --download_cifar
```


## Useful Arguments

Most scripts support:

```text
--epochs N
--strategies scratch,partial_ft,full_ft
--output_dir PATH
--dummy
```

Dataset-size experiments also support:

```text
--train_samples N
--val_samples N
--test_samples N
```

Batch experiment scripts for 4.1, 4.2, and 4.3 also support:

```text
--skip_existing     reuse run folders that already have summary.json and metrics.json
--aggregate_only    rebuild top-level CSV and plots from existing run folders
```

Use `--skip_existing` after an interrupted run when some child runs are complete
and some are missing. Use `--aggregate_only` when all selected child runs are
complete but the top-level CSV or plots are stale. Keep `--strategies`,
`--fractions`, and `--output_dir` consistent with the run you want to recover;
otherwise the script will look for its default output layout.

The ImageNet forgetting script is different: its final rows include an
additional ImageNet re-evaluation step and are not currently recoverable from
per-run `summary.json` files alone.

`--dummy` uses fake data for a quick code-path check:

```bash
python scripts/run_domain_gap.py --config configs/base.yaml --dummy --strategies scratch --epochs 1 --train_samples 32 --val_samples 16 --test_samples 16
```

## Where to Read Results

Start with the top-level CSV in each experiment output folder.

For training curves, use each experiment folder's `val_top1_acc_curve.png` and
`val_macro_f1_curve.png`, or open each run's `metrics.json` for raw epoch-level
values.

For final test metrics of a single run, open `summary.json`.
