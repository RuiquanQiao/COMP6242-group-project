# COMP6242 Transfer Learning Experiments

Terminal-first code for the ResNet18 transfer-learning experiments in `report.md`.

Experiments covered:

- 4.1 EuroSAT strategy ablation
- 4.2 EuroSAT vs CIFAR-10 same-size domain comparison
- 4.3 EuroSAT data-size comparison
- previous-task forgetting analysis after downstream fine-tuning

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

## Prepare ImageNet Validation Set

For catastrophic forgetting, this project now uses the official ImageNet-1K
`ILSVRC2012` validation set as the previous-task evaluation set. This is the
standard 50,000-image validation archive from `image-net.org`, not a third-party
repack.

Download, extract, and create a smaller derivative in one command:

```bash
python scripts/prepare_imagenet.py --root data/imagenet_official --compact_root data/imagenet_official_resized
```

This script:

- downloads the official validation archive from `https://image-net.org`
- downloads the official `ILSVRC2012` devkit
- converts the validation set into ImageFolder layout at `data/imagenet_official/val`
- writes `official_manifest.json` with the official source URLs
- creates a smaller resized derivative at `data/imagenet_official_resized/val`

Useful options:

```bash
python scripts/prepare_imagenet.py --download
python scripts/prepare_imagenet.py --extract
python scripts/prepare_imagenet.py --make_compact --compact_size 256 --jpeg_quality 90
python scripts/prepare_imagenet.py --keep_archives
```

Expected folders after preparation:

```text
data/imagenet_official/
  archives/
  official_manifest.json
  val/<wnid>/*.JPEG

data/imagenet_official_resized/
  compact_manifest.json
  val/<wnid>/*.JPEG
```

Notes:

- `data/` is already ignored by git, so teammates can generate these files locally.
- The official validation tar is about 6.7 GB, which fits the "10+ GB" storage limit.
- The resized derivative is smaller and is intended for storage-constrained teammates.


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


### Previous-Task Forgetting Analysis

The forgetting experiment uses the official ImageNet-1K pretrained ResNet-18
as the previous-task model, fine-tunes it on one of this project's downstream
datasets, and then evaluates the adapted backbone back on the official ImageNet
validation set. This measures how much ImageNet performance is forgotten after
fine-tuning.

How this applies to the three experiments:

```text
4.1 main experiment / strategy ablation:
  downstream dataset = EuroSAT
  compare forgetting across linear_probe, partial_ft, and full_ft

4.2 domain_gap:
  downstream datasets = EuroSAT and CIFAR-10
  run forgetting once for EuroSAT and once for CIFAR-10, then compare them

4.3 data_size:
  downstream dataset = EuroSAT
  compare forgetting across different EuroSAT training sizes
```

First prepare the official ImageNet validation set:

```bash
python scripts/prepare_imagenet.py --root data/imagenet_official --compact_root data/imagenet_official_resized
```

Then run the forgetting experiment against the official validation set. For
the main experiment and data_size experiment, use EuroSAT:

```bash
python scripts/run_forgetting.py --config configs/base.yaml --source_dataset imagenet --target_dataset eurosat --imagenet_root data/imagenet_official --output_dir outputs/forgetting_imagenet_to_eurosat
```

For the domain_gap experiment, run both EuroSAT and CIFAR-10, then compare the
two forgetting result files:

```bash
python scripts/run_forgetting.py --config configs/base.yaml --source_dataset imagenet --target_dataset eurosat --imagenet_root data/imagenet_official --output_dir outputs/forgetting_imagenet_to_eurosat
python scripts/run_forgetting.py --config configs/base.yaml --source_dataset imagenet --target_dataset cifar10 --imagenet_root data/imagenet_official --download_cifar --output_dir outputs/forgetting_imagenet_to_cifar10
```

Compare:

```text
outputs/forgetting_imagenet_to_eurosat/forgetting_results.csv
outputs/forgetting_imagenet_to_cifar10/forgetting_results.csv
```

This script performs the following sequence:

```text
evaluate pretrained ResNet-18 on official ImageNet validation -> ImageNet_before
fine-tune the same pretrained initialization on the downstream dataset
restore the original 1000-way ImageNet classifier head
load the downstream fine-tuned backbone
evaluate on official ImageNet validation again -> ImageNet_after
compute forgetting = ImageNet_before - ImageNet_after
```

The key columns in `forgetting_results.csv` are:

```text
source_before_top1   ImageNet accuracy before downstream fine-tuning
source_after_top1    ImageNet accuracy after downstream fine-tuning
forgetting_top1      source_before_top1 - source_after_top1
target_test_top1     downstream dataset accuracy after fine-tuning
```

The default downstream fine-tuning strategies are:

```text
linear_probe
partial_ft
full_ft
```

The `scratch` strategy is excluded because this experiment starts from an
already-trained previous-task checkpoint.

Outputs:

```text
outputs/forgetting_imagenet_to_eurosat/forgetting_results.csv
outputs/forgetting_imagenet_to_eurosat/forgetting_top1.png
outputs/forgetting_imagenet_to_eurosat/forgetting_macro_f1.png
outputs/forgetting_imagenet_to_cifar10/forgetting_results.csv
outputs/forgetting_imagenet_to_cifar10/forgetting_top1.png
outputs/forgetting_imagenet_to_cifar10/forgetting_macro_f1.png
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

The forgetting script is different: its final rows include an additional
ImageNet re-evaluation step after downstream fine-tuning and are not currently
recoverable from per-run `summary.json` files alone.

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
