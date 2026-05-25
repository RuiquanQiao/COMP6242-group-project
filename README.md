# COMP6242 Transfer Learning Experiments

Terminal-first code for the ResNet18 transfer-learning experiments outlined in
`report-outline.md` and written up in `report.md`.

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

This CUDA file keeps PyPI as the main package index and adds the PyTorch CUDA
wheel index as an extra source, so general dependencies such as
`matplotlib` still install normally while `torch` and `torchvision` resolve to
the CUDA 12.4 builds.

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

With `--aggregate_only`, the script discovers existing completed strategy
folders automatically, so newly added runs such as `linear_probe` are included
without changing the command. If you want to aggregate only a subset or use a
custom output directory, pass the corresponding options:

```bash
python scripts/run_domain_gap.py --config configs/base.yaml --aggregate_only --strategies scratch,full_ft
python scripts/run_domain_gap.py --config configs/base.yaml --aggregate_only --output_dir outputs/my_domain_gap
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

With `--aggregate_only`, the script discovers existing `frac_*` and strategy
folders automatically. Pass `--fractions`, `--strategies`, or `--output_dir`
only when aggregating a subset or custom output folder.


### Previous-Task Forgetting Analysis

The forgetting experiment uses the official ImageNet-1K pretrained ResNet-18
as the previous-task model and evaluates ImageNet performance after loading
the backbone from completed downstream checkpoints. It does not retrain the
downstream models. You pass the completed run folders directly with
`--run_dirs`. Each argument can be either a completed run folder containing
`best.pt` and `summary.json`, or a parent folder that contains completed run
folders. `scratch` runs are skipped because they were not fine-tuned from
ImageNet pretrained weights.

First prepare the official ImageNet validation set:

```bash
python scripts/prepare_imagenet.py --root data/imagenet_official --compact_root data/imagenet_official_resized
```

Then run the forgetting analysis against the official ImageNet validation set
by passing any completed experiment output folder:

```bash
python scripts/run_forgetting.py --config configs/base.yaml --imagenet_root data/imagenet_official --output_dir outputs/forgetting_<name> --run_dirs outputs/<completed_experiment>
```

Examples:

```bash
python scripts/run_forgetting.py --config configs/base.yaml --imagenet_root data/imagenet_official --output_dir outputs/forgetting_main --run_dirs outputs/eurosat_ablation
python scripts/run_forgetting.py --config configs/base.yaml --imagenet_root data/imagenet_official --output_dir outputs/forgetting_domain_gap --run_dirs outputs/domain_gap
python scripts/run_forgetting.py --config configs/base.yaml --imagenet_root data/imagenet_official --output_dir outputs/forgetting_data_size --run_dirs outputs/data_fraction
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
--aggregate_only    discover completed run folders and rebuild top-level CSV/plots
```

Use `--skip_existing` after an interrupted run when some child runs are complete
and some are missing. Use `--aggregate_only` when completed child runs exist but
the top-level CSV or plots are stale; by default it traverses the existing
completed folders under the selected output directory. Pass `--strategies`,
`--fractions`, or `--output_dir` only when you want a subset or a custom output
layout.

The forgetting script is different: its final rows include an additional
ImageNet re-evaluation step after downstream fine-tuning and are not currently
recoverable from per-run `summary.json` files alone.

`--dummy` uses fake data for a quick code-path check:

```bash
python scripts/run_domain_gap.py --config configs/base.yaml --dummy --strategies scratch --epochs 1 --train_samples 32 --val_samples 16 --test_samples 16
```

## Where to Read Results

Start with the top-level CSV in each experiment output folder.

For training curves, prefer the split plots when an experiment has multiple
comparison axes:

```text
outputs/domain_gap/<dataset>_train_loss_curve.png
outputs/domain_gap/<dataset>_val_loss_curve.png
outputs/domain_gap/<dataset>_val_top1_acc_curve.png
outputs/domain_gap/<dataset>_val_macro_f1_curve.png

outputs/data_fraction/<strategy>_train_loss_curve.png
outputs/data_fraction/<strategy>_val_loss_curve.png
outputs/data_fraction/<strategy>_val_top1_acc_curve.png
outputs/data_fraction/<strategy>_val_macro_f1_curve.png
```

The unsplit training-curve plots are only kept for simpler experiments where
they remain readable, or as quick overview plots. For domain_gap and data_size,
prefer the split plots in the main report. Open each run's `metrics.json` for
raw epoch-level values.

For final test metrics of a single run, open `summary.json`.
