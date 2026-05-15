# COMP6242 Transfer Learning Experiments

Terminal-first code for the ResNet18 transfer-learning experiments in `report.md`.

Experiments covered:

- 4.1 EuroSAT strategy ablation
- 4.2 EuroSAT vs CIFAR-10 same-size domain comparison
- 4.3 EuroSAT data-size comparison
- ImageNet forgetting analysis after every downstream transfer experiment

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

## Strategies

All experiments use ResNet18.

```text
scratch       random initialization, train all layers
linear_probe  ImageNet initialization, train classifier only
partial_ft    ImageNet initialization, train layer4 + classifier
full_ft       ImageNet initialization, train all layers
```

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

Default same-size split:

```text
train=18900, val=4050, test=4050
```

Outputs:

```text
outputs/domain_gap/results.csv
outputs/domain_gap/test_top1_acc.png
outputs/domain_gap/test_macro_f1.png
outputs/domain_gap/val_top1_acc_curve.png
outputs/domain_gap/val_macro_f1_curve.png
```

### 4.3 EuroSAT Data-Size Comparison

```bash
python scripts/run_data_fraction.py --config configs/base.yaml
```

Outputs:

```text
outputs/data_fraction/results.csv
outputs/data_fraction/transfer_gain.csv
outputs/data_fraction/test_top1_acc.png
outputs/data_fraction/transfer_gain_top1.png
outputs/data_fraction/val_top1_acc_curve.png
outputs/data_fraction/val_macro_f1_curve.png
```

### ImageNet Forgetting Analysis

```bash
python scripts/run_forgetting.py --config configs/base.yaml --imagenet_root "PATH/TO/IMAGENET" --download_cifar
```

This script measures how much ImageNet validation performance drops after an
ImageNet-pretrained ResNet18 is fine-tuned on each downstream transfer setting.
ImageNet is always the retained pretraining task. EuroSAT and CIFAR-10 are
downstream fine-tuning tasks.

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
ImageNet pretraining to forget.

For quick checks or limited compute, evaluate on a balanced ImageNet subset:

```bash
python scripts/run_forgetting.py --config configs/base.yaml --imagenet_root "PATH/TO/IMAGENET" --imagenet_samples 5000 --download_cifar
```

Run a subset of scenarios with:

```bash
python scripts/run_forgetting.py --config configs/base.yaml --scenarios domain_gap --imagenet_root "PATH/TO/IMAGENET" --download_cifar
python scripts/run_forgetting.py --config configs/base.yaml --scenarios data_fraction --imagenet_root "PATH/TO/IMAGENET"
```

Outputs:

```text
outputs/forgetting/forgetting_results.csv
outputs/forgetting/forgetting_top1.png
outputs/forgetting/forgetting_macro_f1.png
outputs/forgetting/forgetting_by_fraction_top1.png
outputs/forgetting/forgetting_by_fraction_macro_f1.png
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

