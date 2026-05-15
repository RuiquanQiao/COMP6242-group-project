# Modification Report

This report summarizes the current local codebase changes compared with the
public GitHub version at:

```text
https://github.com/RuiquanQiao/COMP6242-group-project
```

At the time of review, the GitHub version still shows the earlier
MobileNetV2-to-EuroSAT baseline workflow. The local version has been refactored
for the ResNet18 transfer-learning experiments described in `report.md`.

## 1. Main Changes Compared with the GitHub Version

The remote GitHub version is centered on:

- MobileNetV2
- EuroSAT only
- five strategies: `zero_shot`, `from_scratch`, `linear_probe`,
  `partial_unfreeze`, and `full_finetune`
- package path `src/eurosat_baseline`
- one main ablation workflow

The local version is centered on:

- ResNet18
- EuroSAT and CIFAR-10
- four strategies: `scratch`, `linear_probe`, `partial_ft`, and `full_ft`
- package path `src/transfer_learning`
- separate terminal scripts for each report experiment

Major structural changes:

- Renamed the reusable source package from `eurosat_baseline` to
  `transfer_learning`.
- Removed the old zero-shot workflow because it is not part of the current
  report experiments.
- Replaced the MobileNetV2-specific model code with ResNet18-specific transfer
  learning code.
- Added CIFAR-10 support through `torchvision.datasets.CIFAR10`.
- Added fixed-size and balanced sampling support for fair dataset-size
  comparisons.
- Added dedicated scripts for:
  - EuroSAT strategy ablation
  - EuroSAT vs CIFAR-10 domain-gap comparison
  - EuroSAT data-fraction comparison
  - forgetting analysis
- Rewrote `README.md` as a concise experiment-running guide.
- Rewrote `interpretation.md` as an internal code-reading document.

## 2. Newly Added Changes in the Latest Update

The latest update added two report-facing features.

### 2.1 Transfer Gain Line Chart for Experiment 4.3

Experiment 4.3 now produces a transfer-gain line chart:

```text
outputs/data_fraction/transfer_gain_top1.png
```

This complements:

```text
outputs/data_fraction/results.csv
outputs/data_fraction/transfer_gain.csv
outputs/data_fraction/test_top1_acc.png
```

The line chart plots transfer gain over EuroSAT training-data fractions.

Transfer gain is computed as:

```text
transfer_gain = transfer_strategy_score - scratch_score
```

### 2.2 Convergence Metric in Training Summary

Each training run now records a simple convergence-speed metric in
`summary.json`.

The new config field is:

```yaml
training:
  convergence_threshold: 0.9
```

The new summary fields are:

```json
"convergence_threshold": 0.9,
"convergence_epoch": 3
```

`convergence_epoch` means the first epoch where:

```text
val_top1_acc >= convergence_threshold
```

If the model never reaches the threshold, the value is written as `null`.

## 3. Files Involved

Core source files:

```text
src/transfer_learning/config.py
src/transfer_learning/data.py
src/transfer_learning/device.py
src/transfer_learning/evaluate.py
src/transfer_learning/model.py
src/transfer_learning/train.py
```

Experiment scripts:

```text
scripts/prepare_eurosat.py
scripts/train.py
scripts/eval.py
scripts/run_ablation.py
scripts/run_domain_gap.py
scripts/run_data_fraction.py
scripts/run_forgetting.py
scripts/experiment_utils.py
```

Configuration and documentation:

```text
configs/base.yaml
README.md
interpretation.md
requirements.txt
requirements-cpu.txt
requirements-cu124.txt
modification_report.md
```

Key latest-update files:

```text
configs/base.yaml
src/transfer_learning/train.py
scripts/experiment_utils.py
scripts/run_data_fraction.py
README.md
interpretation.md
```

## 4. Experiment Coverage

### Experiment 4.1: EuroSAT Strategy Ablation

Script:

```bash
python scripts/run_ablation.py --config configs/base.yaml
```

Covered strategies:

```text
scratch
linear_probe
partial_ft
full_ft
```

Outputs:

```text
outputs/eurosat_ablation/results.csv
outputs/eurosat_ablation/test_top1_acc.png
```

Each strategy also writes:

```text
best.pt
metrics.json
summary.json
```

### Experiment 4.2: EuroSAT vs CIFAR-10 Same-Size Comparison

Script:

```bash
python scripts/run_domain_gap.py --config configs/base.yaml --download_cifar
```

Covered datasets:

```text
EuroSAT
CIFAR-10
```

Default strategies:

```text
scratch
partial_ft
full_ft
```

Default same-size split:

```text
train=18900
val=4050
test=4050
```

Outputs:

```text
outputs/domain_gap/results.csv
outputs/domain_gap/test_top1_acc.png
outputs/domain_gap/test_macro_f1.png
```

### Experiment 4.3: EuroSAT Data-Size Comparison

Script:

```bash
python scripts/run_data_fraction.py --config configs/base.yaml
```

Default fractions:

```text
0.1
0.3
0.6
1.0
```

Default strategies:

```text
scratch
linear_probe
full_ft
```

Outputs:

```text
outputs/data_fraction/results.csv
outputs/data_fraction/transfer_gain.csv
outputs/data_fraction/test_top1_acc.png
outputs/data_fraction/transfer_gain_top1.png
```

### Forgetting Analysis

Script:

```bash
python scripts/run_forgetting.py --config configs/base.yaml --download_cifar
```

Current task order:

```text
Task A: EuroSAT
Task B: CIFAR-10
```

Reported metrics:

```text
A_before top-1 / macro-F1
A_after top-1 / macro-F1
forgetting top-1 / macro-F1
CIFAR-10 final top-1 / macro-F1
```

Outputs:

```text
outputs/forgetting/forgetting_results.csv
outputs/forgetting/forgetting_top1.png
```

## 5. Remaining Notes

- The code has passed static Python compilation with:

  ```bash
  python -m compileall src scripts
  ```

- Full training was not executed in the current local environment because PyTorch
  previously failed during DLL loading on this machine. This is an environment
  issue, not a Python syntax issue.

- The current forgetting implementation uses a single shared classifier head.
  Since EuroSAT and CIFAR-10 both have 10 classes but different class meanings,
  forgetting results should be interpreted as a single-head sequential transfer
  setting.

- The code does not currently implement repeated runs over multiple random
  seeds. This was intentionally left out based on the current project scope.

- EuroSAT is treated as 27,000 images in the code. The original report outline
  says EuroSAT(30k), which should be interpreted as approximate or updated in
  the final report.

- `eurosat_experiments.ipynb` is an older notebook workflow and still reflects
  the previous project structure. The current maintained workflow is terminal
  based.
