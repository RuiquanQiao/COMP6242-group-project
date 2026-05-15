# Code Interpretation

This document explains how the code works. It is written as a reading guide for
the project, not as a changelog.

The project studies transfer learning with ResNet18 on image classification
datasets. The code is designed around four questions:

1. How do different fine-tuning strategies perform on EuroSAT?
2. Does transfer learning help equally on EuroSAT and CIFAR-10?
3. Does transfer learning help more when EuroSAT has less training data?
4. How much ImageNet performance is forgotten after each downstream fine-tuning experiment?

## High-Level Flow

All experiments follow the same basic pipeline:

```text
YAML config
  -> script argument overrides
  -> DatasetConfig
  -> DataLoader
  -> ResNet18 model
  -> train / validate / save best checkpoint
  -> test evaluation
  -> JSON, CSV, and figure outputs
```

The code separates command-line scripts from reusable training code:

- `scripts/` contains terminal entry points.
- `src/transfer_learning/` contains reusable Python modules.

The scripts decide which experiment to run. The `src` modules decide how data,
models, training, and evaluation work.

## Configuration: `configs/base.yaml`

`configs/base.yaml` stores defaults shared by all scripts.

Important fields:

- `seed`: controls Python and PyTorch randomness.
- `device`: `auto`, `cpu`, or a specific device string.
- `output_dir`: where outputs are written unless a script overrides it.
- `model.name`: kept as `resnet18` for documentation clarity.
- `dataset.name`: `eurosat`, `cifar10`, or `imagenet`.
- `dataset.root`: image root for EuroSAT, torchvision data root for CIFAR-10,
  or an ImageFolder-style ImageNet root.
- `dataset.metadata_csv`: EuroSAT metadata file.
- `dataset.train_samples`, `val_samples`, `test_samples`: optional sample limits.
- `training.strategy`: one of `scratch`, `linear_probe`, `partial_ft`, `full_ft`.
- `training.epochs`, `batch_size`, `lr`, `weight_decay`: optimizer and loop settings.
- `training.convergence_threshold`: validation accuracy threshold used to report convergence speed.

The sample limit values use this convention:

```yaml
train_samples: 0
val_samples: 0
test_samples: 0
```

`0` means "use all samples in that split". A positive number means "use exactly
that many samples if available". This is used when comparing EuroSAT and
CIFAR-10 with the same split sizes.

The current scripts run one seed at a time. They do not automatically repeat an
experiment across multiple random seeds or aggregate mean and standard deviation
statistics. To report repeated runs, the scripts would need to be run separately
with different seed values and the resulting CSV files aggregated.

## Config Loader: `src/transfer_learning/config.py`

This module is intentionally small.

`load_config(path)`:

- opens a YAML file
- parses it with `yaml.safe_load`
- wraps the resulting dictionary in a `Config` object

`Config` exposes a few common fields as properties:

- `seed`
- `device`
- `output_dir`

Other nested values are still accessed through `cfg.raw`. This keeps the code
simple and avoids a large configuration system.

## Data Module: `src/transfer_learning/data.py`

This module turns a dataset name and configuration into a PyTorch `DataLoader`.
It supports EuroSAT, CIFAR-10, ImageNet, and a dummy dataset for quick code-path
checks.

### `DatasetConfig`

`DatasetConfig` is a dataclass that stores all dataset-related settings after
they are read from YAML:

- dataset name
- root path
- number of classes
- image size
- number of DataLoader workers
- EuroSAT metadata path
- CIFAR-10 download flag
- validation split ratio
- optional sample limits

The function `dataset_cfg_from_raw(raw)` builds this dataclass from the raw YAML
dictionary.

### `EuroSatDataset`

`EuroSatDataset` is a custom PyTorch dataset for EuroSAT.

Each item in `self.rows` is:

```python
(image_path, label)
```

When `__getitem__` is called:

1. The image path is read from the row.
2. If the path is relative, it is joined with `dataset.root`.
3. The image is opened with PIL and converted to RGB.
4. The configured transform is applied.
5. The function returns `(image_tensor, label)`.

EuroSAT uses `metadata.csv` because the images are stored in class folders and
the project needs a fixed train/validation/test split.

### `DummyDataset`

`DummyDataset` returns random image tensors and deterministic labels.

It exists only to check that scripts, models, and output writing work without
requiring real data. It should not be used for real results.

### `build_dataloader(...)`

This is the public function used by the training code.

It receives:

- a `DatasetConfig`
- a split name: `train`, `val`, or `test`
- batch size
- random seed
- dummy flag

Then it chooses the correct dataset builder:

- `_build_eurosat(...)` for EuroSAT
- `_build_cifar10(...)` for CIFAR-10
- `_build_imagenet(...)` for ImageNet
- `DummyDataset` if `dummy=True`

Finally, it wraps the dataset in a PyTorch `DataLoader`.

Training loaders are shuffled. Validation and test loaders are not shuffled.

### `_build_transforms(...)`

This function creates the image preprocessing pipeline.

For training:

1. resize to `image_size`
2. random horizontal flip
3. random rotation
4. convert to tensor
5. normalize with ImageNet mean and standard deviation

For validation and testing:

1. resize
2. convert to tensor
3. normalize

ImageNet normalization is used because the transfer-learning model is initialized
from ImageNet pretraining.

### `_build_eurosat(...)`

This function builds a EuroSAT dataset for one split.

It:

1. checks that `metadata_csv` exists in the config
2. reads rows for the requested split using `_read_eurosat_rows`
3. optionally applies balanced sampling using `_balanced_sample`
4. returns a `EuroSatDataset`

Balanced sampling matters when the experiment asks for a subset, such as 10% or
30% of the training data. It reduces the chance that a random subset has a badly
skewed class distribution.

### `_build_cifar10(...)`

Torchvision's CIFAR-10 dataset has only official `train` and `test` splits.
This project also needs a validation split.

For `test`, the code uses the official CIFAR-10 test set.

For `train` and `val`, the code:

1. loads the official CIFAR-10 training set
2. shuffles its indices with the project seed
3. takes the first `val_ratio` portion as validation
4. uses the remaining indices as training
5. optionally applies balanced sampling
6. returns a `Subset`

This makes CIFAR-10 match the same train/validation/test structure as EuroSAT.

### `_build_imagenet(...)`

ImageNet is used by the forgetting analysis to evaluate whether ImageNet
pretraining performance drops after downstream fine-tuning.

The code expects a local ImageFolder-style directory with standard ImageNet WNID
class folders, for example:

```text
imagenet_root/val/n01440764/*.JPEG
imagenet_root/val/n01443537/*.JPEG
```

The `test` split is mapped to the same `val` directory because ImageNet test
labels are not publicly available in the usual benchmark setup. Optional sample
limits are applied with balanced sampling.

### `_balanced_sample(...)`

This helper takes items shaped like:

```python
(something, label)
```

For EuroSAT, `something` is an image path. For CIFAR-10, `something` is a dataset
index.

The function groups items by label, shuffles each class group, then repeatedly
takes one item from each class until the requested limit is reached. The final
selected list is shuffled again.

This is simple class-balanced sampling. It is not a replacement for a rigorous
sampling study, but it is appropriate for this course project and keeps subset
comparisons fairer.

## Model Module: `src/transfer_learning/model.py`

This module builds ResNet18 and controls which layers are trainable.

### `STRATEGIES`

The supported strategy names are:

```text
scratch
linear_probe
partial_ft
full_ft
```

These are the names used by the terminal scripts.

### `build_resnet18(...)`

This function creates a ResNet18 model.

If `pretrained=True`, it loads ImageNet weights:

```python
ResNet18_Weights.IMAGENET1K_V1
```

If `pretrained=False`, the model is randomly initialized.

The original ResNet18 classification layer is replaced only when the requested
number of classes differs from the model's current output size:

```python
if model.fc.out_features != num_classes:
    model.fc = nn.Linear(model.fc.in_features, num_classes)
```

This keeps the original pretrained 1000-class ImageNet head when the code builds
an ImageNet evaluation model, while still replacing the head for EuroSAT and
CIFAR-10.

### `configure_trainable_layers(...)`

This function implements the transfer-learning strategy.

It starts by freezing all parameters:

```python
for param in model.parameters():
    param.requires_grad = False
```

Then it selectively unfreezes parameters:

- `scratch`: all parameters are trainable.
- `linear_probe`: only `model.fc` is trainable.
- `partial_ft`: `model.fc` and `model.layer4` are trainable.
- `full_ft`: all parameters are trainable.

For ResNet18, `layer4` is the final convolutional stage. It is a natural choice
for partial fine-tuning because it is closest to the classifier and contains more
task-specific high-level features.

### `should_use_pretrained(...)`

This helper returns:

```python
strategy != "scratch"
```

So only `scratch` starts from random initialization. The other strategies start
from ImageNet weights.

## Evaluation Module: `src/transfer_learning/evaluate.py`

This module evaluates a model on one DataLoader.

### `evaluate(...)`

The function:

1. sets the model to evaluation mode
2. disables gradient computation with `@torch.inference_mode()`
3. loops over all batches
4. computes cross-entropy loss
5. collects predictions and targets
6. returns loss, top-1 accuracy, and macro-F1

Loss and accuracy are weighted by the number of samples, not by the number of
batches. This avoids bias when the final batch is smaller.

### `_macro_f1(...)`

This function computes F1 per class and then averages over classes.

Macro-F1 is useful here because it gives each class equal weight. That matters
for land-cover classification, where per-class behavior can be more informative
than accuracy alone.

## Device Module: `src/transfer_learning/device.py`

This module chooses the runtime device.

When `device: auto`, the priority is:

1. CUDA GPU
2. Apple MPS
3. CPU

`runtime.gpu_id` selects which CUDA GPU to use when CUDA is available.

`device_summary(...)` converts the selected device into a readable string for
the terminal logs.

## Training Module: `src/transfer_learning/train.py`

This is the central reusable training module.

### `RunArtifacts`

`RunArtifacts` records the three main files produced by a run:

- `best_ckpt`
- `metrics_json`
- `summary_json`

The scripts use this object to find outputs after training finishes.

### `train_main(...)`

This function runs one complete training job.

Its input is a loaded `Config`. Optional arguments are:

- `dummy`: use random data instead of real datasets
- `init_ckpt`: initialize the model from a previous checkpoint

The function does the following:

1. Sets the random seed.
2. Resolves the device.
3. Creates the output directory.
4. Builds the dataset config.
5. Builds train, validation, and test DataLoaders.
6. Builds ResNet18 with or without ImageNet weights.
7. Optionally loads `init_ckpt`.
8. Applies the selected freezing strategy.
9. Creates AdamW over trainable parameters only.
10. Trains for the configured number of epochs.
11. Evaluates on validation after each epoch.
12. Saves the checkpoint with the best validation accuracy.
13. Reloads the best checkpoint.
14. Evaluates on the test split.
15. Writes `metrics.json` and `summary.json`.

The best checkpoint is selected by validation top-1 accuracy.
The convergence indicator is the first epoch whose validation top-1 accuracy is
greater than or equal to `training.convergence_threshold`. If the run never
reaches the threshold, `convergence_epoch` is written as `null` in `summary.json`.

### `_train_one_epoch(...)`

This helper performs one training epoch.

For each batch, it:

1. moves images and labels to the selected device
2. clears gradients
3. computes logits
4. computes cross-entropy loss
5. backpropagates
6. steps the optimizer
7. accumulates sample-weighted loss

It returns the average training loss for the epoch.

### `evaluate_checkpoint(...)`

This function loads a checkpoint and evaluates it on a selected split.

It is used by:

- `scripts/eval.py`

The ImageNet forgetting script performs its own evaluation because it needs to
attach the fine-tuned backbone to the original 1000-class ImageNet head.

## Script Utilities: `scripts/experiment_utils.py`

This file contains small helpers shared by experiment scripts.

### `make_run_config(...)`

This function copies the base config and applies experiment-specific overrides:

- output directory
- dataset name
- data root
- metadata path
- strategy
- sample limits
- CIFAR-10 download flag
- epoch override

It returns a new `Config` object. This avoids modifying the original config while
running many experiments in a loop.

### `read_summary(...)`

Reads a `summary.json` file produced by `train_main`.

### `write_csv(...)`

Writes a list of dictionaries to CSV. Experiment scripts use it for final result
tables.

### `save_bar_chart(...)`

Creates grouped bar charts from CSV-style rows. It is used for accuracy, macro-F1,
and forgetting plots.

## Terminal Scripts

### `scripts/prepare_eurosat.py`

This script prepares EuroSAT metadata.

It can either:

- download EuroSAT through torchvision
- or scan an existing EuroSAT image folder

For each class folder, it:

1. finds image files
2. shuffles them with a fixed seed
3. splits them into train, validation, and test
4. writes rows to `metadata.csv`

The metadata file is what allows the rest of the project to use a stable split.

### `scripts/train.py`

This script runs one training job from the terminal.

It:

1. loads `configs/base.yaml`
2. applies command-line overrides
3. calls `train_main`
4. prints the paths of the checkpoint and output files

Example:

```bash
python scripts/train.py --config configs/base.yaml --strategy full_ft --output_dir outputs/single_full_ft
```

### `scripts/eval.py`

This script evaluates one saved checkpoint.

It:

1. loads the config
2. applies dataset and strategy overrides
3. calls `evaluate_checkpoint`
4. prints loss, top-1 accuracy, and macro-F1

Example:

```bash
python scripts/eval.py --config configs/base.yaml --strategy full_ft --ckpt outputs/single_full_ft/best.pt
```

### `scripts/run_ablation.py`

This runs Experiment 4.1.

It compares transfer strategies on EuroSAT:

```text
scratch
linear_probe
partial_ft
full_ft
```

For each strategy, it creates a run config and calls `train_main`.

Final outputs:

- `results.csv`
- `test_top1_acc.png`

### `scripts/run_domain_gap.py`

This runs Experiment 4.2.

It compares EuroSAT and CIFAR-10 using the same split sizes.

Default strategies:

```text
scratch
partial_ft
full_ft
```

Default split sizes:

```text
train: 18900
val: 4050
test: 4050
```

These numbers match a 70/15/15 split of EuroSAT's 27,000 images. CIFAR-10 is
sampled to the same sizes so the comparison focuses more on domain difference
than dataset size.

Final outputs:

- `results.csv`
- `test_top1_acc.png`
- `test_macro_f1.png`

### `scripts/run_data_fraction.py`

This runs Experiment 4.3.

It studies whether transfer learning helps more when EuroSAT has less training
data.

Default fractions:

```text
0.1, 0.3, 0.6, 1.0
```

For each fraction, the script limits the EuroSAT training set and compares:

```text
scratch
linear_probe
full_ft
```

It also computes transfer gain:

```text
transfer_gain = transfer_strategy_score - scratch_score
```

Final outputs:

- `results.csv`
- `transfer_gain.csv`
- `test_top1_acc.png`
- `transfer_gain_top1.png`

### `scripts/run_forgetting.py`

This runs the ImageNet catastrophic forgetting analysis for every downstream
transfer experiment in the report.

By default, the script runs three scenarios:

```text
eurosat_ablation  4.1 EuroSAT strategy ablation
domain_gap        4.2 EuroSAT and CIFAR-10 same-size comparison
data_fraction     4.3 EuroSAT 10% / 30% / 60% / 100% comparison
```

For each selected scenario and strategy, the script does:

```text
build the original ImageNet-pretrained ResNet18 with its 1000-class head
evaluate it on ImageNet validation data -> ImageNet_before
fine-tune a 10-class ResNet18 copy on the downstream dataset
load the fine-tuned backbone into a fresh 1000-class ImageNet model
evaluate it again on ImageNet validation data -> ImageNet_after
compute forgetting = ImageNet_before - ImageNet_after
```

The output also records downstream test performance, because a low forgetting
score is not useful if the model did not learn the downstream task.

The ImageNet head is not trained during downstream fine-tuning. It is restored
for evaluation so the metric reflects how much the fine-tuned backbone no longer
supports the original ImageNet classifier. This directly addresses forgetting of
the pretraining task rather than sequential transfer between EuroSAT and
CIFAR-10.

The default forgetting strategies are ImageNet-pretrained strategies only:

```text
linear_probe
partial_ft
full_ft
```

The `scratch` strategy is intentionally excluded because it does not start from
ImageNet pretraining, so ImageNet catastrophic forgetting is not defined for it.

Scenario coverage:

```text
4.1: EuroSAT full-data fine-tuning
4.2: EuroSAT same-size fine-tuning and CIFAR-10 same-size fine-tuning
4.3: EuroSAT 10% / 30% / 60% / 100% fine-tuning
```

Final outputs:

- `forgetting_results.csv`
- `forgetting_top1.png`
- `forgetting_macro_f1.png`
- `forgetting_by_fraction_top1.png`
- `forgetting_by_fraction_macro_f1.png`

## Output Files

Each individual training run writes:

```text
best.pt
metrics.json
summary.json
```

`best.pt` contains the model state with the best validation accuracy.

`metrics.json` contains one row per epoch:

- training loss
- validation loss
- validation top-1 accuracy
- validation macro-F1

`summary.json` contains final run-level values:

- dataset
- model
- strategy
- epochs
- parameter counts
- training time
- best validation accuracy
- convergence threshold
- first convergence epoch
- test loss
- test top-1 accuracy
- test macro-F1

Experiment scripts collect these summaries into CSV tables and plots.


