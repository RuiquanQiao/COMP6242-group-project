# COMP6242 EuroSAT Transfer Learning

This repository runs a `MobileNetV2 -> EuroSAT` transfer learning study with 5 reproducible strategies:

- `zero_shot`
- `from_scratch`
- `linear_probe`
- `partial_unfreeze`
- `full_finetune`

## Project Structure

```text
configs/
  base.yaml
eurosat_experiments.ipynb
scripts/
  prepare_eurosat.py
  train.py
  eval.py
  zero_shot.py
  run_ablation.py
src/
  eurosat_baseline/
    config.py
    data.py
    model.py
    evaluate.py
    train.py
```

Optional: run the full workflow in notebook form (with saved logs and summary outputs):

```bash
eurosat_experiments.ipynb
```

## Terminal-First Workflow (Recommended)

No notebook required. No file edits required.

Run one strategy:

```bash
python scripts/train.py --config configs/base.yaml --strategy from_scratch --output_dir outputs/eurosat_mobilenetv2/from_scratch
python scripts/eval.py --config configs/base.yaml --split test --strategy from_scratch --ckpt outputs/eurosat_mobilenetv2/from_scratch/best.pt
```

Run each strategy explicitly:

```bash
python scripts/train.py --config configs/base.yaml --strategy zero_shot --output_dir outputs/eurosat_mobilenetv2/zero_shot
python scripts/train.py --config configs/base.yaml --strategy from_scratch --output_dir outputs/eurosat_mobilenetv2/from_scratch
python scripts/train.py --config configs/base.yaml --strategy linear_probe --output_dir outputs/eurosat_mobilenetv2/linear_probe
python scripts/train.py --config configs/base.yaml --strategy partial_unfreeze --output_dir outputs/eurosat_mobilenetv2/partial_unfreeze
python scripts/train.py --config configs/base.yaml --strategy full_finetune --output_dir outputs/eurosat_mobilenetv2/full_finetune
```

## Installation

Recommended GPU environment (CUDA 12.4):

```bash
pip install -r requirements-cu124.txt
pip install -r requirements.txt
```

CPU environment:

```bash
pip install -r requirements-cpu.txt
pip install -r requirements.txt
```

Verify CUDA availability in PyTorch:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 1) Data Preparation

Download EuroSAT and generate `metadata.csv`:

```bash
python scripts/prepare_eurosat.py --root data --download --out_csv data/metadata.csv
```

If EuroSAT is already downloaded, provide the local image root directly:

```bash
python scripts/prepare_eurosat.py --images_root "E:/datasets/EuroSAT/2750" --out_csv data/metadata.csv
```

`metadata.csv` columns:

- `image_path`: image path relative to `dataset.root`
- `label`: integer class id
- `label_name`: class name
- `split`: `train/val/test`

## 2) Single-Strategy Training

```bash
python scripts/train.py --config configs/base.yaml --strategy linear_probe
```

CLI overrides (no `base.yaml` edits needed):

```bash
python scripts/train.py --config configs/base.yaml --strategy linear_probe
python scripts/train.py --config configs/base.yaml --strategy from_scratch --epochs 12 --output_dir outputs/eurosat_mobilenetv2/from_scratch
```

## 3) Single-Strategy Evaluation

```bash
python scripts/eval.py --config configs/base.yaml --split test --strategy linear_probe
```

Strategy can also be overridden during evaluation:

```bash
python scripts/eval.py --config configs/base.yaml --split test --strategy from_scratch
```

## 4) Zero-Shot Baseline

```bash
python scripts/zero_shot.py --config configs/base.yaml --split test
```

## 5) Run Five-Strategy Ablation

```bash
python scripts/run_ablation.py --config configs/base.yaml
```

Run only selected strategies:

```bash
python scripts/run_ablation.py --config configs/base.yaml --strategies from_scratch,linear_probe,full_finetune
```

Output files:

- `outputs/eurosat_mobilenetv2/<strategy>/summary.json`
- `outputs/eurosat_mobilenetv2/ablation_results.csv`

## Config Notes

Key fields in `configs/base.yaml`:

- `device`: keep `auto` to prefer GPU (CUDA) and fallback to CPU
- `runtime.gpu_id`: GPU index on multi-GPU machines (`0`, `1`, ...)
- `training.strategy`: default placeholder is `__CLI__`; pass `--strategy` explicitly in CLI
- `training.partial_blocks`: number of tail blocks to unfreeze for `partial_unfreeze`
- `dataset.num_classes`: `10` for EuroSAT
