# COMP6242 EuroSAT Transfer Learning

这个仓库用于完成 `MobileNetV2 -> EuroSAT` 的迁移学习实验，并提供可直接复现的 5 组消融对比：

- `zero_shot`
- `from_scratch`
- `linear_probe`
- `partial_unfreeze`
- `full_finetune`

## 项目结构

```text
configs/
  base.yaml
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

## 安装

```bash
pip install -r requirements.txt
```

## 1) 数据准备

自动下载并生成 `metadata.csv`（推荐）：

```bash
python scripts/prepare_eurosat.py --root data --download --out_csv data/metadata.csv
```

如果你已经手动下载过 EuroSAT，可直接指定图片目录：

```bash
python scripts/prepare_eurosat.py --images_root "E:/datasets/EuroSAT/2750" --out_csv data/metadata.csv
```

`metadata.csv` 字段：

- `image_path`：相对 `dataset.root` 的图片路径
- `label`：类别 id
- `label_name`：类别名
- `split`：`train/val/test`

## 2) 单策略训练

```bash
python scripts/train.py --config configs/base.yaml --strategy linear_probe
```

不改 `base.yaml`，直接在命令行指定本次策略：

```bash
python scripts/train.py --config configs/base.yaml --strategy linear_probe
python scripts/train.py --config configs/base.yaml --strategy from_scratch --epochs 12 --output_dir outputs/eurosat_mobilenetv2/from_scratch
```

## 3) 单策略评估

```bash
python scripts/eval.py --config configs/base.yaml --split test --strategy linear_probe
```

评估时也可覆盖策略（用于无 checkpoint 的即时对比）：

```bash
python scripts/eval.py --config configs/base.yaml --split test --strategy from_scratch
```

## 4) Zero-shot 基线

```bash
python scripts/zero_shot.py --config configs/base.yaml --split test
```

## 5) 一键跑五组消融

```bash
python scripts/run_ablation.py --config configs/base.yaml
```

只跑部分策略：

```bash
python scripts/run_ablation.py --config configs/base.yaml --strategies from_scratch,linear_probe,full_finetune
```

输出结果会保存在：

- `outputs/eurosat_mobilenetv2/<strategy>/summary.json`
- `outputs/eurosat_mobilenetv2/ablation_results.csv`

## 配置说明

`configs/base.yaml` 重点字段：

- `device`：建议保持 `auto`，会自动优先使用 GPU（CUDA），否则回退 CPU
- `runtime.gpu_id`：多卡机器可指定卡号（如 `0`、`1`）
- `training.strategy`：默认占位为 `__CLI__`，需通过命令行 `--strategy` 显式指定
- `training.partial_blocks`：`partial_unfreeze` 时解冻末端 block 数量
- `dataset.num_classes`：EuroSAT 为 `10`
