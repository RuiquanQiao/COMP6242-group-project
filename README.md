# COMP6242 MobileNetV2 Baseline Starter

这个仓库是一个先行原型：在还没下载 EGTEA Gaze+ 的情况下，先把训练/评估流程系统化搭好。

## 功能

- `train`：微调 `MobileNetV2`（支持只训练分类头 or 全量微调）。
- `eval`：在验证/测试集上评估 Top-1 准确率。
- `zero-shot` 基线：冻结 backbone，仅用随机初始化分类头直接评估（模拟“啥也不训练”）。
- `dummy` 数据模式：即使没有真实数据，也能先把工程管线跑通。

## 目录结构

```text
configs/
  base.yaml
scripts/
  train.py
  eval.py
  zero_shot.py
src/
  egtea_baseline/
    __init__.py
    config.py
    data.py
    model.py
    train.py
    evaluate.py
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 数据准备（后续）

当前代码支持从一个 `metadata.csv` 读取数据，字段如下：

- `image_path`：图片路径（相对 `dataset.root` 或绝对路径）
- `label`：整数类别 id
- `split`：`train` / `val` / `test`

等你把 EGTEA 下载好，我们再补一个视频解帧脚本，把数据整理成这个格式即可。

## 快速运行（无真实数据）

```bash
python scripts/train.py --config configs/base.yaml --dummy
python scripts/eval.py --config configs/base.yaml --dummy
python scripts/zero_shot.py --config configs/base.yaml --dummy
```

## 使用真实数据

1. 修改 `configs/base.yaml` 里的 `dataset.root` 和 `dataset.metadata_csv`。
2. 去掉命令中的 `--dummy`。
3. 训练与评估：

```bash
python scripts/train.py --config configs/base.yaml
python scripts/eval.py --config configs/base.yaml
python scripts/zero_shot.py --config configs/base.yaml
```
