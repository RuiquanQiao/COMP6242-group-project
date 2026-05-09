# Codebase Structure

```text
COMP6242-group-project
├─configs
│  └─base.yaml                  # 全局默认配置：数据路径、训练超参数、策略占位符、评估 split
├─scripts
│  ├─prepare_eurosat.py         # 生成 metadata.csv，并按类别切分 train/val/test
│  ├─train.py                   # 训练入口：加载配置、应用 CLI 覆盖、调用 train_main
│  ├─eval.py                    # 评估入口：加载模型/ckpt，在指定 split 上输出指标
│  ├─zero_shot.py               # 零训练基线：冻结参数后直接评估
│  └─run_ablation.py            # 批量跑多策略并汇总为 ablation_results.csv
├─src
│  └─eurosat_baseline
│      ├─config.py              # Config 封装与 YAML 加载
│      ├─data.py                # Dataset/DataLoader、图像变换、metadata 读取
│      ├─device.py              # 设备解析：CUDA/MPS/CPU 选择与设备信息展示
│      ├─evaluate.py            # 评估循环与指标计算（loss/top1/macro-f1）
│      ├─model.py               # MobileNetV2 构建与五策略冻结/解冻规则
│      ├─train.py               # 训练主循环、best checkpoint 选择、summary 输出
│      └─__init__.py            # 包初始化文件
├─requirements*.txt             # 依赖清单（CPU/CUDA 的 torch 安装入口 + 通用库）
└─eurosat_experiments.ipynb     # Notebook 版实验流程（调用同一套 src 代码）
```

# Codebase Interpretation

## configs

### `base.yaml`

```yaml
# 文件路径: configs/base.yaml
# 固定随机种子；训练脚本会据此设置 Python/Torch/CUDA 的随机状态
seed: 42
# 设备自动选择；优先 CUDA，其次 MPS，最后 CPU
device: "auto"
# 所有训练产物默认输出目录（可被 CLI 覆盖）
output_dir: "outputs/eurosat_mobilenetv2"
# 运行时配置区：例如指定多卡机器上的 GPU 索引
runtime:
  # 当 device=auto 且 CUDA 可用时，使用该 GPU 编号
  gpu_id: 0

# 数据配置区
dataset:
  # 数据图片根目录；metadata.csv 里的相对路径会拼接到这里
  root: "data/eurosat/2750"
  # 由 prepare_eurosat.py 生成，保存 image_path/label/split
  metadata_csv: "data/metadata.csv"
  # EuroSAT 类别数，决定分类头输出维度
  num_classes: 10
  # 训练前统一 resize 到该尺寸
  image_size: 224
  # DataLoader 并行读取进程数
  num_workers: 4

# 训练配置区
training:
  # 训练策略；这里故意设为 __CLI__，强制命令行显式指定
  strategy: "__CLI__"  # must pass --strategy in CLI; options: zero_shot | from_scratch | linear_probe | partial_unfreeze | full_finetune
  # partial_unfreeze 策略下解冻末尾 block 数量
  partial_blocks: 2
  # 训练 epoch 总轮数
  epochs: 8
  # 训练与评估 batch 大小
  batch_size: 32
  # AdamW 学习率
  lr: 0.0003
  # AdamW 权重衰减
  weight_decay: 0.0001

# 评估配置区
evaluation:
  # 默认评估集（通常为 test）
  split: "test"
```

## scripts

### `prepare_eurosat.py`

```python
# 文件路径: scripts/prepare_eurosat.py
# 启用延迟注解，改善类型标注兼容性
from __future__ import annotations

# 命令行参数解析
import argparse
# 生成 metadata.csv
import csv
# split 前打乱样本顺序
import random
# 跨平台路径处理
from pathlib import Path

# 官方数据下载入口
from torchvision.datasets import EuroSAT

# 允许纳入索引的图像后缀名
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# 定义 CLI 参数：路径、下载、划分比例、随机种子、输出文件
def parse_args() -> argparse.Namespace:
    # 构建脚本说明，便于 `-h` 查看帮助
    parser = argparse.ArgumentParser(description="Prepare EuroSAT metadata and split files.")
    # 数据根目录（下载或扫描都以此为基准）
    parser.add_argument("--root", type=str, default="data", help="Data root directory.")
    parser.add_argument(
        # 若已下载数据，可直接指定图片目录，跳过自动探测
        "--images_root",
        type=str,
        default="",
        help="Optional path to EuroSAT class folders. Auto-discovered if empty.",
    )
    # 开关：是否用 torchvision 自动下载 EuroSAT
    parser.add_argument("--download", action="store_true", help="Download EuroSAT via torchvision.")
    # 验证集比例（按类别内切分）
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio.")
    # 测试集比例（按类别内切分）
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio.")
    # 划分随机种子，确保 split 可复现
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        # metadata 输出路径
        "--out_csv",
        type=str,
        default="data/metadata.csv",
        help="Output metadata CSV path.",
    )
    return parser.parse_args()


# 主流程入口
def main() -> None:
    args = parse_args()
    root = Path(args.root)
    # 确保数据根目录存在
    root.mkdir(parents=True, exist_ok=True)
    # 需要时执行下载
    if args.download:
        EuroSAT(root=str(root), download=True)
        print("EuroSAT downloaded.")

    # 优先使用用户显式给定的图片目录
    if args.images_root:
        images_root = Path(args.images_root)
    else:
        # 否则自动探测常见 2750 目录位置
        images_root = auto_find_images_root(root)

    # 目录不存在直接报错，避免静默失败
    if not images_root.exists():
        # 找不到时明确报错并提示使用 --images_root
        raise FileNotFoundError(f"images root not found: {images_root}")

    # 读取并排序类别文件夹，确保标签映射稳定
    class_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found under: {images_root}")

    # 构建局部随机数发生器，避免污染全局状态
    rng = random.Random(args.seed)
    rows: list[dict[str, str | int]] = []
    # 按目录顺序赋予类别 id
    for label, class_dir in enumerate(class_dirs):
        images = sorted(
            [
                p
                # 递归收集该类别下全部图像
                for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ]
        )
        # 某个类别空目录则跳过
        if not images:
            continue
        # 类别内部打乱再切分，减少顺序偏差
        rng.shuffle(images)
        n = len(images)
        # 测试集样本数
        n_test = int(n * args.test_ratio)
        # 验证集样本数
        n_val = int(n * args.val_ratio)
        # 训练集样本数取剩余
        n_train = n - n_val - n_test
        # 遍历该类别每张图像并打 split 标签
        for idx, image_path in enumerate(images):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            rows.append(
                {
                    # 保存相对路径，便于跨机器迁移
                    "image_path": str(image_path.relative_to(images_root)).replace("\\", "/"),
                    # 保存整数类别 id
                    "label": label,
                    # 保存类别名，便于结果展示
                    "label_name": class_dir.name,
                    # 保存 train/val/test
                    "split": split,
                }
            )

    out_csv = Path(args.out_csv)
    # 确保 metadata 输出目录存在
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        # 按固定列名写 CSV，后续 dataloader 依赖这些列
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "label_name", "split"])
        # 先写表头
        writer.writeheader()
        # 一次写入所有样本
        writer.writerows(rows)

    # 打印数据目录和生成统计，便于核验
    print(f"images_root={images_root}")
    print(f"metadata={out_csv} rows={len(rows)} classes={len(class_dirs)}")


# 自动探测 EuroSAT 图像根目录
def auto_find_images_root(root: Path) -> Path:
    # 先尝试常见目录模式
    candidates = [
        root / "eurosat" / "2750",
        root / "EuroSAT" / "2750",
        root / "2750",
    ]
    for c in candidates:
        if c.exists():
            return c
    # 再全局递归搜索名为 2750 的目录
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() == "2750":
            return path
    # 若已下载数据，可直接指定图片目录，跳过自动探测
    raise FileNotFoundError("Cannot auto-detect EuroSAT image folder. Please provide --images_root.")


if __name__ == "__main__":
    main()
```

### `train.py`

```python
# 文件路径: scripts/train.py
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

# 计算项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 将 src 加入模块搜索路径
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 导入配置加载组件
from eurosat_baseline.config import Config, load_config
# 导入训练主函数
from eurosat_baseline.train import train_main

# 允许的训练策略白名单
STRATEGIES = ["zero_shot", "from_scratch", "linear_probe", "partial_unfreeze", "full_finetune"]
# 特殊标记：表示必须由 CLI 指定策略
UNSET_STRATEGY = "__CLI__"


# 解析训练脚本参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 baseline.")
    # 必填：YAML 配置文件路径
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    # 开关：是否使用随机假数据快速 smoke test
    parser.add_argument("--dummy", action="store_true", help="Use dummy synthetic data.")
    parser.add_argument(
        # 覆盖 training.strategy
        "--strategy",
        type=str,
        default="",
        choices=STRATEGIES,
        help="Override training.strategy from config.",
    )
    # 覆盖训练轮数
    parser.add_argument("--epochs", type=int, default=0, help="Override training.epochs.")
    parser.add_argument(
        # 覆盖输出目录
        "--output_dir",
        type=str,
        default="",
        help="Override output_dir for this run.",
    )
    return parser.parse_args()


# 脚本入口：加载配置并启动训练
def main() -> None:
    args = parse_args()
    # 先应用 CLI 覆盖，确保策略明确
    cfg = apply_overrides(load_config(args.config), args)
    # 调用核心训练流程
    artifacts = train_main(cfg, dummy=args.dummy)
    # 输出最佳模型路径
    print(f"best checkpoint: {artifacts.best_ckpt}")
    # 输出逐 epoch 指标路径
    print(f"metrics log: {artifacts.metrics_json}")
    # 输出总览指标路径
    print(f"summary: {artifacts.summary_json}")


# 对配置进行深拷贝后覆盖，避免污染原对象
def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    # 深拷贝原始配置字典
    raw = deepcopy(cfg.raw)
    # CLI 传了策略就覆盖配置
    if args.strategy:
        raw["training"]["strategy"] = args.strategy
    # 读取当前策略
    current = str(raw["training"].get("strategy", UNSET_STRATEGY))
    # 未指定策略则报错，防止误跑
    if current == UNSET_STRATEGY:
        raise ValueError(
            # 覆盖 training.strategy
            "training.strategy is unset. Please pass --strategy "
            "(zero_shot|from_scratch|linear_probe|partial_unfreeze|full_finetune)."
        )
    # 仅接受正整数 epoch 覆盖
    if args.epochs and args.epochs > 0:
        raw["training"]["epochs"] = int(args.epochs)
    # 输出目录覆盖
    if args.output_dir:
        raw["output_dir"] = args.output_dir
    # 返回新的配置对象
    return Config(raw=raw)


if __name__ == "__main__":
    main()
```

### `eval.py`

```python
# 文件路径: scripts/eval.py
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

# 载入 checkpoint 需要 torch.load
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_baseline.config import Config, load_config
# 构建评估数据加载器
from eurosat_baseline.data import DatasetConfig, build_dataloader
# 设备选择与日志展示
from eurosat_baseline.device import device_summary, resolve_device
# 评估函数
from eurosat_baseline.evaluate import evaluate
# 模型构建与策略一致性设置
from eurosat_baseline.model import build_mobilenetv2, configure_trainable_layers

# 允许覆盖的策略白名单
STRATEGIES = ["zero_shot", "from_scratch", "linear_probe", "partial_unfreeze", "full_finetune"]
UNSET_STRATEGY = "__CLI__"


# 解析评估参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    # 可选：模型权重路径
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint. Optional.")
    # 可选：覆盖 val/test
    parser.add_argument("--split", type=str, default="", help="Override split: val/test.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="",
        choices=STRATEGIES,
        help="Override training.strategy from config.",
    )
    parser.add_argument("--dummy", action="store_true", help="Use dummy synthetic data.")
    return parser.parse_args()


# 评估入口
def main() -> None:
    args = parse_args()
    # 应用策略覆盖并做校验
    cfg = apply_overrides(load_config(args.config), args)
    raw = cfg.raw
    ds = raw["dataset"]
    # CLI split 优先于配置
    split = args.split or raw["evaluation"]["split"]
    # 解析运行设备
    device = resolve_device(raw)
    # 输出设备信息
    print(f"runtime device: {device_summary(device)}")

    # 整理数据配置结构
    ds_cfg = DatasetConfig(
        root=Path(ds["root"]),
        metadata_csv=Path(ds["metadata_csv"]),
        num_classes=int(ds["num_classes"]),
        image_size=int(ds["image_size"]),
        num_workers=int(ds["num_workers"]),
    )
    # 构建指定 split 的 dataloader
    loader = build_dataloader(
        dataset_cfg=ds_cfg,
        split=split,
        batch_size=int(raw["training"]["batch_size"]),
        dummy=args.dummy,
    )
    # 获取当前策略
    strategy = str(raw["training"].get("strategy", "linear_probe"))
    # 构建模型并决定是否用 ImageNet 预训练
    model = build_mobilenetv2(
        num_classes=ds_cfg.num_classes,
        pretrained=(strategy != "from_scratch"),
    ).to(device)
    # 配置可训练层（保持与训练时结构一致）
    configure_trainable_layers(
        model=model,
        strategy=strategy,
        partial_blocks=int(raw["training"].get("partial_blocks", 2)),
    )

    # 传了权重则严格加载
    if args.ckpt:
        # 读取 checkpoint 字典
        state = torch.load(args.ckpt, map_location="cpu")
        # 严格匹配参数键，避免遗漏
        model.load_state_dict(state["model"], strict=True)

    # 执行评估
    metrics = evaluate(model, loader, device)
    # 输出核心指标
    print(
        f"split={split} loss={metrics['loss']:.4f} "
        f"top1={metrics['top1_acc']:.4f} macro_f1={metrics['macro_f1']:.4f}"
    )


# 策略覆盖逻辑（与 train 脚本一致）
def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    raw = deepcopy(cfg.raw)
    if args.strategy:
        raw["training"]["strategy"] = args.strategy
    current = str(raw["training"].get("strategy", UNSET_STRATEGY))
    if current == UNSET_STRATEGY:
        raise ValueError(
            "training.strategy is unset. Please pass --strategy "
            "(zero_shot|from_scratch|linear_probe|partial_unfreeze|full_finetune)."
        )
    return Config(raw=raw)


if __name__ == "__main__":
    main()
```

### `zero_shot.py`

```python
# 文件路径: scripts/zero_shot.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_baseline.config import load_config
from eurosat_baseline.data import DatasetConfig, build_dataloader
from eurosat_baseline.device import device_summary, resolve_device
from eurosat_baseline.evaluate import evaluate
# 构建模型并设置 zero-shot 冻结策略
from eurosat_baseline.model import build_mobilenetv2, configure_trainable_layers


# 解析 zero-shot 评估参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot-like baseline evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--split", type=str, default="", help="Override split: val/test.")
    parser.add_argument("--dummy", action="store_true", help="Use dummy synthetic data.")
    return parser.parse_args()


# zero-shot 入口
def main() -> None:
    args = parse_args()
    # 读取配置
    cfg = load_config(args.config)
    raw = cfg.raw
    ds = raw["dataset"]
    # 支持 CLI 覆盖评估 split
    split = args.split or raw["evaluation"]["split"]
    # 解析设备
    device = resolve_device(raw)
    print(f"runtime device: {device_summary(device)}")

    ds_cfg = DatasetConfig(
        root=Path(ds["root"]),
        metadata_csv=Path(ds["metadata_csv"]),
        num_classes=int(ds["num_classes"]),
        image_size=int(ds["image_size"]),
        num_workers=int(ds["num_workers"]),
    )
    # 构建评估集加载器
    loader = build_dataloader(
        dataset_cfg=ds_cfg,
        split=split,
        batch_size=int(raw["training"]["batch_size"]),
        dummy=args.dummy,
    )

    # 构建带预训练权重的分类模型
    model = build_mobilenetv2(num_classes=ds_cfg.num_classes).to(device)
    # 核心：冻结全部参数，不做训练更新
    configure_trainable_layers(model=model, strategy="zero_shot")
    # 直接在评估集上前向计算指标
    metrics = evaluate(model, loader, device)
    print(
        f"[zero-shot] split={split} loss={metrics['loss']:.4f} "
        f"top1={metrics['top1_acc']:.4f} macro_f1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
```

### `run_ablation.py`

```python
# 文件路径: scripts/run_ablation.py
from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_baseline.config import Config, load_config
from eurosat_baseline.train import train_main

# 默认按顺序运行五种策略
DEFAULT_STRATEGIES = [
    "zero_shot",
    "from_scratch",
    "linear_probe",
    "partial_unfreeze",
    "full_finetune",
]


# 解析消融脚本参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transfer-learning ablation on EuroSAT.")
    parser.add_argument("--config", type=str, required=True, help="Path to base YAML config.")
    parser.add_argument(
        # 支持逗号分隔传入子集策略
        "--strategies",
        type=str,
        default="",
        help="Comma-separated list. Default: zero_shot,from_scratch,linear_probe,partial_unfreeze,full_finetune",
    )
    parser.add_argument("--dummy", action="store_true", help="Use dummy synthetic data.")
    return parser.parse_args()


# 消融入口：循环训练并汇总
def main() -> None:
    args = parse_args()
    # 读取基础配置
    base_cfg = load_config(args.config)
    # 解析并校验策略列表
    strategies = parse_strategies(args.strategies)
    rows: list[dict[str, str | float | int]] = []
    # 消融汇总输出根目录
    base_out = Path(base_cfg.raw["output_dir"])

    # 逐策略执行训练
    for strategy in strategies:
        # 构造当前策略的专用配置
        cfg = with_strategy(base_cfg, strategy=strategy, output_dir=base_out / strategy)
        print(f"\n=== Running strategy: {strategy} ===")
        # 运行训练并拿到产物路径
        artifacts = train_main(cfg, dummy=args.dummy)
        # 读取每个策略 summary.json
        summary = json.loads(Path(artifacts.summary_json).read_text(encoding="utf-8"))
        rows.append(
            {
                # 汇总策略名
                "strategy": strategy,
                # 汇总最优验证精度
                "best_val_top1": summary["best_val_top1"],
                # 汇总测试 top1
                "test_top1_acc": summary["test_top1_acc"],
                # 汇总测试 macro-f1
                "test_macro_f1": summary["test_macro_f1"],
                # 汇总训练耗时
                "train_seconds": summary["train_seconds"],
                # 汇总可训练参数量
                "trainable_params": summary["trainable_params"],
                # 汇总模型总参数量
                "total_params": summary["total_params"],
            }
        )

    # 写入总汇 CSV
    out_csv = base_out / "ablation_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        # 固定列顺序输出
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "best_val_top1",
                "test_top1_acc",
                "test_macro_f1",
                "train_seconds",
                "trainable_params",
                "total_params",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAblation results saved: {out_csv}")


# 解析字符串策略列表并做合法性检查
def parse_strategies(s: str) -> list[str]:
    if not s:
        return DEFAULT_STRATEGIES
    values = [x.strip() for x in s.split(",") if x.strip()]
    # 找出非法策略名
    invalid = [x for x in values if x not in DEFAULT_STRATEGIES]
    if invalid:
        raise ValueError(f"Invalid strategies: {invalid}")
    return values


# 基于基础配置创建策略专用配置
def with_strategy(base_cfg: Config, strategy: str, output_dir: Path) -> Config:
    raw = deepcopy(base_cfg.raw)
    # 统一路径分隔符，提升跨平台稳定性
    raw["output_dir"] = str(output_dir).replace("\\", "/")
    raw["training"]["strategy"] = strategy
    # 对 zero_shot 特判可缩短训练轮数
    if strategy == "zero_shot":
        raw["training"]["epochs"] = 1
    return Config(raw=raw)


if __name__ == "__main__":
    main()
```

## src

### eurosat\_baseline

#### `__init__.py`

```python
# 文件路径: src/eurosat_baseline/__init__.py
# 包级别说明字符串
"""EuroSAT transfer learning baseline package."""
```

#### `config.py`

```python
# 文件路径: src/eurosat_baseline/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


# 使用 dataclass 组织配置对象
@dataclass
# 配置包装器，原始字典保存在 raw
class Config:
    raw: Dict[str, Any]

    @property
    # 提供种子快捷访问
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    # 提供设备配置快捷访问
    def device(self) -> str:
        return str(self.raw.get("device", "auto"))

    @property
    # 提供输出目录快捷访问
    def output_dir(self) -> Path:
        return Path(self.raw.get("output_dir", "outputs/default"))


# 从 YAML 文件读取配置并封装
def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as f:
        # 安全解析 YAML
        data = yaml.safe_load(f)
    return Config(raw=data)
```

#### `device.py`

```python
# 文件路径: src/eurosat_baseline/device.py
from __future__ import annotations

from typing import Any, Dict

import torch


# 核心设备解析函数
def resolve_device(raw_cfg: Dict[str, Any]) -> torch.device:
    """Resolve runtime device with GPU preference by default.

    Priority when device=auto:
    1) CUDA GPU
    2) MPS
    3) CPU
    """
    # 统一转小写处理
    device_setting = str(raw_cfg.get("device", "auto")).lower()
    runtime_cfg = raw_cfg.get("runtime", {})
    # 获取目标 GPU 编号
    gpu_id = int(runtime_cfg.get("gpu_id", 0))

    # 自动模式：按可用性回退
    if device_setting == "auto":
        # CUDA 可用则优先选 GPU
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_id}")
        # 在 Apple 设备上尝试 MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        # 不可用时回退到 CPU
        return torch.device("cpu")

    # 显式请求 CUDA 的处理
    if device_setting.startswith("cuda"):
        # CUDA 可用则优先选 GPU
        if torch.cuda.is_available():
            return torch.device(device_setting)
        # 不可用时回退到 CPU
        return torch.device("cpu")

    return torch.device(device_setting)


# 生成人类可读设备字符串
def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        idx = 0 if device.index is None else int(device.index)
        # 查询 GPU 名称用于日志
        name = torch.cuda.get_device_name(idx)
        return f"{device} ({name})"
    return str(device)
```

#### `data.py`

```python
# 文件路径: src/eurosat_baseline/data.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# 数据配置结构体
@dataclass
# 描述 root/csv/classes/size/workers
class DatasetConfig:
    root: Path
    metadata_csv: Path
    num_classes: int
    image_size: int
    num_workers: int


# 真实数据集实现
class EuroSatDataset(Dataset):
    def __init__(self, rows: list[tuple[str, int]], root: Path, transform: transforms.Compose):
        self.rows = rows
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    # 按 index 读取图像并返回 (tensor, label)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path_str, label = self.rows[index]
        image_path = Path(image_path_str)
        # 支持 metadata 使用相对路径
        if not image_path.is_absolute():
            image_path = self.root / image_path
        # 统一为 RGB 三通道
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), int(label)


# 随机数据集，用于快速验证训练流程
class DummyDataset(Dataset):
    def __init__(self, num_samples: int, num_classes: int, image_size: int):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    # 按 index 读取图像并返回 (tensor, label)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # 随机图像张量
        x = torch.rand(3, self.image_size, self.image_size)
        # 随机类别标签
        y = torch.randint(0, self.num_classes, size=(1,)).item()
        return x, y


# 按 train/val/test 构建图像变换
def _build_transforms(image_size: int, is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose(
            [
                # 统一尺寸到模型输入大小
                transforms.Resize((image_size, image_size), antialias=True),
                # 训练增强：随机水平翻转
                transforms.RandomHorizontalFlip(p=0.5),
                # 训练增强：随机旋转
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                # 使用 ImageNet 均值方差归一化
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            # 统一尺寸到模型输入大小
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            # 使用 ImageNet 均值方差归一化
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


# 从 metadata.csv 读取指定 split 样本
def _read_split_rows(metadata_csv: Path, split: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # 校验 metadata 关键列是否齐全
        required = {"image_path", "label", "split"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("metadata.csv must include columns: image_path,label,split")
        for row in reader:
            # 只保留目标划分
            if row["split"] == split:
                rows.append((row["image_path"], int(row["label"])))
    # 指定 split 为空则报错
    if not rows:
        raise ValueError(f"No rows found for split={split} in {metadata_csv}")
    return rows


# 统一 dataloader 构建入口
def build_dataloader(
    dataset_cfg: DatasetConfig,
    split: str,
    batch_size: int,
    dummy: bool = False,
) -> DataLoader:
    # 训练集决定 shuffle 和样本数量
    is_train = split == "train"
    # dummy 模式用随机数据替代真实图像
    if dummy:
        dataset = DummyDataset(
            num_samples=256 if is_train else 64,
            num_classes=dataset_cfg.num_classes,
            image_size=dataset_cfg.image_size,
        )
    else:
        rows = _read_split_rows(dataset_cfg.metadata_csv, split)
        # 正常模式读取真实数据
        dataset = EuroSatDataset(
            rows=rows,
            root=dataset_cfg.root,
            transform=_build_transforms(dataset_cfg.image_size, is_train=is_train),
        )
    # 创建并返回 DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        # 仅训练集打乱
        shuffle=is_train,
        num_workers=dataset_cfg.num_workers,
        # GPU 训练时加速 Host->Device 拷贝
        pin_memory=torch.cuda.is_available(),
    )


# 读取 label 到类名映射，便于可视化/解释
def read_label_names(metadata_csv: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "label_name" in row and row["label_name"]:
                labels[int(row["label"])] = row["label_name"]
    return labels
```

#### `model.py`

```python
# 文件路径: src/eurosat_baseline/model.py
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


# 构建 MobileNetV2 模型并替换分类头
def build_mobilenetv2(num_classes: int, pretrained: bool = True) -> nn.Module:
    # 根据策略决定是否加载 ImageNet 预训练
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    # 实例化 backbone
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    # 将最终分类层改为 EuroSAT 类别数
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# 五种策略的冻结/解冻逻辑核心
def configure_trainable_layers(model: nn.Module, strategy: str, partial_blocks: int = 2) -> None:
    strategy = strategy.lower()
    # Freeze everything first; then selectively unfreeze based on strategy.
    # 先全部冻结，避免遗漏
    for param in model.parameters():
        param.requires_grad = False

    # zero-shot：保持全冻结
    if strategy in {"zero_shot", "zero-shot"}:
        return

    # Always train classification head except zero-shot.
    # 除 zero-shot 外，默认训练分类头
    for param in model.classifier.parameters():
        param.requires_grad = True

    # linear probe：只训练头部
    if strategy == "linear_probe":
        return
    # 全量微调/从零训练：特征层全解冻
    if strategy in {"full_finetune", "from_scratch"}:
        for param in model.features.parameters():
            param.requires_grad = True
        return
    # 部分解冻：仅解冻末尾 block
    if strategy == "partial_unfreeze":
        # 提取特征层 block 列表
        blocks = list(model.features.children())
        # 从末尾向前解冻指定数量
        for block in blocks[-max(1, int(partial_blocks)) :]:
            for param in block.parameters():
                param.requires_grad = True
        return
    # 策略非法时显式报错
    raise ValueError(f"Unknown training strategy: {strategy}")


@torch.inference_mode()
# 计算 top1 准确率
def compute_top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()
```

#### `evaluate.py`

```python
# 文件路径: src/eurosat_baseline/evaluate.py
from __future__ import annotations

import os
import sys
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model import compute_top1_accuracy


# 评估时关闭梯度，节省显存和计算
@torch.inference_mode()
# 评估主函数
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    # 切换到评估模式（关闭 dropout/bn 训练行为）
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_steps = 0
    # 使用交叉熵计算分类损失
    criterion = nn.CrossEntropyLoss()
    # 累积预测类别用于 macro-f1
    all_preds: list[torch.Tensor] = []
    # 累积真实标签用于 macro-f1
    all_targets: list[torch.Tensor] = []
    # 环境变量控制是否强制显示进度条
    force_progress = os.environ.get("EUROSAT_FORCE_PROGRESS", "0") == "1"
    show_progress = sys.stdout.isatty() or force_progress

    # 按 batch 遍历评估集
    for images, labels in tqdm(dataloader, desc="eval", leave=False, disable=not show_progress):
        # 将数据搬运到目标设备
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # 前向推理得到分类 logits
        logits = model(images)
        # 计算 batch 损失
        loss = criterion(logits, labels)
        # 计算 batch top1
        acc = compute_top1_accuracy(logits, labels)
        # 收集预测标签
        all_preds.append(logits.argmax(dim=1).detach().cpu())
        # 收集真实标签
        all_targets.append(labels.detach().cpu())
        total_loss += loss.item()
        total_acc += acc
        total_steps += 1

    # 防御性检查：空 dataloader 直接报错
    if total_steps == 0:
        raise RuntimeError("Empty dataloader in evaluation.")
    # 汇总 macro-f1
    macro_f1 = _macro_f1(torch.cat(all_preds), torch.cat(all_targets))
    return {
        "loss": total_loss / total_steps,
        "top1_acc": total_acc / total_steps,
        "macro_f1": macro_f1,
    }


# 手工实现 macro-f1，避免额外依赖
def _macro_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    # 只在目标中出现过的类别上统计
    classes = torch.unique(targets)
    f1s: list[float] = []
    for cls in classes:
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fp = ((preds == cls) & (targets != cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()
        # 每类精确率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # 每类召回率
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            # 每类 F1
            f1s.append(2 * precision * recall / (precision + recall))
    return float(sum(f1s) / max(len(f1s), 1))
```

#### `train.py`

```python
# 文件路径: src/eurosat_baseline/train.py
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from tqdm.auto import tqdm

from .config import Config
from .data import DatasetConfig, build_dataloader
from .device import device_summary, resolve_device
from .evaluate import evaluate
# 根据策略冻结/解冻参数
from .model import build_mobilenetv2, configure_trainable_layers


# 训练产物 dataclass
@dataclass
# 封装关键输出路径，供脚本层打印/复用
class RunArtifacts:
    best_ckpt: Path
    metrics_json: Path
    summary_json: Path


# 设置随机种子，提升可复现性
def _set_seed(seed: int) -> None:
    # Python 随机数种子
    random.seed(seed)
    # CPU 张量随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # CUDA 随机种子
        torch.cuda.manual_seed_all(seed)


# 从 raw 配置中提取数据配置
def _dataset_cfg_from_raw(raw: Dict) -> DatasetConfig:
    ds = raw["dataset"]
    return DatasetConfig(
        root=Path(ds["root"]),
        metadata_csv=Path(ds["metadata_csv"]),
        num_classes=int(ds["num_classes"]),
        image_size=int(ds["image_size"]),
        num_workers=int(ds["num_workers"]),
    )


# 训练主流程入口
def train_main(cfg: Config, dummy: bool = False) -> RunArtifacts:
    _set_seed(cfg.seed)
    # 解析并选择运行设备
    device = resolve_device(cfg.raw)
    print(f"runtime device: {device_summary(device)}", flush=True)
    out_dir = cfg.output_dir
    # 确保输出目录存在
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cfg = _dataset_cfg_from_raw(cfg.raw)
    train_cfg = cfg.raw["training"]
    strategy = str(train_cfg.get("strategy", "linear_probe"))
    partial_blocks = int(train_cfg.get("partial_blocks", 2))

    # 构建训练集加载器
    train_loader = build_dataloader(
        dataset_cfg=ds_cfg,
        split="train",
        batch_size=int(train_cfg["batch_size"]),
        dummy=dummy,
    )
    # 构建验证集加载器
    val_loader = build_dataloader(
        dataset_cfg=ds_cfg,
        split="val",
        batch_size=int(train_cfg["batch_size"]),
        dummy=dummy,
    )
    # 构建测试集加载器
    test_loader = build_dataloader(
        dataset_cfg=ds_cfg,
        split="test",
        batch_size=int(train_cfg["batch_size"]),
        dummy=dummy,
    )

    # from_scratch 不加载预训练权重
    use_pretrained = strategy != "from_scratch"
    # 构建模型
    model = build_mobilenetv2(num_classes=ds_cfg.num_classes, pretrained=use_pretrained).to(device)
    # 根据策略冻结/解冻参数
    configure_trainable_layers(model, strategy=strategy, partial_blocks=partial_blocks)

    # 统计可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 统计总参数量
    total_params = sum(p.numel() for p in model.parameters())
    # 训练损失函数
    criterion = nn.CrossEntropyLoss()
    # 先置空，zero-shot 可能不需要优化器
    optimizer = None
    # 只有存在可训练参数才创建优化器
    if trainable_params > 0:
        # 使用 AdamW 优化器
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )

    # 追踪最佳验证 top1
    best_val_acc = -1.0
    # 最佳模型 checkpoint 路径
    best_ckpt = out_dir / "best.pt"
    # 记录逐 epoch 指标
    history = []
    # 记录训练开始时间
    start_time = time.perf_counter()
    force_progress = os.environ.get("EUROSAT_FORCE_PROGRESS", "0") == "1"
    runtime_force = bool(cfg.raw.get("runtime", {}).get("force_progress", False))
    # 终端环境决定是否显示进度条
    show_progress = sys.stdout.isatty() or force_progress or runtime_force

    # epoch 主循环
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        # zero-shot 情况不做反向传播
        if optimizer is None:
            # 无优化器时保持 eval 模式
            model.eval()
        else:
            # 常规训练模式
            model.train()
        total_loss = 0.0
        total_steps = 0
        # 迭代训练 batch
        for images, labels in tqdm(
            train_loader,
            desc=f"train epoch {epoch}",
            leave=False,
            disable=not show_progress,
        ):
            # 数据搬运到设备
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 前向计算
            logits = model(images)
            # 计算训练损失
            loss = criterion(logits, labels)
            if optimizer is not None:
                # 清梯度，减少显存碎片
                optimizer.zero_grad(set_to_none=True)
                # 反向传播
                loss.backward()
                # 参数更新
                optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        train_loss = total_loss / max(total_steps, 1)
        # 每轮结束后验证
        val_metrics = evaluate(model, val_loader, device=device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_top1_acc": val_metrics["top1_acc"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        # 记录该轮训练/验证指标
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1_acc']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

        # 验证精度提升则保存最佳模型
        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            # 保存 checkpoint（模型参数+epoch+策略）
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "strategy": strategy},
                best_ckpt,
            )

    metrics_json = out_dir / "metrics.json"
    # 写入逐 epoch 历史到 metrics.json
    metrics_json.write_text(json.dumps(history, indent=2), encoding="utf-8")
    # 计算总耗时
    elapsed = time.perf_counter() - start_time

    # 读取最佳 checkpoint
    state = torch.load(best_ckpt, map_location="cpu")
    # 严格加载最佳参数
    model.load_state_dict(state["model"], strict=True)
    # 在测试集做最终评估
    test_metrics = evaluate(model, test_loader, device=device)
    # 组织最终汇总指标：这一段会直接写入 summary.json，供后续对比实验读取
    summary = {
        # 当前训练策略名称（zero_shot / linear_probe / ...）
        "strategy": strategy,
        # 本次实际训练轮数（来自配置）
        "epochs": int(train_cfg["epochs"]),
        # 本次真正参与更新的参数总量（受冻结策略影响）
        "trainable_params": trainable_params,
        # 模型总参数量（与策略无关）
        "total_params": total_params,
        # 端到端耗时（从训练开始到测试评估结束，单位秒）
        "train_seconds": elapsed,
        # 验证集历史最优 top1（用于反映最佳 checkpoint 质量）
        "best_val_top1": best_val_acc,
        # 用最佳 checkpoint 在测试集上的平均交叉熵损失
        "test_loss": test_metrics["loss"],
        # 用最佳 checkpoint 在测试集上的 top-1 准确率
        "test_top1_acc": test_metrics["top1_acc"],
        # 用最佳 checkpoint 在测试集上的 macro-f1（类别均衡视角）
        "test_macro_f1": test_metrics["macro_f1"],
    }
    summary_json = out_dir / "summary.json"
    # 写 summary.json：这是 run_ablation.py 汇总 CSV 时读取的核心输入文件
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # 打印最终结果摘要，方便在终端快速确认当前策略效果与耗时
    print(
        f"strategy={strategy} test_top1={test_metrics['top1_acc']:.4f} "
        f"test_f1={test_metrics['macro_f1']:.4f} time={elapsed:.1f}s",
        flush=True,
    )
    # 返回关键产物路径：外层脚本会打印路径或继续做汇总处理
    return RunArtifacts(best_ckpt=best_ckpt, metrics_json=metrics_json, summary_json=summary_json)
```
