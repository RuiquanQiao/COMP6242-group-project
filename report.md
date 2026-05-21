# Report Outline: Transfer Learning, When Transfer Helps, and Catastrophic Forgetting

## 1. Introduction (做什么 + 为什么)
- 研究问题:
  - transfer learning 是否优于从零训练?
  - transfer 在什么条件下更有帮助?
  - 微调后是否出现 catastrophic forgetting?
- 任务范围:
  - 图像分类任务
  - 主线数据集为 EuroSAT
  - 对比跨领域数据集 CIFAR-10
- 本文贡献(简写):
  - 设计控制变量实验: 同领域不同数据量、同数据量不同领域
  - 系统比较不同微调策略
  - 统一度量 forgetting 并分析 trade-off

## 2. Related Work (引用相关工作)
- Transfer learning 与 fine-tuning 基础文献
- Catastrophic forgetting / continual learning 相关文献
- 数据集与模型来源文献:
  - ResNet18
  - EuroSAT
  - CIFAR-10
- 可选: 一个 forgetting 缓解方法文献(如 L2 regularization / rehearsal)

## 3. Experimental Setup (实验设置)

### 3.1 模型
- Backbone: ResNet18
- 预训练来源:
  - ImageNet pre-trained
  - random initialization (scratch baseline)

### 3.2 数据集与划分
- EuroSAT (主实验)
- CIFAR-10 (跨领域对比)
- 关键设置:
  - 同领域不同大小: EuroSAT 10% / 30% / 60% / 100%
  - 同样大小不同领域: EuroSAT(27,000) vs CIFAR-10(27,000)

### 3.3 训练策略(核心变量)
- Scratch: 从零训练
- Linear Probe: 冻结 backbone，仅训练分类头
- Partial Fine-tuning: 解冻最后一个 stage + 分类头
- Full Fine-tuning: 全参数微调

### 3.4 统一训练配置
- 相同 epoch、batch size、optimizer、学习率调度
- 每组实验固定随机种子并重复运行(建议 3 次)
- 评价指标:
  - Top-1 accuracy / macro-F1
  - 收敛速度(达到某阈值所需 epoch)
  - Forgetting 指标

## 4. Experiments and Results (实验与结果)

### 4.1 Main Experiment (0): EuroSAT 主实验
- 模型: ResNet18
- 数据: EuroSAT (建议先用 100% 或 60%)
- 策略: Scratch vs Linear Probe vs Partial FT vs Full FT
- 目标:
  - 建立主结论: transfer 是否有效
  - 不同策略性能与稳定性比较
- 输出:
  - 主结果表(accuracy/F1)
  - 训练曲线(收敛速度)

### 4.2 When Transfer Helps (1): 同样大小不同领域
- 模型: ResNet18
- 数据: EuroSAT(27,000) vs CIFAR-10(27,000)
- 策略: 至少比较 Scratch / Partial FT / Full FT
- 目标:
  - 观察 domain gap 对 transfer 收益的影响
  - 解释为什么某些领域迁移更有效
- 输出:
  - 按数据集分组的柱状图/表格

### 4.3 When Transfer Helps (2): 同领域不同数据量
- 模型: ResNet18
- 数据: EuroSAT 10% / 30% / 60% / 100%
- 策略: Scratch vs Full FT (可加 Linear Probe)
- 目标:
  - 分析数据量变化下 transfer 增益曲线
  - 回答“低资源是否更依赖 transfer”
- 输出:
  - 数据量-性能曲线图
  - transfer gain = (FT - Scratch) 折线

## 5. Catastrophic Forgetting Analysis (三个主实验都要做)

### 5.1 顺序任务设计
- 标准顺序:
  - Task A: ImageNet-1K 预训练任务
  - Task B: EuroSAT 微调任务
- 具体做法:
  - 以 torchvision 的 ImageNet pre-trained ResNet18 作为 Task A 模型
  - 在官方 `ILSVRC2012` validation set 上测得 `A_before`
  - 用同一预训练初始化在 EuroSAT 上做 linear probe / partial FT / full FT
  - 将微调后的 backbone 接回原始 1000-way ImageNet 分类头
  - 再回到官方 `ILSVRC2012` validation set 上测得 `A_after`
- 这样得到的 forgetting 是“迁移到 EuroSAT 之后，对原始 ImageNet 识别能力损失了多少”，比用 CIFAR-10 近似更直接也更标准

### 5.2 指标定义
- A_before: 训练 B 前在 A 上性能
- A_after: 训练 B 后在 A 上性能
- Forgetting = A_before - A_after
- A 任务指标:
  - ImageNet Top-1 accuracy
  - ImageNet macro-F1
- B 任务指标:
  - EuroSAT test Top-1 accuracy
  - EuroSAT test macro-F1
- 同时报 B_task 性能，避免“只保留旧知识但学不会新任务”

### 5.3 对比重点
- 哪种策略 forgetting 最严重?
- 哪种策略在“学新任务”与“保留旧任务”之间最平衡?
- forgetting 与 fine-tuning 深度之间的关系
- 预期现象:
  - `linear_probe` 通常最能保留 ImageNet 能力，因为 backbone 基本不变
  - `full_ft` 往往在 EuroSAT 上适应最强，但也最可能牺牲原有 ImageNet 表征
  - `partial_ft` 可能提供较好的 trade-off

## 6. Discussion (解释现象与局限)
- 回答核心问题:
  - transfer 什么时候最有帮助?
  - 什么时候帮助有限或无明显优势?
  - forgetting 何时更严重?
- 误差来源与局限:
  - 数据集差异、样本量、训练预算
  - 单一 backbone 的外推性限制
- 可选缓解方向:
  - 正则化约束 / rehearsal / 冻结更多层

## 7. Conclusion (结论)
- 用 3-4 条结论总结:
  - transfer 是否有效
  - when transfer helps 的条件
  - forgetting 的规律与建议策略

## 8. References
- 预计 8-15 篇，覆盖:
  - transfer learning 基础
  - catastrophic forgetting
  - ResNet18 / EuroSAT / CIFAR-10 数据与模型来源
