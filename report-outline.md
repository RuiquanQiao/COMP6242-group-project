# Report Outline: Transfer Learning, When Transfer Helps, and Catastrophic Forgetting

## 0. Project Mainline and Chapter Ownership

- 项目主线:
  - 整个小组作业的核心任务始终是: 将 ImageNet pre-trained `ResNet18`
    迁移到 `EuroSAT`，使模型能够识别遥感场景图像
  - `when transfer helps` 和 `catastrophic forgetting` 是围绕这条主线展开的
    分析章节，不是脱离主线的独立项目
- 最终报告篇幅限制:
  - 根据 guideline，final report 必须控制在 `4-8 pages (including references)`
  - 实际写作时要把“最终定稿不得超过 8 页（包括引用）”当成硬约束
  - 因此正文里要优先保留主线结果、关键图表和最必要的分析，避免展开过度
- 章节负责人总览:
  - `Chapter 1 Introduction`: Ruiquan Qiao 主笔，其他成员核对
  - `Chapter 2 Related Work`: Ruiquan Qiao 主笔，其他成员核对
  - `Chapter 3 Experimental Setup`: Ruiquan Qiao 主笔，其他成员核对
  - `Section 4.1 Main Experiment`: Ruiquan Qiao 负责
  - `Section 4.2 Same-Size Cross-Domain`: Tiancheng Xia 负责
  - `Section 4.3 Same-Domain Data-Size`: Guangde Shi 负责
  - `Section 5.1 Forgetting for 4.1`: Ruiquan Qiao 负责
  - `Section 5.2 Forgetting for 4.2`: Tiancheng Xia 负责
  - `Section 5.3 Forgetting for 4.3`: Guangde Shi 负责
  - `Chapter 6 Discussion`: Tiancheng Xia 主笔，三人提供各自实验解释
  - `Chapter 7 Conclusion`: Tiancheng Xia 主笔
  - `Chapter 8 References`: Guangde Shi 整理

## 1. Introduction (做什么 + 为什么)
- 负责人: Ruiquan Qiao 主笔，其他成员核对
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
- 负责人: Ruiquan Qiao 主笔，其他成员核对
- Transfer learning 与 fine-tuning 基础文献
- Catastrophic forgetting / continual learning 相关文献
- 数据集与模型来源文献:
  - ResNet18
  - EuroSAT
  - CIFAR-10
- 可补充一个 forgetting 缓解方法文献，用来说明本项目没有实现缓解方法

## 3. Experimental Setup (实验设置)
- 负责人: Ruiquan Qiao 主笔，其他成员核对

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
- 相同 epoch、batch size、optimizer
- 当前代码默认固定随机种子为 `42`
- 评价指标:
  - Top-1 accuracy / macro-F1
  - 收敛速度(达到某阈值所需 epoch)
  - Forgetting 指标

## 4. Experiments and Results (实验与结果)

- 章节说明:
  - 第四章的三个实验共享同一主线: `ImageNet -> EuroSAT`
  - `4.2` 和 `4.3` 不是偏题，而是为了回答 guideline 里要求的
    `when transfer helps`
  - 每个实验做完之后，都要在第五章额外回答: 这种迁移是否让模型遗忘了
    原来的 ImageNet 能力

### 4.1 Main Experiment (0): EuroSAT 主实验
- 负责人: Ruiquan Qiao
- 模型: ResNet18
- 数据: EuroSAT
- 策略: Scratch vs Linear Probe vs Partial FT vs Full FT
- 目标:
  - 建立主结论: transfer 是否有效
  - 不同策略性能与稳定性比较
- 输出:
  - 主结果表(accuracy/F1)
  - 训练曲线(收敛速度)

### 4.2 When Transfer Helps: 同样大小不同领域
- 负责人: Tiancheng Xia
- 模型: ResNet18
- 数据: EuroSAT(27,000) vs CIFAR-10(27,000)
- 策略: 至少比较 Scratch / Partial FT / Full FT
- 目标:
  - 观察 domain gap 对 transfer 收益的影响
  - 解释为什么某些领域迁移更有效
- 输出:
  - 按数据集分组的柱状图/表格

### 4.3 When Transfer Helps: 同领域不同数据量
- 负责人: Guangde Shi
- 模型: ResNet18
- 数据: EuroSAT 10% / 30% / 60% / 100%
- 默认策略: Scratch vs Linear Probe vs Full FT
- 目标:
  - 分析数据量变化下 transfer 增益曲线
  - 回答“低资源是否更依赖 transfer”
- 输出:
  - 数据量-性能曲线图
  - transfer gain = (FT - Scratch) 折线

## 5. Catastrophic Forgetting Analysis (每个主实验都要回到 ImageNet 检查遗忘)

- 本章总说明:
  - 这一章不是另起炉灶做一个新任务，而是对第四章每个主实验做“后测”
  - 我们已经下载官方 `ILSVRC2012` validation set，整理后约 `6.7GB`
  - 目的非常直接: 看模型在完成下游迁移之后，是否遗忘了它最初在
    `ImageNet-1K` 上学到的能力
  - 也就是说，第四章回答“迁移后 EuroSAT 学得怎么样”，第五章回答
    “学会 EuroSAT 的同时，ImageNet 忘了多少”

### 5.0 统一评测协议
- 负责人: Ruiquan Qiao 主写，三人共用
- 标准顺序:
  - Task A: `ImageNet-1K` 预训练任务
  - Task B: 第四章中的某个下游实验
- 统一做法:
  - 先以 `torchvision` 的 ImageNet pre-trained `ResNet18` 作为基线模型
  - 在官方 `ILSVRC2012` validation set 上测得 `A_before`
  - 再读取某个已经完成的下游 checkpoint，例如 `outputs/eurosat_ablation`
    或 `outputs/domain_gap` 或 `outputs/data_fraction` 下的 `best.pt`
  - 将该 checkpoint 的 backbone 权重加载回 `ResNet18`
  - 接回原始 `1000` 类 ImageNet 分类头
  - 回到官方 `ILSVRC2012` validation set 上再次评测，得到 `A_after`
- 为什么这样最清楚:
  - 因为这里测的不是“另一个代理任务”，而是模型原本真正预训练过的
    `ImageNet` 任务本身
  - 所以 forgetting 的含义非常直接:
    “迁移学习之后，原始 ImageNet 识别能力下降了多少”

### 5.1 Forgetting After 4.1 Main Experiment
- 负责人: Ruiquan Qiao
- 对应主实验:
  - `Section 4.1` 的 `linear_probe / partial_ft / full_ft`
- 具体问题:
  - 在 EuroSAT 主实验中，哪种微调策略最容易损伤原始 ImageNet 表征?
  - `partial_ft` 在 EuroSAT 上效果最好时，是否也比 `full_ft` 保留了更多
    ImageNet 能力?
- 报告方式:
  - 一张表同时报告:
    - `ImageNet before`
    - `ImageNet after`
    - `forgetting = before - after`
    - `EuroSAT final performance`
  - 一张 forgetting 柱状图，按策略比较

### 5.2 Forgetting After 4.2 Same-Size Cross-Domain Experiment
- 负责人: Tiancheng Xia
- 对应主实验:
  - `Section 4.2` 的 `EuroSAT vs CIFAR-10` 同样大小不同领域对比
- 具体问题:
  - 如果下游任务更接近自然图像分布，是否会更少破坏 ImageNet 表征?
  - `EuroSAT` 和 `CIFAR-10` 的 forgetting 是否能从 domain gap 角度解释?
- 报告方式:
  - 按数据集分组比较 `forgetting_top1` 和 `forgetting_macro_f1`
  - 和 `4.2` 的 transfer gain 一起讨论，回答:
    - “更容易迁移的任务，是否也更不容易遗忘?”

### 5.3 Forgetting After 4.3 Same-Domain Data-Size Experiment
- 负责人: Guangde Shi
- 对应主实验:
  - `Section 4.3` 的 `EuroSAT 10% / 30% / 60% / 100%`
- 具体问题:
  - 下游数据越多，模型是否会为了适应 EuroSAT 而改动更多，从而遗忘更多?
  - 小数据量微调是否反而更能保留原始 ImageNet 能力?
- 报告方式:
  - 以训练比例为横轴，画出 forgetting 曲线
  - 同时对照 `4.3` 的 transfer gain 曲线，分析“数据量增加”对
    新任务收益和旧任务遗忘的双重影响

### 5.4 指标定义与共同结论
- 负责人: Ruiquan Qiao 统稿
- A_before:
  - 微调前在官方 ImageNet validation set 上的性能
- A_after:
  - 微调后回到官方 ImageNet validation set 上的性能
- Forgetting:
  - `Forgetting = A_before - A_after`
- A 任务指标:
  - ImageNet Top-1 accuracy
  - ImageNet macro-F1
- B 任务指标:
  - 对应第四章实验的 test Top-1 accuracy
  - 对应第四章实验的 test macro-F1
- 共同要回答的结论:
  - 哪种策略 forgetting 最严重?
  - 哪种策略在“学新任务”和“保留旧任务”之间最平衡?
  - forgetting 与 fine-tuning 深度、领域差异、数据量之间分别是什么关系?

## 6. Discussion (解释现象与局限)
- 负责人: Tiancheng Xia 主笔，三人提供各自实验解释
- 回答核心问题:
  - transfer 什么时候最有帮助?
  - 什么时候帮助有限或无明显优势?
  - forgetting 何时更严重?
- 误差来源与局限:
  - 数据集差异、样本量、训练预算
  - 单一 backbone 的外推性限制
- 可讨论缓解方向:
  - 正则化约束 / rehearsal / 冻结更多层

## 7. Conclusion (结论)
- 负责人: Tiancheng Xia 主笔
- 用 3-4 条结论总结:
  - transfer 是否有效
  - when transfer helps 的条件
  - forgetting 的规律与可行策略

## 8. References
- 负责人: Guangde Shi 整理
- 参考文献需覆盖:
  - transfer learning 基础
  - catastrophic forgetting
  - ResNet18 / EuroSAT / CIFAR-10 数据与模型来源
