# 5.3 Forgetting After 4.3 Same-Domain Data-Size Experiment 中文报告内容

## 实验目的

本节对应 report outline 中的 Section 5.3，目标是在 4.3 的 EuroSAT 数据量实验之后，回到 ImageNet-1K validation set 上评估模型对原始预训练任务的遗忘程度。这里的核心问题是：当下游 EuroSAT 训练数据从 10%、30%、60% 增加到 100% 时，模型为了适应新任务是否会更多改变原始 ImageNet 表征，从而产生更强的 catastrophic forgetting。

## 实验设置

- Backbone：torchvision ImageNet 预训练 ResNet18。
- 下游任务：Section 4.3 的 EuroSAT 10 类分类。
- 数据比例：10%、30%、60%、100%。
- 训练策略：Scratch、Linear Probe、Full Fine-tuning。
- 遗忘评估数据：官方 ILSVRC2012 validation set，共 50,000 张图像、1000 类。
- 评估方法：先测 ImageNet 预训练 ResNet18 得到 A_before；再读取每个 EuroSAT checkpoint，把 backbone 权重加载回带原始 1000 类 ImageNet 分类头的 ResNet18，得到 A_after。
- 指标：Forgetting = A_before - A_after，以 ImageNet top-1 accuracy 的百分点下降表示。

本次 A_before 为 69.76% top-1，macro-F1 为 69.30%。

## 实验结果

| 训练比例 | 策略 | EuroSAT Top-1 | ImageNet A_before | ImageNet A_after | Forgetting | Transfer gain vs Scratch |
|---:|---|---:|---:|---:|---:|---:|
| 10% | Scratch | 68.02% | 69.76% | 0.12% | 69.64 pp | 0.00 pp |
| 10% | Linear probe | 76.20% | 69.76% | 32.50% | 37.26 pp | 8.18 pp |
| 10% | Full fine-tune | 93.58% | 69.76% | 1.24% | 68.52 pp | 25.56 pp |
| 30% | Scratch | 82.83% | 69.76% | 0.08% | 69.68 pp | 0.00 pp |
| 30% | Linear probe | 85.10% | 69.76% | 32.11% | 37.65 pp | 2.27 pp |
| 30% | Full fine-tune | 96.65% | 69.76% | 0.25% | 69.51 pp | 13.82 pp |
| 60% | Scratch | 89.46% | 69.76% | 0.13% | 69.63 pp | 0.00 pp |
| 60% | Linear probe | 86.99% | 69.76% | 31.62% | 38.14 pp | -2.47 pp |
| 60% | Full fine-tune | 96.88% | 69.76% | 0.25% | 69.51 pp | 7.41 pp |
| 100% | Scratch | 89.39% | 69.76% | 0.09% | 69.67 pp | 0.00 pp |
| 100% | Linear probe | 87.93% | 69.76% | 32.13% | 37.63 pp | -1.47 pp |
| 100% | Full fine-tune | 97.15% | 69.76% | 0.20% | 69.56 pp | 7.75 pp |

![Forgetting curve](forgetting_curve_top1.svg)

![Transfer gain curve](transfer_gain_curve_4_3.svg)

## 结果分析

首先，scratch 组的 ImageNet A_after 接近 0%。这一结果不应被解释为严格意义上的 catastrophic forgetting，因为 scratch 模型并没有继承 ImageNet 预训练知识；它只是说明从随机初始化训练出的 EuroSAT backbone 与原始 ImageNet 分类头几乎不兼容。因此，scratch 更适合作为“没有 ImageNet 表征保留”的下界参照。

对于真正继承 ImageNet 预训练的两种策略，Linear Probe 和 Full Fine-tuning 呈现出明显不同的遗忘模式。Linear Probe 的 ImageNet A_after 稳定在约 31%-32%，对应遗忘约 37.26-38.14 pp。它比 full fine-tuning 保留了更多 ImageNet 能力，但并没有完全保留原始表现。一个可能原因是，虽然 linear probe 冻结了 backbone 参数，但训练过程中 BatchNorm 的 running statistics 仍可能随 EuroSAT 数据分布更新，使得 ImageNet validation 上的特征归一化不再完全匹配原始预训练分布。

Full Fine-tuning 在 EuroSAT 上表现最好，但 ImageNet 遗忘最严重。其 A_after 只有 0.20%-1.24%，遗忘约 68.52-69.56 pp，几乎相当于失去了原始 ImageNet 分类能力。这说明端到端微调会强烈重塑 ResNet18 的特征空间，使其非常适合 EuroSAT，但这些特征已经难以被原来的 1000 类 ImageNet head 使用。

从数据量角度看，遗忘并没有随 EuroSAT 数据比例单调增加。Full fine-tuning 的遗忘在 10%、30%、60%、100% 下都接近 69 pp；Linear probe 的遗忘也基本稳定在 37-38 pp。这说明在当前 8 epochs 设置下，是否更新 backbone 比下游数据量大小更决定 ImageNet 遗忘程度。也就是说，训练策略是主要因素：full fine-tuning 带来最高 EuroSAT 收益，但代价是几乎完全遗忘 ImageNet；linear probe 的 EuroSAT 收益较小，甚至在 60% 和 100% 数据量下低于 scratch，但能保留更多 ImageNet 能力。

## 与 4.3 结果的联系

Section 4.3 表明，full fine-tuning 在 10% 数据量下对 EuroSAT 的收益最大，相比 scratch 提升 25.56 pp；随着数据比例增加，收益下降到 60% 的 7.41 pp 和 100% 的 7.75 pp。然而在 5.3 中，full fine-tuning 的 ImageNet 遗忘始终保持在约 69 pp。这意味着 full fine-tuning 的收益-遗忘权衡在低数据量时最划算：10% 数据已经获得最大迁移收益，但遗忘程度并不比 100% 数据更高。

Linear probe 的情况相反。它在 10% 数据时相对 scratch 仍有 8.18 pp 提升，但在 60% 和 100% 时分别为 -2.47 pp 和 -1.47 pp，即低于 scratch。与此同时，它的 ImageNet 遗忘保持在约 37-38 pp。因此，如果目标是最大化 EuroSAT 性能，linear probe 不如 full fine-tuning；如果目标是在学习 EuroSAT 的同时尽量保留 ImageNet 能力，linear probe 是更保守的折中方案。

## 可写入报告的结论

本实验显示，在 EuroSAT same-domain data-size 设置中，catastrophic forgetting 主要由微调策略决定，而不是由下游数据比例单独决定。Full fine-tuning 在所有数据比例下都取得最高 EuroSAT accuracy，但将 ImageNet top-1 从 69.76% 降到约 0.2%-1.2%，说明端到端适配 EuroSAT 会严重破坏原始 ImageNet 分类能力。Linear probe 保留了更多 ImageNet 能力，A_after 约为 31%-32%，但其 EuroSAT 收益有限，并在较大数据量下弱于 scratch。

因此，5.3 可以和 4.3 一起形成一个 trade-off 结论：full fine-tuning 是获得下游任务性能的最佳方法，尤其在 10% 低数据量下迁移收益最大；但这种性能提升伴随严重 ImageNet 遗忘。Linear probe 则牺牲部分 EuroSAT 适应能力，换来更好的原任务保留。对于需要持续保留 ImageNet 通用识别能力的场景，未来应考虑冻结 BatchNorm、正则化、rehearsal 或参数高效微调方法来降低遗忘。

## 局限性

- 每组只运行一个随机种子，遗忘数值可能存在随机波动。
- 本实验按照 checkpoint 后测协议评估 ImageNet，不重新训练 ImageNet head；因此结果反映的是“EuroSAT 微调后的 backbone 与原始 ImageNet head 的兼容性”。
- Linear probe 中 BatchNorm running statistics 的更新可能影响 ImageNet 表现；如果要严格测试“完全冻结 backbone”，应补充一组冻结 BatchNorm 的 linear probe。
- Full fine-tuning 的高遗忘不代表模型完全失去所有通用视觉特征，而是说明这些特征已经不再适配原始 ImageNet 分类头。
