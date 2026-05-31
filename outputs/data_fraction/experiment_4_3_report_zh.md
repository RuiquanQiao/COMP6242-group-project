# 4.3 Same-Domain Data-Size Experiment 中文报告内容

## 实验目的

本实验对应报告大纲中的 Section 4.3，目标是在同一数据领域内分析训练数据量变化时迁移学习的收益。具体来说，我们使用 EuroSAT 数据集，并分别采用 10%、30%、60% 和 100% 的训练数据，比较三种训练策略：

- Scratch：ResNet18 随机初始化，从零训练全部参数。
- Linear Probe：使用 ImageNet 预训练 ResNet18，冻结 backbone，只训练最后的分类层。
- Full Fine-tuning：使用 ImageNet 预训练 ResNet18，并微调全部参数。

该实验用于回答一个核心问题：在低资源场景下，模型是否更依赖迁移学习？

## 实验设置

- Backbone：ResNet18。
- 预训练来源：ImageNet。
- 数据集：EuroSAT，10 类遥感图像分类。
- 训练数据比例：10%、30%、60%、100%。
- 训练轮数：每组 8 epochs。
- 评价指标：测试集 Top-1 accuracy 和 macro-F1。
- 运行设备：Apple MPS。
- 随机种子：42。

## 实验结果

| 训练数据比例 | Scratch Acc. | Linear Probe Acc. | Full FT Acc. | Scratch F1 | Linear Probe F1 | Full FT F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 68.02% | 76.20% | 93.58% | 66.88% | 75.51% | 93.51% |
| 30% | 82.83% | 85.10% | 96.65% | 82.29% | 84.81% | 96.52% |
| 60% | 89.46% | 86.99% | 96.88% | 88.89% | 86.67% | 96.78% |
| 100% | 89.39% | 87.93% | 97.15% | 89.24% | 87.71% | 97.04% |

相对于 scratch 的准确率提升如下：

| 训练数据比例 | Linear Probe Gain | Full FT Gain |
|---:|---:|---:|
| 10% | +8.18 pp | +25.56 pp |
| 30% | +2.27 pp | +13.82 pp |
| 60% | -2.47 pp | +7.41 pp |
| 100% | -1.47 pp | +7.75 pp |

## 结果分析

实验结果表明，full fine-tuning 在所有数据比例下都取得了最高的测试准确率。这说明 ImageNet 预训练特征对于 EuroSAT 遥感图像分类任务具有明显帮助，尤其是在训练数据较少时优势最明显。

在 10% 数据量下，scratch 的测试准确率为 68.02%，而 full fine-tuning 达到 93.58%，提升了 25.56 个百分点。这说明在低资源场景下，从零训练很难仅依靠少量 EuroSAT 样本学习到稳定的视觉特征，而预训练模型提供了更好的初始化和通用视觉表示。

随着训练数据量增加，scratch 的表现明显提升：从 10% 数据下的 68.02% 提升到 30% 数据下的 82.83%，并在 60% 和 100% 数据下达到约 89%。与此同时，full fine-tuning 相对 scratch 的收益逐渐下降，从 10% 下的 +25.56 pp 下降到 60% 下的 +7.41 pp 和 100% 下的 +7.75 pp。这说明迁移学习的主要优势在于提升低数据量场景下的 sample efficiency；当同领域训练数据变多时，scratch 模型能够逐渐学习到更适合任务的特征，因此与预训练微调之间的差距缩小。

Linear probe 的结果提供了一个更细的对比。在 10% 和 30% 数据量下，linear probe 分别比 scratch 高 8.18 和 2.27 个百分点，说明冻结的 ImageNet 特征在低数据量下已经包含有用的视觉表示。然而，在 60% 和 100% 数据量下，linear probe 反而低于 scratch。这说明当训练数据较多时，只训练分类头会限制模型对 EuroSAT 任务的适配能力；相比之下，scratch 可以学习任务相关特征，而 full fine-tuning 既保留预训练初始化优势，又允许全部参数适配新任务，因此表现最好。

## 收敛速度观察

从训练过程看，full fine-tuning 的收敛速度明显快于 scratch。例如，在 10% 数据量下，full fine-tuning 第 1 个 epoch 的验证准确率已经达到 89.27%，而 scratch 训练到第 8 个 epoch 的测试准确率只有 68.02%。在 100% 数据量下，full fine-tuning 第 2 个 epoch 已达到 96.26% 验证准确率，而 scratch 在 8 个 epochs 后测试准确率为 89.39%。这说明迁移学习不仅提升最终性能，也能显著减少达到高准确率所需的训练时间。

## 可写入报告的结论

本实验支持“低资源场景更依赖迁移学习”的结论。对于 EuroSAT 同领域分类任务，当训练数据只有 10% 时，full fine-tuning 相比 scratch 有 25.56 个百分点的准确率提升，是所有数据比例中最大的收益。随着训练数据比例增加，scratch 模型逐渐变强，full fine-tuning 的相对收益下降。这说明预训练模型的主要价值在于降低对标注样本数量的依赖，提高模型的数据效率。

同时，linear probe 的结果表明，预训练特征本身在低数据量下已经有帮助，但冻结 backbone 会限制模型在数据较多时的任务适配能力。因此，在本实验中，full fine-tuning 是最有效的迁移策略：它既利用了 ImageNet 预训练表示，又允许模型根据 EuroSAT 数据进行端到端调整。

## 局限性

- 本实验只使用一个随机种子，结果可能受到随机初始化、数据顺序和优化波动影响。
- 每组实验固定为 8 epochs。Scratch 在 100% 数据量下可能尚未完全收敛，因此更长训练可能进一步提升 scratch 表现。
- 本节只分析同一领域内数据量变化，不包含跨领域迁移，也不包含 catastrophic forgetting 分析。
- Linear probe 使用冻结 backbone，因此它更适合评估预训练特征的可分性，而不是最终最优性能。

## 可直接放进报告的英文段落草稿

In the same-domain data-size experiment, we evaluated ResNet18 on EuroSAT using 10%, 30%, 60%, and 100% of the training data. We compared three strategies: training from scratch, linear probing with an ImageNet-pretrained backbone, and full fine-tuning. Full fine-tuning achieved the best performance across all data sizes. The largest improvement appeared in the low-data regime: with only 10% training data, full fine-tuning achieved 93.58% test accuracy, compared with 68.02% for scratch training, giving a gain of 25.56 percentage points. As the amount of labelled data increased, the scratch baseline improved and the fine-tuning gain decreased to about 7-8 percentage points at 60% and 100% data. This suggests that transfer learning mainly improves sample efficiency and is most valuable when labelled data is limited.

Linear probing also improved over scratch at 10% and 30% data, but became worse than scratch at 60% and 100%. This indicates that frozen ImageNet features are useful in low-data settings, but limiting adaptation to only the classification head can become restrictive when more in-domain data is available. Full fine-tuning provides the best trade-off because it benefits from pretrained representations while still allowing the whole network to adapt to EuroSAT.
