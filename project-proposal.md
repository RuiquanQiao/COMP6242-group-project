# COMP6242 Project Proposal

***

## 📄 Formal Academic Submission (For Professor/TA)

**Title:** Bridging the Aerial Domain Gap: A Comparative Analysis of Transfer Learning Strategies for Satellite Image Classification

### Abstract

The application of deep learning to remote sensing has revolutionized Earth observation tasks, yet training models from scratch on specialized satellite imagery remains computationally prohibitive. While transfer learning from models pre-trained on natural images (e.g., ImageNet) offers a practical solution, it introduces a significant domain gap. Unlike natural images, which are typically object-centric and captured from a horizontal perspective, satellite images (such as those from Sentinel-2) present a top-down, bird's-eye view where features are heavily dependent on dense textures, spectral bands, and spatial patterns rather than object shapes.

This project investigates efficient transfer learning strategies to adapt lightweight convolutional neural networks (e.g., MobileNetV2 or ResNet18) to the EuroSAT land-use and land-cover classification dataset. Adhering to strict computational constraints (CPU/Edge-friendly), we will conduct a rigorous ablation study comparing five paradigms: (1) Zero-shot inference (baseline), (2) Training from scratch (random initialization), (3) Linear probing (training only the classification head), (4) Partial unfreezing of high-level convolutional blocks, and (5) Full fine-tuning. By analyzing the accuracy versus computational cost trade-offs, this study aims to determine the optimal layer-freezing strategy that effectively mitigates the horizontal-to-aerial domain shift while minimizing training overhead.

***

## 💬 团队内部讨论版 (For Team Discussion)

**一句话目标：** 使用极低算力（迁移学习），将平时用来“认猫认狗”的通用 AI 模型，改造为能从高空卫星图上认出“森林、高速公路和工业区”的遥感分析专家。

### 📝 极简开题报告 (Abstract)

- **核心痛点 (Domain Gap)：** 现成模型（基于 ImageNet 训练）习惯了看“平视、有明确轮廓”的物体。但卫星遥感图是“从上往下看”的，没有上下之分，且全是一堆纹理和颜色块。直接拿去识别遥感图，模型会“水土不服”。
- **解决方案：** 拿一个极其好跑的轻量级模型 (**MobileNetV2**)，在超级干净、完美的遥感分类数据集 (**EuroSAT**，全是切好的 64x64 纯图片，0 预处理) 上进行微调。
- **消融实验 (高分关键)：** 为了满足课程 "重分析" 的要求，我们将跑 5 个对比实验：
  1. **Zero-shot (基线)：** 啥也不训练，直接测准确率有多惨（证明卫星图和日常图的差异有多大）。
  2. **From Scratch (从零训练)：** 随机初始化模型，从头训练，作为 transfer learning 的关键对照。
  3. **Linear Probing (线性探测)：** 冻结全网，只微调最后的一层分类头。
  4. **Partial Unfreezing (部分解冻)：** 解冻最后 1-2 个卷积块（这是算力和准确率性价比最高的商业 Sweet Spot）。
  5. **Full Fine-Tuning (全参数微调)：** 所有层全部重新训练。
- **最终产出：** 画一张图表，对比这 5 种策略的“准确率 vs 训练耗时”。

<br />
