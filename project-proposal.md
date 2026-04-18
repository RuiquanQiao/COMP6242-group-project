# COMP6242 Project Proposal

***

## 📄 Formal Academic Submission (For Professor/TA)

**Title:** Bridging the Domain Gap in Egocentric Vision: A Comparative Study of Transfer Learning Strategies for Edge Devices

### Abstract

The proliferation of smart glasses and wearable devices has highlighted a critical challenge in computer vision: models pre-trained on conventional, third-person perspective datasets (e.g., ImageNet) suffer significant performance degradation when applied to egocentric (first-person) views. This domain gap is primarily driven by severe hand occlusions, perspective distortions, and motion blur inherent in wearable camera data. This project investigates efficient transfer learning strategies to adapt lightweight convolutional neural networks, specifically MobileNetV2, to egocentric vision tasks under strict computational constraints.

Rather than relying on computationally expensive full-parameter fine-tuning, this study aims to evaluate the optimal trade-off between model accuracy and computational cost (CPU-friendly training). We will conduct a rigorous ablation study comparing four transfer learning paradigms on a subset of the EGTEA Gaze+ dataset: (1) Zero-shot inference (baseline), (2) Linear probing, (3) Partial unfreezing of high-level convolutional blocks, and (4) Full fine-tuning. By analyzing the learning curves and catastrophic forgetting across these strategies, this project hypothesizes that partial unfreezing will offer a "sweet spot" that significantly mitigates the egocentric domain gap while maintaining the low computational overhead required for practical edge-device deployment.

***

## 💬 团队内部讨论版 (For Team Discussion)

**一句话目标：** 使用极低算力（迁移学习），将普通的图像模型改造为能在“智能眼镜（第一人称视角）”上高效运行的边缘 AI。

### 📝 极简开题报告 (Abstract)

- **核心痛点 (Domain Gap)：** 现成模型（基于第三人称视角训练）在智能眼镜上会因为手部遮挡、运动模糊和奇怪的俯视角度导致准确率暴跌。
- **解决方案：** 拿一个专为手机设计的轻量级模型 (**MobileNetV2**)，在第一人称视角数据集（如 **EGTEA Gaze+**）上进行微调。
- **消融实验 ：** 为了满足课程 "注重分析" 的要求，我们将跑 4 个对比实验：
  1. **Zero-shot (基线)：** 啥也不训练，直接测准确率有多惨。
  2. **Linear Probing (线性探测)：** 冻结全网，只微调最后的一层分类头。
  3. **Partial Unfreezing (部分解冻)：** 解冻最后 1-2 个卷积块（这是算力和准确率性价比最高的商业 Sweet Spot）。
  4. **Full Fine-Tuning (全参数微调)：** 所有层全部重新训练。
- **最终产出：** 画一张图表，对比这 4 种策略的“准确率 vs 训练耗时”，在报告里讨论商业落地时的成本 Trade-off。

