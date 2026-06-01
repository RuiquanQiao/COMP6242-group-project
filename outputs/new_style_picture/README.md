# 新生成的论文图片

本目录保存按论文原始文件名重新生成的图像，使用统一的 `neurips_modern` 风格。

输出规则：

- 保留论文 Markdown 中的相对路径和文件名，例如 `outputs/data_fraction/test_top1_acc.png`。
- 同时导出 PNG、PDF、SVG，便于论文最终排版。
- 所有准确率、F1、gain、forgetting 都直接使用百分数单位，不使用 0-1 小数。
- 未覆盖原始报告文件，也未修改训练代码或报告正文。

`outputs/eurosat_ablation/val_top1_acc_curve.png` 的逐 epoch validation accuracy 来自 GitHub 项目中的 `outputs/eurosat_ablation/*/metrics.json`。

已生成图像文件数量：42
