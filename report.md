## From ImageNet to EuroSAT: Transfer Learning, Forgetting, and When Transfer Helps

### 1. Introduction

Transfer learning is a practical way to adapt deep models to domains where labeled data is limited. In computer vision, ImageNet pretraining is widely used because large-scale supervised training provides reusable visual features [1, 2]. However, transfer is not uniformly effective: usefulness depends on the distance between source and target tasks and on how much of the pretrained network is allowed to adapt [1].

This question is important for remote sensing, where satellite scenes differ substantially from natural images in viewpoint, texture, and semantics [3]. Our main task is therefore straightforward: adapt an ImageNet-pretrained ResNet-18 [5, 6] to EuroSAT [4] and determine which transfer strategy works best. We compare four strategies: `scratch`, `linear_probe`, `partial_ft`, and `full_ft`.

Around this mainline task, the report addresses two supporting questions required by the project guideline. First, when does transfer help most? We study this through a same-size cross-domain comparison with CIFAR-10 [8] and a same-domain data-size study on EuroSAT. Second, how much original ImageNet ability is lost after downstream adaptation? We answer this with a catastrophic forgetting analysis on the official ILSVRC2012 validation set [2, 7].

The report therefore makes three contributions: a controlled EuroSAT transfer study, an analysis of when transfer helps, and a direct measurement of forgetting on the original ImageNet task.

### 2. Related Work

#### 2.1 Transfer Learning and Fine-Tuning in Deep Vision

Modern vision transfer learning usually starts from ImageNet-pretrained models and adapts them through feature extraction or fine-tuning. Yosinski et al. show that lower layers transfer more generally than higher layers and that transfer quality depends on task distance [1]. This directly motivates our comparison of `linear_probe`, `partial_ft`, and `full_ft`. ImageNet itself remains the standard source task because of its scale and influence on visual representation learning [2, 6].

#### 2.2 Remote Sensing Scene Classification Under Domain Shift

Remote sensing scene classification differs from natural-image recognition because classes are defined by large-scale spatial patterns rather than object-centric appearance [3]. EuroSAT is a standard benchmark in this setting, containing 27,000 labeled Sentinel-2 image patches from 10 classes [4]. This makes it a suitable test bed for asking whether ImageNet representations remain useful under a clear domain shift.

#### 2.3 Catastrophic Forgetting and Sequential Adaptation

Catastrophic forgetting refers to performance loss on a previous task after training on a new one [7]. Methods such as EWC [9] and Learning without Forgetting [10] aim to reduce this effect, but our goal is simpler: measure how much forgetting ordinary downstream fine-tuning already causes. By evaluating on the official ImageNet validation set before and after adaptation, we tie forgetting directly to the actual source task rather than to a proxy benchmark.

### 3. Experimental Setup

#### 3.1 Backbone and Initialization

All experiments use ResNet-18 [5]. Transfer runs start from the official torchvision ImageNet-1K weights, while `scratch` uses random initialization. The final layer is replaced with a 10-way classifier for EuroSAT and CIFAR-10, and the original 1000-way head is restored for ImageNet forgetting evaluation. We compare four strategies: `scratch`, `linear_probe`, `partial_ft`, and `full_ft`. For the 10-class downstream tasks, their trainable parameter counts are 11,181,642, 5,130, 8,398,858, and 11,181,642 respectively.

#### 3.2 Datasets and Splits

The main downstream dataset is EuroSAT [4], using a fixed class-balanced split of 27,000 RGB images into 18,900 train, 4,050 validation, and 4,050 test samples. The same split sizes are used for the CIFAR-10 same-size comparison [8]. For the data-size analysis, the EuroSAT validation and test splits stay fixed while the training set is reduced to 10%, 30%, 60%, and 100% of the full training pool. For forgetting, the previous task is the original ImageNet-1K problem [2, 6], evaluated on the official ILSVRC2012 validation set prepared in ImageFolder format.

#### 3.3 Preprocessing and Data Loading

All inputs are resized to 224 x 224. EuroSAT and CIFAR-10 training use resize, random horizontal flip, random rotation, tensor conversion, and ImageNet normalization; validation and test use deterministic resize plus the same normalization. ImageNet forgetting evaluation follows the standard resize-center-crop pipeline. Whenever a subset of a split is requested, sampling is class balanced.

#### 3.4 Training Protocol and Metrics

Unless overridden, all runs use the shared `configs/base.yaml` settings: seed 42, batch size 32, 8 epochs, AdamW, learning rate 3e-4, and weight decay 1e-4. The best checkpoint is selected by validation top-1 accuracy. We report test top-1 accuracy, test macro-F1, training time, and convergence behavior. In Chapter 5, forgetting is defined as the drop in ImageNet top-1 accuracy or macro-F1 after downstream fine-tuning.

#### 3.5 Experiment Structure

The study has one main experiment and two supporting analyses. Section 4.1 compares four transfer strategies on EuroSAT. Section 4.2 studies when transfer helps by comparing EuroSAT and CIFAR-10 at the same sample size. Section 4.3 studies when transfer helps by varying the amount of EuroSAT training data. Chapter 5 then evaluates the resulting checkpoints on the official ImageNet validation set to measure catastrophic forgetting.

### 4. Experiments and Results

#### 4.1 Main Experiment: EuroSAT Strategy Ablation

Section 4.1 asks the core question of the project: on EuroSAT, does ImageNet pretraining help ResNet-18, and which transfer strategy works best? We compare `scratch`, `linear_probe`, `partial_ft`, and `full_ft` under the same training budget.

| Strategy | Best Val Top-1 | Test Top-1 | Test Macro-F1 | Train Time (s) | Trainable Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scratch` | 93.36 | 93.21 | 93.04 | 1100.5 | 11,181,642 |
| `linear_probe` | 90.05 | 88.54 | 88.33 | 560.2 | 5,130 |
| `partial_ft` | 97.68 | **97.78** | **97.71** | 641.2 | 8,398,858 |
| `full_ft` | **97.80** | 97.56 | 97.50 | 1068.3 | 11,181,642 |

The results show that transfer is helpful, but only when the pretrained model is allowed to adapt. `partial_ft` achieves the best test performance at 97.78% top-1 accuracy and 97.71% macro-F1, with `full_ft` a close second. Both strategies clearly outperform `scratch`, while `linear_probe` is worst and even falls below the scratch baseline. This indicates that ImageNet features are useful as initialization, but not as a fully frozen feature extractor for EuroSAT.

`partial_ft` is also the best trade-off overall. It slightly outperforms `full_ft` on the test set while using fewer trainable parameters and less training time, suggesting that adapting the top stage is sufficient for most of the domain shift. The convergence logs tell the same story: `partial_ft` and `full_ft` pass the 90% validation threshold in epoch 1, whereas `scratch` reaches it in epoch 6 and `linear_probe` only in epoch 8.

<table>
  <tr>
    <td width="50%" align="center">
      <img src="outputs/eurosat_ablation/test_top1_acc.png" alt="EuroSAT test top-1 accuracy by training strategy" width="100%" />
    </td>
    <td width="50%" align="center">
      <img src="outputs/eurosat_ablation/val_top1_acc_curve.png" alt="EuroSAT validation top-1 accuracy across epochs" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center"><em>(a) Final EuroSAT test top-1 accuracy by strategy.</em></td>
    <td align="center"><em>(b) Validation top-1 accuracy across training epochs.</em></td>
  </tr>
</table>

*Figure 1. Side-by-side summary of Experiment 4.1. The left panel shows final EuroSAT test top-1 accuracy, while the right panel shows validation top-1 trajectories. Together they show that `partial_ft` and `full_ft` both outperform `scratch` and `linear_probe`, and that deeper fine-tuning converges faster.*

Overall, Experiment 4.1 shows that ImageNet transfer improves EuroSAT classification, but the best result comes from selective or full fine-tuning rather than from freezing the backbone. Among the tested strategies, `partial_ft` provides the best balance of accuracy, speed, and efficiency, so it is a strong reference point for the later experiments and forgetting analysis.

#### 4.2 When Transfer Helps: Same-Size Cross-Domain Comparison

This section compares EuroSAT and CIFAR-10 under the same sample budget in order to isolate the role of source-target domain similarity. The goal is to determine whether transfer from ImageNet is more effective, and less destructive to the pretrained representation, when the downstream task remains closer to natural-image statistics.

#### 4.3 When Transfer Helps: Same-Domain Data-Size Comparison

This section varies the amount of EuroSAT training data in order to study whether transfer becomes more valuable in lower-data regimes. The key quantity of interest is transfer gain relative to scratch as the target sample budget increases from 10% to 100% of the full EuroSAT training split.

### 5. Catastrophic Forgetting Analysis

Chapter 5 returns each downstream-adapted model to the official ILSVRC2012 validation set and measures how much of the original ImageNet capability has been lost after fine-tuning. Rather than introducing a new task, this chapter serves as a direct post-test for Chapter 4: Chapter 4 asks how well the model learns EuroSAT, while Chapter 5 asks how much ImageNet performance is forgotten in the process.

#### 5.1 Forgetting After the Main EuroSAT Experiment

We evaluate the `linear_probe`, `partial_ft`, and `full_ft` checkpoints from Section 4.1 on the official ILSVRC2012 validation set. For each strategy, forgetting is measured as the drop from the original ImageNet-pretrained ResNet-18 (`A_before`) to the EuroSAT-adapted model evaluated back on ImageNet (`A_after`). `scratch` is excluded because it does not start from ImageNet-pretrained weights.

| Strategy | ImageNet Before Top-1 | ImageNet After Top-1 | Forgetting Top-1 | ImageNet Before Macro-F1 | ImageNet After Macro-F1 | Forgetting Macro-F1 | EuroSAT Test Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear_probe` | TBD | TBD | TBD | TBD | TBD | TBD | 88.54 |
| `partial_ft` | TBD | TBD | TBD | TBD | TBD | TBD | 97.78 |
| `full_ft` | TBD | TBD | TBD | TBD | TBD | TBD | 97.56 |

The key question is whether the best EuroSAT strategy from Section 4.1 also gives the best balance between adaptation and retention. In general, `linear_probe` is expected to forget least, `full_ft` most, and `partial_ft` may provide the best compromise if it preserves substantially more ImageNet performance than `full_ft` while keeping its strong EuroSAT accuracy.

#### 5.2 Forgetting After the Same-Size Cross-Domain Experiment

This section studies whether forgetting differs between EuroSAT and CIFAR-10 when the downstream sample budget is matched. The analysis complements Section 4.2 by asking whether tasks that are easier to transfer to are also less likely to damage the original ImageNet representation.

#### 5.3 Forgetting After the Data-Size Experiment

This section evaluates how forgetting changes as the amount of EuroSAT downstream data increases. Combined with Section 4.3, it allows the report to analyze the trade-off between improved downstream adaptation and greater modification of the pretrained representation.

### 6. Discussion

The discussion chapter synthesizes the evidence from the EuroSAT main experiment, the two supporting transfer analyses, and the ImageNet forgetting results. Its role is to interpret when transfer helps most, why frozen features are insufficient in this project, and where the main limitations of the current study remain.

### 7. Conclusion

The conclusion summarizes the report's main answers to the three overarching questions: whether transfer helps EuroSAT classification, under what conditions transfer helps most, and how strongly different fine-tuning strategies induce catastrophic forgetting on the original ImageNet task.

### 8. References

[1] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How Transferable Are Features in Deep Neural Networks?," in *Advances in Neural Information Processing Systems 27*, 2014, pp. 3320-3328.

[2] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei, "ImageNet Large Scale Visual Recognition Challenge," *International Journal of Computer Vision*, vol. 115, no. 3, pp. 211-252, 2015. doi: 10.1007/s11263-015-0816-y.

[3] G. Cheng, X. Xie, J. Han, L. Guo, and G.-S. Xia, "Remote Sensing Image Scene Classification Meets Deep Learning: Challenges, Methods, Benchmarks, and Opportunities," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 13, pp. 3735-3756, 2020. doi: 10.1109/JSTARS.2020.3005403.

[4] P. Helber, B. Bischke, A. Dengel, and D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 12, no. 7, pp. 2217-2226, 2019. doi: 10.1109/JSTARS.2019.2918242.

[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016, pp. 770-778. doi: 10.1109/CVPR.2016.90.

[6] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2009, pp. 248-255. doi: 10.1109/CVPR.2009.5206848.

[7] I. J. Goodfellow, M. Mirza, X. Da, A. Courville, and Y. Bengio, "An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks," arXiv:1312.6211, 2013.

[8] A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Technical Report, University of Toronto, 2009.

[9] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell, "Overcoming Catastrophic Forgetting in Neural Networks," *Proceedings of the National Academy of Sciences*, vol. 114, no. 13, pp. 3521-3526, 2017. doi: 10.1073/pnas.1611835114.

[10] Z. Li and D. Hoiem, "Learning Without Forgetting," in *European Conference on Computer Vision*, 2016, pp. 614-629. doi: 10.1007/978-3-319-46493-0_37.
