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

All experiments use ResNet-18 as the backbone architecture [5]. This choice keeps the study computationally feasible while still providing a standard CNN baseline whose transfer behavior is well understood. Unless otherwise stated, transferred models are initialized with the official torchvision ImageNet-1K pretrained weights, while the scratch baseline uses random initialization. The final fully connected layer is replaced to match the target number of classes. For EuroSAT and CIFAR-10, this means a 10-way classifier; for the forgetting analysis, the original 1000-way ImageNet head is restored when evaluating the previous task.

The report studies four training strategies that differ only in which parameters are allowed to update during downstream training. In `scratch`, the entire ResNet-18 is trained from random initialization. In `linear_probe`, the ImageNet-pretrained backbone is frozen and only the classifier is optimized. In `partial_ft`, the classifier and the final residual stage (`layer4`) are trainable while earlier layers remain frozen. In `full_ft`, all layers are fine-tuned from the ImageNet initialization. For the 10-class downstream tasks, these settings correspond to 11,181,642 trainable parameters for `scratch`, 5,130 for `linear_probe`, 8,398,858 for `partial_ft`, and 11,181,642 for `full_ft`. This design makes adaptation depth the primary experimental variable.

#### 3.2 Datasets and Splits

The main downstream benchmark is EuroSAT [4], using the RGB image collection derived from Sentinel-2 satellite imagery. The repository uses a metadata file that defines a fixed split of 27,000 total images into 18,900 training samples, 4,050 validation samples, and 4,050 test samples. The split is class balanced, yielding 2,100 training images, 450 validation images, and 450 test images per class across the 10 land-use categories. This fixed split is important because it ensures that the strategy comparison in Section 4.1 is not confounded by different train/validation/test partitions.

Two additional dataset settings are used for the later analyses. First, CIFAR-10 [8] serves as a same-size natural-image comparison dataset for studying domain similarity under a matched sample budget. In this experiment, balanced subsampling reduces CIFAR-10 to the same split sizes used for EuroSAT: 18,900 training, 4,050 validation, and 4,050 test images. Second, the data-size analysis keeps the EuroSAT validation and test sets fixed while varying the number of EuroSAT training samples through balanced fractions of 10%, 30%, 60%, and 100% of the 18,900-image training pool. This makes it possible to examine transfer gain as a function of target data availability.

For the catastrophic forgetting analysis, the previous task is the original ImageNet-1K classification problem [2, 6]. Rather than using a proxy previous task, the project evaluates the pretrained and fine-tuned backbones on the official ILSVRC2012 validation set prepared in ImageFolder format. This choice makes the forgetting score directly interpretable as lost performance on the source task that produced the pretrained weights.

#### 3.3 Preprocessing and Data Loading

All downstream images are resized to 224 x 224 to match the ResNet-18 input resolution. During EuroSAT and CIFAR-10 training, the pipeline applies resizing, random horizontal flipping, random rotation up to 15 degrees, tensor conversion, and normalization with ImageNet channel statistics. Validation and test data use deterministic resizing and the same normalization without augmentation. For ImageNet evaluation in the forgetting experiment, the project uses the standard validation preprocessing pattern of resize, center crop, tensor conversion, and ImageNet normalization.

The code uses class-balanced sampling whenever a subset of a split is requested. This implementation detail matters for the domain-gap and data-fraction experiments because it reduces the chance that observed transfer gains are driven by accidental class imbalance rather than true differences in domain similarity or data scale.

#### 3.4 Training Protocol and Metrics

Unless a script explicitly overrides the settings, all runs inherit a shared configuration from `configs/base.yaml`. The default setup uses seed 42, batch size 32, 8 training epochs, AdamW optimization, learning rate 3e-4, and weight decay 1e-4. No learning-rate scheduler is introduced, which keeps the optimization budget identical across strategies and makes the role of initialization and freezing policy easier to interpret. Model selection is based on validation top-1 accuracy: after each epoch, the current checkpoint is evaluated on the validation split, and the best-performing checkpoint is saved and later used for test evaluation.

The report tracks four main performance quantities. The first two are test top-1 accuracy and test macro-F1, which measure final predictive quality on the target task. The third is validation convergence behavior, summarized through the per-epoch curves and, when relevant, the first epoch at which validation accuracy exceeds a predefined threshold. The fourth is training time, reported in seconds, which provides a simple proxy for computational cost. In the forgetting analysis, two additional metrics are recorded on ImageNet before and after EuroSAT fine-tuning: previous-task top-1 accuracy and previous-task macro-F1. Forgetting is then defined as the difference between pre-fine-tuning and post-fine-tuning ImageNet performance.

#### 3.5 Experiment Structure

The full study is organized around a single mainline task plus supporting analyses. Experiment 4.1 is the central EuroSAT strategy ablation over `scratch`, `linear_probe`, `partial_ft`, and `full_ft`, and it establishes the primary conclusion about how best to transfer ResNet-18 from ImageNet to EuroSAT. Experiments 4.2 and 4.3 are supporting analyses motivated by the course guideline to study when transfer helps: Experiment 4.2 tests the role of domain similarity by comparing EuroSAT and CIFAR-10 at the same sample size, while Experiment 4.3 tests the role of target data size by varying the EuroSAT training fraction. After these downstream experiments, Chapter 5 evaluates the completed checkpoints on the official ImageNet validation set, allowing the report to study the trade-off between adaptation to EuroSAT and retention of source-domain knowledge.

### 4. Experiments and Results

#### 4.1 Main Experiment: EuroSAT Strategy Ablation

The main experiment asks the most basic question in the project: under a fixed training budget on EuroSAT, does ImageNet pretraining improve downstream classification relative to training ResNet-18 from scratch? To answer this, we compare four strategies on the full EuroSAT split defined in Section 3: `scratch`, `linear_probe`, `partial_ft`, and `full_ft`. Every run uses the same optimizer, batch size, number of epochs, data split, and evaluation protocol. The only factor that changes is how much of the pretrained backbone is allowed to adapt.

This experiment is the anchor for the rest of the report because each strategy corresponds to a distinct hypothesis about transfer. If `linear_probe` performs competitively, then the ImageNet representation is already well aligned with EuroSAT and only a new classifier is needed. If `partial_ft` closes most of the gap to `full_ft`, then adapting the highest-level semantic layers may be sufficient and full-network optimization may be unnecessary. If `full_ft` clearly outperforms both `linear_probe` and `partial_ft`, then the natural-image representation requires substantial domain-specific adjustment before it becomes optimal for satellite scenes. Finally, if all pretrained variants outperform `scratch`, then the report can conclude that positive transfer exists even under a meaningful domain shift.

The implementation reflects these hypotheses directly. `linear_probe` updates only 5,130 classifier parameters, making it the cheapest transfer option and the cleanest test of feature reuse. `partial_ft` updates the classifier and `layer4`, totaling 8,398,858 trainable parameters, and therefore tests whether limited high-level adaptation is enough. `full_ft` updates all 11.2 million parameters, maximizing flexibility but also increasing the risk of overwriting source-task structure that may later matter for forgetting. Because `scratch` has the same number of trainable parameters as `full_ft` but starts without pretrained weights, the gap between these two settings isolates the contribution of initialization rather than model capacity.

Table 1 reports the realized EuroSAT results from `outputs/eurosat_ablation/results.csv`. The outcome is clear: transfer learning is helpful, but the amount of adaptation matters substantially. The best downstream performance comes from `partial_ft`, which achieves 97.78% test top-1 accuracy and 97.71% test macro-F1. `full_ft` is a close second at 97.56% top-1 and 97.50% macro-F1. Both strategies outperform `scratch` by a large margin of more than four percentage points in top-1 accuracy. By contrast, `linear_probe` performs worst at 88.54% test top-1 accuracy, even below the scratch baseline.

| Strategy | Best Val Top-1 | Test Top-1 | Test Macro-F1 | Train Time (s) | Trainable Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scratch` | 93.36 | 93.21 | 93.04 | 1100.5 | 11,181,642 |
| `linear_probe` | 90.05 | 88.54 | 88.33 | 560.2 | 5,130 |
| `partial_ft` | 97.68 | **97.78** | **97.71** | 641.2 | 8,398,858 |
| `full_ft` | **97.80** | 97.56 | 97.50 | 1068.3 | 11,181,642 |

The strategy ranking supports a nuanced view of transfer. First, ImageNet initialization is clearly beneficial when the network is allowed to adapt at least its highest layers: both `partial_ft` and `full_ft` strongly outperform `scratch`. This shows that the domain gap between ImageNet and EuroSAT is not so severe that pretraining becomes useless. Second, the poor result of `linear_probe` indicates that frozen ImageNet features alone are not well aligned with satellite scene semantics. In other words, transfer is useful here not as a fixed feature extractor, but as an initialization that still requires domain-specific representation adjustment.

The comparison between `partial_ft` and `full_ft` is especially informative. `full_ft` attains the best validation accuracy (97.80%), but `partial_ft` achieves slightly better test performance while using fewer trainable parameters and requiring much less training time. This pattern suggests that adapting only the top residual stage plus the classifier may provide a better bias-variance trade-off than updating the entire network. A plausible explanation is that EuroSAT benefits from preserving more low- and mid-level ImageNet features while still allowing the highest-level semantic features to reorganize around remote sensing classes. Because the margin between `partial_ft` and `full_ft` is small, we do not interpret this as definitive evidence that partial fine-tuning is always superior; however, it strongly suggests that full-network adaptation is not necessary to obtain near-optimal performance on this task.

The convergence behavior further strengthens this conclusion. Using the summary logs, both `partial_ft` and `full_ft` exceed the 90% validation threshold in the very first epoch, whereas `scratch` does not reach the threshold until epoch 6 and `linear_probe` only does so at epoch 8. This means that transfer not only improves final accuracy but also accelerates optimization substantially. Figure 1 shows the final test accuracy comparison, while Figure 2 shows the validation top-1 trajectories over training. The curves make it visually clear that deeper fine-tuning strategies start from much stronger validation performance and stabilize at a much higher level than either `scratch` or `linear_probe`.

![Figure 1. EuroSAT test top-1 accuracy by training strategy.](outputs/eurosat_ablation/test_top1_acc.png)

*Figure 1. Final EuroSAT test top-1 accuracy for `scratch`, `linear_probe`, `partial_ft`, and `full_ft`. The main performance jump comes from allowing at least the top residual block to adapt.*

![Figure 2. EuroSAT validation top-1 accuracy across epochs.](outputs/eurosat_ablation/val_top1_acc_curve.png)

*Figure 2. Validation top-1 accuracy across training epochs. `partial_ft` and `full_ft` converge faster and to substantially better optima than the other two strategies.*

Overall, Experiment 4.1 answers the project's first research question positively but with an important qualification. ImageNet transfer does help EuroSAT classification, yet the benefit depends on allowing nontrivial adaptation of the pretrained representation. The strongest result is not obtained by freezing the backbone, but by selectively or fully fine-tuning it. Among the tested strategies, `partial_ft` offers the best overall trade-off between accuracy, convergence speed, and computational cost, making it a strong candidate for the remaining experiments and for the later forgetting analysis.

#### 4.2 When Transfer Helps: Same-Size Cross-Domain Comparison

This section compares EuroSAT and CIFAR-10 under the same sample budget in order to isolate the role of source-target domain similarity. The goal is to determine whether transfer from ImageNet is more effective, and less destructive to the pretrained representation, when the downstream task remains closer to natural-image statistics.

#### 4.3 When Transfer Helps: Same-Domain Data-Size Comparison

This section varies the amount of EuroSAT training data in order to study whether transfer becomes more valuable in lower-data regimes. The key quantity of interest is transfer gain relative to scratch as the target sample budget increases from 10% to 100% of the full EuroSAT training split.

### 5. Catastrophic Forgetting Analysis

Chapter 5 returns each downstream-adapted model to the official ILSVRC2012 validation set and measures how much of the original ImageNet capability has been lost after fine-tuning. Rather than introducing a new task, this chapter serves as a direct post-test for the experiments in Chapter 4: Chapter 4 asks how well the model learns EuroSAT, while Chapter 5 asks how much ImageNet performance is forgotten in the process.

#### 5.1 Forgetting After the Main EuroSAT Experiment

This section evaluates the checkpoints from Section 4.1 on the official ImageNet validation set and compares forgetting across `linear_probe`, `partial_ft`, and `full_ft`. The objective is to identify which transfer strategy best balances strong EuroSAT adaptation against retention of source-task knowledge.

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
