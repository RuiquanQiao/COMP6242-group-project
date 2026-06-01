## From ImageNet to EuroSAT: Transfer Learning, Forgetting, and When Transfer Helps

### 0. Project Information

#### AI Use Declaration

AI tools were used as supporting aids for code review, result checking, and language refinement. All experimental design, code execution, interpretation of results, and final report decisions were completed and verified by the group members.

#### Group Contribution

| Member | Main Responsibility |
| --- |---------------------|
| Ruiquan Qiao | TBD                 |
| Tiancheng Xia | TBD                 |
| Guangde Shi | TBD                 |

#### Source Code

The source code and experiment artifacts are available at: `<https://github.com/RuiquanQiao/COMP6242-group-project>`.

### 1. Introduction

ImageNet pretraining is often treated as a reliable starting point for vision tasks, but this assumption becomes less obvious when the target images are satellite scenes rather than object-centric photographs. EuroSAT images are viewed from above and are defined by land-cover patterns, texture, and spatial layout, so the features learned from ImageNet may not transfer automatically.

This project studies that problem by adapting an ImageNet-pretrained ResNet-18 [5, 6] to EuroSAT [4] and comparing it with training from scratch. We test linear probing, partial fine-tuning, and full fine-tuning to understand how much of the pretrained model should be frozen or updated.

Beyond the main EuroSAT experiment, we ask when transfer helps and what it costs. We compare EuroSAT with CIFAR-10 under the same sample budget [8], vary the amount of EuroSAT training data, and evaluate ImageNet performance after downstream adaptation to measure catastrophic forgetting on the original source task [2, 7].

### 2. Related Work

ImageNet-pretrained models are widely used as starting points for transfer learning because large-scale supervised training produces reusable visual features [2, 6]. However, transfer quality depends on task distance, and earlier layers tend to be more general than later layers [1]. This motivates our comparison of linear probing, partial fine-tuning, and full fine-tuning.

Remote sensing scene classification is a useful stress test for transfer because its classes are defined by overhead land-cover patterns rather than object-centric appearance [3]. EuroSAT provides a standard benchmark for this setting with 27,000 Sentinel-2 image patches across 10 classes [4].

Fine-tuning also raises the issue of catastrophic forgetting: performance on a previous task can drop after training on a new task [7]. Although methods such as EWC and Learning without Forgetting address this problem [9, 10], our focus is to measure how much ordinary downstream fine-tuning changes the original ImageNet capability.

### 3. Experimental Setup

<p align="center">
  <img src="figures/experimental_pipeline.svg" alt="Experimental pipeline for transfer learning and forgetting evaluation" width="100%" />
</p>

*Figure 1. Experimental workflow used in Chapters 4 and 5. The downstream experiments compare transfer strategies, target-domain distance, and training-set size. The downstream-trained checkpoints are then re-evaluated on ImageNet to measure catastrophic forgetting.*

#### 3.1 Backbone and Initialization

All experiments use ResNet-18 [5]. Transfer runs start from the official torchvision ImageNet-1K weights, while training from scratch uses random initialization. ResNet-18 first extracts local visual patterns through convolutional layers, then builds increasingly task-specific representations through deeper residual blocks. In this structure, earlier layers tend to capture general low-level features such as edges, corners, and textures, while later layers combine them into higher-level semantic patterns. The final fully connected layer maps the learned representation to class logits, so it must be replaced when the number of target classes changes.

For EuroSAT and CIFAR-10, the original 1000-way ImageNet classifier is replaced with a 10-way classifier. For ImageNet forgetting evaluation, the model is rebuilt with the original 1000-way head and the downstream-adapted backbone is loaded back into it. This allows us to test whether the feature extractor remains compatible with the original ImageNet task after downstream training.

We compare four strategies that update different parts of the network. Training from scratch updates the entire randomly initialized model and serves as the no-transfer baseline. Linear probing freezes the pretrained backbone and trains only the new classifier, testing whether ImageNet features are already separable for the target task. Partial fine-tuning trains the classifier and the final residual stage of ResNet-18, allowing higher-level features to adapt while keeping earlier visual filters fixed. Full fine-tuning updates the entire pretrained network, giving the model maximum flexibility but also the greatest risk of overwriting source-task representations. For the 10-class downstream tasks, their trainable parameter counts are 11,181,642, 5,130, 8,398,858, and 11,181,642 respectively.

#### 3.2 Data Preparation

The main downstream dataset is EuroSAT [4], using a fixed class-balanced split of 27,000 RGB images into 18,900 train, 4,050 validation, and 4,050 test samples. The same split sizes are used for the CIFAR-10 same-size comparison [8], so the domain-gap experiment compares target domains under the same amount of supervision. For the data-size analysis, the EuroSAT validation and test splits stay fixed while the training set is reduced to 10%, 30%, 60%, and 100% of the full training pool. This design changes only the amount of training evidence, while keeping evaluation conditions unchanged. For forgetting, the previous task is the original ImageNet-1K problem [2, 6], evaluated on the official ILSVRC2012 validation set prepared in ImageFolder format.

All inputs are resized to 224 x 224 because ResNet-18 pretrained on ImageNet expects image tensors at this scale. EuroSAT and CIFAR-10 training use random horizontal flip and random rotation as data augmentation. These transformations expose the model to small appearance variations and reduce overfitting, especially when the training set is limited. Validation and test data use deterministic preprocessing so that evaluation measures the learned model rather than random augmentation effects.

All datasets are converted to tensors and normalized with ImageNet mean and standard deviation. This is important for transfer learning because the pretrained ResNet-18 weights were learned under ImageNet-style input normalization; using the same scale keeps the input distribution compatible with the pretrained filters. ImageNet forgetting evaluation follows the standard resize-center-crop pipeline. Whenever a subset of a split is requested, sampling is class balanced so that accuracy and Macro-F1 are not dominated by class-frequency differences.

#### 3.3 Training Protocol and Metrics

Unless overridden, all runs use the shared `configs/base.yaml` settings: seed 42, batch size 32, 8 epochs, AdamW, learning rate 3e-4, and weight decay 1e-4. Each training run follows the standard supervised learning pipeline: images are passed through the network to produce class logits, cross-entropy loss compares these logits with the ground-truth labels, backpropagation computes gradients for the trainable parameters, and AdamW updates those parameters while applying weight decay regularization. The difference between strategies is therefore not the learning objective, but which parameters are allowed to receive gradient updates.

Validation is used to monitor generalization during training. The best checkpoint is selected by validation top-1 accuracy rather than final-epoch accuracy, because the validation set gives an estimate of performance on unseen data and reduces dependence on a single last epoch. Top-1 accuracy is the primary metric throughout the report because all downstream tasks are single-label classification tasks with balanced sampling. Macro-F1 is reported as a secondary check of class-balanced performance in the main strategy, domain-gap, and corresponding forgetting experiments. The data-size experiments focus on top-1 trends for compactness. Transfer gain is computed against the training-from-scratch baseline, and forgetting is measured as the ImageNet performance drop after downstream adaptation. Loss curves are used only as diagnostic evidence for training stability.

#### 3.4 Experiment Structure

The study has one main experiment and two supporting analyses. Section 4.1 compares four transfer strategies on EuroSAT. Section 4.2 studies when transfer helps by comparing EuroSAT and CIFAR-10 at the same sample size. Section 4.3 studies when transfer helps by varying the amount of EuroSAT training data. Chapter 5 then evaluates the resulting checkpoints on the official ImageNet validation set to measure catastrophic forgetting.

### 4. Experiments and Results

#### 4.1 Main Experiment: EuroSAT Strategy Ablation

Section 4.1 asks the core question of the project: on EuroSAT, does ImageNet pretraining help ResNet-18, and which transfer strategy works best? We compare training from scratch, linear probing, partial fine-tuning, and full fine-tuning under the same training budget.

| Strategy | Test Top-1 | Test Macro-F1 | Transfer Gain Top-1 | Train Time (s) | Trainable Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training from scratch | 93.21 | 93.04 | - | 1100.5 | 11,181,642 |
| Linear probing | 88.54 | 88.33 | -4.67 | 560.2 | 5,130 |
| Partial fine-tuning | **97.78** | **97.71** | **+4.57** | 641.2 | 8,398,858 |
| Full fine-tuning | 97.56 | 97.50 | +4.35 | 1068.3 | 11,181,642 |

The results show that transfer is helpful only when the pretrained model is allowed to adapt. Partial fine-tuning gives the best test performance, with full fine-tuning close behind, while linear probing is worst and even falls below the training-from-scratch baseline.

Partial fine-tuning is also the best trade-off overall. It slightly outperforms full fine-tuning on the test set while using fewer trainable parameters and less training time, suggesting that adapting the top stage is sufficient for most of the domain shift. The convergence logs tell the same story: partial fine-tuning and full fine-tuning pass the 90% validation threshold in epoch 1, whereas training from scratch reaches it in epoch 6 and linear probing only in epoch 8.

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

*Figure 2. Side-by-side summary of Experiment 4.1. The left panel shows final EuroSAT test top-1 accuracy, while the right panel shows validation top-1 trajectories. Together they show that partial fine-tuning and full fine-tuning both outperform training from scratch and linear probing, and that deeper fine-tuning converges faster.*

Overall, the EuroSAT ablation shows that transfer is beneficial only when the backbone is allowed to adapt. Linear probing is insufficient for this domain shift, while partial fine-tuning gives the strongest accuracy-efficiency trade-off.

#### 4.2 When Transfer Helps: Same-Size Cross-Domain Comparison

This section compares EuroSAT and CIFAR-10 under the same sample budget in order to isolate the role of source-target domain similarity. Both datasets use 18,900 training, 4,050 validation, and 4,050 test samples, so the main difference is the target domain rather than the amount of supervision. EuroSAT represents a remote-sensing scene classification task with a clear shift from ImageNet, while CIFAR-10 remains closer to natural-image object recognition.

| Dataset | Strategy | Test Top-1 | Test Macro-F1 | Transfer Gain Top-1 |
| --- | --- | ---: | ---: | ---: |
| EuroSAT | Training from scratch | 92.86 | 92.67 | - |
| EuroSAT | Linear probing | 88.54 | 88.33 | -4.32 |
| EuroSAT | Partial fine-tuning | **97.51** | **97.41** | **+4.64** |
| EuroSAT | Full fine-tuning | 97.31 | 97.25 | +4.44 |
| CIFAR-10 | Training from scratch | 78.12 | 78.27 | - |
| CIFAR-10 | Linear probing | 78.20 | 78.00 | +0.07 |
| CIFAR-10 | Partial fine-tuning | **91.19** | **91.18** | **+13.06** |
| CIFAR-10 | Full fine-tuning | 90.07 | 90.03 | +11.95 |

The same-size comparison preserves the main pattern from Section 4.1: transfer works best when the pretrained backbone is allowed to adapt. On EuroSAT, partial fine-tuning and full fine-tuning improve test accuracy over training from scratch by 4.64 and 4.44 percentage points respectively, while linear probing falls below the training-from-scratch baseline. On CIFAR-10, the gains from partial fine-tuning and full fine-tuning are larger, at 13.06 and 11.95 points over training from scratch.

This comparison separates absolute accuracy from transfer benefit. EuroSAT reaches higher final accuracy, but CIFAR-10 receives the larger gain over training from scratch. The broader implications of this domain difference are discussed in Chapter 6.

<table>
  <tr>
    <td width="50%" align="center">
      <img src="outputs/domain_gap/test_top1_acc.png" alt="Same-size EuroSAT and CIFAR-10 test top-1 accuracy" width="100%" />
    </td>
    <td width="50%" align="center">
      <img src="outputs/domain_gap/transfer_gain_top1.png" alt="Same-size EuroSAT and CIFAR-10 top-1 transfer gain" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center"><em>(a) Test top-1 accuracy by dataset and strategy.</em></td>
    <td align="center"><em>(b) Top-1 transfer gain over training from scratch.</em></td>
  </tr>
</table>

*Figure 3. Same-size cross-domain comparison. The left panel shows downstream test top-1 accuracy, while the right panel shows transfer gain over training from scratch. CIFAR-10 receives a larger gain from ImageNet transfer than EuroSAT.*

#### 4.3 When Transfer Helps: Same-Domain Data-Size Comparison

This section varies the amount of EuroSAT training data in order to study whether transfer becomes more valuable in lower-data regimes. The model and target domain are fixed, while the EuroSAT training fraction is changed from 10% to 30%, 60%, and 100%. We compare training from scratch, linear probing, and full fine-tuning to represent no transfer, frozen-feature transfer, and full adaptation. Because this experiment focuses on how performance changes with training-set size, we report top-1 accuracy and top-1 transfer gain as the primary trend metrics.

| Train Fraction | Training from Scratch Top-1 | Linear Probing Top-1 | Full Fine-Tuning Top-1 | Full Fine-Tuning Gain |
| ---: | ---: | ---: | ---: | ---: |
| 10% | 68.02 | 76.20 | **93.58** | +25.56 |
| 30% | 82.83 | 85.10 | **96.65** | +13.82 |
| 60% | 89.46 | 86.99 | **96.88** | +7.41 |
| 100% | 89.39 | 87.93 | **97.15** | +7.75 |

<table>
  <tr>
    <td width="50%" align="center">
      <img src="outputs/data_fraction/test_top1_acc.png" alt="EuroSAT test top-1 accuracy by training fraction" width="100%" />
    </td>
    <td width="50%" align="center">
      <img src="outputs/data_fraction/transfer_gain_top1.png" alt="Transfer gain over training from scratch by training fraction" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center"><em>(a) Test top-1 accuracy by training fraction.</em></td>
    <td align="center"><em>(b) Transfer gain over training from scratch.</em></td>
  </tr>
</table>

*Figure 4. Same-domain data-size comparison. Full fine-tuning performs best at every training fraction, while its transfer gain is largest in the 10% low-data setting.*

The results show that transfer is most valuable when labelled data is limited. With only 10% of the EuroSAT training set, full fine-tuning improves over training from scratch by 25.56 percentage points. As the training fraction increases, the model trained from scratch becomes stronger, and the gain from full fine-tuning decreases to about 7-8 points. This shows that transfer mainly improves sample efficiency in this setting.

Linear probing helps at 10% and 30% data, but it becomes worse than training from scratch at 60% and 100%. The data-size trend shows that transfer is most valuable in the low-data regime; as more EuroSAT labels are available, training from scratch becomes stronger and the relative gain from full fine-tuning decreases.

### 5. Catastrophic Forgetting Analysis

Chapter 5 returns each downstream-adapted model to the official ILSVRC2012 validation set and measures how much of the original ImageNet capability has been lost after fine-tuning. Rather than introducing a new task, this chapter serves as a direct post-test for Chapter 4: Chapter 4 asks how well the model learns EuroSAT, while Chapter 5 asks how much ImageNet performance is forgotten in the process.

In this chapter, a trade-off refers to the balance between downstream adaptation and source-task retention. A desirable model would achieve high downstream accuracy or transfer gain while keeping ImageNet forgetting low. In the scatter plots in Sections 5.1 and 5.2, the horizontal axis is ImageNet forgetting and the vertical axis is downstream test accuracy, so points closer to the upper-left region are preferable. Points on the right side indicate stronger forgetting, even if their downstream accuracy is high. In Section 5.3, the trade-off is shown across training fractions by plotting transfer gain and forgetting as separate trends.

#### 5.1 Forgetting After the Main EuroSAT Experiment

We evaluate the linear probing, partial fine-tuning, and full fine-tuning checkpoints from Section 4.1 on the official ILSVRC2012 validation set. The original ImageNet-pretrained ResNet-18 obtains 69.76% top-1 accuracy and 69.30 macro-F1 before downstream adaptation. For each strategy, forgetting is measured as the drop from this baseline to the EuroSAT-adapted model evaluated back on ImageNet. Training from scratch is excluded because it does not start from ImageNet-pretrained weights.

| Strategy | ImageNet After Top-1 | Forgetting Top-1 | ImageNet After Macro-F1 | Forgetting Macro-F1 | EuroSAT Test Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear probing | 32.03 | 37.73 | 33.87 | 35.43 | 88.54 |
| Partial fine-tuning | 0.50 | 69.26 | 0.24 | 69.06 | **97.78** |
| Full fine-tuning | 0.12 | 69.64 | 0.02 | 69.27 | 97.56 |

The main forgetting experiment reveals a clear adaptation-retention trade-off. Linear probing preserves more source-task ability, while partial fine-tuning and full fine-tuning achieve higher EuroSAT accuracy but almost eliminate ImageNet performance.

<table>
  <tr>
    <td width="50%" align="center">
      <img src="outputs/forgetting_main/forgetting_top1.png" alt="ImageNet top-1 forgetting after the main EuroSAT experiment" width="100%" />
    </td>
    <td width="50%" align="center">
      <img src="outputs/forgetting_main/transfer_forgetting_tradeoff.png" alt="EuroSAT downstream accuracy and ImageNet forgetting trade-off" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center"><em>(a) ImageNet top-1 forgetting.</em></td>
    <td align="center"><em>(b) Downstream accuracy and forgetting trade-off.</em></td>
  </tr>
</table>

*Figure 5. Forgetting after the main EuroSAT experiment. Fine-tuning improves EuroSAT performance but causes much larger ImageNet forgetting than linear probing.*

#### 5.2 Forgetting After the Same-Size Cross-Domain Experiment

This section studies whether forgetting differs between EuroSAT and CIFAR-10 when the downstream sample budget is matched. For each pretrained strategy from Section 4.2, we evaluate the adapted backbone on the official ImageNet validation set and compare it with the original torchvision ResNet-18 baseline. Training from scratch is excluded because it was not initialized from ImageNet.

| Dataset | Strategy | ImageNet After Top-1 | Forgetting Top-1 | ImageNet After Macro-F1 | Forgetting Macro-F1 | Downstream Test Top-1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | Linear probing | 28.53 | 41.23 | 30.40 | 38.90 | 78.20 |
| CIFAR-10 | Partial fine-tuning | 0.59 | 69.17 | 0.35 | 68.95 | **91.19** |
| CIFAR-10 | Full fine-tuning | 0.23 | 69.53 | 0.09 | 69.21 | 90.07 |
| EuroSAT | Linear probing | 32.03 | 37.73 | 33.87 | 35.43 | 88.54 |
| EuroSAT | Partial fine-tuning | 0.23 | 69.53 | 0.07 | 69.23 | **97.51** |
| EuroSAT | Full fine-tuning | 0.13 | 69.63 | 0.03 | 69.27 | 97.31 |

The forgetting results follow the same strategy-level pattern as Section 5.1. Partial fine-tuning and full fine-tuning obtain the strongest downstream accuracy, but both reduce ImageNet top-1 accuracy from 69.76% to below 1% on both downstream datasets. Linear probing retains more ImageNet performance because the backbone remains frozen, but it is also the weakest pretrained strategy on the downstream tasks.

The domain-gap forgetting results show that a downstream task being closer to ImageNet does not prevent forgetting after fine-tuning. CIFAR-10 receives larger transfer gains than EuroSAT, but fine-tuning on either dataset still severely damages ImageNet performance.

<table>
  <tr>
    <td width="50%" align="center">
      <img src="outputs/forgetting_domain_gap/forgetting_top1.png" alt="ImageNet top-1 forgetting after same-size domain-gap experiment" width="100%" />
    </td>
    <td width="50%" align="center">
      <img src="outputs/forgetting_domain_gap/transfer_forgetting_tradeoff.png" alt="Downstream accuracy and ImageNet forgetting trade-off after same-size domain-gap experiment" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center"><em>(a) ImageNet top-1 forgetting.</em></td>
    <td align="center"><em>(b) Downstream accuracy and forgetting trade-off.</em></td>
  </tr>
</table>

*Figure 6. Forgetting after same-size downstream adaptation. Fine-tuning causes much larger ImageNet forgetting than linear probing on both EuroSAT and CIFAR-10, even when the target domain changes.*

#### 5.3 Forgetting After the Data-Size Experiment

This section evaluates how forgetting changes as the amount of EuroSAT downstream data increases. Combined with Section 4.3, it allows the report to analyze the trade-off between improved downstream adaptation and greater modification of the pretrained representation. The original ImageNet-pretrained ResNet-18 obtains 69.76% top-1 accuracy before EuroSAT adaptation, and forgetting is measured as `A_before - A_after`. For consistency with Section 4.3, this section focuses on top-1 forgetting across training fractions.

| Train Fraction | Strategy | ImageNet After Top-1 | Forgetting Top-1 | EuroSAT Top-1 |
| ---: | --- | ---: | ---: | ---: |
| 10% | Linear probing | 32.50 | 37.26 | 76.20 |
| 10% | Full fine-tuning | 1.24 | 68.52 | 93.58 |
| 30% | Linear probing | 32.11 | 37.65 | 85.10 |
| 30% | Full fine-tuning | 0.25 | 69.51 | 96.65 |
| 60% | Linear probing | 31.62 | 38.14 | 86.99 |
| 60% | Full fine-tuning | 0.25 | 69.51 | 96.88 |
| 100% | Linear probing | 32.13 | 37.63 | 87.93 |
| 100% | Full fine-tuning | 0.20 | 69.56 | 97.15 |

<table>
  <tr>
    <td width="50%" align="center">
      <img src="outputs/forgetting_data_fraction/forgetting_top1_transfer_methods.png" alt="ImageNet top-1 forgetting by transfer method" width="100%" />
    </td>
    <td width="50%" align="center">
      <img src="outputs/forgetting_data_fraction/transfer_forgetting_tradeoff.png" alt="Transfer gain and forgetting trade-off" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center"><em>(a) ImageNet top-1 forgetting by transfer method.</em></td>
    <td align="center"><em>(b) Transfer gain and forgetting trade-off.</em></td>
  </tr>
</table>

*Figure 7. Forgetting after the data-size experiment. Full fine-tuning gives the strongest EuroSAT adaptation, but it also causes much larger ImageNet forgetting than linear probing.*

The main result is that forgetting depends more on the training strategy than on the amount of EuroSAT data. Full fine-tuning forgets heavily at every data size: ImageNet top-1 drops from 69.76% to about 0.20%-1.24%, meaning that the adapted backbone is no longer compatible with the original ImageNet classifier.

Linear probing preserves more ImageNet ability. Its ImageNet top-1 stays around 31%-32%, giving about 37-38 points of forgetting. The data-size forgetting results show that forgetting is driven more by the fine-tuning strategy than by the amount of EuroSAT data: full fine-tuning forgets heavily at every data fraction, while linear probing preserves more ImageNet accuracy but gives lower EuroSAT performance.

### 6. Discussion

The experimental findings suggest several explanations for when transfer helps and why catastrophic forgetting occurs.

Linear probing performs poorly on EuroSAT because the fixed ImageNet representation is not directly aligned with remote-sensing scene categories. A linear classifier can only draw new decision boundaries on top of the existing feature space; it cannot reshape the feature extractor itself. If EuroSAT classes are not already well separated by ImageNet features, the classifier head has limited capacity to correct the mismatch. This is plausible because EuroSAT relies on overhead texture, land-cover patterns, and spatial layout, whereas ImageNet pretraining is dominated by object-centric natural images [3, 4]. Linear probing therefore protects the pretrained representation, but it also exposes the limitation of frozen features under domain shift.

Partial fine-tuning works well because it matches the hierarchical structure of convolutional networks. Earlier convolutional layers usually encode reusable visual primitives such as edges, corners, color contrast, and local texture, while deeper residual blocks encode more task-specific combinations of those features [1]. Updating the final residual stage and the classifier lets the model move the high-level representation toward EuroSAT without disturbing all low-level filters. This gives enough flexibility to adapt to the target task, while avoiding some unnecessary parameter updates in the earlier backbone. Under the short training budget used here, the extra flexibility of full fine-tuning does not produce a clear EuroSAT advantage over partial fine-tuning.

The CIFAR-10 comparison also shows why absolute accuracy and transfer gain need to be separated. CIFAR-10 has lower absolute accuracy than EuroSAT, but larger transfer gain. This is not contradictory. EuroSAT may be easier under the current split and preprocessing, so both training from scratch and fine-tuned models reach high accuracy. CIFAR-10 remains harder in absolute terms, partly because resizing small 32 x 32 images to 224 x 224 cannot create new visual detail. However, CIFAR-10 still shares more natural-image structure with ImageNet than EuroSAT does, so the pretrained filters provide a stronger optimization starting point and a larger improvement over training from scratch.

The data-size results reflect the same principle from another angle. Deep networks have many parameters, so training from scratch requires enough labelled examples to learn useful low-level and high-level features without overfitting. When EuroSAT labels are limited, pretrained ImageNet features act as a strong inductive bias: the model starts from filters that already encode useful visual regularities instead of learning all representations from random initialization. As the training set grows, the training-from-scratch baseline receives more evidence, learns stronger features, and narrows the gap. This explains why transfer mainly improves sample efficiency in this project.

Catastrophic forgetting arises because fine-tuning optimizes the shared backbone for a new 10-class objective, not for the original 1000-class ImageNet objective. Backpropagation changes any unfrozen weights in the direction that reduces downstream cross-entropy loss, even if those changes remove information that was useful for ImageNet. This behavior is consistent with prior observations that optimization on a new task can reduce performance on a previous task [7]. Without an explicit constraint to preserve ImageNet performance, the adapted feature extractor becomes less compatible with the original ImageNet classifier. Linear probing forgets less because it freezes the backbone and only learns the downstream classification head. Since the forgetting evaluation reloads the adapted backbone into the ImageNet classifier, a mostly unchanged backbone remains more compatible with the original ImageNet task. The cost is weaker downstream performance because the representation cannot adapt deeply to EuroSAT or CIFAR-10.

This explains the trade-off plots in Sections 5.1 and 5.2. Linear probing appears on the lower-forgetting side because the pretrained backbone is preserved, but it does not reach the highest downstream accuracy because only the classifier head can adapt. Partial fine-tuning and full fine-tuning move toward higher downstream accuracy, but they also move to the higher-forgetting side because many more backbone parameters are updated. The experiments therefore do not find a strategy in the ideal upper-left region of the trade-off plots, where downstream accuracy would be high and ImageNet forgetting would be low.

The main limitation is that these experiments use a single architecture, one random seed, and a short fixed training budget. We also measure forgetting directly through ImageNet re-evaluation, but we do not test mitigation methods such as regularization, rehearsal, Learning without Forgetting, EWC, or freezing additional layers [9, 10].

### 7. Conclusion

This project shows that ImageNet transfer can substantially improve EuroSAT classification, but only when the pretrained backbone is allowed to adapt. Partial fine-tuning gives the strongest balance of EuroSAT accuracy and efficiency, while linear probing retains more ImageNet ability at the cost of weaker downstream performance.

Transfer is most useful when labelled target data are limited or when the target domain is closer to ImageNet. However, stronger adaptation also causes severe catastrophic forgetting on the original ImageNet task. Future work should therefore focus on mitigation methods that preserve source-task performance while retaining the downstream gains of fine-tuning.

### 8. References

[1] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How Transferable Are Features in Deep Neural Networks?," in *Advances in Neural Information Processing Systems 27*, 2014, pp. 3320-3328.

[2] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei, "ImageNet Large Scale Visual Recognition Challenge," *International Journal of Computer Vision*, vol. 115, no. 3, pp. 211-252, 2015. doi: 10.1007/s11263-015-0816-y.

[3] G. Cheng, X. Xie, J. Han, L. Guo, and G.-S. Xia, "Remote Sensing Image Scene Classification Meets Deep Learning: Challenges, Methods, Benchmarks, and Opportunities," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 13, pp. 3735-3756, 2020. doi: 10.1109/JSTARS.2020.3005403.

[4] P. Helber, B. Bischke, A. Dengel, and D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 12, no. 7, pp. 2217-2226, 2019. doi: 10.1109/JSTARS.2019.2918242.

[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016, pp. 770-778. doi: 10.1109/CVPR.2016.90.

[6] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2009, pp. 248-255. doi: 10.1109/CVPR.2009.5206848.

[7] I. J. Goodfellow, M. Mirza, D. Xiao, A. Courville, and Y. Bengio, "An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks," arXiv:1312.6211, 2013.

[8] A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Technical Report, University of Toronto, 2009.

[9] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell, "Overcoming Catastrophic Forgetting in Neural Networks," *Proceedings of the National Academy of Sciences*, vol. 114, no. 13, pp. 3521-3526, 2017. doi: 10.1073/pnas.1611835114.

[10] Z. Li and D. Hoiem, "Learning Without Forgetting," in *European Conference on Computer Vision*, 2016, pp. 614-629. doi: 10.1007/978-3-319-46493-0_37.
