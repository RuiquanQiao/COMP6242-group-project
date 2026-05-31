#### 4.3 When Transfer Helps: Same-Domain Data-Size Comparison

This section varies the amount of EuroSAT training data in order to study whether transfer becomes more valuable in lower-data regimes. The model and target domain are fixed, so the main variable is the target sample budget. We compare `scratch`, `linear_probe`, and `full_ft` at 10%, 30%, 60%, and 100% of the EuroSAT training split.

| Train Fraction | `scratch` Top-1 | `linear_probe` Top-1 | `full_ft` Top-1 | `linear_probe` Gain | `full_ft` Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 68.02 | 76.20 | **93.58** | +8.18 | +25.56 |
| 30% | 82.83 | 85.10 | **96.65** | +2.27 | +13.82 |
| 60% | 89.46 | 86.99 | **96.88** | -2.47 | +7.41 |
| 100% | 89.39 | 87.93 | **97.15** | -1.47 | +7.75 |

<table>
  <tr>
    <td width="50%" align="center">
      <img src="outputs/outline_figures/section_4_3_accuracy_curve.svg" alt="EuroSAT accuracy across training fractions" width="100%" />
    </td>
    <td width="50%" align="center">
      <img src="outputs/outline_figures/section_4_3_transfer_gain_curve.svg" alt="Transfer gain over scratch across training fractions" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center"><em>(a) EuroSAT test top-1 accuracy as the training set grows.</em></td>
    <td align="center"><em>(b) Transfer gain relative to the scratch baseline.</em></td>
  </tr>
</table>

*Figure 3. Same-domain data-size comparison on EuroSAT. The left panel shows final test accuracy, while the right panel shows transfer gain over `scratch`. Transfer is most useful when the target training set is small.*

The results show that `full_ft` is the strongest strategy at every data size. With only 10% of the EuroSAT training data, `scratch` reaches 68.02% test accuracy, while `full_ft` reaches 93.58%. This is a gain of 25.56 percentage points. The large gap suggests that ImageNet pretraining supplies useful visual features when the target dataset is too small to learn strong representations from random initialization.

As the amount of EuroSAT data increases, the scratch model improves quickly. Its accuracy rises from 68.02% at 10% data to about 89% at 60% and 100% data. At the same time, the gain from `full_ft` becomes smaller, falling from +25.56 points at 10% data to about +7 to +8 points at 60% and 100% data. This pattern supports the main hypothesis of this section: transfer helps most in the low-data regime, because pretraining improves sample efficiency.

`linear_probe` gives a different pattern. It helps at 10% and 30% data, but becomes worse than `scratch` at 60% and 100% data. This means that frozen ImageNet features are useful when few target samples are available, but they are not flexible enough when more EuroSAT data can be used. In larger-data settings, the model needs to adapt its representation to remote-sensing textures and scene layouts, not only train a new classifier head.

Overall, Section 4.3 shows that the value of transfer depends on the target data budget. `full_ft` gives the best downstream performance, but its relative advantage is largest when labeled EuroSAT data is limited. This makes full fine-tuning especially important for low-resource remote-sensing classification.

#### 5.3 Forgetting After the Data-Size Experiment

This section evaluates how forgetting changes as the amount of EuroSAT downstream data increases. It uses the checkpoints from Section 4.3 and returns each adapted model to the official ILSVRC2012 validation set. The ImageNet-pretrained ResNet-18 obtains 69.76% top-1 accuracy before adaptation. Forgetting is measured as the drop from this value after EuroSAT training.

| Train Fraction | Strategy | ImageNet After Top-1 | Forgetting Top-1 | EuroSAT Top-1 |
| ---: | --- | ---: | ---: | ---: |
| 10% | `scratch` | 0.12 | 69.64 | 68.02 |
| 10% | `linear_probe` | 32.50 | 37.26 | 76.20 |
| 10% | `full_ft` | 1.24 | 68.52 | 93.58 |
| 30% | `scratch` | 0.08 | 69.68 | 82.83 |
| 30% | `linear_probe` | 32.11 | 37.65 | 85.10 |
| 30% | `full_ft` | 0.25 | 69.51 | 96.65 |
| 60% | `scratch` | 0.13 | 69.63 | 89.46 |
| 60% | `linear_probe` | 31.62 | 38.14 | 86.99 |
| 60% | `full_ft` | 0.25 | 69.51 | 96.88 |
| 100% | `scratch` | 0.09 | 69.67 | 89.39 |
| 100% | `linear_probe` | 32.13 | 37.63 | 87.93 |
| 100% | `full_ft` | 0.20 | 69.56 | 97.15 |

<table>
  <tr>
    <td width="50%" align="center">
      <img src="outputs/outline_figures/section_5_3_forgetting_curve_transfer_methods.svg" alt="ImageNet forgetting for transfer methods" width="100%" />
    </td>
    <td width="50%" align="center">
      <img src="outputs/outline_figures/section_5_3_transfer_forgetting_tradeoff.svg" alt="Transfer gain and forgetting trade-off" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center"><em>(a) ImageNet forgetting for the transfer methods.</em></td>
    <td align="center"><em>(b) Downstream gain compared with ImageNet forgetting.</em></td>
  </tr>
</table>

*Figure 4. Forgetting after the Section 4.3 data-size experiment. `full_ft` gives the highest EuroSAT accuracy but almost completely removes compatibility with the original ImageNet classifier. `linear_probe` preserves more ImageNet performance, but its EuroSAT gain is smaller.*

The main result is that forgetting is controlled more by the training strategy than by the amount of EuroSAT data. `full_ft` shows severe forgetting at every data size. Its ImageNet top-1 accuracy drops from 69.76% to between 0.20% and 1.24%. This means that full fine-tuning strongly changes the pretrained representation. The resulting backbone works very well for EuroSAT, but it no longer matches the original ImageNet classification head.

`linear_probe` forgets less, but it still loses a large amount of ImageNet performance. Its ImageNet top-1 accuracy after adaptation stays near 31% to 32%, giving about 37 to 38 percentage points of forgetting. This is much better than `full_ft`, but it is not zero forgetting. A likely reason is that BatchNorm statistics can still shift toward the EuroSAT distribution during training, even when most backbone weights are frozen.

The `scratch` rows should be interpreted carefully. These models were not initialized from ImageNet, so they cannot strictly forget ImageNet knowledge. Their near-zero ImageNet accuracy only acts as a lower reference point. It shows what happens when an EuroSAT-trained backbone is paired with the original ImageNet head without having preserved the ImageNet representation.

Combining Sections 4.3 and 5.3 gives a clear trade-off. At 10% EuroSAT data, `full_ft` gives the largest transfer gain (+25.56 points), but it also causes almost complete ImageNet forgetting. As the data fraction increases, the transfer gain becomes smaller, but the forgetting remains high. Therefore, more target data does not produce a clear monotonic increase in forgetting in this experiment. Instead, the depth of fine-tuning is the main factor.

Overall, `full_ft` is the best choice if the goal is EuroSAT accuracy only. `linear_probe` is more conservative because it preserves more ImageNet ability, but its downstream adaptation is weaker and becomes worse than `scratch` at larger data sizes. This result suggests that future work should test intermediate strategies, such as partial fine-tuning or frozen BatchNorm, to improve the balance between learning the new remote-sensing task and retaining the original ImageNet capability.
