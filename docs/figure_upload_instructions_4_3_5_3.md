# Figure Upload Instructions for Sections 4.3 and 5.3

This note lists the files that should be uploaded to the GitHub `outputs` folder and the figures that should be referenced in `report.md`.

## 1. Upload These Output Folders

Upload the whole folder:

```text
outputs/data_fraction/
```

This folder belongs to Section 4.3, the same-domain data-size experiment.

Upload the whole folder:

```text
outputs/forgetting_data_fraction/
```

This folder belongs to Section 5.3, the forgetting analysis after the data-size experiment.

## 2. Images for Section 4.3

In `report.md`, Section `4.3 When Transfer Helps: Same-Domain Data-Size Comparison`, insert these two images:

```text
outputs/data_fraction/test_top1_acc.png
outputs/data_fraction/transfer_gain_top1.png
```

Recommended layout:

```html
<table>
  <tr>
    <td width="50%" align="center">
      <img src="outputs/data_fraction/test_top1_acc.png" alt="EuroSAT accuracy across training fractions" width="100%" />
    </td>
    <td width="50%" align="center">
      <img src="outputs/data_fraction/transfer_gain_top1.png" alt="Transfer gain over scratch across training fractions" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center"><em>(a) EuroSAT test top-1 accuracy as the training set grows.</em></td>
    <td align="center"><em>(b) Transfer gain relative to the scratch baseline.</em></td>
  </tr>
</table>
```

Optional extra figure, if space allows:

```text
outputs/data_fraction/test_macro_f1.png
```

## 3. Images for Section 5.3

In `report.md`, Section `5.3 Forgetting After the Data-Size Experiment`, insert these two images:

```text
outputs/forgetting_data_fraction/forgetting_top1_transfer_methods.png
outputs/forgetting_data_fraction/transfer_forgetting_tradeoff.png
```

Recommended layout:

```html
<table>
  <tr>
    <td width="50%" align="center">
      <img src="outputs/forgetting_data_fraction/forgetting_top1_transfer_methods.png" alt="ImageNet forgetting for transfer methods" width="100%" />
    </td>
    <td width="50%" align="center">
      <img src="outputs/forgetting_data_fraction/transfer_forgetting_tradeoff.png" alt="Transfer gain and forgetting trade-off" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center"><em>(a) ImageNet forgetting for the transfer methods.</em></td>
    <td align="center"><em>(b) Downstream gain compared with ImageNet forgetting.</em></td>
  </tr>
</table>
```

Optional extra figures, if space allows:

```text
outputs/forgetting_data_fraction/forgetting_top1.png
outputs/forgetting_data_fraction/forgetting_macro_f1.png
```

## 4. Result Tables and Text

Use this file as the source text for Sections 4.3 and 5.3:

```text
outputs/data_fraction/report_section_4_3_and_5_3.md
```

The same file is also copied here for convenience:

```text
outputs/forgetting_data_fraction/report_section_4_3_and_5_3.md
```

Use these CSV files if the report needs exact numbers:

```text
outputs/data_fraction/results.csv
outputs/forgetting_data_fraction/results.csv
```

## 5. Git Commands

Because `outputs/` is ignored by `.gitignore`, use forced add:

```bash
git add -f outputs/data_fraction outputs/forgetting_data_fraction
git add docs/figure_upload_instructions_4_3_5_3.md
git commit -m "Add data fraction results and forgetting figures"
git push origin main
```

If `main` is behind the remote branch, first pull or push to a feature branch:

```bash
git switch -c guangde-data-fraction-results
git push origin guangde-data-fraction-results
```
