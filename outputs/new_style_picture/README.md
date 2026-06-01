# New Style Picture

This directory contains regenerated publication-style report figures.

Data source:

- `eurosat_ablation/*`: generated from `outputs/eurosat_ablation/results.csv` and per-run `metrics.json`.
- `domain_gap/*`: generated from `outputs/domain_gap/results.csv`.
- `forgetting_main/*`: generated from `outputs/forgetting_main/forgetting_results.csv`.
- `forgetting_domain_gap/*`: generated from `outputs/forgetting_domain_gap/forgetting_results.csv`.
- `data_fraction/*` and `forgetting_data_fraction/*`: generated from the paper table values because the repository does not include `outputs/data_fraction/results.csv`, `transfer_gain.csv`, or the original `frac_*/*/summary.json` artifacts.

All percentage values are plotted directly as percentages, not 0-1 decimals.
Each figure is exported as PNG, PDF, and SVG.
