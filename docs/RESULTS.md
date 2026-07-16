# Results

## Reliable 16-seed confirmation

Reference configuration: SNV + ANOVA Top-125 + EasyEnsemble-50 + threshold 0.5.

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| F1 | 0.7615 | 0.0357 | 0.6923 | 0.8214 |
| ROC-AUC | 0.8971 | — | — | — |
| PR-AUC | 0.8268 | — | — | — |

The complete per-seed metrics are stored in `results/reliable_16seed_rows.csv` and the aggregate table in `results/reliable_16seed_summary.csv`.

## Exploratory alternatives

A three-selector ensemble produced F1 0.7644 in the same 16-seed round, but the paired improvement over ANOVA Top-125 was only +0.0029, with 7 wins, 1 tie and 8 losses; Wilcoxon `p = 0.776`. It was not retained because the gain was not reliable and the training cost was roughly tripled.

A chemistry-keyword aggregation experiment reached F1 0.7676 on a separate 12-seed confirmation. The paired gain over the corresponding Easy-50 baseline was about +0.005 and was unstable, so it remains exploratory rather than the default.

## Interpretation

The evidence supports a stable performance level around F1 0.75–0.76. Higher values near 0.80 occurred in selected development splits but did not consistently survive independent repeated validation.
