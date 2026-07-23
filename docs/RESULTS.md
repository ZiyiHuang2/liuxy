# Results

## Current highest enhanced candidate: round 6

Configuration:

```text
SNV
+ fixed ANOVA Top-125 EasyEnsemble-50
+ ANOVA-weighted random-150 EasyEnsemble-50
+ Top-250 random-125 EasyEnsemble-50
→ equal probability average
→ threshold 0.5
```

Thirty-two previously unused shuffle seeds were evaluated with five-fold outer cross-validation.

| Method | F1 mean | F1 std | F1 minimum | F1 maximum | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Feature-diverse three-branch ensemble** | **0.7714** | 0.0286 | **0.7037** | 0.8319 | **0.9018** | **0.8319** |
| Fixed Top-125 EasyEnsemble-50 | 0.7658 | 0.0308 | 0.6852 | 0.8246 | 0.8993 | 0.8297 |

Paired difference, enhanced minus fixed:

- mean F1 difference: `+0.00556`;
- 17 wins, 3 ties and 12 losses;
- bootstrap 95% CI: approximately `[-0.00002, +0.01126]`;
- Wilcoxon `p = 0.0938`.

The gain improves the minimum F1 but remains small and near rather than beyond the conventional significance boundary. The enhanced model is the current highest confirmed candidate, not proof of definitive superiority.

Complete records:

- `results/round6/confirm32_rows_*.csv`;
- `results/round6/confirm32_summary.csv`;
- `results/round6/confirm32_paired.csv`.

## Canonical seed 42

| Method | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Feature-diverse three-branch ensemble | **0.7593** | 0.8884 | 0.8264 |
| Fixed Top-125 | 0.7321 | 0.8873 | 0.8205 |

This fixed split supports the enhanced direction but is not the primary generalization claim.

## Threshold confirmation

| Threshold | F1 mean | F1 std | Minimum F1 |
|---|---:|---:|---:|
| **0.500** | **0.7565** | 0.0422 | 0.6726 |
| 0.495 | 0.7520 | 0.0350 | 0.6783 |

The lower threshold failed independent confirmation. All deployable models continue to use 0.5.

## Earlier stable baseline

The original frozen SNV + ANOVA Top-125 + EasyEnsemble-50 pipeline reached F1 `0.7615 ± 0.0357` in its original 16-seed confirmation. The round-6 paired baseline is `0.7658 ± 0.0308` because it uses a different, fully fresh set of 32 shuffle seeds.

## Rejected round-6 directions

- PLS, Elastic Net/L1 Logistic, RBF-SVM, shrinkage LDA and regularized QDA;
- explicit VOC log-ratios, pairwise differences and correlation-module aggregates;
- inner-OOF adaptive branch weights and thresholds;
- chemistry-keyword aggregation at material weight;
- threshold 0.495.

Development values near 0.80 are not reported as generalization results.

## Retained Unknown VOC comparison

An offline payload reconstructs the 780-feature combined matrix and the 335-feature Unknown-only matrix from the original raw VOC table. All candidates use five-fold outer CV, fold-local SNV and ANOVA, a fixed threshold of 0.5, and identical model random states across feature representations.

The lightweight six-seed discovery stage ranked direct retention below the 445-feature known baseline: combined F1 `0.7452`, appended F1 `0.7415`, and Unknown-only F1 `0.6301`, versus known F1 `0.7828`. The frozen full confirmation therefore evaluated the two strongest probability fusions against the known baseline on 32 new seeds.

| Method | F1 mean | F1 std | Minimum F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| known + 5% Unknown probability | **0.7773** | **0.0191** | **0.7434** | **0.9034** | **0.8365** |
| known + 20% Unknown probability | 0.7765 | 0.0193 | 0.7273 | 0.9021 | 0.8353 |
| known only | 0.7763 | 0.0216 | 0.7339 | 0.9032 | 0.8357 |

For the 5% fusion, paired `ΔF1 = +0.00096`, bootstrap 95% CI `[-0.00188, +0.00370]`, with 10 wins, 16 ties and 6 losses; Wilcoxon `p = 0.642`. The effect is too small and uncertain to justify replacing the known-only model. The formal pipeline continues to remove Unknown.

Complete records are stored in `results/unknown_voc/`.
