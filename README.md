# VOC EasyEnsemble Classification

A complete, reproducible binary VOC classification project for a high-dimensional, small-sample setting:

- 159 samples;
- 445 known VOC features after removing exact-name `Unknown` columns;
- class counts 106 vs 53;
- original MATLAB dataset and a hash-verified retained-Unknown payload included.

## Current frozen model

| Model | Pipeline | Status | Repeated-CV F1 |
|---|---|---|---:|
| Stable baseline | SNV → ANOVA Top-125 → EasyEnsemble-50 → threshold 0.5 | simplest reliable model | about 0.76 |
| **Feature-diverse enhancement** | SNV → three VOC subspace branches → 3 × EasyEnsemble-50 → equal probability mean → threshold 0.5 | **current highest confirmed candidate** | **0.7714 ± 0.0286** |

The enhanced model is numerically stronger, but its paired 32-seed advantage over the fixed Top-125 baseline is small and not conventionally significant (`ΔF1 = +0.00556`, 95% CI approximately `[-0.00002, 0.01126]`, Wilcoxon `p = 0.0938`). It is the current best enhanced candidate, not conclusively superior.

## Feature-diverse model

The enhanced model trains three branches inside every training fold:

1. **Fixed strong panel:** ANOVA Top-125;
2. **Weighted exploration:** each balanced submodel samples 150 of all 445 VOCs using training-fold ANOVA scores as probabilities;
3. **Top-pool diversity:** each balanced submodel samples 125 VOCs from the training-fold ANOVA Top-250.

Each branch contains 50 AdaBoost models with depth-1 trees. Every submodel receives all positive samples and an equally sized random subset of negatives. The three branch probabilities are averaged at threshold 0.5.

## Confirmed round-6 result

Thirty-two previously unused shuffle seeds were evaluated with five-fold outer cross-validation. All SNV transformations, ANOVA scores and feature subspaces were fitted only on the current outer training fold.

| Method | F1 mean | F1 std | Minimum F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **Feature-diverse three-branch ensemble** | **0.7714** | 0.0286 | **0.7037** | **0.9018** | **0.8319** |
| Fixed ANOVA Top-125 EasyEnsemble-50 | 0.7658 | 0.0308 | 0.6852 | 0.8993 | 0.8297 |

## First retained-Unknown audit

The raw source contained 1,734 VOC columns, including 746 exact-name `Unknown` columns. Frozen filtering produced 445 known features, 335 separately filtered Unknown features and 780-feature combined/appended representations.

| Representation | Features | Discovery F1 |
|---|---:|---:|
| known | 445 | **0.7828** |
| combined | 780 | 0.7452 |
| appended | 780 | 0.7415 |
| unknown-only | 335 | 0.6301 |

The best initial retained-Unknown candidate was a 5% probability fusion. Across 32 new seeds it reached F1 `0.7773` versus `0.7763` for known-only, with paired `ΔF1=+0.00096`, 95% CI `[-0.00188, 0.00370]` and Wilcoxon `p=0.642`. This was not a stable improvement, so the deployable model still removes Unknown.

## Replanned Unknown experiment

The new experiment addresses the narrow two-champion funnel and tests whether only a stable subset of Unknown features is useful.

- **22 candidates** plus the known baseline;
- fold-local stable Unknown selection at top 25, 50, 100 and 150;
- appended-subset and 5%/10%/20% probability-fusion families;
- disjoint **12-seed screen → 16-seed verification → 32-seed final** stages;
- champion pools of **6 → 4 → 3** with per-family caps;
- full final budget of three branches × 50 submodels × 50 AdaBoost estimators;
- one checkpoint file per seed for safe local resume.

The formal known-only model remains frozen until a final candidate has a positive paired effect, confidence interval entirely above zero, Wilcoxon `p<0.05`, more wins than losses, and no material worst-seed regression.

Quick integrity run:

```bash
make unknown-replan-quick
```

Full automatic run:

```bash
make unknown-replan-full
```

Detailed protocol: [`docs/UNKNOWN_VOC_REPLAN.md`](docs/UNKNOWN_VOC_REPLAN.md).

## Repository structure

```text
.
├── configs/
│   ├── default.json
│   ├── feature_diverse.json
│   └── unknown_voc_replanned.json
├── data/voc_dataset_1+2_vs_3.mat
├── experiments/
│   ├── run_unknown_voc_comparison.py
│   └── run_unknown_voc_replanned.py
├── results/
│   ├── round6/
│   ├── unknown_voc/
│   └── unknown_voc_extract/
├── src/voc_easyensemble/
└── tests/
```

## Installation

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Reproduce the enhanced 32-seed experiment

```bash
voc-easy evaluate \
  --config configs/feature_diverse.json \
  --seeds 85001:85016,86001:86016 \
  --output outputs/feature_diverse_32seed
```

## Train and predict

```bash
voc-easy train --config configs/feature_diverse.json --model artifacts/voc_feature_diverse.joblib
voc-easy predict --model artifacts/voc_feature_diverse.joblib --input new_samples.csv --output outputs/predictions.csv
```

## Tests

```bash
pytest
```

## Important limitation

The dataset does not include subject identifiers, collection batches, devices or sampling dates. Cross-validation is row-level and cannot establish subject-level independence or exclude hidden batch effects.

## Documentation

- [Method details](docs/METHOD.md)
- [Results](docs/RESULTS.md)
- [Experiment history](docs/EXPERIMENT_HISTORY.md)
- [Round-6 report](docs/ROUND6_EXPLORATION.md)
- [First retained-Unknown report](results/unknown_voc/REPORT.md)
- [Replanned Unknown protocol](docs/UNKNOWN_VOC_REPLAN.md)
- [Dataset card](data/README.md)
