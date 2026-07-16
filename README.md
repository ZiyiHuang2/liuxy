# VOC EasyEnsemble Classification

A complete, reproducible binary VOC classification project for a high-dimensional, small-sample setting:

- 159 samples;
- 445 VOC features;
- class counts 106 vs 53;
- original MATLAB dataset included in the repository.

The project now contains two frozen model lines.

| Model | Pipeline | Status | Repeated-CV F1 |
|---|---|---|---:|
| Stable baseline | SNV → ANOVA Top-125 → EasyEnsemble-50 → threshold 0.5 | simplest reliable model | about 0.76 |
| **Feature-diverse enhancement** | SNV → three VOC subspace branches → 3 × EasyEnsemble-50 → equal probability mean → threshold 0.5 | **current highest confirmed candidate** | **0.7714 ± 0.0286** |

The enhanced model is numerically stronger, but its paired 32-seed advantage over the fixed Top-125 baseline is small and not conventionally significant (`ΔF1 = +0.00556`, 95% CI approximately `[-0.00002, 0.01126]`, Wilcoxon `p = 0.0938`). It should be described as the current best enhanced candidate, not as conclusively superior.

## Feature-diverse model

The enhanced model trains three branches inside every training fold:

1. **Fixed strong panel:** ANOVA Top-125;
2. **Weighted exploration:** each balanced submodel samples 150 of all 445 VOCs without replacement, using training-fold ANOVA scores as probabilities;
3. **Top-pool diversity:** each balanced submodel samples 125 VOCs from the training-fold ANOVA Top-250.

Each branch contains 50 AdaBoost models with depth-1 trees. Every submodel receives all positive samples and an equally sized random subset of negatives. The three branch probabilities are averaged and classified with a fixed threshold of 0.5.

This extends the original EasyEnsemble in two dimensions:

- sample diversity through repeated majority-class subsampling;
- feature diversity through repeated high-evidence VOC subspaces.

## Confirmed round-6 result

Thirty-two previously unused shuffle seeds were evaluated with five-fold outer cross-validation. All SNV transformations, ANOVA scores and feature subspaces were fitted using only the current outer training fold.

| Method | F1 mean | F1 std | Minimum F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **Feature-diverse three-branch ensemble** | **0.7714** | 0.0286 | **0.7037** | **0.9018** | **0.8319** |
| Fixed ANOVA Top-125 EasyEnsemble-50 | 0.7658 | 0.0308 | 0.6852 | 0.8993 | 0.8297 |

On the canonical `seed=42` split, the enhanced model reached F1 `0.7593`, compared with `0.7321` for the fixed branch. A single seed is not used as the primary claim; the 32-seed paired result is the main evidence.

## Repository structure

```text
.
├── configs/
│   ├── default.json
│   └── feature_diverse.json
├── data/voc_dataset_1+2_vs_3.mat
├── docs/ROUND6_EXPLORATION.md
├── results/round6/
│   ├── confirm32_rows_*.csv
│   ├── confirm32_summary.csv
│   ├── confirm32_paired.csv
│   └── experiment_manifest.json
├── src/voc_easyensemble/
│   ├── model.py
│   └── feature_diverse.py
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
- [Dataset card](data/README.md)
