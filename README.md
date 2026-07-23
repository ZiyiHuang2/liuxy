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

## Retained Unknown VOC audit

The repository also contains an offline reconstruction of the VOC columns whose exact feature name was `Unknown`. The original source had 1,734 VOC columns, including 746 `Unknown` columns. After the frozen abundance and IQR filters, the retained matrices are:

| Representation | Features | Construction | Discovery F1 |
|---|---:|---|---:|
| known | 445 | existing frozen matrix with `Unknown` removed | **0.7828** |
| combined | 780 | rerun the filters over known and `Unknown` columns together | 0.7452 |
| appended | 780 | frozen 445 known features plus 335 separately filtered `Unknown` features | 0.7415 |
| unknown-only | 335 | separately filtered `Unknown` features | 0.6301 |

The direct retained-Unknown representations were clearly weaker in the frozen discovery stage. The two best low-weight probability-fusion candidates were then frozen and evaluated on 32 completely new shuffle seeds with the full 50 × 50 model budget:

| Method | F1 mean | F1 std | Minimum F1 | Mean paired ΔF1 | 95% CI |
|---|---:|---:|---:|---:|---:|
| known + 5% unknown probability | **0.7773** | **0.0191** | **0.7434** | +0.0010 | [-0.0019, 0.0037] |
| known + 20% unknown probability | 0.7765 | 0.0193 | 0.7273 | +0.0002 | [-0.0062, 0.0072] |
| known only | 0.7763 | 0.0216 | 0.7339 | reference | — |

The 5% fusion produced 10 wins, 16 ties and 6 losses against the paired known-only baseline, with Wilcoxon `p = 0.642`. This is not evidence of a stable improvement. **The deployable model therefore continues to remove `Unknown`; retaining it is documented as an exploratory result, not a new champion.** These F1 values use a new seed set and should only be compared within this paired table, not directly against the earlier round-6 mean.

Reproduce the offline comparison with:

```bash
make unknown-full
```

A smoke-sized integrity run is available as `make unknown-quick`. Full records are in [`results/unknown_voc`](results/unknown_voc), and the Chinese report is [`results/unknown_voc/REPORT.md`](results/unknown_voc/REPORT.md).

## Repository structure

```text
.
├── configs/
│   ├── default.json
│   └── feature_diverse.json
├── data/voc_dataset_1+2_vs_3.mat
├── experiments/run_unknown_voc_comparison.py
├── results/
│   ├── round6/
│   ├── unknown_voc/
│   └── unknown_voc_extract/
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
- [Retained Unknown VOC report](results/unknown_voc/REPORT.md)
- [Dataset card](data/README.md)
