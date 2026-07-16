# VOC EasyEnsemble Classification

A complete, reproducible binary VOC classification project built around the most reliable configuration found during repeated small-sample experiments:

```text
SNV + ANOVA Top-125 + EasyEnsemble-50 + fixed threshold 0.5
```

The repository directly includes the original MATLAB dataset, reusable Python package, command-line tools, tests, reference results and methodology notes.

## Reference result

Frozen configuration evaluated with 5-fold outer cross-validation over 16 independent shuffle seeds:

| Metric | Result |
|---|---:|
| F1 | **0.7615 ± 0.0357** |
| ROC-AUC | **0.8971** |
| PR-AUC | **0.8268** |
| Lowest seed F1 | 0.6923 |
| Highest seed F1 | 0.8214 |

These are repeated out-of-fold results, not the best single split. Full tables are in [`results/`](results/).

## Why this method

The dataset has 159 samples, 445 VOC features and a 106:53 class imbalance. The final method focuses on the actual statistical constraints:

- **SNV** reduces sample-wide intensity differences;
- **ANOVA Top-125** limits the high-dimensional search space inside each training fold;
- **EasyEnsemble-50** repeatedly trains on all positives and different balanced negative subsets;
- **fixed 0.5 threshold** supports independent single-sample prediction without test-cohort tuning.

## Repository structure

```text
.
├── configs/default.json
├── data/
│   ├── voc_dataset_1+2_vs_3.mat
│   └── README.md
├── docs/
│   ├── METHOD.md
│   ├── RESULTS.md
│   └── EXPERIMENT_HISTORY.md
├── results/
│   ├── reliable_16seed_rows.csv
│   ├── reliable_16seed_summary.csv
│   ├── paired_tests.csv
│   └── feature_selection_stability.csv
├── scripts/
├── src/voc_easyensemble/
├── tests/
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Inspect the included data

```bash
voc-easy inspect
```

Expected core properties:

```text
samples: 159
features: 445
class 0: 106
class 1: 53
```

Verify the bundled dataset hash:

```bash
python scripts/materialize_dataset.py
```

Expected SHA-256:

```text
5abfb996395fc9814cddb266cbde93efab7993dc551450507312469ab0ef2635
```

## Reproduce the reliable repeated-CV experiment

```bash
voc-easy evaluate \
  --data data/voc_dataset_1+2_vs_3.mat \
  --config configs/default.json \
  --seeds 71001:71016 \
  --output outputs/reliable_16seed
```

This writes per-seed metrics, per-fold metrics, complete OOF predictions and a JSON summary.

A faster single-seed check:

```bash
voc-easy evaluate --seeds 42 --output outputs/smoke
```

## Train the final deployable model

```bash
voc-easy train \
  --data data/voc_dataset_1+2_vs_3.mat \
  --config configs/default.json \
  --model artifacts/voc_easyensemble.joblib
```

The command also writes a JSON metadata file containing the dataset hash, configuration and selected VOC feature names.

## Predict new samples

Prepare a CSV containing the same 445 feature-name columns as the bundled dataset, then run:

```bash
voc-easy predict \
  --model artifacts/voc_easyensemble.joblib \
  --input path/to/new_samples.csv \
  --output outputs/predictions.csv
```

The output adds `positive_probability` and `prediction` columns.

## Tests

```bash
pytest
```

## Data provenance

The bundled MAT file was imported from the original project repository by `.github/workflows/import-dataset.yml`. The workflow verifies the fixed SHA-256 before committing the binary file. The dataset stored here has Git blob SHA `d72ecbd7e4770375b76a37224a919c517e0befc3`, matching the source file.

## Data and evaluation limitation

The included dataset has no subject identifiers, batch labels, device metadata or sampling dates. Current cross-validation is row-level and cannot establish subject-level independence or exclude hidden batch effects. This limitation should remain explicit in any report or publication.

## Documentation

- [Method details](docs/METHOD.md)
- [Reference and exploratory results](docs/RESULTS.md)
- [Model-development history](docs/EXPERIMENT_HISTORY.md)
- [Dataset card](data/README.md)
