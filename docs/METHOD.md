# Method

The frozen default pipeline is:

```text
raw 445-dimensional VOC vector
→ sample-wise standard normal variate (SNV)
→ training-fold ANOVA Top-125 feature selection
→ 50 balanced AdaBoost submodels
→ mean positive-class probability
→ fixed threshold 0.5
```

## 1. SNV

For sample `i` and feature `j`:

```text
z_ij = (x_ij - mean_i) / max(std_i, 1e-9)
```

SNV is applied independently to each sample. It reduces sample-wide intensity differences and emphasizes the relative VOC profile.

## 2. ANOVA Top-125

The ANOVA F statistic is calculated using only the current training fold. The 125 features with the largest F statistic are retained. The test fold is transformed using that training-fold selection.

This placement is essential: selecting features on the full dataset before cross-validation would leak test-fold label information.

## 3. EasyEnsemble-50

Each submodel receives:

- every positive training sample;
- a random subset of negative training samples with the same size as the positive set.

The base learner is AdaBoost with 50 depth-1 decision trees. Fifty independently resampled balanced submodels are trained. The final score is their mean positive-class probability.

## 4. Decision rule

The frozen single-sample decision rule is:

```text
positive probability >= 0.5 → class 1
positive probability < 0.5  → class 0
```

No test-cohort prevalence or test-label threshold fitting is used.

## 5. Evaluation protocol

The reference experiment uses:

- 5-fold stratified outer cross-validation;
- 16 independent shuffle seeds: 71001 through 71016;
- all preprocessing and feature selection fitted separately inside each outer training fold;
- pooled out-of-fold predictions for each seed;
- F1 as the primary metric, with ROC-AUC and PR-AUC as secondary metrics.

The configuration was frozen before this 16-seed confirmation round.
