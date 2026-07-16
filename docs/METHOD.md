# Method

The repository retains two frozen pipelines: a compact baseline and a higher-scoring feature-diverse candidate.

## Stable baseline

```text
raw 445-dimensional VOC vector
→ sample-wise SNV
→ training-fold ANOVA Top-125
→ 50 balanced AdaBoost submodels
→ mean positive-class probability
→ fixed threshold 0.5
```

## Feature-diverse enhancement

```text
raw 445-dimensional VOC vector
→ sample-wise SNV
→ training-fold ANOVA scores
→ branch A: fixed Top-125
→ branch B: ANOVA-weighted random 150-feature subsets
→ branch C: random 125-feature subsets from Top-250
→ 50 balanced AdaBoost submodels per branch
→ equal mean of the three branch probabilities
→ fixed threshold 0.5
```

## 1. Sample-wise SNV

For sample `i` and VOC feature `j`:

```text
z_ij = (x_ij - mean_i) / max(std_i, 1e-9)
```

SNV is applied independently to each sample and therefore does not estimate population statistics from the test fold. It reduces sample-wide intensity variation and emphasizes the relative VOC profile.

## 2. Training-fold ANOVA evidence

The ANOVA F statistic is computed only on the current training fold. The test fold never contributes labels or feature ranks.

The fixed baseline retains the 125 highest-scoring features. The enhanced model reuses the same training-fold score vector in three different ways.

### Branch A — fixed strong panel

Every submodel uses the training-fold ANOVA Top-125. This anchors the ensemble in the most stable univariate evidence.

### Branch B — weighted feature exploration

For every balanced submodel, 150 features are sampled without replacement from all 445 VOCs. The sampling probability is proportional to the nonnegative training-fold ANOVA score plus a small numerical floor.

This allows lower-ranked but still informative VOCs to enter some submodels without treating every feature equally.

### Branch C — Top-pool diversity

For every balanced submodel, 125 features are sampled uniformly without replacement from the training-fold ANOVA Top-250.

This preserves a quality filter while reducing dependence on one deterministic Top-125 panel.

## 3. Balanced EasyEnsemble training

Every submodel receives:

- all positive training samples;
- an equally sized random subset of negative training samples.

The base learner is AdaBoost with 50 depth-1 decision trees. The stable baseline trains 50 submodels. The enhanced model trains 50 submodels in each of three branches.

Within a branch, submodel probabilities are averaged. The enhanced model then equally averages the three branch probabilities.

## 4. Fixed decision rule

```text
positive probability >= 0.5 → class 1
positive probability < 0.5  → class 0
```

A development signal suggested threshold 0.495, but a separate 16-seed confirmation rejected it. The deployable rule therefore remains 0.5. No test-cohort prevalence or test-label threshold fitting is used.

## 5. Evaluation protocol

The round-6 enhanced confirmation uses:

- five-fold stratified outer cross-validation;
- two independent groups of 16 shuffle seeds: `85001–85016` and `86001–86016`;
- all SNV transformations, ANOVA scores and feature subsets fitted inside each outer training fold;
- pooled OOF predictions for each seed;
- F1 as the primary metric, with ROC-AUC and PR-AUC as secondary metrics;
- paired comparisons against the fixed Top-125 branch on identical folds.

The enhanced configuration uses the frozen fold seed `seed × 10000 + fold × 100`; branch roots are separated by `1,000,003`. This convention matches the archived 32-seed result files.

## 6. Interpretation boundary

The enhanced model improves mean F1 and worst-seed F1, but the paired 95% interval narrowly crosses zero and Wilcoxon `p=0.0938`. It is the current highest confirmed candidate, not proof of a statistically definitive superiority.
