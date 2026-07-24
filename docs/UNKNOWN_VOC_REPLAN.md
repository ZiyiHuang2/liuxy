# Unknown VOC Experiment Replan

## Why the previous protocol was insufficient

The first retained-Unknown audit was useful for rejecting naive concatenation, but its candidate funnel was narrow: seven Unknown-related candidates were screened and only the top two were sent to the 32-seed confirmation. The confirmed 5% and 20% all-Unknown probability fusions were adequately evaluated, but direct and feature-selected Unknown representations were not explored at comparable depth.

The replanned protocol does not merely increase random seeds. It changes the search axis from “use all Unknown columns or not” to “which Unknown subset is stable inside the current training fold, and how should it enter the model?”

## Candidate space

The screen evaluates 22 Unknown-related candidates plus the frozen known-only baseline.

- Direct controls: combined filtering, appended-all, and Unknown-only.
- Stable append: append the top 25, 50, 100, or 150 Unknown features.
- Stable fusion: train an Unknown-subset model at each top-k and fuse it with known-only probabilities at weights 5%, 10%, or 20%.
- All-Unknown fusion: retain the previous 5%, 10%, and 20% controls.

Unknown feature rankings are recomputed inside every outer-training fold. They are based on repeated stratified subsamples of that fold only; the outer test fold is never used for feature selection.

## Frozen three-stage funnel

| Stage | Seeds | Model budget | Stability resamples | Champion count |
|---|---:|---:|---:|---:|
| Screen | 12 new seeds | 12 submodels × 25 estimators per branch | 12 | 6 |
| Verify | 16 new seeds | 25 submodels × 35 estimators per branch | 20 | 4 |
| Final | 32 new seeds | 50 submodels × 50 estimators per branch | 30 | 3 reported champions |

All stages use five-fold outer cross-validation and threshold 0.5. Seed ranges are disjoint from each other and from the earlier Unknown comparison.

Champion selection ranks mean F1 and then minimum F1. A per-family cap prevents the pool from being filled by near-duplicate fusion weights. The final stage still evaluates every candidate frozen by verification; its top three are reported as the final champion ranking.

## Replacement rule

Unknown replaces the known-only pipeline only when the best final candidate simultaneously satisfies all of the following:

1. positive paired mean F1 difference;
2. bootstrap 95% confidence interval lower bound above zero;
3. two-sided Wilcoxon p-value below 0.05;
4. more wins than losses across seeds;
5. minimum F1 no more than 0.005 below the known baseline.

Until this rule is met, the deployable pipeline remains the 445-feature known-only model.

## Execution

Smoke test:

```bash
make unknown-replan-quick
```

Full three-stage run:

```bash
make unknown-replan-full
```

The runner writes one checkpoint CSV per seed. Re-running the same command resumes missing seeds. A configuration mismatch fails safely; use `--overwrite` only when intentionally starting a new protocol.

Run an individual stage after earlier stage outputs exist:

```bash
python experiments/run_unknown_voc_replanned.py --stage verify
python experiments/run_unknown_voc_replanned.py --stage final
```

Outputs are written to `results/unknown_voc_replanned/`. Large checkpoints and temporary payloads remain outside the committed source history; the GitHub Actions workflow uploads them as run artifacts.
