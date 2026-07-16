# Experiment history

The project moved through five main modeling ideas.

1. **Initial neural feature-selection models** attempted to learn feature masks and interactions from 159 samples. Capacity was high relative to the sample count and positive-class recall was unstable.
2. **Traditional-model nested CV** corrected the evaluation protocol and established that threshold tuning alone could not reach the target F1.
3. **Heterogeneous stable ensembles** improved repeated-CV F1 to roughly 0.72–0.73 but became complex and produced only small gains from additional model families.
4. **Balanced EasyEnsemble** directly addressed the 106:53 class imbalance. Repeated negative-class resampling produced the clearest reliable improvement, reaching approximately 0.76 F1.
5. **Feature-diverse EasyEnsemble** retained the balanced sampling mechanism and added controlled feature-subspace diversity. A fixed strong panel, an ANOVA-weighted exploration branch and a Top-250 random branch increased 32-seed mean F1 to 0.7714 and raised the worst-seed result, although the paired improvement remained near rather than beyond conventional significance.

The current direction remains variance control rather than model-capacity growth. Deep architectures, supervised latent-variable models, explicit feature-pair construction, adaptive stacking and threshold tuning did not provide a stronger independently confirmed result.
