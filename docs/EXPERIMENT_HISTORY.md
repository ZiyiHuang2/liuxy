# Experiment history

The project moved through four main modeling ideas.

1. **Initial neural feature-selection models** attempted to learn feature masks and interactions from 159 samples. Capacity was high relative to the sample count and positive-class recall was unstable.
2. **Traditional-model nested CV** corrected the evaluation protocol and established that threshold tuning alone could not reach the target F1.
3. **Heterogeneous stable ensembles** improved repeated-CV F1 to roughly 0.72–0.73 but became complex and produced only small gains from additional branches.
4. **Balanced EasyEnsemble** directly addressed the 106:53 class imbalance. Repeated negative-class resampling produced the clearest reliable improvement, reaching approximately 0.76 F1.

The final design is intentionally simpler than the initial neural architecture. Its main objective is variance control under high-dimensional, small-sample conditions.
