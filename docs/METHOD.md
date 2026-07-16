# Method

## Final model

The final model is:

SNV normalization -> ANOVA Top-125 feature selection -> EasyEnsemble-50 -> threshold 0.5.

## Why

The task is a high-dimensional small-sample VOC classification problem. Deep neural networks were replaced by variance-controlled ensemble learning.

## EasyEnsemble

Each weak learner receives all positive samples and a random balanced subset of negative samples. Fifty AdaBoost decision-stump models are trained and their probabilities are averaged.

## Leakage prevention

All preprocessing and feature selection must be fitted inside training folds only.
