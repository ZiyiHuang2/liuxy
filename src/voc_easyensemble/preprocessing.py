from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_array, check_is_fitted


class SampleSNV(BaseEstimator, TransformerMixin):
    """Standard-normal-variate transform applied independently to each sample."""

    def __init__(self, eps: float = 1e-9):
        self.eps = eps

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SampleSNV":
        X = check_array(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        means = X.mean(axis=1, keepdims=True)
        scales = np.maximum(X.std(axis=1, keepdims=True), self.eps)
        return (X - means) / scales


class ANOVATopK(BaseEstimator, TransformerMixin):
    """Select the top-k features by training-fold ANOVA F statistic."""

    def __init__(self, k: int = 125):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANOVATopK":
        X = check_array(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if len(y) != X.shape[0]:
            raise ValueError("X and y have different numbers of samples")
        if self.k <= 0:
            raise ValueError("k must be positive")

        scores, p_values = f_classif(X, y)
        scores = np.nan_to_num(
            scores, nan=0.0, posinf=np.finfo(float).max, neginf=0.0
        )
        p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=0.0)
        k = min(int(self.k), X.shape[1])
        selected = np.argsort(scores, kind="mergesort")[::-1][:k]

        self.n_features_in_ = X.shape[1]
        self.scores_ = scores
        self.p_values_ = p_values
        self.selected_indices_ = selected.astype(np.int64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "selected_indices_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return X[:, self.selected_indices_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        check_is_fitted(self, "selected_indices_")
        if indices:
            return self.selected_indices_.copy()
        support = np.zeros(self.n_features_in_, dtype=bool)
        support[self.selected_indices_] = True
        return support
