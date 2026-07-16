from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted

from .preprocessing import ANOVATopK, SampleSNV


@dataclass(frozen=True)
class EasyEnsembleConfig:
    top_k: int = 125
    n_submodels: int = 50
    n_estimators: int = 50
    learning_rate: float = 1.0
    max_depth: int = 1
    threshold: float = 0.5
    random_state: int = 42
    snv_eps: float = 1e-9


class VOCEasyEnsemble(BaseEstimator, ClassifierMixin):
    """SNV + ANOVA Top-K + balanced AdaBoost ensemble."""

    def __init__(
        self,
        top_k: int = 125,
        n_submodels: int = 50,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 1,
        threshold: float = 0.5,
        random_state: int = 42,
        snv_eps: float = 1e-9,
    ):
        self.top_k = top_k
        self.n_submodels = n_submodels
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.threshold = threshold
        self.random_state = random_state
        self.snv_eps = snv_eps

    @classmethod
    def from_config(cls, config: EasyEnsembleConfig) -> "VOCEasyEnsemble":
        return cls(**asdict(config))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> "VOCEasyEnsemble":
        X = check_array(X, dtype=float)
        y = np.asarray(y).reshape(-1).astype(np.int64)
        if len(y) != X.shape[0]:
            raise ValueError("X and y have different numbers of samples")
        if not np.array_equal(np.unique(y), np.array([0, 1])):
            raise ValueError("VOCEasyEnsemble requires binary labels encoded as 0/1")
        if self.n_submodels <= 0 or self.n_estimators <= 0:
            raise ValueError("n_submodels and n_estimators must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.snv_ = SampleSNV(eps=self.snv_eps).fit(X)
        X_snv = self.snv_.transform(X)
        self.selector_ = ANOVATopK(k=self.top_k).fit(X_snv, y)
        X_selected = self.selector_.transform(X_snv)

        if feature_names is not None:
            names = np.asarray(feature_names).reshape(-1).astype(str)
            if len(names) != self.n_features_in_:
                raise ValueError("feature_names length does not match X")
            self.feature_names_in_ = names
            self.selected_feature_names_ = names[self.selector_.selected_indices_]
        else:
            self.feature_names_in_ = None
            self.selected_feature_names_ = None

        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        if len(positive) == 0 or len(negative) == 0:
            raise ValueError("Both classes must be present")

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.sample_indices_ = []
        replace_negative = len(negative) < len(positive)

        for index in range(int(self.n_submodels)):
            sampled_negative = rng.choice(
                negative, size=len(positive), replace=replace_negative
            )
            sampled = np.concatenate([positive, sampled_negative])
            rng.shuffle(sampled)

            seed = int(self.random_state) + index
            stump = DecisionTreeClassifier(
                max_depth=int(self.max_depth), random_state=seed
            )
            estimator = AdaBoostClassifier(
                estimator=stump,
                n_estimators=int(self.n_estimators),
                learning_rate=float(self.learning_rate),
                random_state=seed,
            )
            estimator.fit(X_selected[sampled], y[sampled])
            self.estimators_.append(estimator)
            self.sample_indices_.append(sampled.astype(np.int64))

        self.training_class_counts_ = {
            0: int(len(negative)),
            1: int(len(positive)),
        }
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return self.selector_.transform(self.snv_.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_selected = self._transform(X)
        positive_probabilities = np.column_stack(
            [estimator.predict_proba(X_selected)[:, 1] for estimator in self.estimators_]
        ).mean(axis=1)
        return np.column_stack([1.0 - positive_probabilities, positive_probabilities])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= float(self.threshold)).astype(np.int64)

    def save(self, path: str | Path) -> Path:
        check_is_fitted(self, "estimators_")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)
        return output

    @staticmethod
    def load(path: str | Path) -> "VOCEasyEnsemble":
        model = joblib.load(Path(path))
        if not isinstance(model, VOCEasyEnsemble):
            raise TypeError(f"File does not contain a VOCEasyEnsemble: {path}")
        return model
