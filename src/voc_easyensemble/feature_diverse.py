from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted

from .preprocessing import SampleSNV


@dataclass(frozen=True)
class FeatureDiverseConfig:
    n_submodels: int = 50
    n_estimators: int = 50
    learning_rate: float = 1.0
    max_depth: int = 1
    threshold: float = 0.5
    random_state: int = 42
    snv_eps: float = 1e-9
    fixed_top_k: int = 125
    weighted_subset_size: int = 150
    random_pool_size: int = 250
    random_subset_size: int = 125


class FeatureDiverseEasyEnsemble(BaseEstimator, ClassifierMixin):
    """Three balanced EasyEnsemble branches with distinct VOC subspaces.

    Branches are frozen to the round-6 design:
    1. fixed ANOVA Top-K;
    2. ANOVA-score-weighted random feature subsets;
    3. uniform random subsets drawn from an ANOVA-ranked top pool.
    """

    def __init__(self, n_submodels: int = 50, n_estimators: int = 50,
                 learning_rate: float = 1.0, max_depth: int = 1,
                 threshold: float = 0.5, random_state: int = 42,
                 snv_eps: float = 1e-9, fixed_top_k: int = 125,
                 weighted_subset_size: int = 150, random_pool_size: int = 250,
                 random_subset_size: int = 125):
        self.n_submodels = n_submodels
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.threshold = threshold
        self.random_state = random_state
        self.snv_eps = snv_eps
        self.fixed_top_k = fixed_top_k
        self.weighted_subset_size = weighted_subset_size
        self.random_pool_size = random_pool_size
        self.random_subset_size = random_subset_size

    @classmethod
    def from_config(cls, config: FeatureDiverseConfig) -> "FeatureDiverseEasyEnsemble":
        return cls(**asdict(config))

    @staticmethod
    def _anova_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        scores, _ = f_classif(X, y)
        finite = np.isfinite(scores)
        cap = float(np.max(scores[finite])) if np.any(finite) else 1.0
        return np.nan_to_num(scores, nan=0.0, posinf=cap + 1.0, neginf=0.0)

    def _validate_hyperparameters(self, n_features: int) -> None:
        integer_fields = {
            "n_submodels": self.n_submodels,
            "n_estimators": self.n_estimators,
            "fixed_top_k": self.fixed_top_k,
            "weighted_subset_size": self.weighted_subset_size,
            "random_pool_size": self.random_pool_size,
            "random_subset_size": self.random_subset_size,
        }
        if any(int(value) <= 0 for value in integer_fields.values()):
            raise ValueError("All ensemble and feature-subspace sizes must be positive")
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        if self.fixed_top_k > n_features or self.weighted_subset_size > n_features:
            raise ValueError("Feature subset size exceeds the input dimensionality")
        if self.random_pool_size > n_features:
            raise ValueError("random_pool_size exceeds the input dimensionality")
        if self.random_subset_size > self.random_pool_size:
            raise ValueError("random_subset_size must not exceed random_pool_size")

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Sequence[str] | None = None) -> "FeatureDiverseEasyEnsemble":
        X = check_array(X, dtype=float)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if X.shape[0] != y.size:
            raise ValueError("X and y have inconsistent sample counts")
        if not np.array_equal(np.unique(y), np.array([0, 1])):
            raise ValueError("Expected binary labels encoded as 0/1")
        self._validate_hyperparameters(X.shape[1])

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.snv_ = SampleSNV(eps=float(self.snv_eps)).fit(X)
        X_snv = self.snv_.transform(X)
        self.anova_scores_ = self._anova_scores(X_snv, y)
        self.anova_rank_ = np.argsort(-self.anova_scores_, kind="mergesort")

        if feature_names is not None:
            names = np.asarray(feature_names).reshape(-1).astype(str)
            if names.size != self.n_features_in_:
                raise ValueError("feature_names length does not match X")
            self.feature_names_in_ = names
        else:
            self.feature_names_in_ = None

        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        if positive.size == 0 or negative.size == 0:
            raise ValueError("Both classes must be present")
        replace_negative = negative.size < positive.size

        self.branch_names_ = np.asarray(
            [
                f"fixed_top{int(self.fixed_top_k)}",
                f"anova_weighted{int(self.weighted_subset_size)}",
                f"top{int(self.random_pool_size)}_random{int(self.random_subset_size)}",
            ],
            dtype=str,
        )
        self.branch_estimators_: list[list[AdaBoostClassifier]] = []
        self.branch_feature_indices_: list[list[np.ndarray]] = []

        weighted_probability = self.anova_scores_.astype(float) + 1e-8
        weighted_probability /= weighted_probability.sum()
        random_pool = self.anova_rank_[: int(self.random_pool_size)]
        fixed_features = self.anova_rank_[: int(self.fixed_top_k)]

        for branch_index in range(3):
            branch_seed = int(self.random_state) + branch_index * 1_000_003
            rng = np.random.default_rng(branch_seed)
            estimators: list[AdaBoostClassifier] = []
            features_per_model: list[np.ndarray] = []
            for model_index in range(int(self.n_submodels)):
                sampled_negative = rng.choice(
                    negative, size=positive.size, replace=replace_negative
                )
                sampled = np.concatenate([positive, sampled_negative])
                rng.shuffle(sampled)

                if branch_index == 0:
                    features = fixed_features.copy()
                elif branch_index == 1:
                    features = np.sort(
                        rng.choice(
                            np.arange(self.n_features_in_),
                            size=int(self.weighted_subset_size),
                            replace=False,
                            p=weighted_probability,
                        )
                    )
                else:
                    features = np.sort(
                        rng.choice(
                            random_pool,
                            size=int(self.random_subset_size),
                            replace=False,
                        )
                    )

                seed = branch_seed + model_index
                stump = DecisionTreeClassifier(max_depth=int(self.max_depth), random_state=seed)
                estimator = AdaBoostClassifier(
                    estimator=stump,
                    n_estimators=int(self.n_estimators),
                    learning_rate=float(self.learning_rate),
                    random_state=seed,
                )
                estimator.fit(X_snv[sampled][:, features], y[sampled])
                estimators.append(estimator)
                features_per_model.append(np.asarray(features, dtype=np.int64))
            self.branch_estimators_.append(estimators)
            self.branch_feature_indices_.append(features_per_model)

        return self

    def branch_probabilities(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["branch_estimators_", "branch_feature_indices_", "snv_"])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        X_snv = self.snv_.transform(X)
        branch_scores = []
        for estimators, feature_sets in zip(
            self.branch_estimators_, self.branch_feature_indices_
        ):
            probabilities = [
                estimator.predict_proba(X_snv[:, features])[:, 1]
                for estimator, features in zip(estimators, feature_sets)
            ]
            branch_scores.append(np.mean(probabilities, axis=0))
        return np.column_stack(branch_scores)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = self.branch_probabilities(X).mean(axis=1)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= float(self.threshold)).astype(np.int64)

    def save(self, path: str | Path) -> Path:
        check_is_fitted(self, "branch_estimators_")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)
        return output

    @staticmethod
    def load(path: str | Path) -> "FeatureDiverseEasyEnsemble":
        model = joblib.load(Path(path))
        if not isinstance(model, FeatureDiverseEasyEnsemble):
            raise TypeError(f"File does not contain a FeatureDiverseEasyEnsemble: {path}")
        return model
