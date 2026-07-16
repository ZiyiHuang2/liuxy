from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import ModelConfig, build_model, config_to_dict
from .data import VOCDataset
from .feature_diverse import FeatureDiverseConfig
from .metrics import binary_metrics


def default_fold_seed_step(config: ModelConfig) -> int:
    """Return the frozen fold-seed convention used by each experiment line."""
    return 100 if isinstance(config, FeatureDiverseConfig) else 101


def evaluate_seed(
    dataset: VOCDataset,
    seed: int,
    config: ModelConfig,
    n_splits: int = 5,
    fold_seed_step: int | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """Evaluate one shuffle seed with fold-local preprocessing and selection."""
    step = default_fold_seed_step(config) if fold_seed_step is None else int(fold_seed_step)
    oof_probability = np.full(dataset.n_samples, np.nan, dtype=float)
    fold_rows: list[dict[str, float | int]] = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(cv.split(dataset.X, dataset.y)):
        fold_config = replace(config, random_state=int(seed) * 10000 + fold * step)
        model = build_model(fold_config)
        model.fit(
            dataset.X[train_idx],
            dataset.y[train_idx],
            feature_names=dataset.feature_names,
        )
        probabilities = model.predict_proba(dataset.X[test_idx])[:, 1]
        oof_probability[test_idx] = probabilities
        fold_rows.append(
            {
                "seed": int(seed),
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                **binary_metrics(
                    dataset.y[test_idx], probabilities, float(config.threshold)
                ),
            }
        )

    if np.isnan(oof_probability).any():
        raise RuntimeError("Some samples did not receive an OOF prediction")

    pooled = {
        "seed": int(seed),
        **binary_metrics(dataset.y, oof_probability, float(config.threshold)),
    }
    predictions = pd.DataFrame(
        {
            "seed": int(seed),
            "sample_index": np.arange(dataset.n_samples, dtype=int),
            "y_true": dataset.y,
            "probability": oof_probability,
            "prediction": (oof_probability >= float(config.threshold)).astype(int),
        }
    )
    return pooled, pd.DataFrame(fold_rows), predictions


def run_repeated_cv(
    dataset: VOCDataset,
    seeds: Iterable[int],
    config: ModelConfig,
    output_dir: str | Path,
    n_splits: int = 5,
    fold_seed_step: int | None = None,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, float | int]] = []
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for seed in seeds:
        pooled, folds, predictions = evaluate_seed(
            dataset,
            int(seed),
            config,
            n_splits=n_splits,
            fold_seed_step=fold_seed_step,
        )
        seed_rows.append(pooled)
        fold_frames.append(folds)
        prediction_frames.append(predictions)
        print(
            f"seed={seed} f1={pooled['f1']:.4f} "
            f"auc={pooled['roc_auc']:.4f} pr_auc={pooled['pr_auc']:.4f}",
            flush=True,
        )

    seed_df = pd.DataFrame(seed_rows)
    fold_df = pd.concat(fold_frames, ignore_index=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    metric_columns = [
        "f1",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]
    resolved_step = (
        default_fold_seed_step(config) if fold_seed_step is None else int(fold_seed_step)
    )
    summary = {
        "n_seeds": int(len(seed_df)),
        "n_splits": int(n_splits),
        "fold_seed_step": resolved_step,
        "config": config_to_dict(config),
        "metrics": {
            metric: {
                "mean": float(seed_df[metric].mean()),
                "std": float(seed_df[metric].std(ddof=1)),
                "min": float(seed_df[metric].min()),
                "max": float(seed_df[metric].max()),
            }
            for metric in metric_columns
        },
    }
    paths = {
        "seed_metrics": output / "seed_metrics.csv",
        "fold_metrics": output / "fold_metrics.csv",
        "oof_predictions": output / "oof_predictions.csv",
        "summary": output / "summary.json",
    }
    seed_df.to_csv(paths["seed_metrics"], index=False)
    fold_df.to_csv(paths["fold_metrics"], index=False)
    prediction_df.to_csv(paths["oof_predictions"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return paths
