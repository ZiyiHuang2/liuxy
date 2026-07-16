from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from voc_easyensemble.data import load_voc_mat
from voc_easyensemble.metrics import binary_metrics
from voc_easyensemble.model import VOCEasyEnsemble


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "voc_dataset_1+2_vs_3.mat"


def test_dataset_schema() -> None:
    dataset = load_voc_mat(DATA)
    assert dataset.X.shape == (159, 445)
    assert dataset.y.shape == (159,)
    assert dataset.feature_names.shape == (445,)
    assert np.array_equal(np.unique(dataset.y), np.array([0, 1]))


def test_small_end_to_end_model(tmp_path: Path) -> None:
    dataset = load_voc_mat(DATA)
    train_idx, test_idx = train_test_split(
        np.arange(dataset.n_samples),
        test_size=0.2,
        stratify=dataset.y,
        random_state=7,
    )
    model = VOCEasyEnsemble(
        top_k=20,
        n_submodels=3,
        n_estimators=5,
        random_state=7,
    )
    model.fit(
        dataset.X[train_idx],
        dataset.y[train_idx],
        feature_names=dataset.feature_names,
    )
    probabilities = model.predict_proba(dataset.X[test_idx])[:, 1]
    assert probabilities.shape == (len(test_idx),)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    metrics = binary_metrics(dataset.y[test_idx], probabilities)
    assert 0.0 <= metrics["f1"] <= 1.0

    saved = model.save(tmp_path / "model.joblib")
    loaded = VOCEasyEnsemble.load(saved)
    np.testing.assert_allclose(
        loaded.predict_proba(dataset.X[test_idx]),
        model.predict_proba(dataset.X[test_idx]),
    )
