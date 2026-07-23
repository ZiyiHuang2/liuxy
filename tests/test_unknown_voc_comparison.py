import importlib.util
import sys
from pathlib import Path

import numpy as np
from voc_easyensemble.feature_diverse import FeatureDiverseConfig

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "run_unknown_voc_comparison.py"
SPEC = importlib.util.spec_from_file_location("unknown_voc_comparison", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_retained_unknown_payload_schema():
    matrices, y, audit = MODULE.load_matrices(ROOT)
    assert matrices["known"].shape == (159, 445)
    assert matrices["combined"].shape == (159, 780)
    assert matrices["unknown"].shape == (159, 335)
    assert matrices["appended"].shape == (159, 780)
    assert np.array_equal(np.unique(y), np.array([0, 1]))
    assert audit["class_counts"] == {"0": 106, "1": 53}


def test_retained_unknown_oof_smoke():
    matrices, y, _ = MODULE.load_matrices(ROOT)
    rows = MODULE.evaluate_seed(
        17,
        matrices,
        y,
        FeatureDiverseConfig(n_submodels=1, n_estimators=2),
        (
            MODULE.BASELINE,
            "combined_diverse",
            "unknown_diverse",
            "known_unknown_w10",
        ),
    )
    assert {row["method"] for row in rows} == {
        MODULE.BASELINE,
        "combined_diverse",
        "unknown_diverse",
        "known_unknown_w10",
    }
    assert all(0.0 <= float(row["f1"]) <= 1.0 for row in rows)
