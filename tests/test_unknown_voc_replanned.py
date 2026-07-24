from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "run_unknown_voc_replanned.py"
SPEC = importlib.util.spec_from_file_location("unknown_voc_replanned", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_plan_has_disjoint_6_4_3_champion_pools() -> None:
    raw, stages = MODULE.load_plan(ROOT / "configs" / "unknown_voc_replanned.json", quick=False)
    assert stages["screen"].champion_count == 6
    assert stages["verify"].champion_count == 4
    assert stages["final"].champion_count == 3
    seed_sets = [set(stages[name].seeds) for name in MODULE.STAGES]
    assert not (seed_sets[0] & seed_sets[1])
    assert not (seed_sets[0] & seed_sets[2])
    assert not (seed_sets[1] & seed_sets[2])
    methods = MODULE.candidate_methods(raw)
    assert methods[0] == MODULE.BASELINE
    assert len(methods) == 23
    assert len(set(methods)) == len(methods)


def test_diverse_selection_respects_family_cap() -> None:
    summary = pd.DataFrame(
        [
            {"method": MODULE.BASELINE, "family": "baseline", "f1_mean": 0.80, "f1_min": 0.70},
            {"method": "known_unknown_stable_k25_w05", "family": "stable_fusion", "f1_mean": 0.79, "f1_min": 0.69},
            {"method": "known_unknown_stable_k50_w05", "family": "stable_fusion", "f1_mean": 0.78, "f1_min": 0.68},
            {"method": "known_unknown_stable_k100_w05", "family": "stable_fusion", "f1_mean": 0.77, "f1_min": 0.67},
            {"method": "appended_stable_k25", "family": "stable_append", "f1_mean": 0.76, "f1_min": 0.66},
            {"method": "combined_diverse", "family": "direct", "f1_mean": 0.75, "f1_min": 0.65},
        ]
    )
    selected = MODULE.select_champions(summary, count=4, max_per_family=2)
    families = [MODULE.candidate_family(method) for method in selected]
    assert len(selected) == 4
    assert families.count("stable_fusion") == 2
    assert "stable_append" in families
    assert "direct" in families


def test_stable_unknown_ranking_is_deterministic_and_complete() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 30))
    y = np.asarray([0] * 20 + [1] * 20)
    X[y == 1, :3] += 1.5
    first = MODULE.stable_unknown_ranking(X, y, seed=123, n_bootstraps=6, sample_fraction=0.8)
    second = MODULE.stable_unknown_ranking(X, y, seed=123, n_bootstraps=6, sample_fraction=0.8)
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), np.arange(X.shape[1]))
    assert len(set(first[:3]) & {0, 1, 2}) >= 2


def test_model_config_adapts_to_small_unknown_panels() -> None:
    config = MODULE.FeatureDiverseConfig(n_submodels=2, n_estimators=3)
    adapted = MODULE.adapt_config(config, n_features=25, seed=99)
    assert adapted.fixed_top_k == 25
    assert adapted.weighted_subset_size == 25
    assert adapted.random_pool_size == 25
    assert adapted.random_subset_size == 25
    assert adapted.random_state == 99
