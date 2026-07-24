from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.stats import rankdata, wilcoxon
from sklearn.feature_selection import f_classif
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from voc_easyensemble.feature_diverse import (
    FeatureDiverseConfig,
    FeatureDiverseEasyEnsemble,
)
from voc_easyensemble.preprocessing import SampleSNV

PAYLOAD_SHA256 = "2040c38df075e71b3a588bc94a588dc5d3d3e0c0bb7158808c2bea7da8dabff8"
KNOWN_MAT_SHA256 = "5abfb996395fc9814cddb266cbde93efab7993dc551450507312469ab0ef2635"
BASELINE = "known_diverse"
STAGES = ("screen", "verify", "final")


@dataclass(frozen=True)
class StageSpec:
    name: str
    seeds: tuple[int, ...]
    model: FeatureDiverseConfig
    stability_bootstraps: int
    champion_count: int
    max_per_family: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def materialize_payload(root: Path, output: Path | None = None) -> Path:
    output = output or root / ".work" / "unknown_voc_payload.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and sha256_file(output) == PAYLOAD_SHA256:
        return output

    manifest_path = root / "results" / "unknown_voc_extract" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    encoded = "".join(
        (manifest_path.parent / part).read_text(encoding="ascii").strip()
        for part in manifest["parts"]
    )
    output.write_bytes(base64.b64decode(encoded, validate=True))
    actual = sha256_file(output)
    if actual != PAYLOAD_SHA256 or actual != manifest["payload_sha256"]:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Unknown VOC payload hash mismatch: {actual}")
    return output


def load_matrices(root: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    known_path = root / "data" / "voc_dataset_1+2_vs_3.mat"
    known_sha = sha256_file(known_path)
    if known_sha != KNOWN_MAT_SHA256:
        raise RuntimeError(f"Known MAT hash mismatch: {known_sha}")

    known_raw = loadmat(known_path)
    known = np.asarray(known_raw["X"], dtype=np.float64)
    y = np.asarray(known_raw["y"]).reshape(-1).astype(np.int64)

    payload_path = materialize_payload(root)
    with np.load(payload_path, allow_pickle=False) as payload:
        combined = np.asarray(payload["combined_X"], dtype=np.float64)
        unknown = np.asarray(payload["unknown_X"], dtype=np.float64)
        payload_y = np.asarray(payload["y"], dtype=np.int64).reshape(-1)

    if not np.array_equal(y, payload_y):
        raise RuntimeError("Known MAT and Unknown VOC payload labels differ")
    if known.shape != (159, 445):
        raise RuntimeError(f"Unexpected known matrix shape: {known.shape}")
    if combined.shape != (159, 780) or unknown.shape != (159, 335):
        raise RuntimeError(
            f"Unexpected retained matrices: combined={combined.shape}, unknown={unknown.shape}"
        )
    if not all(np.isfinite(matrix).all() for matrix in (known, combined, unknown)):
        raise RuntimeError("A retained VOC matrix contains NaN or infinite values")

    matrices = {
        "known": known,
        "combined": combined,
        "appended_all": np.column_stack([known, unknown]),
        "unknown_all": unknown,
    }
    audit = {
        "n_samples": int(len(y)),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(y, return_counts=True))
        },
        "known_shape": list(known.shape),
        "combined_shape": list(combined.shape),
        "unknown_shape": list(unknown.shape),
        "appended_all_shape": list(matrices["appended_all"].shape),
        "known_mat_sha256": known_sha,
        "payload_sha256": sha256_file(payload_path),
    }
    return matrices, y, audit


def load_plan(path: Path, quick: bool) -> tuple[dict[str, Any], dict[str, StageSpec]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    stages: dict[str, StageSpec] = {}
    for stage_name in STAGES:
        item = raw[stage_name]
        if quick:
            seeds = tuple(int(value) for value in item["seeds"][:2])
            model = FeatureDiverseConfig(n_submodels=2, n_estimators=3)
            bootstraps = 3
            count = min(3 if stage_name == "screen" else 2, int(item["champion_count"]))
        else:
            seeds = tuple(int(value) for value in item["seeds"])
            model = FeatureDiverseConfig(
                n_submodels=int(item["n_submodels"]),
                n_estimators=int(item["n_estimators"]),
            )
            bootstraps = int(item["stability_bootstraps"])
            count = int(item["champion_count"])
        stages[stage_name] = StageSpec(
            name=stage_name,
            seeds=seeds,
            model=model,
            stability_bootstraps=bootstraps,
            champion_count=count,
            max_per_family=int(item["max_per_family"]),
        )
    return raw, stages


def candidate_family(method: str) -> str:
    if method == BASELINE:
        return "baseline"
    if method in {"combined_diverse", "appended_all_diverse", "unknown_all_diverse"}:
        return "direct"
    if method.startswith("appended_stable_k"):
        return "stable_append"
    if method.startswith("known_unknown_stable_k"):
        return "stable_fusion"
    if method.startswith("known_unknown_all_w"):
        return "all_fusion"
    raise ValueError(f"Unknown method: {method}")


def candidate_methods(plan: dict[str, Any]) -> tuple[str, ...]:
    top_k = tuple(int(value) for value in plan["stable_top_k"])
    weights = tuple(float(value) for value in plan["fusion_weights"])
    methods = [
        BASELINE,
        "combined_diverse",
        "appended_all_diverse",
        "unknown_all_diverse",
    ]
    methods.extend(f"appended_stable_k{k}" for k in top_k)
    methods.extend(
        f"known_unknown_stable_k{k}_w{int(round(weight * 100)):02d}"
        for k in top_k
        for weight in weights
    )
    methods.extend(
        f"known_unknown_all_w{int(round(weight * 100)):02d}" for weight in weights
    )
    return tuple(methods)


def adapt_config(config: FeatureDiverseConfig, n_features: int, seed: int) -> FeatureDiverseConfig:
    fixed = min(int(config.fixed_top_k), n_features)
    weighted = min(int(config.weighted_subset_size), n_features)
    pool = min(int(config.random_pool_size), n_features)
    random_subset = min(int(config.random_subset_size), pool)
    return replace(
        config,
        random_state=int(seed),
        fixed_top_k=fixed,
        weighted_subset_size=weighted,
        random_pool_size=pool,
        random_subset_size=random_subset,
    )


def stable_unknown_ranking(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_bootstraps: int,
    sample_fraction: float,
) -> np.ndarray:
    if not 0.5 <= sample_fraction <= 1.0:
        raise ValueError("selection_fraction must be in [0.5, 1.0]")
    X_snv = SampleSNV().fit_transform(X)
    n_features = X.shape[1]
    frequency = np.zeros(n_features, dtype=float)
    rank_score = np.zeros(n_features, dtype=float)
    rng = np.random.default_rng(seed)
    top_for_frequency = min(150, n_features)

    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    for _ in range(n_bootstraps):
        sampled_parts = []
        for indices in class_indices:
            size = max(2, int(np.ceil(len(indices) * sample_fraction)))
            sampled_parts.append(rng.choice(indices, size=size, replace=False))
        sampled = np.concatenate(sampled_parts)
        scores, _ = f_classif(X_snv[sampled], y[sampled])
        finite = np.isfinite(scores)
        cap = float(np.max(scores[finite])) if np.any(finite) else 1.0
        scores = np.nan_to_num(scores, nan=0.0, posinf=cap + 1.0, neginf=0.0)
        order = np.argsort(-scores, kind="mergesort")
        frequency[order[:top_for_frequency]] += 1.0
        ranks = rankdata(-scores, method="average")
        rank_score += (n_features + 1.0 - ranks) / n_features

    mean_rank_score = rank_score / float(n_bootstraps)
    return np.lexsort((np.arange(n_features), -mean_rank_score, -frequency))


def parse_weight(method: str) -> float:
    return int(method.rsplit("_w", 1)[1]) / 100.0


def required_components(methods: Iterable[str]) -> set[str]:
    components: set[str] = set()
    for method in methods:
        if method == BASELINE:
            components.add("known")
        elif method == "combined_diverse":
            components.add("combined")
        elif method == "appended_all_diverse":
            components.add("appended_all")
        elif method == "unknown_all_diverse":
            components.add("unknown_all")
        elif method.startswith("appended_stable_k"):
            components.add(method.replace("_diverse", ""))
        elif method.startswith("known_unknown_stable_k"):
            prefix = method.rsplit("_w", 1)[0]
            components.update(("known", prefix.replace("known_unknown", "unknown")))
        elif method.startswith("known_unknown_all_w"):
            components.update(("known", "unknown_all"))
        else:
            raise ValueError(method)
    return components


def combine_probability(method: str, cache: dict[str, np.ndarray]) -> np.ndarray:
    if method == BASELINE:
        return cache["known"]
    if method == "combined_diverse":
        return cache["combined"]
    if method == "appended_all_diverse":
        return cache["appended_all"]
    if method == "unknown_all_diverse":
        return cache["unknown_all"]
    if method.startswith("appended_stable_k"):
        return cache[method]
    if method.startswith("known_unknown_stable_k"):
        component = method.rsplit("_w", 1)[0].replace("known_unknown", "unknown")
        weight = parse_weight(method)
        return (1.0 - weight) * cache["known"] + weight * cache[component]
    if method.startswith("known_unknown_all_w"):
        weight = parse_weight(method)
        return (1.0 - weight) * cache["known"] + weight * cache["unknown_all"]
    raise ValueError(method)


def metric_row(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(np.int64)
    return {
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
    }


def evaluate_seed(
    seed: int,
    stage: StageSpec,
    matrices: dict[str, np.ndarray],
    y: np.ndarray,
    methods: tuple[str, ...],
    stable_top_k: tuple[int, ...],
    selection_fraction: float,
) -> pd.DataFrame:
    oof = {method: np.full(len(y), np.nan, dtype=np.float64) for method in methods}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(cv.split(matrices["known"], y)):
        fold_seed = int(seed) * 10000 + fold * 100
        ranking = stable_unknown_ranking(
            matrices["unknown_all"][train_idx],
            y[train_idx],
            seed=fold_seed + 7,
            n_bootstraps=stage.stability_bootstraps,
            sample_fraction=selection_fraction,
        )
        fold_matrices = dict(matrices)
        for k in stable_top_k:
            indices = ranking[:k]
            fold_matrices[f"unknown_stable_k{k}"] = matrices["unknown_all"][:, indices]
            fold_matrices[f"appended_stable_k{k}"] = np.column_stack(
                [matrices["known"], matrices["unknown_all"][:, indices]]
            )

        cache: dict[str, np.ndarray] = {}
        for component in sorted(required_components(methods)):
            X = fold_matrices[component]
            config = adapt_config(stage.model, X.shape[1], fold_seed)
            model = FeatureDiverseEasyEnsemble.from_config(config)
            model.fit(X[train_idx], y[train_idx])
            cache[component] = model.predict_proba(X[test_idx])[:, 1]

        for method in methods:
            oof[method][test_idx] = combine_probability(method, cache)

    rows = []
    for method in methods:
        probability = oof[method]
        if np.isnan(probability).any():
            raise RuntimeError(f"Missing OOF predictions: stage={stage.name}, seed={seed}, method={method}")
        rows.append(
            {
                "stage": stage.name,
                "seed": int(seed),
                "method": method,
                "family": candidate_family(method),
                **metric_row(y, probability, float(stage.model.threshold)),
            }
        )
    return pd.DataFrame(rows)


def checkpoint_fingerprint(
    stage: StageSpec,
    methods: tuple[str, ...],
    stable_top_k: tuple[int, ...],
    selection_fraction: float,
) -> str:
    return sha256_json(
        {
            "stage": asdict(stage),
            "methods": methods,
            "stable_top_k": stable_top_k,
            "selection_fraction": selection_fraction,
            "known_sha": KNOWN_MAT_SHA256,
            "payload_sha": PAYLOAD_SHA256,
        }
    )


def run_stage(
    output: Path,
    stage: StageSpec,
    matrices: dict[str, np.ndarray],
    y: np.ndarray,
    methods: tuple[str, ...],
    stable_top_k: tuple[int, ...],
    selection_fraction: float,
    n_jobs: int,
    overwrite: bool,
) -> pd.DataFrame:
    checkpoint_dir = output / "checkpoints" / stage.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = checkpoint_fingerprint(stage, methods, stable_top_k, selection_fraction)
    metadata_path = checkpoint_dir / "metadata.json"
    expected_metadata = {"fingerprint": fingerprint, "methods": list(methods), "stage": asdict(stage)}

    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            if not overwrite:
                raise RuntimeError(
                    f"Checkpoint configuration changed for {stage.name}; use --overwrite to replace it"
                )
            for path in checkpoint_dir.glob("seed_*.csv"):
                path.unlink()
    metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")

    pending = []
    for seed in stage.seeds:
        path = checkpoint_dir / f"seed_{seed}.csv"
        if overwrite or not path.exists():
            pending.append(seed)

    def run_one(seed: int) -> str:
        frame = evaluate_seed(
            seed,
            stage,
            matrices,
            y,
            methods,
            stable_top_k,
            selection_fraction,
        )
        target = checkpoint_dir / f"seed_{seed}.csv"
        temporary = target.with_suffix(".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(target)
        return str(target)

    if pending:
        Parallel(n_jobs=n_jobs, verbose=10)(delayed(run_one)(seed) for seed in pending)

    frames = [pd.read_csv(checkpoint_dir / f"seed_{seed}.csv") for seed in stage.seeds]
    return pd.concat(frames, ignore_index=True)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["method", "family"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            f1_min=("f1", "min"),
            f1_max=("f1", "max"),
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
        )
        .sort_values(["f1_mean", "f1_min"], ascending=False)
        .reset_index(drop=True)
    )


def paired_test(rows: pd.DataFrame, candidate: str, baseline: str = BASELINE) -> dict[str, Any]:
    wide = rows.pivot(index="seed", columns="method", values="f1")
    difference = (wide[candidate] - wide[baseline]).dropna().to_numpy(dtype=float)
    if len(difference) == 0:
        raise RuntimeError(f"No paired observations for {candidate}")
    rng = np.random.default_rng(20260724)
    bootstrap = np.asarray(
        [rng.choice(difference, size=len(difference), replace=True).mean() for _ in range(20000)]
    )
    if np.allclose(difference, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
    return {
        "candidate": candidate,
        "family": candidate_family(candidate),
        "baseline": baseline,
        "n": int(len(difference)),
        "mean_delta_f1": float(difference.mean()),
        "median_delta_f1": float(np.median(difference)),
        "ci95_low": float(np.percentile(bootstrap, 2.5)),
        "ci95_high": float(np.percentile(bootstrap, 97.5)),
        "wins": int(np.sum(difference > 1e-12)),
        "ties": int(np.sum(np.abs(difference) <= 1e-12)),
        "losses": int(np.sum(difference < -1e-12)),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
    }


def paired_table(rows: pd.DataFrame) -> pd.DataFrame:
    candidates = [method for method in rows["method"].unique() if method != BASELINE]
    return pd.DataFrame([paired_test(rows, candidate) for candidate in candidates])


def select_champions(
    summary: pd.DataFrame,
    count: int,
    max_per_family: int,
) -> list[str]:
    ranked = summary.loc[summary["method"] != BASELINE].copy()
    selected: list[str] = []
    family_count: dict[str, int] = {}
    for row in ranked.itertuples(index=False):
        family = str(row.family)
        if family_count.get(family, 0) >= max_per_family:
            continue
        selected.append(str(row.method))
        family_count[family] = family_count.get(family, 0) + 1
        if len(selected) == count:
            return selected
    for method in ranked["method"].astype(str):
        if method not in selected:
            selected.append(method)
        if len(selected) == count:
            break
    return selected


def write_report(
    path: Path,
    audit: dict[str, Any],
    stage_summaries: dict[str, pd.DataFrame],
    stage_paired: dict[str, pd.DataFrame],
    selections: dict[str, list[str]],
) -> None:
    final_summary = stage_summaries["final"]
    final_paired = stage_paired["final"]
    baseline = final_summary.loc[final_summary["method"] == BASELINE].iloc[0]
    best = final_summary.iloc[0]
    best_method = str(best["method"])
    supported = False
    decision = "正式模型继续删除 Unknown。"
    if best_method != BASELINE:
        test = final_paired.loc[final_paired["candidate"] == best_method].iloc[0]
        supported = bool(
            float(test["mean_delta_f1"]) > 0.0
            and float(test["ci95_low"]) > 0.0
            and float(test["wilcoxon_p"]) < 0.05
            and int(test["wins"]) > int(test["losses"])
            and float(best["f1_min"]) >= float(baseline["f1_min"]) - 0.005
        )
        if supported:
            decision = f"`{best_method}` 满足冻结替换标准，可作为新的 Unknown 使用方案。"

    lines = [
        "# Unknown VOC 重新规划实验",
        "",
        "## 最终判定",
        "",
        decision,
        "",
        "该实验不再用一次 `.head(2)` 决定全部结论，而是采用分阶段冠军池：",
        "",
        f"- 筛选阶段进入复核（{len(selections['screen'])} 个）：{', '.join(f'`{x}`' for x in selections['screen'])}；",
        f"- 复核阶段进入最终确认（{len(selections['verify'])} 个）：{', '.join(f'`{x}`' for x in selections['verify'])}；",
        f"- 最终阶段保留报告冠军（{len(selections['final'])} 个）：{', '.join(f'`{x}`' for x in selections['final'])}。",
        "",
        "所有 Unknown 特征选择均只使用当前外层训练折；筛选、复核和最终确认使用互不重叠的随机种子。",
        "",
        "## 数据审计",
        "",
        f"- known：{audit['known_shape']}；",
        f"- combined：{audit['combined_shape']}；",
        f"- unknown-only：{audit['unknown_shape']}；",
        f"- appended-all：{audit['appended_all_shape']}。",
        "",
    ]
    for stage_name in STAGES:
        lines.extend(
            [
                f"## {stage_name} 阶段",
                "",
                stage_summaries[stage_name].to_markdown(index=False, floatfmt=".6f"),
                "",
                "### 相对 known 基线的配对结果",
                "",
                stage_paired[stage_name].to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )
    lines.extend(
        [
            "## 替换标准",
            "",
            "最终候选必须同时满足：配对均值提升为正、bootstrap 95% 区间下界大于 0、"
            "Wilcoxon p<0.05、胜场多于负场，且最低 F1 不得比 known 基线低超过 0.005。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-stage retained-Unknown VOC experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/unknown_voc_replanned.json"))
    parser.add_argument("--output", type=Path, default=Path("results/unknown_voc_replanned"))
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--n-jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    plan, stages = load_plan(config_path, args.quick)
    matrices, y, audit = load_matrices(root)
    methods = candidate_methods(plan)
    stable_top_k = tuple(int(value) for value in plan["stable_top_k"])
    selection_fraction = float(plan["selection_fraction"])

    stage_rows: dict[str, pd.DataFrame] = {}
    stage_summaries: dict[str, pd.DataFrame] = {}
    stage_paired: dict[str, pd.DataFrame] = {}
    selections: dict[str, list[str]] = {}

    for stage_name in STAGES:
        stage = stages[stage_name]
        if stage_name == "screen":
            stage_methods = methods
        else:
            previous = "screen" if stage_name == "verify" else "verify"
            selected_path = output / f"{previous}_selected.json"
            if not selected_path.exists():
                raise RuntimeError(f"Missing {selected_path}; run earlier stages first")
            stage_methods = (BASELINE, *json.loads(selected_path.read_text(encoding="utf-8"))["selected"])

        should_run = args.stage in ("all", stage_name)
        rows_path = output / f"{stage_name}_rows.csv"
        if should_run:
            rows = run_stage(
                output,
                stage,
                matrices,
                y,
                stage_methods,
                stable_top_k,
                selection_fraction,
                args.n_jobs,
                args.overwrite,
            )
            rows.to_csv(rows_path, index=False)
        elif rows_path.exists():
            rows = pd.read_csv(rows_path)
        else:
            continue

        summary = summarize(rows)
        paired = paired_table(rows)
        summary.to_csv(output / f"{stage_name}_summary.csv", index=False)
        paired.to_csv(output / f"{stage_name}_paired.csv", index=False)
        stage_rows[stage_name] = rows
        stage_summaries[stage_name] = summary
        stage_paired[stage_name] = paired

        selected = select_champions(summary, stage.champion_count, stage.max_per_family)
        selections[stage_name] = selected
        (output / f"{stage_name}_selected.json").write_text(
            json.dumps(
                {
                    "stage": stage_name,
                    "selected": selected,
                    "champion_count": stage.champion_count,
                    "max_per_family": stage.max_per_family,
                    "selection_rule": "rank by mean F1 then minimum F1, with family cap",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if args.stage != "all":
            break

    if all(stage in stage_summaries for stage in STAGES):
        manifest = {
            "config": plan,
            "config_sha256": sha256_file(config_path),
            "quick": bool(args.quick),
            "n_splits": 5,
            "threshold": 0.5,
            "candidate_count": len(methods) - 1,
            "screen_champion_count": stages["screen"].champion_count,
            "verify_champion_count": stages["verify"].champion_count,
            "final_champion_count": stages["final"].champion_count,
            "selections": selections,
            "audit": audit,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output / "audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_report(output / "REPORT.md", audit, stage_summaries, stage_paired, selections)

    print(
        json.dumps(
            {
                "output": str(output),
                "candidate_count": len(methods) - 1,
                "selections": selections,
                "completed_stages": list(stage_summaries),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
