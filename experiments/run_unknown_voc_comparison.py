from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from voc_easyensemble.feature_diverse import (
    FeatureDiverseConfig,
    FeatureDiverseEasyEnsemble,
)

PAYLOAD_SHA256 = "2040c38df075e71b3a588bc94a588dc5d3d3e0c0bb7158808c2bea7da8dabff8"
KNOWN_MAT_SHA256 = "5abfb996395fc9814cddb266cbde93efab7993dc551450507312469ab0ef2635"
BASELINE = "known_diverse"
CANDIDATES = (
    "combined_diverse",
    "appended_diverse",
    "unknown_diverse",
    "known_unknown_w05",
    "known_unknown_w10",
    "known_unknown_w20",
    "known_unknown_w30",
)


@dataclass(frozen=True)
class StageSpec:
    seeds: tuple[int, ...]
    model: FeatureDiverseConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_payload(root: Path, output: Path | None = None) -> Path:
    output = output or root / ".work" / "unknown_voc_payload.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and sha256_file(output) == PAYLOAD_SHA256:
        return output

    manifest_path = root / "results" / "unknown_voc_extract" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts_dir = manifest_path.parent
    encoded = "".join(
        (parts_dir / part).read_text(encoding="ascii").strip()
        for part in manifest["parts"]
    )
    output.write_bytes(base64.b64decode(encoded, validate=True))
    actual = sha256_file(output)
    if actual != PAYLOAD_SHA256 or actual != manifest["payload_sha256"]:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Unknown VOC payload hash mismatch: {actual}")
    return output


def load_matrices(root: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
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

    appended = np.column_stack([known, unknown])
    matrices = {
        "known": known,
        "combined": combined,
        "appended": appended,
        "unknown": unknown,
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
        "appended_shape": list(appended.shape),
        "known_mat_sha256": known_sha,
        "payload_sha256": sha256_file(payload_path),
        "protocol_note": (
            "combined reruns the original abundance/IQR filters over all raw VOCs; "
            "appended concatenates the frozen 445 known panel with the separately "
            "filtered 335 Unknown panel"
        ),
    }
    return matrices, y, audit


def model_components(methods: Iterable[str]) -> set[str]:
    components: set[str] = set()
    for method in methods:
        if method == BASELINE:
            components.add("known")
        elif method == "combined_diverse":
            components.add("combined")
        elif method == "appended_diverse":
            components.add("appended")
        elif method == "unknown_diverse":
            components.add("unknown")
        elif method.startswith("known_unknown_w"):
            components.update(("known", "unknown"))
        else:
            raise ValueError(f"Unknown method: {method}")
    return components


def combine_probability(method: str, cache: dict[str, np.ndarray]) -> np.ndarray:
    if method == BASELINE:
        return cache["known"]
    if method == "combined_diverse":
        return cache["combined"]
    if method == "appended_diverse":
        return cache["appended"]
    if method == "unknown_diverse":
        return cache["unknown"]
    if method.startswith("known_unknown_w"):
        unknown_weight = int(method.rsplit("w", 1)[1]) / 100.0
        return (1.0 - unknown_weight) * cache["known"] + unknown_weight * cache["unknown"]
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
    matrices: dict[str, np.ndarray],
    y: np.ndarray,
    config: FeatureDiverseConfig,
    methods: tuple[str, ...],
) -> list[dict[str, float | int | str]]:
    components = model_components(methods)
    oof = {method: np.full(len(y), np.nan, dtype=np.float64) for method in methods}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(cv.split(matrices["known"], y)):
        # Identical model seed across representations keeps negative subsampling aligned.
        fold_config = replace(config, random_state=int(seed) * 10000 + fold * 100)
        component_probability: dict[str, np.ndarray] = {}
        for component in sorted(components):
            model = FeatureDiverseEasyEnsemble.from_config(fold_config)
            model.fit(matrices[component][train_idx], y[train_idx])
            component_probability[component] = model.predict_proba(
                matrices[component][test_idx]
            )[:, 1]

        for method in methods:
            oof[method][test_idx] = combine_probability(method, component_probability)

    rows: list[dict[str, float | int | str]] = []
    for method in methods:
        probability = oof[method]
        if np.isnan(probability).any():
            raise RuntimeError(f"Missing OOF predictions: seed={seed}, method={method}")
        rows.append(
            {
                "seed": int(seed),
                "method": method,
                **metric_row(y, probability, float(config.threshold)),
            }
        )
    return rows


def run_stage(
    stage: StageSpec,
    matrices: dict[str, np.ndarray],
    y: np.ndarray,
    methods: tuple[str, ...],
    n_jobs: int,
) -> pd.DataFrame:
    batches = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(evaluate_seed)(seed, matrices, y, stage.model, methods)
        for seed in stage.seeds
    )
    return pd.DataFrame(row for batch in batches for row in batch)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby("method", as_index=False)
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


def paired_test(
    rows: pd.DataFrame,
    candidate: str,
    baseline: str = BASELINE,
) -> dict[str, float | int | str]:
    wide = rows.pivot(index="seed", columns="method", values="f1")
    difference = (wide[candidate] - wide[baseline]).dropna().to_numpy(dtype=float)
    if len(difference) == 0:
        raise RuntimeError(f"No paired observations for {candidate}")

    rng = np.random.default_rng(20260723)
    bootstrap = np.asarray(
        [rng.choice(difference, size=len(difference), replace=True).mean() for _ in range(20000)]
    )
    if np.allclose(difference, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
    return {
        "candidate": candidate,
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


def write_report(
    path: Path,
    audit: dict[str, object],
    discovery: pd.DataFrame,
    selected: list[str],
    confirmation: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    best = confirmation.iloc[0]
    baseline = confirmation.loc[confirmation["method"] == BASELINE].iloc[0]
    best_method = str(best["method"])
    best_is_baseline = best_method == BASELINE
    best_test = None
    if not best_is_baseline:
        best_test = paired.loc[paired["candidate"] == best_method].iloc[0]
    supported = bool(
        best_test is not None
        and float(best_test["mean_delta_f1"]) > 0.0
        and float(best_test["ci95_low"]) > 0.0
        and int(best_test["wins"]) > int(best_test["losses"])
    )

    lines = [
        "# Unknown VOC 保留实验",
        "",
        "## 结论",
        "",
    ]
    if best_is_baseline:
        lines.append(
            f"在冻结协议和 32 个全新随机种子下，删除 Unknown 的 `{BASELINE}` 仍为最高方案，"
            f"平均 F1 为 {baseline['f1_mean']:.4f}。当前证据不支持在正式模型中保留 Unknown。"
        )
    elif supported:
        lines.append(
            f"`{best_method}` 在 32 个全新种子上获得可重复的配对提升，平均 F1 为 "
            f"{best['f1_mean']:.4f}，因此可考虑替换 `{BASELINE}`。"
        )
    else:
        lines.append(
            f"`{best_method}` 的平均 F1 为 {best['f1_mean']:.4f}，仅比 `{BASELINE}` 的 "
            f"{baseline['f1_mean']:.4f} 高 {float(best_test['mean_delta_f1']):.4f}；配对 95% 区间为 "
            f"[{float(best_test['ci95_low']):.4f}, {float(best_test['ci95_high']):.4f}]，"
            f"Wilcoxon p={float(best_test['wilcoxon_p']):.3f}。该差异不稳定，正式模型继续删除 Unknown。"
        )
    lines.extend(
        [
            "",
            "## 数据矩阵",
            "",
            f"- known：{audit['known_shape']}，即现有删除 Unknown 后的冻结数据；",
            f"- combined：{audit['combined_shape']}，对全部原始 VOC 一起重新执行丰度和 IQR 过滤；",
            f"- unknown-only：{audit['unknown_shape']}，仅对 Unknown 列独立过滤；",
            f"- appended：{audit['appended_shape']}，冻结 known 与独立过滤后的 Unknown 直接拼接。",
            "",
            "所有特征处理均在仓库数据构造阶段完成。模型内部的 SNV 和 ANOVA 只在当前外层训练折拟合，"
            "各表示在同一折使用相同模型随机状态，阈值固定为 0.5。",
            "",
            "## 轻量发现阶段",
            "",
            discovery.to_markdown(index=False, floatfmt=".6f"),
            "",
            "冻结进入正式确认的两个候选：" + "、".join(f"`{item}`" for item in selected) + "。",
            "",
            "## 32 种子正式确认",
            "",
            confirmation.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## 相对 known 基线的配对检验",
            "",
            paired.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## 判定边界",
            "",
            "只有候选的配对均值提升为正、bootstrap 95% 区间不跨越 0，且改进在多数种子中出现，"
            "才足以支持替换当前删除 Unknown 的正式模型。否则 Unknown 仅可视为探索性信息。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline retained-Unknown VOC comparison")
    parser.add_argument("--output", type=Path, default=Path("results/unknown_voc"))
    parser.add_argument("--n-jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--quick", action="store_true", help="Run a smoke-sized protocol")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    matrices, y, audit = load_matrices(root)

    if args.quick:
        discovery = StageSpec(
            seeds=(107001, 107002),
            model=FeatureDiverseConfig(n_submodels=2, n_estimators=3),
        )
        confirmation_seeds = (108001, 108002)
        confirmation_model = FeatureDiverseConfig(n_submodels=2, n_estimators=3)
    else:
        discovery = StageSpec(
            seeds=tuple(range(107001, 107007)),
            model=FeatureDiverseConfig(n_submodels=8, n_estimators=20),
        )
        confirmation_seeds = tuple(range(108001, 108033))
        confirmation_model = FeatureDiverseConfig(n_submodels=50, n_estimators=50)

    discovery_methods = (BASELINE, *CANDIDATES)
    discovery_rows = run_stage(discovery, matrices, y, discovery_methods, args.n_jobs)
    discovery_summary = summarize(discovery_rows)
    selected = (
        discovery_summary.loc[discovery_summary["method"] != BASELINE, "method"]
        .head(2)
        .tolist()
    )
    confirmation = StageSpec(
        seeds=confirmation_seeds,
        model=confirmation_model,
    )
    confirmation_methods = (BASELINE, *selected)
    confirmation_rows = run_stage(
        confirmation, matrices, y, confirmation_methods, args.n_jobs
    )
    confirmation_summary = summarize(confirmation_rows)
    paired = pd.DataFrame(
        [paired_test(confirmation_rows, candidate) for candidate in selected]
    )

    discovery_rows.to_csv(output / "discovery_rows.csv", index=False)
    discovery_summary.to_csv(output / "discovery_summary.csv", index=False)
    confirmation_rows.to_csv(output / "confirmation_rows.csv", index=False)
    confirmation_summary.to_csv(output / "confirmation_summary.csv", index=False)
    paired.to_csv(output / "paired_tests.csv", index=False)
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "baseline": BASELINE,
        "candidate_methods": list(CANDIDATES),
        "selected_candidates": selected,
        "selection_rule": "top two discovery mean F1 values excluding known_diverse",
        "discovery": {
            "seeds": list(discovery.seeds),
            "model": asdict(discovery.model),
        },
        "confirmation": {
            "seeds": list(confirmation.seeds),
            "model": asdict(confirmation.model),
        },
        "n_splits": 5,
        "threshold": float(confirmation.model.threshold),
        "quick": bool(args.quick),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_report(
        output / "REPORT.md",
        audit,
        discovery_summary,
        selected,
        confirmation_summary,
        paired,
    )

    print(
        json.dumps(
            {
                "selected": selected,
                "confirmation": confirmation_summary.to_dict("records"),
                "paired": paired.to_dict("records"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
