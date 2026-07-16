from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import load_voc_mat
from .evaluation import run_repeated_cv
from .model import VOCEasyEnsemble


def _parse_seeds(value: str) -> list[int]:
    value = value.strip()
    if ":" in value and "," not in value:
        start, end = (int(part) for part in value.split(":", maxsplit=1))
        if end < start:
            raise argparse.ArgumentTypeError("Seed range end must be >= start")
        return list(range(start, end + 1))
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Seeds must be comma-separated integers or START:END"
        ) from exc


def command_inspect(args: argparse.Namespace) -> None:
    dataset = load_voc_mat(args.data)
    counts = dict(zip(*np.unique(dataset.y, return_counts=True)))
    payload = {
        "path": str(dataset.source),
        "sha256": dataset.sha256,
        "n_samples": dataset.n_samples,
        "n_features": dataset.n_features,
        "class_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "zero_fraction": float(np.mean(dataset.X == 0)),
        "minimum": float(dataset.X.min()),
        "maximum": float(dataset.X.max()),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def command_evaluate(args: argparse.Namespace) -> None:
    dataset = load_voc_mat(args.data)
    config = load_config(args.config)
    paths = run_repeated_cv(
        dataset,
        seeds=_parse_seeds(args.seeds),
        config=config,
        output_dir=args.output,
        n_splits=args.folds,
    )
    print("\nSaved:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def command_train(args: argparse.Namespace) -> None:
    dataset = load_voc_mat(args.data)
    config = load_config(args.config)
    model = VOCEasyEnsemble.from_config(config)
    model.fit(dataset.X, dataset.y, feature_names=dataset.feature_names)
    model_path = model.save(args.model)

    metadata = {
        "dataset": str(dataset.source),
        "dataset_sha256": dataset.sha256,
        "n_samples": dataset.n_samples,
        "n_features": dataset.n_features,
        "selected_feature_indices": model.selector_.selected_indices_.tolist(),
        "selected_feature_names": (
            model.selected_feature_names_.tolist()
            if model.selected_feature_names_ is not None
            else None
        ),
        "config": config.__dict__,
    }
    metadata_path = model_path.with_suffix(model_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")


def command_predict(args: argparse.Namespace) -> None:
    model = VOCEasyEnsemble.load(args.model)
    frame = pd.read_csv(args.input)

    if model.feature_names_in_ is not None:
        missing = [name for name in model.feature_names_in_ if name not in frame.columns]
        if missing:
            raise ValueError(
                f"Input CSV is missing {len(missing)} required feature columns; "
                f"first: {missing[:10]}"
            )
        X = frame.loc[:, model.feature_names_in_].to_numpy(dtype=float)
    else:
        X = frame.to_numpy(dtype=float)

    probability = model.predict_proba(X)[:, 1]
    output = frame.copy()
    output["positive_probability"] = probability
    output["prediction"] = (probability >= model.threshold).astype(int)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved predictions: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voc-easy",
        description="VOC classification with SNV + ANOVA Top-K + EasyEnsemble",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser("inspect", help="Inspect the bundled MAT dataset")
    inspect_parser.add_argument("--data", default="data/voc_dataset_1+2_vs_3.mat")
    inspect_parser.set_defaults(func=command_inspect)

    evaluate_parser = sub.add_parser(
        "evaluate", help="Run repeated outer cross-validation"
    )
    evaluate_parser.add_argument("--data", default="data/voc_dataset_1+2_vs_3.mat")
    evaluate_parser.add_argument("--config", default="configs/default.json")
    evaluate_parser.add_argument("--seeds", default="71001:71016")
    evaluate_parser.add_argument("--folds", type=int, default=5)
    evaluate_parser.add_argument("--output", default="outputs/repeated_cv")
    evaluate_parser.set_defaults(func=command_evaluate)

    train_parser = sub.add_parser("train", help="Fit the frozen model on all data")
    train_parser.add_argument("--data", default="data/voc_dataset_1+2_vs_3.mat")
    train_parser.add_argument("--config", default="configs/default.json")
    train_parser.add_argument("--model", default="artifacts/voc_easyensemble.joblib")
    train_parser.set_defaults(func=command_train)

    predict_parser = sub.add_parser(
        "predict", help="Predict a CSV with the original 445 feature columns"
    )
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", default="outputs/predictions.csv")
    predict_parser.set_defaults(func=command_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
