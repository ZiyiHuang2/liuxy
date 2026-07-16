from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TypeAlias

from .feature_diverse import FeatureDiverseConfig, FeatureDiverseEasyEnsemble
from .model import EasyEnsembleConfig, VOCEasyEnsemble

ModelConfig: TypeAlias = EasyEnsembleConfig | FeatureDiverseConfig
ModelType: TypeAlias = VOCEasyEnsemble | FeatureDiverseEasyEnsemble


def load_config(path: str | Path) -> ModelConfig:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    model_type = str(values.pop("model_type", "fixed")).strip().lower()
    if model_type in {"fixed", "easyensemble", "voc_easyensemble"}:
        return EasyEnsembleConfig(**values)
    if model_type in {"feature_diverse", "feature-diverse", "enhanced"}:
        return FeatureDiverseConfig(**values)
    raise ValueError(f"Unsupported model_type: {model_type}")


def build_model(config: ModelConfig) -> ModelType:
    if isinstance(config, EasyEnsembleConfig):
        return VOCEasyEnsemble.from_config(config)
    if isinstance(config, FeatureDiverseConfig):
        return FeatureDiverseEasyEnsemble.from_config(config)
    raise TypeError(f"Unsupported configuration type: {type(config).__name__}")


def config_to_dict(config: ModelConfig) -> dict[str, object]:
    payload: dict[str, object] = asdict(config)
    payload["model_type"] = (
        "feature_diverse" if isinstance(config, FeatureDiverseConfig) else "fixed"
    )
    return payload
