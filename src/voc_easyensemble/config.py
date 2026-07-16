from __future__ import annotations

import json
from pathlib import Path

from .model import EasyEnsembleConfig


def load_config(path: str | Path) -> EasyEnsembleConfig:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return EasyEnsembleConfig(**values)
