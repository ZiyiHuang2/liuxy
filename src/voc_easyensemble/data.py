from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class VOCDataset:
    """Validated VOC feature matrix loaded from the project MAT file."""

    X: np.ndarray
    y: np.ndarray
    feature_names: np.ndarray
    source: Path
    sha256: str
    encoded_parts: tuple[Path, ...] = ()

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def _normalize_feature_names(values: Iterable[object]) -> np.ndarray:
    names = np.asarray(list(values)).reshape(-1).astype(str)
    return np.asarray([name.strip() for name in names], dtype=str)


def load_voc_mat(path: str | Path) -> VOCDataset:
    """Load and validate X, y and feat_names from MAT or encoded parts."""

    source = Path(path).expanduser().resolve()
    encoded_parts: tuple[Path, ...] = ()
    if source.exists():
        payload = source.read_bytes()
    else:
        encoded_parts = tuple(sorted(source.parent.glob(source.name + ".b64.part*")))
        if not encoded_parts:
            raise FileNotFoundError(
                f"Dataset not found: {source}; no base64 parts matched "
                f"{source.name}.b64.part*"
            )
        encoded = "".join(
            part.read_text(encoding="ascii").strip() for part in encoded_parts
        )
        payload = base64.b64decode(encoded, validate=True)

    raw = loadmat(BytesIO(payload))
    missing = {key for key in ("X", "y", "feat_names") if key not in raw}
    if missing:
        raise KeyError(f"MAT file is missing required keys: {sorted(missing)}")

    X = np.asarray(raw["X"], dtype=np.float64)
    y = np.asarray(raw["y"]).reshape(-1).astype(np.int64)
    feature_names = _normalize_feature_names(np.asarray(raw["feat_names"]).reshape(-1))

    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape {X.shape}")
    if len(y) != X.shape[0]:
        raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {len(y)}")
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"Feature-name count mismatch: {len(feature_names)} vs {X.shape[1]}"
        )
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    unique = np.unique(y)
    if not np.array_equal(unique, np.array([0, 1])):
        raise ValueError(f"Expected binary labels [0, 1], got {unique.tolist()}")

    return VOCDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        source=source,
        sha256=hashlib.sha256(payload).hexdigest(),
        encoded_parts=encoded_parts,
    )
