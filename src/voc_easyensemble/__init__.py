"""Reliable small-sample VOC classification pipelines."""

from .data import VOCDataset, load_voc_mat
from .feature_diverse import FeatureDiverseConfig, FeatureDiverseEasyEnsemble
from .model import EasyEnsembleConfig, VOCEasyEnsemble

__all__ = [
    "VOCDataset",
    "load_voc_mat",
    "EasyEnsembleConfig",
    "VOCEasyEnsemble",
    "FeatureDiverseConfig",
    "FeatureDiverseEasyEnsemble",
]

__version__ = "1.1.0"
