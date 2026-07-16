"""Reliable small-sample VOC classification pipeline."""

from .data import VOCDataset, load_voc_mat
from .model import EasyEnsembleConfig, VOCEasyEnsemble

__all__ = [
    "VOCDataset",
    "load_voc_mat",
    "EasyEnsembleConfig",
    "VOCEasyEnsemble",
]

__version__ = "1.0.0"
