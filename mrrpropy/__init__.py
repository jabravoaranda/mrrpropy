"""
Top-level package for `mrrpropy`.

The public API is centered on :class:`mrrpropy.dataset.MRRProData`, which loads
MRR-PRO files, owns the raw and processed datasets, and exposes processing,
analysis, classification and plotting namespaces.
"""

from pathlib import Path

from mrrpropy.config import MicrophysicsConfig, PlotConfig
from mrrpropy.dataset import MRRProData

PACKAGE_DIR = Path(__file__).resolve().parent
__version__ = "0.3.0"

__all__ = [
    "MRRProData",
    "MicrophysicsConfig",
    "PACKAGE_DIR",
    "PlotConfig",
    "__version__",
]
