"""
Top-level package for `mrrpropy`.

The project provides:

- a high-level `MRRProData` API for loading, processing and plotting METEK MRR-PRO
  datasets,
- the published RaProMPro scientific processing implementation,
- utilities for microphysical process analysis and hexagram-based classification.

The public API is intentionally concentrated around `mrrpropy.raw_class.MRRProData`.
The lower-level `preprocessing/RaProMPro_*` modules are retained mainly for scientific reference and
workflow-controlled optimization work.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
__version__ = "0.3.0"

__all__ = ["PACKAGE_DIR", "__version__"]
