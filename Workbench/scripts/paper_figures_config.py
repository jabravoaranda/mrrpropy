"""Editable defaults for ``build_paper_figures.py``.

Command-line options override these values.  Keeping the defaults here makes
the paper workflow behave like the configuration cells in the full notebook.
"""

from pathlib import Path


RAW_DIR = Path(r"Workbench/test files/")
RAW_PATTERN = "*.nc"
OUTPUT_DIR = Path("Workbench/output/paper_figures")
PRODUCT_DIR = OUTPUT_DIR / "products"

K = 11
WINDOW_THICKNESS_M = 500.0
WINDOW_STEP_M = None
MIN_TAU_STRENGTH = 0.3
ZE_TH = -5.0
MIN_POINTS_TREND = 10
SPECTRUM_VAR = "spe_3D"
RANGE_LIMITS = (0.0, 5000.0)
SHORT_RANGE_LIMITS = (2, 2.5)
TARGET_TIME_OFFSET_MINUTES = 30.0

PROCESSES = None
SAVE_SPE_3D = True
SAVE_DSD_3D = False
FORCE_PROCESS = False
# Set to 1 while testing to rebuild processed products and sliding scans.
RESET_CACHE = 0
FIGURE_DPI = 300
# Approximate KDE domains as mean +/- this many standard deviations per process.
KDE_DOMAIN_SIGMA = 3.0
