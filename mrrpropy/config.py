"""Configuration objects for the public MRR-PRO data interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MicrophysicsConfig:
    """
    Default thresholds and RGB/hexagram settings for rain-process analysis.

    Scan-mode workflows use ``window_thickness_m`` and ``window_step_m`` from
    this configuration by default. ``window_step_m=None`` means the native range
    grid spacing is used as the scan step.
    """

    variable_threshold: str = "Ze"
    threshold_value: float = -5.0
    window_thickness_m: float = 500.0
    window_step_m: float | None = None
    trend_method: str = "kendall_theilsen"
    tau_zero_tol: float = 0.05
    min_points_trend: int = 10
    min_points_ols: int = 10
    min_tau_strength: float = 0.5
    max_tau_pvalue: float | None = None
    eps_q: float = 0.01
    rgb_q: float = 0.02
    eps_mode: str = "global_quantile"
    tol_center: float = 0.05
    min_strength: float = 0.10
    vars_trend: tuple[str, str, str] = ("Dm", "Nw", "LWC")
    k: int = 11


@dataclass
class PlotConfig:
    """Default plotting configuration shared by plotting methods."""

    figsize: tuple[float, float] = (10, 10)
    figsize_hex: tuple[float, float] = (10, 10)
    figsize_summary: tuple[float, float] = (14, 10)
    figsize_quicklook: tuple[float, float] = (16, 8)
    figsize_spectrogram: tuple[float, float] = (10, 14)
    figsize_profiles: tuple[float, float] = (14, 10)
    figsize_multipanel: tuple[float, float] = (14, 10)
    cmap: str = "jet"
    marker: str = "o"
    markersize: float = 10.0
    legendfontsize: float = 12.0
    alpha_points: float = 0.9
    alpha_hexagram: float = 0.25
    show_path_line: bool = True
    linewidth: float = 0.8
    dpi: int = 200
