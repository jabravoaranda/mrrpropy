"""
High-level API for METEK MRR-PRO data access, processing, plotting and analysis.

The package is organized around :class:`MRRProData`, which wraps an xarray
dataset and exposes three main workflows:

1. Load and inspect raw MRR-PRO NetCDF files.
2. Run or load RaProMPro processed products.
3. Generate diagnostic plots and rain-process analyses from processed variables.

The lower-level scientific processing kernel retained by the package is wrapped
here through a user-facing interface built on top of xarray and matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Union
import warnings

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

from mrrpropy.analysis import classified_rain_process_metrics as classified_metrics
from mrrpropy.analysis import rain_process_features as rain_feature_analysis
from mrrpropy.analysis import rain_processes_classification as rain_classification
from mrrpropy.analysis import sliding as sliding_analysis
from mrrpropy.analysis import trends as trend_analysis
from mrrpropy.config import MicrophysicsConfig, PlotConfig
from mrrpropy.plotting.api import PlotAPI
from mrrpropy.processing import raprompro as raprompro_processing
from mrrpropy.workflow.api import WorkflowAPI

DatetimeLike = Union[str, np.datetime64, datetime]


class _UnsetType:
    pass


_UNSET = _UnsetType()

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 32,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 14,
    }
)


@dataclass
class MRRProData:
    """
    User-facing container for raw and processed METEK MRR-PRO datasets.

    The object holds the raw xarray dataset in :attr:`ds` and, when available,
    a processed RaProMPro product in :attr:`raprompro`. Most public methods fall
    into one of four groups:

    - raw-data access and subsetting,
    - RaProMPro processing or loading,
    - radar/spectral plotting,
    - microphysical and hexagram-based rain-process analysis.
    """

    path: str | Path
    ds: xr.Dataset

    micro_cfg: MicrophysicsConfig = field(default_factory=MicrophysicsConfig)
    plot_cfg: PlotConfig = field(default_factory=PlotConfig)
    plot: PlotAPI = field(init=False)
    workflow: WorkflowAPI = field(init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.raprompro: xr.Dataset | None = None
        self.plot = PlotAPI(self)
        self.workflow = WorkflowAPI(self)

    # -------------------------
    # Constructors
    # -------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> "MRRProData":
        """
        Open a raw MRR-PRO NetCDF file and wrap it in :class:`MRRProData`.

        Parameters
        ----------
        path:
            Path to a raw MRR-PRO NetCDF file readable by :mod:`xarray`.

        Returns
        -------
        MRRProData
            Object holding the opened dataset and ready for plotting, processing
            or loading an existing RaProMPro product.
        """
        ds = xr.open_dataset(path)
        return cls(path=path, ds=ds)

    # -------------------------
    # Basic Properties
    # -------------------------
    @property
    def time(self) -> pd.DatetimeIndex:
        """Time index as pandas DatetimeIndex."""
        return self.ds["time"].to_index()

    @property
    def range(self) -> np.ndarray:
        """
        Range of bins (m above radar, typically).
        """
        return self.ds["range"].values

    @property
    def n_time(self) -> int:
        return self.ds.sizes["time"]

    @property
    def n_range(self) -> int:
        return self.ds.sizes["range"]

    @property
    def variables(self) -> List[str]:
        """List of data variables (Za, Z, Ze, RR, VEL, etc.)."""
        return [str(name) for name in self.ds.data_vars]

    # -------------------------
    # Data Access
    # -------------------------

    def get_field(self, name: str) -> xr.DataArray:
        """
        Return a dataset variable (e.g., 'Ze', 'RR', 'VEL').
        """
        if name not in self.ds:
            raise KeyError(
                f"Variable '{name}' does not exist. Available variables: {list(self.ds.data_vars)}"
            )
        return self.ds[name]

    # -------------------------
    # Subsets
    # -------------------------

    def subset(
        self,
        time_slice: Optional[slice] = None,
        range_slice: Optional[slice] = None,
    ) -> "MRRProData":
        """
        Return a new instance with a subset in time and/or range.

        Examples
        --------
        mrr_sub = mrr.subset(time_slice=slice('2025-02-05T00:10', '2025-02-05T00:30'))
        mrr_sub = mrr.subset(range_slice=slice(0, 50))   # first 50 bins
        """
        sel_kwargs: dict[str, Any] = {}
        if time_slice is not None:
            sel_kwargs["time"] = time_slice
        if range_slice is not None:
            sel_kwargs["range"] = range_slice

        ds_sub = self.ds.sel(sel_kwargs)
        return MRRProData(path=self.path, ds=ds_sub)

    # -------------------------
    # Temporal Utilities
    # -------------------------

    def nearest_time_index(self, when: DatetimeLike) -> int:
        """
        Return the time index closest to 'when'.

        Parameters
        ----------
        when : str, np.datetime64 or datetime
        """
        t = self.ds["time"]
        when_np = np.datetime64(when)
        idx = int(np.argmin(np.abs(t.values - when_np)))
        return idx

    def profile_at(
        self,
        when: DatetimeLike,
        field: str = "Ze",
    ) -> xr.DataArray:
        """
        Return the vertical profile of a variable for the nearest time.

        Parameters
        ----------
        when : reference instant (str, np.datetime64, datetime)
        field : variable name (default 'Ze').

        Returns
        -------
        xr.DataArray with 'range' dimension.
        """
        if field not in self.ds:
            raise KeyError(f"Variable '{field}' does not exist in the dataset.")
        i = self.nearest_time_index(when)
        return self.ds[field].isel(time=i)

    # -------------------------
    # Doppler Spectra
    # -------------------------

    def gate_spectrum(
        self,
        time_idx: int,
        range_idx: int,
        use_raw: bool = False,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Return the Doppler spectrum for a gate (time_idx, range_idx).

        Uses:
          - index_spectra(time, range) -> index of 'n_spectra'
          - D(n_spectra, spectrum_n_samples) -> Doppler velocity axis
          - N(time, n_spectra, spectrum_n_samples) or spectrum_raw(...)

        Parameters
        ----------
        time_idx : time index (0 .. n_time-1)
        range_idx : range index (0 .. n_range-1)
        use_raw : if True, use 'spectrum_raw' instead of 'N'.

        Returns
        -------
        (vel, spec)
        vel  : DataArray with raw-file Doppler velocity (m/s, typically).
               Plotting methods expose velocity with negative values downward.
        spec : DataArray with spectrum (N or spectrum_raw)
        """
        if "index_spectra" not in self.ds:
            raise RuntimeError(
                "Dataset does not contain 'index_spectra'; cannot retrieve spectrum."
            )

        idx_spec = int(
            self.ds["index_spectra"].isel(time=time_idx, range=range_idx).values
        )

        # Velocity axis (only n_spectra, spectrum_n_samples)
        vel = self.ds["D"].isel(n_spectra=idx_spec)

        if use_raw:
            var_name = "spectrum_raw"
        else:
            var_name = "N"

        if var_name not in self.ds:
            raise RuntimeError(
                f"Dataset does not contain spectral variable '{var_name}'."
            )

        spec = self.ds[var_name].isel(time=time_idx, n_spectra=idx_spec)
        return vel, spec

    def process_raprompro(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Run the canonical RaProMPro processing path used by the package.

        This is the default processing entry point for ``mrrpropy``.
        """
        return raprompro_processing.process_raprompro(self, *args, **kwargs)

    def _is_processed(
        self,
        *,
        required: Iterable[str] = ("Ze", "Zea", "Za", "Z_all", "Dm", "Nw", "LWC", "RR"),
    ) -> bool:
        """
        Minimal heuristic: if the key RaProMPro variables exist, treat the
        dataset as processed.

        A stricter check could also require a global processing attribute, for
        example ``ds.attrs.get("processing") == "RaProMPro"``.
        """
        if self.raprompro is None:
            return False

        return all(v in self.raprompro.data_vars for v in required)

    # -------------------------
    # Resource Management
    # -------------------------

    def close(self) -> None:
        """Close the xarray dataset (e.g., at the end of the script)."""
        self.ds.close()

    # -------------------------------------------------------------------------
    # Internal helpers for MRR-PRO spectra
    # -------------------------------------------------------------------------

    def load_raprompro(
        self,
        path: str | Path,
        *,
        chunks: str | dict | None = "auto",
        validate: bool = True,
        required_vars: tuple[str, ...] = (
            "Ze",
            "Dm",
            "Nw",
            "LWC",
            "RR",
            "Nw_all",
            "Dm_all",
            "N_da",
        ),
        assign: bool = True,
    ) -> xr.Dataset:
        """
        Load an existing RaProMPro NetCDF product and optionally validate it.

        Parameters
        ----------
        path : str | Path
            Path to the ``*_raprompro.nc`` file, for example
            ``20250308_120000_raprompro.nc``.
        chunks : "auto" | dict | None
            If not None, open lazily with dask to speed up I/O and avoid
            loading the full dataset into memory.
        validate : bool
            If True, check that the dataset has the expected dimensions and
            coordinates and matches ``self.ds``.
        required_vars : tuple[str, ...]
            Minimum variables expected in the processed dataset.
        assign : bool
            If True, store the dataset in ``self.raprompro``.

        Returns
        -------
        xr.Dataset
            Loaded processed dataset. If ``assign=True``, it is also stored in
            :attr:`raprompro`.
        """
        return raprompro_processing.load_raprompro(
            self,
            path,
            chunks=chunks,
            validate=validate,
            required_vars=required_vars,
            assign=assign,
        )

    def compute_layer_trend_ols(
        self,
        *,
        z_bottom_m: float | None = None,
        z_top_m: float | None = None,
        z_top: float | None = None,
        z_base: float | None = None,
        time_dim: str = "time",
        variable_threshold: str = "Ze",
        threshold_value: float = -5.0,
        vars: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
        eps_mode: str = "hourly_quantile",
        q: float = 0.01,
        eps_floor_mode: str = "global_min",
        min_points_ols: int = 10,
    ) -> xr.Dataset:
        """
        Compute layer-wise legacy OLS trends of selected microphysical variables.

        For each time step, the method fits ``ln(X)`` versus depth from the top
        of the selected layer, after thresholding on a reflectivity field such as
        ``Ze``. It returns slopes, intercepts, fit quality and the masks actually
        used in each regression.

        The output is kept for backward compatibility and diagnostic comparison.
        The recommended microphysical method is :meth:`compute_layer_trend`,
        which uses Kendall's tau plus Theil-Sen slope.

        Use ``z_bottom_m`` and ``z_top_m`` to define the physical layer bounds.
        Legacy ``z_top`` / ``z_base`` aliases are still accepted for
        compatibility.
        """
        return trend_analysis.compute_layer_trend_ols(
            self,
            z_bottom_m=z_bottom_m,
            z_top_m=z_top_m,
            z_top=z_top,
            z_base=z_base,
            time_dim=time_dim,
            variable_threshold=variable_threshold,
            threshold_value=threshold_value,
            vars=vars,
            eps_mode=eps_mode,
            q=q,
            eps_floor_mode=eps_floor_mode,
            min_points_ols=min_points_ols,
        )

    def compute_layer_trend(
        self,
        *,
        z_bottom_m: float | None = None,
        z_top_m: float | None = None,
        z_top: float | None = None,
        z_base: float | None = None,
        time_dim: str = "time",
        variable_threshold: str = "Ze",
        threshold_value: float = -5.0,
        vars: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
        trend_method: str = "kendall_theilsen",
        tau_zero_tol: float = 0.05,
        min_points_trend: int | None = None,
        min_points_ols: int | None = None,
        eps_mode: str = "hourly_quantile",
        q: float = 0.01,
        eps_floor_mode: str = "global_min",
    ) -> xr.Dataset:
        """
        Compute layer-wise microphysical trends.

        The returned dataset always exposes canonical downstream fields such as
        ``trend_mag_*``, ``trend_sign_*``, ``trend_strength_*``,
        ``trend_score_*`` and ``trend_p_*``. By default, the underlying trend
        summary is non-parametric: Kendall's tau captures monotonic direction
        and consistency, while Theil-Sen slope captures robust magnitude.
        ``trend_method="ols"`` keeps the legacy fit available for comparison.

        The fixed layer is defined with ``z_bottom_m`` and ``z_top_m`` in
        meters, with positive change meaning increase while descending from
        ``z_top_m`` to ``z_bottom_m``.
        """
        return trend_analysis.compute_layer_trend(
            self,
            z_bottom_m=z_bottom_m,
            z_top_m=z_top_m,
            z_top=z_top,
            z_base=z_base,
            time_dim=time_dim,
            variable_threshold=variable_threshold,
            threshold_value=threshold_value,
            vars=vars,
            trend_method=trend_method,
            tau_zero_tol=tau_zero_tol,
            min_points_trend=min_points_trend,
            min_points_ols=min_points_ols,
            eps_mode=eps_mode,
            q=q,
            eps_floor_mode=eps_floor_mode,
        )

    def rain_process_analyze(
        self,
        *,
        period: tuple[datetime, datetime],
        k: int,
        selection_mode: str = "sliding",
        window_thickness_m: float | None = None,
        window_step_m: float | None | _UnsetType = _UNSET,
        z_bottom_m: float | None = None,
        z_top_m: float | None = None,
        layer: tuple[float, float] | None = None,
        ze_th: float = -5.0,
        trend_method: str = "kendall_theilsen",
        tau_zero_tol: float = 0.05,
        min_points_trend: int | None = None,
        min_points_ols: int | None = None,
        eps_q: float = 0.01,
        rgb_q: float = 0.02,
        vars_trend: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
        min_tau_strength: float | None | _UnsetType = _UNSET,
        max_tau_pvalue: float | None = None,
    ) -> xr.Dataset | pd.DataFrame:
        """
        Analyse rain-process evolution with a sliding-first public workflow.

        ``selection_mode="sliding"`` is the default public interface and returns a
        dataframe built from sliding windows defined by ``window_thickness_m``
        and ``window_step_m``.

        ``selection_mode="fixed_layer"`` keeps the explicit-layer workflow for
        advanced use and returns the fixed-layer analysis dataset. In that mode,
        use ``z_bottom_m`` and ``z_top_m`` to define the layer. Legacy
        ``layer=(z_bottom_m, z_top_m)`` remains supported with a warning.

        The workflow is:

        1. compute trend diagnostics for ``vars_trend``,
        2. map those diagnostics into RGB space,
        3. project the RGB samples onto the package hexagram grid.

        The pipeline consumes method-neutral canonical trend variables, so the
        downstream RGB and classification steps do not depend on whether the
        diagnostics came from Kendall/Theil-Sen or from the legacy OLS method.

        Returns
        -------
        xr.Dataset | pd.DataFrame
            Sliding mode returns the column-sliding dataframe. Fixed-layer mode
            returns the analysis dataset containing the trend diagnostics, RGB
            channels, elapsed minutes and the hexagram coordinates used
            downstream for plotting and classification.

        Notes
        -----
        In sliding mode, window geometry and the tau-strength threshold default to
        :attr:`micro_cfg` unless overridden by explicit arguments.

        ``window_step_m=None`` means "raw resolution": use the native range grid
        spacing (median of the range-coordinate differences).
        """
        mode = str(selection_mode).strip().lower()
        if mode not in {"sliding", "fixed_layer"}:
            raise ValueError("selection_mode must be either 'sliding' or 'fixed_layer'.")

        has_fixed_layer_args = (
            layer is not None or z_bottom_m is not None or z_top_m is not None
        )
        if mode == "sliding" and has_fixed_layer_args:
            warnings.warn(
                "Fixed-layer arguments were provided to rain_process_analyze(). "
                "Running in selection_mode='fixed_layer'. For the default "
                "public workflow, prefer sliding mode with `window_thickness_m` "
                "and `window_step_m`.",
                FutureWarning,
                stacklevel=2,
            )
            mode = "fixed_layer"

        if mode == "sliding":
            thickness_m = (
                float(window_thickness_m)
                if window_thickness_m is not None
                else float(self.micro_cfg.window_thickness_m)
            )
            step_m: float | None
            if isinstance(window_step_m, _UnsetType):
                step_m = self.micro_cfg.window_step_m
            else:
                step_m = window_step_m
            tau_strength: float | None
            if isinstance(min_tau_strength, _UnsetType):
                tau_strength = self.micro_cfg.min_tau_strength
            else:
                tau_strength = min_tau_strength
            return sliding_analysis.build_sliding_process_dataframe(
                self,
                period=period,
                k=k,
                window_thickness_m=thickness_m,
                window_step_m=step_m,
                min_tau_strength=tau_strength,
                ze_th=ze_th,
                trend_method=trend_method,
                tau_zero_tol=tau_zero_tol,
                min_points_trend=min_points_trend,
                min_points_ols=min_points_ols,
                eps_q=eps_q,
                rgb_q=rgb_q,
                vars_trend=vars_trend,
                max_tau_pvalue=max_tau_pvalue,
            )

        return rain_classification.rain_process_analyze(
            self,
            period=period,
            z_bottom_m=z_bottom_m,
            z_top_m=z_top_m,
            layer=layer,
            k=k,
            ze_th=ze_th,
            trend_method=trend_method,
            tau_zero_tol=tau_zero_tol,
            min_points_trend=min_points_trend,
            min_points_ols=min_points_ols,
            eps_q=eps_q,
            rgb_q=rgb_q,
            vars_trend=vars_trend,
        )

    def classify_rain_process(
        self,
        *,
        analysis: xr.Dataset,
        tol_center: float = 0.05,
        min_strength: float = 0.10,
        min_tau_strength: float | None | _UnsetType = _UNSET,
        max_p_value: float | None = None,
        max_tau_pvalue: float | None = None,
    ) -> xr.Dataset:
        """
        Classify each time sample into a rain-process category.

        The method expects the RGB mapping created by
        :meth:`rain_process_analyze`, with the convention ``R -> Dm``,
        ``G -> Nw`` and ``B -> LWC``. When canonical trend diagnostics are
        present, classification uses ``trend_sign_*`` and ``trend_strength_*``
        independently of the underlying trend method. RGB-centre classification
        is retained as a compatibility fallback for legacy analyses.
        """

        tau_strength: float | None
        if isinstance(min_tau_strength, _UnsetType):
            tau_strength = self.micro_cfg.min_tau_strength
        else:
            tau_strength = min_tau_strength
        return rain_classification.classify_rain_process(
            self,
            analysis=analysis,
            tol_center=tol_center,
            min_strength=min_strength,
            min_tau_strength=tau_strength,
            max_p_value=max_p_value,
            max_tau_pvalue=max_tau_pvalue,
        )

    def build_rain_process_features(
        self,
        *,
        ds: xr.Dataset | None = None,
        mode: Literal["fixed_layer", "scan"],
        range_coord: str = "range",
        window_thickness_m: float | None = None,
        window_step_m: float | None | _UnsetType = _UNSET,
        fixed_layer_top_m: float | None = None,
        fixed_layer_bottom_m: float | None = None,
        bb_bottom_m: float | xr.DataArray,
        bb_peak_m: float | xr.DataArray,
        bb_top_m: float | xr.DataArray,
        Dm_var: str = "Dm",
        Nw_var: str = "Nw",
        LWC_var: str = "LWC",
        RR_var: str = "RR",
        spectrum_var: str = "spectrum",
        velocity_coord: str = "velocity",
    ) -> xr.Dataset:
        """
        Build rain-process feature variables from a dataset.

        In scan mode, ``window_thickness_m`` and ``window_step_m`` default to
        :attr:`micro_cfg` when not explicitly provided. ``window_step_m=None``
        means "raw resolution" (native range-grid spacing).
        """
        ds_in = (
            ds
            if ds is not None
            else (self.raprompro if self.raprompro is not None else self.ds)
        )
        step_m: float | None
        if isinstance(window_step_m, _UnsetType):
            step_m = self.micro_cfg.window_step_m
        else:
            step_m = window_step_m
        return rain_feature_analysis.build_rain_process_features(
            ds_in,
            mode=mode,
            range_coord=range_coord,
            window_thickness_m=window_thickness_m,
            window_step_m=step_m,
            fixed_layer_top_m=fixed_layer_top_m,
            fixed_layer_bottom_m=fixed_layer_bottom_m,
            bb_bottom_m=bb_bottom_m,
            bb_peak_m=bb_peak_m,
            bb_top_m=bb_top_m,
            micro_cfg=self.micro_cfg,
            Dm_var=Dm_var,
            Nw_var=Nw_var,
            LWC_var=LWC_var,
            RR_var=RR_var,
            spectrum_var=spectrum_var,
            velocity_coord=velocity_coord,
        )

    def classify_rain_process_features(
        self,
        *,
        rain_process_features: xr.Dataset,
        refiners: list[Any] | None = None,
        min_strength: float = 0.10,
        min_tau_strength: float | None | _UnsetType = _UNSET,
        max_p_value: float | None = None,
        max_tau_pvalue: float | None = None,
    ) -> xr.Dataset:
        """
        Classify rain-process labels directly from `rain_process_features`.

        This is the classification step paired with
        :meth:`build_rain_process_features`.
        """
        tau_strength: float | None
        if isinstance(min_tau_strength, _UnsetType):
            tau_strength = self.micro_cfg.min_tau_strength
        else:
            tau_strength = min_tau_strength
        return rain_classification.classify_rain_process_features(
            rain_process_features,
            refiners=refiners,
            min_strength=min_strength,
            min_tau_strength=tau_strength,
            max_p_value=max_p_value,
            max_tau_pvalue=max_tau_pvalue,
        )

    def build_process_dynamics_dataframe(
        self,
        *,
        analysis: xr.Dataset,
        classified: xr.Dataset,
        variables: tuple[str, ...] = ("Dm", "Nw", "LWC"),
    ) -> pd.DataFrame:
        """
        Build a per-sample dataframe for quantitative process analysis.

        The dataframe follows the descending-rain convention used by the
        microphysical pipeline, so ``*_delta`` means bottom minus top inside the
        selected layer.
        """
        return classified_metrics.build_process_dynamics_dataframe(
            self,
            analysis=analysis,
            classified=classified,
            variables=variables,
        )

    def summarize_process_dynamics(
        self,
        *,
        analysis: xr.Dataset,
        classified: xr.Dataset,
        variables: tuple[str, ...] = ("Dm", "Nw", "LWC"),
    ) -> pd.DataFrame:
        """
        Summarize rain-process dynamics grouped by ``proc_label``.

        This is a compact table-oriented companion to the process figures and is
        intended for exploratory scientific analysis.
        """
        return classified_metrics.summarize_process_dynamics(
            self,
            analysis=analysis,
            classified=classified,
            variables=variables,
        )

    def build_sliding_process_dataframe(
        self,
        *,
        period: tuple[datetime, datetime],
        k: int,
        window_thickness_m: float | None = None,
        window_step_m: float | None | _UnsetType = _UNSET,
        min_tau_strength: float | None | _UnsetType = _UNSET,
        ze_th: float = -5.0,
        trend_method: str = "kendall_theilsen",
        tau_zero_tol: float = 0.05,
        min_points_trend: int | None = None,
        min_points_ols: int | None = None,
        eps_q: float = 0.01,
        rgb_q: float = 0.02,
        vars_trend: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
        max_tau_pvalue: float | None = None,
    ) -> pd.DataFrame:
        """
        Apply the rain-process analysis across the column with a sliding
        vertical window.

        By default, ``window_thickness_m``, ``window_step_m`` and
        ``min_tau_strength`` are taken from :attr:`micro_cfg` unless overridden
        by explicit arguments. ``window_step_m=None`` means "raw resolution"
        (native range-grid spacing).

        The output dataframe contains one row per ``time x window`` and is the
        recommended input for :meth:`detect_sliding_process_episodes`.
        """
        thickness_m = (
            float(window_thickness_m)
            if window_thickness_m is not None
            else float(self.micro_cfg.window_thickness_m)
        )
        step_m: float | None
        if isinstance(window_step_m, _UnsetType):
            step_m = self.micro_cfg.window_step_m
        else:
            step_m = window_step_m
        tau_strength: float | None
        if isinstance(min_tau_strength, _UnsetType):
            tau_strength = self.micro_cfg.min_tau_strength
        else:
            tau_strength = min_tau_strength
        return sliding_analysis.build_sliding_process_dataframe(
            self,
            period=period,
            k=k,
            window_thickness_m=thickness_m,
            window_step_m=step_m,
            min_tau_strength=tau_strength,
            ze_th=ze_th,
            trend_method=trend_method,
            tau_zero_tol=tau_zero_tol,
            min_points_trend=min_points_trend,
            min_points_ols=min_points_ols,
            eps_q=eps_q,
            rgb_q=rgb_q,
            vars_trend=vars_trend,
            max_tau_pvalue=max_tau_pvalue,
        )

    def detect_sliding_process_episodes(
        self,
        *,
        sliding_df: pd.DataFrame,
        min_consecutive_profiles: int = 6,
    ) -> pd.DataFrame:
        """
        Detect persistent process episodes from a column sliding dataframe.

        Episodes are defined independently in each sliding window and require a
        minimum number of consecutive profiles with the same process label.
        """
        return sliding_analysis.detect_sliding_process_episodes(
            self,
            sliding_df=sliding_df,
            min_consecutive_profiles=min_consecutive_profiles,
        )
