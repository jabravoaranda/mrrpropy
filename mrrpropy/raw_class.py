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
from typing import Any, Iterable, List, Optional, Union
import warnings

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

from mrrpropy.analysis import process_features as process_feature_analysis
from mrrpropy.analysis import processes as process_analysis
from mrrpropy.plotting import _spectra as spectral_plotting
from mrrpropy.plotting import processes as process_plotting
from mrrpropy.plotting import processed as processed_plotting
from mrrpropy.plotting import raw as raw_plotting
from mrrpropy.processing import raprompro as raprompro_processing

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
class MicrophysicsConfig:
    """
    Default thresholds and RGB/hexagram settings for rain-process analysis.

    Notes
    -----
    Scan-mode workflows (sliding vertical windows) use ``window_thickness_m`` and
    ``window_step_m`` from this configuration by default when explicit arguments
    are not provided.

    ``window_step_m=None`` means "raw resolution": the scan step is inferred
    from the native range grid spacing (median of the range-coordinate
    differences).
    """

    variable_threshold: str = "Ze"
    threshold_value: float = -5.0
    window_thickness_m: float = 500.0
    window_step_m: float | None = None  # None means "use raw vertical resolution"
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
    k: int = 11  # default hex resolution

@dataclass
class PlotConfig:
    """Default plotting configuration shared by the high-level plotting methods."""

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

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.raprompro: xr.Dataset | None = None

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
        return list(self.ds.data_vars)

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
        sel_kwargs = {}
        if time_slice is not None:
            sel_kwargs["time"] = time_slice
        if range_slice is not None:
            sel_kwargs["range"] = range_slice

        ds_sub = self.ds.sel(**sel_kwargs)
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

    def process_raprompro_optimized(
        self,
        *,
        adjust_m: float = 1.0,
        save_spe_3d: bool = False,
        save_dsd_3d: bool = False,
        save: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Compatibility alias for :meth:`process_raprompro`.
        """
        return raprompro_processing.process_raprompro_optimized(
            self,
            adjust_m=adjust_m,
            save_spe_3d=save_spe_3d,
            save_dsd_3d=save_dsd_3d,
            save=save,
            **kwargs,
        )

    def _is_processed(
        self,
        *,
        required: Iterable[str] = ("Ze", "Zea", "Za", "Z_all", "Dm", "Nw", "LWC", "RR"),
    ) -> bool:
        """
        Heurística mínima: si existen las variables clave de RaProMPro,
        consideramos que el Dataset está preprocesado.

        Si quieres hacerlo más robusto, puedes además exigir algún atributo global:
        ds.attrs.get("processing") == "RaProMPro" o similar.
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
    # Helpers internos para espectros MRR-PRO
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
            Ruta al fichero *_raprompro.nc (p.ej. '20250308_120000_raprompro.nc').
        chunks : "auto" | dict | None
            Si no es None, abre en modo dask (lazy) para acelerar I/O y evitar cargar todo a RAM.
        validate : bool
            Si True, comprueba que el dataset tiene dims/coords esperadas y que encaja con self.ds.
        required_vars : tuple[str, ...]
            Variables mínimas que deben existir en el dataset procesado.
        assign : bool
            Si True, guarda el dataset en self.raprompro.

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

    def _nearest_period(
        self, target_datetime: datetime | np.datetime64, target_range: float
    ) -> tuple[np.datetime64, float]:
        """Devuelve el time y range reales seleccionados por nearest."""
        return spectral_plotting.nearest_period(self, target_datetime, target_range)

    def _get_velocity_axis(self, n_bins: int) -> np.ndarray:
        """
        Construye el eje de velocidades Doppler (m/s) en ausencia de un eje explícito.

        Nota: MRR-Pro a menudo no guarda el vector de velocidades por bin como coord.
        Usamos fold_limit_upper si está en attrs de VEL, si no asumimos 12 m/s.
        """
        return spectral_plotting.get_velocity_axis(self, n_bins)

    def _get_spectrum_1d(
        self,
        target_datetime: datetime | np.datetime64,
        target_range: float,
        *,
        spectrum_var: str = "spectrum_reflectivity",
    ) -> tuple[np.datetime64, float, np.ndarray, np.ndarray, str]:
        """
        Extrae el espectro 1D más cercano a (time, range), soportando:
          - cubo: spectrum_var(time, range, spectrum_n_samples)
          - indexado: spectrum_var(time, n_spectra, spectrum_n_samples) + index_spectra(time, range)

        Returns:
          t_sel, r_sel, vel_axis, spec_1d, units
        """
        return spectral_plotting.get_spectrum_1d(
            self,
            target_datetime,
            target_range,
            spectrum_var=spectrum_var,
        )

    def _get_spectrogram_2d(
        self,
        target_datetime: datetime | np.datetime64,
        *,
        spectrum_var: str,
        range_limits: tuple[float, float] | None = None,
    ) -> tuple[np.datetime64, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Extrae un espectrograma 2D (range x doppler_bin) para el instante más cercano.

        Returns:
          t_sel, ranges, vel_axis, spec2d, units
        """
        return spectral_plotting.get_spectrogram_2d(
            self,
            target_datetime,
            spectrum_var=spectrum_var,
            range_limits=range_limits,
        )
    # -------------------------
    # Quick Plot (optional)
    # -------------------------

    def quicklook(
        self,
        variable: str = "Ze",
        source: str = "raprompro",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plot a quick time-height view of a raw or processed 2D field.

        This is the fastest visual diagnostic for fields such as ``Ze``, ``Zea``,
        ``Za``, ``RR`` or other variables stored on the ``(time, range)`` grid.

        Parameters
        ----------
        variable:
            Name of the variable to plot.
        source:
            ``"raw"`` to read from :attr:`ds`, or ``"raprompro"`` to read from
            :attr:`raprompro`.
        vmin, vmax:
            Optional color limits in data units.

        Returns
        -------
        tuple[Figure, Axes, Path | None]
            Matplotlib figure, axes, and optional saved path.
        """
        return raw_plotting.quicklook(
            self,
            variable=variable,
            source=source,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_spectrum(
        self,
        target_datetime: datetime | np.datetime64,
        target_range: float,
        *,
        spectrum_var: str = "spectrum_reflectivity",
        velocity_limits: tuple[float, float] | None = None,
        label_type: str = "both",  # both|time|range
        fig: Figure | None = None,
        ax: Axes | None = None,
        savefig: bool = False,
        output_dir: Path | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plot a single-gate Doppler spectrum at a selected time and range.

        The method supports the spectral variables already exposed by the raw
        MRR-PRO files, typically ``spectrum_reflectivity`` or ``spectrum_raw``.
        The plotted Doppler velocity axis follows the public mrrpropy convention:
        negative values indicate downward hydrometeor motion.

        Parameters
        ----------
        target_datetime : datetime | np.datetime64
            The target time for which to extract the spectrum.
        target_range : float
            The target range (in meters) for which to extract the spectrum.
        spectrum_var : str, optional
            The spectrum variable to plot. Default is "spectrum_reflectivity".
        velocity_limits : tuple[float, float] | None, optional
            The velocity limits for the x-axis as (min, max). If None, limits are
            automatically determined from the data. Default is None.
        label_type : str, optional
            The type of label to display. Options are "both" (time and range),
            "time", or "range". Default is "both".
        fig : Figure | None, optional
            An existing matplotlib Figure object. If None, a new figure is created.
            Default is None.
        ax : Axes | None, optional
            An existing matplotlib Axes object. If None and fig is None, a new
            axes is created. If fig is provided but ax is None, the first axes
            from fig is used. Default is None.
        savefig : bool, optional
            Whether to save the figure to a file. Default is False.
        output_dir : Path | None, optional
            The directory where the figure will be saved. Required if savefig is True.
            Default is None.
            Additional keyword arguments passed to matplotlib plotting functions.
            - color : str, optional
                Line color for the spectrum plot. Default is 'black'.
            - dpi : int, optional
                DPI for saved figure. If not provided, uses the plot configuration dpi.
        Returns
        -------
        tuple[Figure, Path | None]
            A tuple containing:
            - fig : matplotlib Figure object
            - filepath : Path to the saved figure if savefig is True, otherwise None
        Raises
        ------
        ValueError
            If savefig is True but output_dir is None.
        """
        return raw_plotting.plot_spectrum(
            self,
            target_datetime,
            target_range,
            spectrum_var=spectrum_var,
            velocity_limits=velocity_limits,
            label_type=label_type,
            fig=fig,
            ax=ax,
            savefig=savefig,
            output_dir=output_dir,
            **kwargs,
        )

    def plot_spectra_by_range(
        self,
        target_datetime,
        ranges: list[float] | np.ndarray,
        *,
        use_db: bool = True,
        label_type: str = "range",
        ncol: int = 2,
        fig=None,
        ax=None,
        savefig: bool = False,
        output_dir=None,
        **kwargs,
    ):
        """
        Plot several MRR-PRO Doppler spectra at a fixed time for multiple ranges.

        This method overlays spectra for the nearest (time, range) gates.
        It relies on the RAW spectral variable 'spectrum_reflectivity' (preferred) or
        falls back to 'spectrum' if present.
        The plotted Doppler velocity axis uses negative values for downward motion.

        Parameters
        ----------
        target_datetime : datetime | np.datetime64 | str
            Time to plot. Nearest time gate is used.
        ranges : list[float] | np.ndarray
            List of ranges [m]. Nearest range gate is used for each value.
        use_db : bool, default True
            Plot spectrum in dB if True (10*log10), else linear.
        label_type : {"range","time","both"}, default "range"
            Legend label formatting.
        ncol : int, default 2
            Legend columns.
        figsize : tuple, default (10,7)
            Figure size if fig/ax not provided.
        fig, ax : matplotlib Figure/Axes, optional
            Reuse existing axes.
        output_dir : Path, optional
            Where to save if savefig=True.
        savefig : bool, default False
            Save figure if True.
        dpi : int, default 200
            Save DPI.
        kwargs :
            Optional plot kwargs forwarded to ax.plot (e.g., linewidth, alpha).

        Returns
        -------
        (fig, ax, filepath) : (Figure, Axes, Path | None)
        """
        return raw_plotting.plot_spectra_by_range(
            self,
            target_datetime,
            ranges,
            use_db=use_db,
            label_type=label_type,
            ncol=ncol,
            fig=fig,
            ax=ax,
            savefig=savefig,
            output_dir=output_dir,
            **kwargs,
        )

    def plot_spectrogram(
        self,
        target_datetime: datetime | np.datetime64,
        *,
        spectrum_var: str,
        variable_threshold: str = "spectrum_raw",
        threshold_value: float = 0,
        range_limits: tuple[float, float] | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        fig: Figure | None = None,
        output_dir: Path | None = None,
        savefig: bool = False,
        **kwargs,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plot a range-by-velocity spectrogram at the nearest requested time.
        The displayed velocity axis uses negative values for downward motion.

        Parameters
        ----------
        target_datetime : datetime | np.datetime64
            The target time for which to generate the spectrogram.
        spectrum_var : str, optional
            The spectrum variable to plot. Default is "spectrum_raw".
        range_limits : tuple[float, float] | None, optional
            Range limits in meters as (min, max). If None, uses full range. Default is None.
        vmin : float | None, optional
            Minimum value for the colorbar scale. If None, uses automatic scaling. Default is None.
        vmax : float | None, optional
            Maximum value for the colorbar scale. If None, uses automatic scaling. Default is None.
        cmap : str, optional
            Matplotlib colormap name. Default is "jet".
        fig : Figure | None, optional
            Matplotlib Figure object. If None, a new figure is created. Default is None.
        ax : optional
            Matplotlib Axes object. If None and fig is provided, uses first axes of fig.
            If both are None, creates new figure and axes. Default is None.
        output_dir : Path | None, optional
            Output directory for saving the figure. Required if savefig=True. Default is None.
        savefig : bool, optional
            Whether to save the figure to disk. Default is False.
        dpi : int, optional
            Resolution in dots per inch for saved figure. Default is 200.

        Returns
        -------
        tuple[Figure, Path | None]
            A tuple containing:
            - Figure: The matplotlib Figure object containing the spectrogram.
            - Path | None: Path to the saved figure if savefig=True, otherwise None.

        Raises
        ------
        ValueError
            If savefig=True but output_dir is None.
        """
        return raw_plotting.plot_spectrogram(
            self,
            target_datetime,
            spectrum_var=spectrum_var,
            variable_threshold=variable_threshold,
            threshold_value=threshold_value,
            range_limits=range_limits,
            vmin=vmin,
            vmax=vmax,
            fig=fig,
            output_dir=output_dir,
            savefig=savefig,
            **kwargs,
        )

    def plot_DSDgram(
        self,
        *,
        target_datetime: datetime.datetime,
        range_limits: tuple[float, float] | None = None,
        drop_limits: tuple[float, float] | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        savefig: bool = False,
        output_dir: Path | None = None,
        **kwargs,
    ):
        """
        DSD-gram: X=DropSize (mm), Y=range (m), color=dsd_3D, a un instante target_datetime.

        Requiere self.raprompro con variable 'dsd_3D' dims ('time','range','DropSize').
        """
        return processed_plotting.plot_dsdgram(
            self,
            target_datetime=target_datetime,
            range_limits=range_limits,
            drop_limits=drop_limits,
            vmin=vmin,
            vmax=vmax,
            savefig=savefig,
            output_dir=output_dir,
            **kwargs,
        )

    def plot_DSD_by_range(
        self,
        target_datetime,
        ranges: list[float] | np.ndarray,
        *,
        use_log10: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        ncol: int = 2,
        savefig: bool = False,
        output_dir=None,
        fig=None,
        ax: Axes | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plot several N(D) curves at a fixed time for multiple provided ranges,
        using raprompro dsd_3D(time, range, DropSize).

        Parameters
        ----------
        target_datetime : datetime | np.datetime64 | str
            Target time. Nearest time gate is used.
        ranges : list[float] | np.ndarray
            List of ranges in meters. Nearest range gate is used for each.
        use_log10 : bool, default False
            If True, plot log10(N). If False, plot N in linear units (log y-scale).
            NOTE: If dsd_3D is stored already in log10, conversion is handled automatically.
        vmin, vmax : float | None
            Optional y-limits (applied as ylim). If both are None, no limits set.
        ncol : int, default 2
            Legend columns.
        fig, ax : matplotlib Figure/Axes, optional
            Reuse existing axes.
        output_dir : Path, optional
            Output directory if savefig=True.
        savefig : bool, default False
            Save to disk if True.

        Returns
        -------
        (fig, ax, filepath) : (Figure, Axes, Path | None)
        """
        return processed_plotting.plot_dsd_by_range(
            self,
            target_datetime,
            ranges,
            use_log10=use_log10,
            vmin=vmin,
            vmax=vmax,
            ncol=ncol,
            savefig=savefig,
            output_dir=output_dir,
            fig=fig,
            ax=ax,
            **kwargs,
        )
    
    def plot_DSD_by_range_3d(
        self,
        target_datetime,
        ranges: list[float] | np.ndarray,
        *,
        use_log10: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        ncol: int = 2,
        savefig: bool = False,
        output_dir=None,
        fig=None,
        ax: Axes | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plot several N(D) curves at a fixed time for multiple provided ranges,
        using raprompro dsd_3D(time, range, DropSize).

        Parameters
        ----------
        target_datetime : datetime | np.datetime64 | str
            Target time. Nearest time gate is used.
        ranges : list[float] | np.ndarray
            List of ranges in meters. Nearest range gate is used for each.
        use_log10 : bool, default False
            If True, plot log10(N). If False, plot N in linear units (log y-scale).
            NOTE: If dsd_3D is stored already in log10, conversion is handled automatically.
        vmin, vmax : float | None
            Optional y-limits (applied as ylim). If both are None, no limits set.
        ncol : int, default 2
            Legend columns.
        fig, ax : matplotlib Figure/Axes, optional
            Reuse existing axes.
        output_dir : Path, optional
            Output directory if savefig=True.
        savefig : bool, default False
            Save to disk if True.

        Returns
        -------
        (fig, ax, filepath) : (Figure, Axes, Path | None)
        """
        return processed_plotting.plot_dsd_by_range_3d(
            self,
            target_datetime,
            ranges,
            use_log10=use_log10,
            vmin=vmin,
            vmax=vmax,
            ncol=ncol,
            savefig=savefig,
            output_dir=output_dir,
            fig=fig,
            ax=ax,
            **kwargs,
        )

    def plot_microphysical_properties_profiles(
        self,
        target_datetime: datetime.datetime,
        savefig: bool = False,
        output_dir: Path | None = None,
        **kwargs,
    ) -> tuple[Figure, np.ndarray, Path | None]:
        """
        RaProMPro diagnostic profile (single figure, 4 axes; Y = height):
        1) Ze, Zea, Z_all, Za
        2) Dm
        3) Nw
        4) LWC, LWC_all

        Uses self.ds and selects the nearest profile to `target_datetime`.
        Raises if the dataset does not look RaProMPro-preprocessed.
        """
        return processed_plotting.plot_microphysical_properties_profiles(
            self,
            target_datetime=target_datetime,
            savefig=savefig,
            output_dir=output_dir,
            **kwargs,
        )

    def plot_rain_process_in_layer_2D(
        self,
        target_datetime: datetime | tuple[datetime, datetime],
        layer: tuple[float, float],
        x: str = "Dm",
        y: str = "LwC",
        z: str = "Nw",
        use_relative_difference: bool = True,
        savefig: bool = False,
        **kwargs,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plots the rain process in a specified atmospheric layer at a given datetime.
        This method generates a scatter plot of two selected variables (x and y) from the dataset,
        vertical layer. The plot can optionally be saved to disk.
        Parameters
        ----------
        target_datetime : datetime
            The target datetime for which to select the data profile.
        layer : tuple[float, float]
            The vertical layer (zmin, zmax) in meters to analyze.
        x : str, optional
            The variable name to plot on the x-axis (default is 'Dm').
        y : str, optional
            The variable name to plot on the y-axis (default is 'LwC').
        z : str, optional
            The variable name to use for color mapping (default is 'Nw').
        savefig : bool, optional
            Whether to save the generated figure to disk (default is False).
            Additional keyword arguments:
                - figsize: tuple, optional
                    Figure size (default is (12, 6)).
                - cmap: str, optional
                    Colormap for the scatter plot (default is 'viridis').
                - markersize: int or float, optional
                    Marker size for the scatter plot (default is 50).
                - output_dir: Path or str, optional
                    Directory to save the figure if savefig is True (default is current working directory).
        Returns
        -------
        tuple[Figure, Axes, Path | None]
            A tuple containing the matplotlib Figure object, Axes object, and output Path if saved, otherwise None.
        Raises
        ------
        KeyError
            If any of the specified variables (x, y, z) are not found in the dataset.
        """

        return process_plotting.plot_rain_process_in_layer_2d(
            self,
            target_datetime=target_datetime,
            layer=layer,
            x=x,
            y=y,
            z=z,
            use_relative_difference=use_relative_difference,
            savefig=savefig,
            **kwargs,
        )

    def plot_event_scatter(
        self,
        *,
        target_datetime: datetime | tuple[datetime, datetime],
        layer: tuple[float, float],
        x: str = "Dm",
        y: str = "Nw",
        color: str = "LWC",
        use_relative_difference: bool = True,
        savefig: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plot a single event scatter for one time window and one layer.

        This helper is intended for presentation-ready standalone figures rather
        than multi-panel summaries. For the more explicit public API, prefer
        :meth:`plot_region_scatter`.
        """
        return process_plotting.plot_event_scatter(
            self,
            target_datetime=target_datetime,
            layer=layer,
            x=x,
            y=y,
            color=color,
            use_relative_difference=use_relative_difference,
            savefig=savefig,
            **kwargs,
        )

    def plot_region_scatter(
        self,
        *,
        target_datetime: datetime | tuple[datetime, datetime],
        layer: tuple[float, float] | None = None,
        z_bottom_m: float | None = None,
        z_top_m: float | None = None,
        x: str = "Dm",
        y: str = "Nw",
        color: str = "LWC",
        processes: list[str] | None = None,
        classified: xr.Dataset | None = None,
        use_relative_difference: bool = True,
        savefig: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plot the scatter of one selected time-height region of the quicklook.

        Optionally filter the selected region to one or more classified rain
        processes.
        """
        return process_plotting.plot_region_scatter(
            self,
            target_datetime=target_datetime,
            layer=layer,
            z_bottom_m=z_bottom_m,
            z_top_m=z_top_m,
            x=x,
            y=y,
            color=color,
            processes=processes,
            classified=classified,
            use_relative_difference=use_relative_difference,
            savefig=savefig,
            **kwargs,
        )

    def plot_process_scatter(
        self,
        *,
        classified: xr.Dataset,
        process: str,
        target_datetime: datetime | tuple[datetime, datetime],
        layer: tuple[float, float],
        x: str = "Dm",
        y: str = "Nw",
        color: str = "LWC",
        use_relative_difference: bool = True,
        savefig: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plot a single event scatter filtered to one classified rain process.
        """
        return process_plotting.plot_process_scatter(
            self,
            classified=classified,
            process=process,
            target_datetime=target_datetime,
            layer=layer,
            x=x,
            y=y,
            color=color,
            use_relative_difference=use_relative_difference,
            savefig=savefig,
            **kwargs,
        )

    def plot_scan_process_scatter_compare(
        self,
        *,
        scan_df: pd.DataFrame,
        processes: list[str],
        x: str = "Dm_layer_mean",
        y: str = "Nw_layer_mean",
        color: str = "LWC_layer_mean",
        period: tuple[datetime, datetime] | None = None,
        z_bottom_m: float | None = None,
        z_top_m: float | None = None,
        show_centroids: bool = False,
        show_density: bool = False,
        savefig: bool = False,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Compare several classified scan processes in a shared microphysical scatter.

        Each point corresponds to one ``time x window`` sample from ``scan_df``.
        Marker shape encodes the process, while color encodes the selected
        numeric variable.
        """
        return process_plotting.plot_scan_process_scatter_compare(
            self,
            scan_df=scan_df,
            processes=processes,
            x=x,
            y=y,
            color=color,
            period=period,
            z_bottom_m=z_bottom_m,
            z_top_m=z_top_m,
            show_centroids=show_centroids,
            show_density=show_density,
            savefig=savefig,
            output_dir=output_dir,
            **kwargs,
        )

    def plot_event_vertical_percent_profiles(
        self,
        *,
        target_datetime: datetime | tuple[datetime, datetime],
        layer: tuple[float, float],
        variables: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
        use_relative_difference: bool = True,
        savefig: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plot vertical percent profiles for one event window and one layer.

        The figure shows one line per selected variable and can include
        interquartile-range shading when multiple times are available.
        """
        return process_plotting.plot_event_vertical_percent_profiles(
            self,
            target_datetime=target_datetime,
            layer=layer,
            variables=variables,
            use_relative_difference=use_relative_difference,
            savefig=savefig,
            **kwargs,
        )

    def plot_process_vertical_percent_profiles(
        self,
        *,
        classified: xr.Dataset,
        process: str,
        target_datetime: datetime | tuple[datetime, datetime],
        layer: tuple[float, float],
        variables: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
        use_relative_difference: bool = True,
        savefig: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plot vertical percent profiles for one classified process in one layer.
        """
        return process_plotting.plot_process_vertical_percent_profiles(
            self,
            classified=classified,
            process=process,
            target_datetime=target_datetime,
            layer=layer,
            variables=variables,
            use_relative_difference=use_relative_difference,
            savefig=savefig,
            **kwargs,
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
        return process_analysis.compute_layer_trend_ols(
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
        return process_analysis.compute_layer_trend(
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
        selection_mode: str = "scan",
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
        Analyse rain-process evolution with a scan-first public workflow.

        ``selection_mode="scan"`` is the default public interface and returns a
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
            Scan mode returns the column-scan dataframe. Fixed-layer mode
            returns the analysis dataset containing the trend diagnostics, RGB
            channels, elapsed minutes and the hexagram coordinates used
            downstream for plotting and classification.

        Notes
        -----
        In scan mode, window geometry and the tau-strength threshold default to
        :attr:`micro_cfg` unless overridden by explicit arguments.

        ``window_step_m=None`` means "raw resolution": use the native range grid
        spacing (median of the range-coordinate differences).
        """
        mode = str(selection_mode).strip().lower()
        if mode not in {"scan", "fixed_layer"}:
            raise ValueError("selection_mode must be either 'scan' or 'fixed_layer'.")

        has_fixed_layer_args = (
            layer is not None or z_bottom_m is not None or z_top_m is not None
        )
        if mode == "scan" and has_fixed_layer_args:
            warnings.warn(
                "Fixed-layer arguments were provided to rain_process_analyze(). "
                "Running in selection_mode='fixed_layer'. For the default "
                "public workflow, prefer scan mode with `window_thickness_m` "
                "and `window_step_m`.",
                FutureWarning,
                stacklevel=2,
            )
            mode = "fixed_layer"

        if mode == "scan":
            thickness_m = (
                float(window_thickness_m)
                if window_thickness_m is not None
                else float(self.micro_cfg.window_thickness_m)
            )
            step_m = (
                window_step_m
                if window_step_m is not _UNSET
                else self.micro_cfg.window_step_m
            )
            tau_strength = (
                min_tau_strength
                if min_tau_strength is not _UNSET
                else self.micro_cfg.min_tau_strength
            )
            return process_analysis.build_column_process_scan_dataframe(
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

        return process_analysis.rain_process_analyze(
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

    def plot_rain_process_in_layer_hexagram(
        self,
        *,
        analysis: xr.Dataset,
        use_snapped_colors: bool = True,
        savefig: bool = False,
        output_dir=None,
        **kwargs,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        SOLO plotting: dibuja el hexagrama base (RGB) y superpone la trayectoria temporal (puntos)
        usando el resultado precomputado `analysis` (salida de rain_process_analyze).

        Requiere en `analysis`:
        - hex_x, hex_y (coords en rejilla del hexagrama)
        - minutes (para colorear por tiempo)
        - R,G,B (0..1) y opcional snap_R,snap_G,snap_B
        - attrs: period_start, period_end, z_bottom_m, z_top_m (opcionales pero recomendados)

        Parameters
        ----------
        analysis : xr.Dataset
            Resultado de rain_process_analyze(...)
        k : int
            Resolución del hexagrama (debe coincidir con la usada en el análisis para que la LUT cuadre).
        use_snapped_colors : bool
            Si True y existen snap_R/G/B, colorea con el color “snapeado” a la celda.
            Si False, usa RGB continuo.
        """
        return process_plotting.plot_rain_process_in_layer_hexagram(
            self,
            analysis=analysis,
            use_snapped_colors=use_snapped_colors,
            savefig=savefig,
            output_dir=output_dir,
            **kwargs,
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

        tau_strength = (
            min_tau_strength
            if min_tau_strength is not _UNSET
            else self.micro_cfg.min_tau_strength
        )
        return process_analysis.classify_rain_process(
            self,
            analysis=analysis,
            tol_center=tol_center,
            min_strength=min_strength,
            min_tau_strength=tau_strength,
            max_p_value=max_p_value,
            max_tau_pvalue=max_tau_pvalue,
        )

    def build_process_features(
        self,
        *,
        ds: xr.Dataset | None = None,
        mode: str,
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
        Build Phase A `process_features` from a dataset.

        In scan mode, ``window_thickness_m`` and ``window_step_m`` default to
        :attr:`micro_cfg` when not explicitly provided. ``window_step_m=None``
        means "raw resolution" (native range-grid spacing).
        """
        ds_in = ds if ds is not None else (self.raprompro if self.raprompro is not None else self.ds)
        return process_feature_analysis.build_process_features(
            ds_in,
            mode=mode,
            range_coord=range_coord,
            window_thickness_m=window_thickness_m,
            window_step_m=window_step_m,
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

    def classify_process_from_features(
        self,
        *,
        process_features: xr.Dataset,
        refiners: list[Any] | None = None,
        min_strength: float = 0.10,
        min_tau_strength: float | None | _UnsetType = _UNSET,
        max_p_value: float | None = None,
        max_tau_pvalue: float | None = None,
    ) -> xr.Dataset:
        """
        Classify rain process labels directly from Phase A `process_features`.

        This is the recommended entry point for the new two-stage pipeline:
        Phase A builds `process_features`, Phase B classifies them.
        """
        tau_strength = (
            min_tau_strength
            if min_tau_strength is not _UNSET
            else self.micro_cfg.min_tau_strength
        )
        return process_analysis.classify_process_from_features(
            process_features,
            refiners=refiners,
            min_strength=min_strength,
            min_tau_strength=tau_strength,
            max_p_value=max_p_value,
            max_tau_pvalue=max_tau_pvalue,
        )

    def plot_processes_evolution(
        self,
        *,
        classified: xr.Dataset,
        analysis: xr.Dataset | None = None,
        savefig: bool = False,
        output_dir: Path | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plot a temporal summary of the classified rain-process evolution.

        Panels
        ------
        (a) Process timeline vs time (color = strength)
        (b) Signs heatmap 3×time (-1 / 0 / +1)

        Notes
        -----
        - Process codes A, B, C... are shown on the y-axis of panel (a).
        - Their meaning is shown in a figure legend below the panels.
        - If a process exists in mrrpropy.hexagram.PROCESS_SIGNATURES, its signature is
        appended in the legend.
        - Colorbars live in fixed GridSpec columns, so subplot widths remain aligned.
        - The function does not classify anything; it only visualizes
          ``classified``.
        """

        return process_plotting.plot_processes_evolution(
            self,
            classified=classified,
            analysis=analysis,
            savefig=savefig,
            output_dir=output_dir,
            **kwargs,
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
        return process_analysis.build_process_dynamics_dataframe(
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
        return process_analysis.summarize_process_dynamics(
            self,
            analysis=analysis,
            classified=classified,
            variables=variables,
        )

    def build_column_process_scan_dataframe(
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
        Scan the whole column with a sliding vertical window.

        By default, ``window_thickness_m``, ``window_step_m`` and
        ``min_tau_strength`` are taken from :attr:`micro_cfg` unless overridden
        by explicit arguments. ``window_step_m=None`` means "raw resolution"
        (native range-grid spacing).

        The output dataframe contains one row per ``time × window`` and is the
        recommended input for :meth:`detect_column_process_episodes`.
        """
        thickness_m = (
            float(window_thickness_m)
            if window_thickness_m is not None
            else float(self.micro_cfg.window_thickness_m)
        )
        step_m = (
            window_step_m
            if window_step_m is not _UNSET
            else self.micro_cfg.window_step_m
        )
        tau_strength = (
            min_tau_strength
            if min_tau_strength is not _UNSET
            else self.micro_cfg.min_tau_strength
        )
        return process_analysis.build_column_process_scan_dataframe(
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

    def detect_column_process_episodes(
        self,
        *,
        scan_df: pd.DataFrame,
        min_consecutive_profiles: int = 6,
    ) -> pd.DataFrame:
        """
        Detect persistent process episodes from a column scan dataframe.

        Episodes are defined independently in each sliding window and require a
        minimum number of consecutive profiles with the same process label.
        """
        return process_analysis.detect_column_process_episodes(
            self,
            scan_df=scan_df,
            min_consecutive_profiles=min_consecutive_profiles,
        )

    def plot_classified_processes_on_hexagram(
        self,
        *,
        classified: xr.Dataset,
        analysis: xr.Dataset | None = None,
        processes: str | None = None,
        show_background: bool = False,
        show_process_masks: bool = True,
        savefig: bool = False,
        output_dir: Path | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plot classified samples on the RGB hexagram used by the package.

        Parameters
        ----------
        classified : xr.Dataset
            Output of classify_rain_process(...). Must contain:
            - proc_label
            - hex_x, hex_y
        analysis : xr.Dataset | None
            Optional analysis dataset. Used to retrieve k if not present in classified.attrs.
        show_background : bool
            If True, show full RGB hexagram background.
        show_process_masks : bool
            If True, overlay the theoretical process masks derived from
            ``PROCESS_SIGNATURES``.
        """

        return process_plotting.plot_classified_processes_on_hexagram(
            self,
            classified=classified,
            analysis=analysis,
            processes=processes,
            show_background=show_background,
            show_process_masks=show_process_masks,
            savefig=savefig,
            output_dir=output_dir,
            **kwargs,
        )

    def plot_column_process_scan(
        self,
        *,
        scan_df: pd.DataFrame,
        processes: list[str] | None = None,
        savefig: bool = False,
        output_dir: Path | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes, Path | None]:
        """
        Plot a time-height curtain of process labels from a whole-column scan.

        The input is the dataframe returned by
        :meth:`build_column_process_scan_dataframe`.

        Common marker options:
        - ``marker_mode="process"`` uses one marker per process label from
          ``mrrpropy.processes.PROCESS_MARKERS``.
        - ``marker_mode="square"`` uses square markers for every process label.
        - ``marker_mode="single"`` uses the value passed in ``marker`` for every
          process label.
        """
        return process_plotting.plot_column_process_scan(
            self,
            scan_df=scan_df,
            processes=processes,
            savefig=savefig,
            output_dir=output_dir,
            **kwargs,
        )

    def plot_za_range_histogram(
            ds_za, 
            za_bins=None, 
            range_bins=None,
            cmap: str | None = 'jet', 
            fig_title: str | None = "Za vs Range — 2D histogram", 
            output_dir: Path | None = None
    ):
            
        return process_plotting.plot_za_range_histogram(
            ds_za,
            za_bins,
            range_bins,
            cmap,
            fig_title,
            output_dir,
        )
    
    