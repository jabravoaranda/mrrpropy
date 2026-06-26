from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import xarray as xr

from mrrpropy.plotting import _spectra


class SupportsRawPlotting(_spectra.SupportsSpectralAccess, Protocol):
    plot_cfg: Any


def quicklook(
    subject: SupportsRawPlotting,
    variable: str = "Ze",
    source: str = "raprompro",
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """Plot a quick time-height view of a raw or processed 2D field."""
    pcfg = subject.plot_cfg
    cmap = kwargs.get("cmap", pcfg.cmap)
    figsize = kwargs.get("figsize", pcfg.figsize_quicklook)

    if source == "raw":
        if variable not in subject.ds:
            raise KeyError(f"Variable '{variable}' not found in raw Dataset.")
        da = subject.ds[variable]
    else:
        if subject.raprompro is None or variable not in subject.raprompro:
            raise KeyError(f"Variable '{variable}' not found in raprompro Dataset.")
        da = subject.raprompro[variable]

    fig, ax = plt.subplots(figsize=figsize)
    data_array_plot = cast(Any, da.plot)
    data_array_plot(
        ax=ax,
        x="time",
        y="range",
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cmap=cmap,
    )
    ax.set_title(f"{variable} (MRR-PRO)")
    ax.set_ylabel("Range (m)")
    ax.set_xlabel("Time")
    plt.tight_layout()
    return fig, ax, None


def plot_spectrum(
    subject: SupportsRawPlotting,
    target_datetime: datetime | np.datetime64,
    target_range: float,
    *,
    spectrum_var: str = "spectrum_reflectivity",
    velocity_limits: tuple[float, float] | None = None,
    label_type: str = "both",
    fig: Figure | None = None,
    ax: Axes | None = None,
    savefig: bool = False,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """Plot a single-gate Doppler spectrum at a selected time and range."""
    pcfg = subject.plot_cfg
    dpi = kwargs.get("dpi", pcfg.dpi)
    figsize = kwargs.get("figsize", pcfg.figsize)

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif fig is not None and ax is None:
        ax = fig.get_axes()[0]

    if fig is None or ax is None:
        raise ValueError("A matplotlib figure and axes could not be prepared.")

    t_sel, r_sel, vel, spec, units = _spectra.get_spectrum_1d(
        subject,
        target_datetime,
        target_range,
        spectrum_var=spectrum_var,
    )

    t_txt = np.datetime_as_string(t_sel, unit="s")
    if label_type == "both":
        label = f"{t_txt} | {r_sel:.1f} m"
    elif label_type == "range":
        label = f"{r_sel:.1f} m"
    else:
        label = f"{t_txt}"

    if not np.isnan(spec).all():
        ax.plot(vel, spec, color=kwargs.get("color", "black"), label=label)

    if velocity_limits is not None:
        ax.set_xlim(*velocity_limits)
    else:
        ax.set_xlim(float(np.nanmin(vel)), float(np.nanmax(vel)))

    ax.set_xlabel("Doppler velocity [m/s, negative downward]")
    ax.set_ylabel(f"Spectrum [{units}]" if units else "Spectrum")
    ax.set_title("MRR-PRO spectrum")
    ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1.0)
    ax.legend(loc="upper right")

    fig.tight_layout()

    filepath: Path | None = None
    if savefig:
        if output_dir is None:
            raise ValueError("output_dir must be provided if savefig=True.")
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / Path(subject.path).name.replace(
            ".nc", f"_spectrum_{t_txt.replace(':', '')}_{r_sel:.1f}m.png"
        )
        fig.savefig(filepath, dpi=dpi)

    return fig, ax, filepath


def plot_spectra_by_range(
    subject: SupportsRawPlotting,
    target_datetime: datetime | np.datetime64 | str,
    ranges: list[float] | np.ndarray,
    *,
    use_db: bool = True,
    label_type: str = "range",
    ncol: int = 2,
    fig: Figure | None = None,
    ax: Axes | None = None,
    savefig: bool = False,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """Plot several MRR-PRO Doppler spectra at a fixed time for multiple ranges."""
    pcfg = subject.plot_cfg
    dpi = kwargs.get("dpi", pcfg.dpi)
    figsize = kwargs.get("figsize", pcfg.figsize)

    ds = subject.ds
    if "time" not in ds or "range" not in ds:
        raise KeyError("Dataset must contain 'time' and 'range' coordinates.")
    if "spectrum_n_samples" not in ds.dims:
        raise KeyError("Dataset must contain dimension 'spectrum_n_samples'.")

    spec_var: str | None = None
    for candidate in ("spectrum_reflectivity", "spectrum", "spectra", "spectrum_raw"):
        if candidate in ds:
            spec_var = candidate
            break
    if spec_var is None:
        raise KeyError(
            "No spectral variable found. Expected one of: "
            "'spectrum_reflectivity', 'spectrum', 'spectra', 'spectrum_raw'."
        )

    t_sel = cast(
        np.datetime64,
        ds["time"].sel(time=target_datetime, method="nearest").values,
    )

    vel_internal: np.ndarray | None = None
    for vname in ("velocity", "doppler_velocity", "velocity_vectors", "vel"):
        if vname in ds:
            velocity_da = ds[vname]
            if "spectrum_n_samples" in velocity_da.dims and len(velocity_da.dims) == 1:
                vel_internal = velocity_da.values.astype(float)
            break
    if vel_internal is None:
        vel_internal = _spectra.get_velocity_axis(
            subject, int(ds.sizes["spectrum_n_samples"])
        )
    vel, _ = _spectra._as_negative_downward_axis(
        vel_internal, np.empty_like(vel_internal)
    )

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif fig is not None and ax is None:
        axes = fig.get_axes()
        ax = axes[0] if axes else fig.add_subplot(111)
    elif fig is None and ax is not None:
        fig = cast(Figure, ax.figure)

    if fig is None or ax is None:
        raise ValueError("A matplotlib figure and axes could not be prepared.")

    def _label(selected_time: np.datetime64, selected_range: float, mode: str) -> str:
        time_text = np.datetime_as_string(selected_time, unit="s")
        if mode == "both":
            return f"{time_text} | {selected_range:.1f} m"
        if mode == "time":
            return f"{time_text}"
        return f"{selected_range:.1f} m"

    range_values = np.asarray(ranges, dtype=float)
    if range_values.size == 0:
        raise ValueError("ranges must contain at least one range value.")

    has_index = "index_spectra" in ds and "n_spectra" in ds.dims
    for requested_range in range_values:
        selected_range = float(
            ds["range"].sel(range=requested_range, method="nearest").values.item()
        )

        if has_index:
            idx_raw = (
                ds["index_spectra"]
                .sel(time=t_sel, range=selected_range, method="nearest")
                .values
            )
            if not np.isfinite(idx_raw):
                continue
            spectrum_idx = int(idx_raw)
            if not (0 <= spectrum_idx < ds.sizes["n_spectra"]):
                continue
            spectrum = (
                ds[spec_var].sel(time=t_sel).values.astype(float)[spectrum_idx, :]
            )
        else:
            spectrum_da = ds[spec_var].sel(
                time=t_sel, range=selected_range, method="nearest"
            )
            if "spectrum_n_samples" not in spectrum_da.dims:
                raise ValueError(
                    f"{spec_var} does not have 'spectrum_n_samples' dimension."
                )
            spectrum = spectrum_da.values.astype(float)

        _, plotted_spectrum = _spectra._as_negative_downward_axis(
            vel_internal, spectrum
        )
        if use_db:
            with np.errstate(divide="ignore", invalid="ignore"):
                plotted_spectrum = 10.0 * np.log10(
                    np.where(plotted_spectrum > 0, plotted_spectrum, np.nan)
                )

        if np.all(~np.isfinite(plotted_spectrum)):
            continue

        ax.plot(
            vel,
            plotted_spectrum,
            label=_label(t_sel, selected_range, label_type),
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"title", "dpi", "figsize"}
            },
        )

    ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Doppler velocity [m/s, negative downward]")
    ax.set_ylabel("Spectrum [dB]" if use_db else "Spectrum [linear]")

    title = kwargs.get("title")
    if title is None:
        time_text = str(np.datetime_as_string(t_sel, unit="s"))
        ax.set_title(f"MRR-PRO spectra by range | time={time_text}")
    else:
        ax.set_title(title)

    ax.legend(ncol=ncol, loc="best", fontsize=9)
    fig.tight_layout()

    filepath: Path | None = None
    if savefig:
        if output_dir is None:
            raise ValueError("output_dir must be provided if savefig=True.")
        output_dir.mkdir(parents=True, exist_ok=True)
        time_text = str(np.datetime_as_string(t_sel, unit="s")).replace(":", "")
        filepath = output_dir / Path(subject.path).name.replace(
            ".nc", f"_spectra_by_range_{time_text}.png"
        )
        fig.savefig(filepath, dpi=dpi)

    return fig, ax, filepath


def plot_spectrogram(
    subject: SupportsRawPlotting,
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
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """Plot a range-by-velocity spectrogram at the nearest requested time."""
    del variable_threshold, threshold_value

    pcfg = subject.plot_cfg
    dpi = kwargs.get("dpi", pcfg.dpi)
    cmap = kwargs.get("cmap", pcfg.cmap)
    figsize = kwargs.get("figsize", pcfg.figsize)

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = fig.get_axes()[0]

    if fig is None:
        raise ValueError("A matplotlib figure could not be prepared.")

    t_sel, ranges, vel, spec2d, units = _spectra.get_spectrogram_2d(
        subject,
        target_datetime,
        spectrum_var=spectrum_var,
        range_limits=range_limits,
    )
    t_txt = np.datetime_as_string(t_sel, unit="s")

    extent = (
        float(vel[0]),
        float(vel[-1]),
        float(ranges[0]),
        float(ranges[-1]),
    )
    im = ax.imshow(
        spec2d,
        aspect="auto",
        extent=extent,
        cmap=cmap,
        origin="lower",
    )

    if vmin is not None or vmax is not None:
        im.set_clim(vmin=vmin, vmax=vmax)

    ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Doppler velocity [m/s, negative downward]")
    ax.set_ylabel("Range [m]")
    ax.set_title(f"MRR-PRO spectrogram \n {t_txt}")
    ax.set_xlim(kwargs.get("x_limits", (-12, 4)))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Spectrum [{units}]" if units else "Spectrum")

    fig.tight_layout()

    filepath: Path | None = None
    if savefig:
        if output_dir is None:
            raise ValueError("output_dir must be provided if savefig=True.")
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / Path(subject.path).name.replace(
            ".nc", f"_{spectrum_var}_spectrogram_{t_txt.replace(':', '')}.png"
        )
        fig.savefig(filepath, dpi=dpi)

    return fig, ax, filepath
