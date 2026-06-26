from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import xarray as xr

from mrrpropy.processes import PROCESS_CODES, PROCESS_MARKERS, PROCESS_SIGNATURES
from mrrpropy.hexagram import (
    get_hexagram_assets,
    get_process_hexagram_mask,
)


class SupportsProcessPlotting(Protocol):
    path: str | Path
    raprompro: xr.Dataset | None
    plot_cfg: Any

    def _is_processed(self) -> bool: ...


def _resolve_processed_dataset(subject: SupportsProcessPlotting) -> xr.Dataset:
    if not subject._is_processed():
        raise RuntimeError("Dataset is not processed.")

    ds = subject.raprompro
    if ds is None:
        raise RuntimeError("raprompro not loaded. Use load_raprompro().")
    return ds


def _resolve_height_limits_km(
    subject: SupportsProcessPlotting,
    explicit_limits: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    if explicit_limits is not None:
        return explicit_limits

    ds = subject.raprompro
    if ds is None or "range" not in ds.coords:
        return None

    heights_km = np.asarray(ds["range"].values, dtype=float) / 1000.0
    finite = heights_km[np.isfinite(heights_km)]
    if finite.size == 0:
        return None
    return float(np.min(finite)), float(np.max(finite))


def _layer_bounds_from_attrs(attrs: dict[str, Any]) -> tuple[float | None, float | None]:
    z_bottom_m = attrs.get("z_bottom_m", attrs.get("z_top", None))
    z_top_m = attrs.get("z_top_m", attrs.get("z_base", None))
    if z_bottom_m is None or z_top_m is None:
        return None, None
    return float(z_bottom_m), float(z_top_m)


def _select_layer_event_data(
    subject: SupportsProcessPlotting,
    *,
    target_datetime: datetime | tuple[datetime, datetime],
    layer: tuple[float, float],
    variables: tuple[str, str, str],
    use_relative_difference: bool,
) -> xr.Dataset:
    ds = _resolve_processed_dataset(subject)

    for var in variables:
        if var not in ds:
            raise KeyError(f"Variable '{var}' not found in dataset.")

    if isinstance(target_datetime, tuple) and target_datetime[0] >= target_datetime[1]:
        raise ValueError(
            "target_datetime tuple must be in increasing order (start, end)."
        )

    if isinstance(target_datetime, datetime):
        data = ds.sel(time=target_datetime, method="nearest").sel(range=slice(*layer))
    else:
        data = ds.sel(time=slice(*target_datetime)).sel(range=slice(*layer))

    if data.sizes.get("time", 0) == 0:
        raise ValueError("Temporal selection is empty.")
    if data.sizes.get("range", 0) == 0:
        raise ValueError("Layer selection is empty.")

    top_range = data["range"].max()
    if use_relative_difference:
        baseline = data.sel(range=top_range)
        layer_mean = np.abs(data.mean("range"))
        scale = xr.where(np.abs(baseline) > 0.0, np.abs(baseline), layer_mean)
        data = 100.0 * (data - baseline) / scale
    else:
        data = data - data.sel(range=top_range)

    data.attrs["profile_reference"] = "top_of_layer"
    data.attrs["profile_direction"] = (
        "positive means increase while descending from the top of the layer"
    )

    return data


def _filter_data_by_process(
    data: xr.Dataset,
    *,
    classified: xr.Dataset,
    process: str,
) -> xr.Dataset:
    if not isinstance(classified, xr.Dataset):
        raise TypeError("classified must be an xr.Dataset.")
    if "time" not in classified.coords or "proc_label" not in classified:
        raise KeyError("classified must contain 'time' and 'proc_label'.")

    data_aligned, classified_aligned = xr.align(
        data,
        classified[["proc_label"]],
        join="inner",
    )

    labels = classified_aligned["proc_label"].values.astype(str)
    mask = labels == process
    if not np.any(mask):
        present = sorted({label for label in labels.tolist() if label})
        raise ValueError(
            f"No samples found for process '{process}'. Available labels: {present}"
        )

    return data_aligned.isel(time=np.flatnonzero(mask))


def _filter_data_by_processes(
    data: xr.Dataset,
    *,
    classified: xr.Dataset,
    processes: list[str],
) -> xr.Dataset:
    if not isinstance(classified, xr.Dataset):
        raise TypeError("classified must be an xr.Dataset.")
    if "time" not in classified.coords or "proc_label" not in classified:
        raise KeyError("classified must contain 'time' and 'proc_label'.")

    selected_processes = {str(process) for process in processes if process is not None}
    if not selected_processes:
        return data

    data_aligned, classified_aligned = xr.align(
        data,
        classified[["proc_label"]],
        join="inner",
    )

    labels = classified_aligned["proc_label"].values.astype(str)
    mask = np.isin(labels, list(selected_processes))
    if not np.any(mask):
        present = sorted({label for label in labels.tolist() if label})
        raise ValueError(
            "No samples found for the requested processes. "
            f"Requested: {sorted(selected_processes)}. Available labels: {present}"
        )

    return data_aligned.isel(time=np.flatnonzero(mask))


def _max_abs_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return float(np.max(np.abs(finite)))


def _robust_abs_limit(
    values: np.ndarray,
    *,
    quantile: float = 0.98,
    floor: float = 1.0,
) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return floor
    limit = float(np.nanquantile(np.abs(finite), quantile))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = _max_abs_finite(finite)
    return max(limit, floor)


def _time_window_text(
    target_datetime: datetime | tuple[datetime, datetime],
) -> tuple[str, str]:
    if isinstance(target_datetime, tuple):
        title_text = (
            f"{target_datetime[0].strftime('%Y-%m-%d %H:%M')} - "
            f"{target_datetime[1].strftime('%H:%M')}"
        )
        file_text = (
            f"{target_datetime[0].strftime('%Y%m%d_%H%M%S')}_to_"
            f"{target_datetime[1].strftime('%Y%m%d_%H%M%S')}"
        )
    else:
        title_text = str(target_datetime)
        file_text = target_datetime.strftime("%Y%m%d_%H%M%S")
    return title_text, file_text


def _plot_layer_scatter(
    subject: SupportsProcessPlotting,
    *,
    target_datetime: datetime | tuple[datetime, datetime],
    layer: tuple[float, float],
    x: str,
    y: str,
    color: str,
    use_relative_difference: bool,
    classified: xr.Dataset | None,
    process: str | None,
    processes: list[str] | None,
    savefig: bool,
    filename_prefix: str,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    pcfg = subject.plot_cfg
    figsize = kwargs.get("figsize", pcfg.figsize)
    markersize = kwargs.get("marker_size", kwargs.get("markersize", 50))
    alpha = kwargs.get("alpha", 0.9)
    edgecolors = kwargs.get("edgecolors", "black")
    linewidths = kwargs.get("linewidths", 0.35)
    title_fs = kwargs.get("title_fs", 15)
    info_fs = kwargs.get("info_fs", 11)
    label_fs = kwargs.get("label_fs", 16)
    tick_fs = kwargs.get("tick_fs", 12)
    robust_quantile = float(kwargs.get("robust_quantile", 0.98))

    full_data = _select_layer_event_data(
        subject,
        target_datetime=target_datetime,
        layer=layer,
        variables=(x, y, color),
        use_relative_difference=use_relative_difference,
    )

    data = full_data

    if process is not None:
        if classified is None:
            raise TypeError("classified must be provided when filtering by process.")
        data = _filter_data_by_process(data, classified=classified, process=process)
        data = data.sortby("time")
    elif processes is not None:
        if classified is None:
            raise TypeError("classified must be provided when filtering by processes.")
        data = _filter_data_by_processes(data, classified=classified, processes=processes)
        data = data.sortby("time")

    limits_source = full_data if process is not None else data
    x_abs_max = _robust_abs_limit(limits_source[x].values, quantile=robust_quantile)
    y_abs_max = _robust_abs_limit(limits_source[y].values, quantile=robust_quantile)
    c_abs_max = _robust_abs_limit(
        limits_source[color].values,
        quantile=robust_quantile,
    )

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    scatter_plot = cast(Any, data.plot.scatter)
    scatter_plot(
        x=x,
        y=y,
        hue=color,
        cmap=kwargs.get("cmap", "viridis"),
        s=markersize,
        vmin=-c_abs_max,
        vmax=c_abs_max,
        alpha=alpha,
        edgecolors=edgecolors,
        linewidths=linewidths,
        ax=ax,
    )
    if process is not None:
        x_values = np.asarray(data[x].values, dtype=float)
        y_values = np.asarray(data[y].values, dtype=float)
        range_values = np.asarray(data["range"].values, dtype=float)
        if x_values.ndim > 1:
            x_values = np.nanmedian(x_values, axis=0)
        if y_values.ndim > 1:
            y_values = np.nanmedian(y_values, axis=0)
        range_order = np.argsort(range_values)[::-1]
        x_values = x_values[range_order]
        y_values = y_values[range_order]
        finite_xy = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[finite_xy]
        y_values = y_values[finite_xy]
        if x_values.size:
            ax.plot(x_values, y_values, color="black", linewidth=0.8, alpha=0.6)
            ax.scatter(x_values[0], y_values[0], c="black", marker="^", s=markersize, label="start")
            ax.scatter(x_values[-1], y_values[-1], c="black", marker="s", s=markersize, label="end")
            ax.legend(loc="best", fontsize=kwargs.get("legend_fs", 11))
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    figure = ax.get_figure()
    if figure is None:
        raise RuntimeError("Scatter plot figure could not be resolved.")

    axes = figure.get_axes()
    if len(axes) > 1:
        axes[1].set_ylabel(color, fontsize=label_fs)
        axes[1].tick_params(labelsize=tick_fs)

    zmin, zmax = layer
    title_time, file_time = _time_window_text(target_datetime)
    ax.set_title(
        f"Layer {zmin/1000:.1f}-{zmax/1000:.1f} km | {title_time}",
        fontsize=title_fs,
        pad=12,
    )

    n_points = int(data.sizes.get("time", 0))
    if process is None and processes is None:
        info_text = f"event | n={n_points}"
    elif processes is not None:
        info_text = f"region | n={n_points} | processes={len(processes)}"
    else:
        info_text = f"{process} | n={n_points}"
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=info_fs,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
    )

    ax.set_xlim(-x_abs_max, x_abs_max)
    ax.set_ylim(-y_abs_max, y_abs_max)
    ax.set_xlabel(x, fontsize=label_fs)
    ax.set_ylabel(y, fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    output_path: Path | None = None
    if savefig:
        output_dir = Path(kwargs.get("output_dir", Path.cwd()))
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_process = (
            ""
            if process is None
            else f"_{process.replace(' ', '_').replace('/', '_').lower()}"
        )
        output_path = output_dir / (
            f"{filename_prefix}{safe_process}_{x}_vs_{y}_color_{color}_"
            f"{file_time}_{zmin}-{zmax}m.png"
        )
        fig.savefig(output_path)

    return fig, ax, output_path


def _plot_vertical_percent_profiles(
    subject: SupportsProcessPlotting,
    *,
    target_datetime: datetime | tuple[datetime, datetime],
    layer: tuple[float, float],
    variables: tuple[str, str, str],
    use_relative_difference: bool,
    classified: xr.Dataset | None,
    process: str | None,
    savefig: bool,
    filename_prefix: str,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    colors = kwargs.get(
        "profile_colors",
        {
            variables[0]: "#1f77b4",
            variables[1]: "#2ca02c",
            variables[2]: "#d62728",
        },
    )
    title_fs = kwargs.get("title_fs", 15)
    info_fs = kwargs.get("info_fs", 11)
    label_fs = kwargs.get("label_fs", 16)
    tick_fs = kwargs.get("tick_fs", 12)
    linewidth = float(kwargs.get("linewidth", 2.2))
    alpha_band = float(kwargs.get("alpha_band", 0.18))
    figsize = kwargs.get("figsize", (7.0, 6.5))

    full_data = _select_layer_event_data(
        subject,
        target_datetime=target_datetime,
        layer=layer,
        variables=variables,
        use_relative_difference=use_relative_difference,
    )
    data = full_data
    if process is not None:
        if classified is None:
            raise TypeError("classified must be provided when filtering by process.")
        data = _filter_data_by_process(data, classified=classified, process=process)

    y = data["range"].values.astype(float) / 1000.0
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    x_limits: list[float] = []
    for variable in variables:
        values = data[variable].values.astype(float)
        median = np.nanmedian(values, axis=0)
        q25 = np.nanquantile(values, 0.25, axis=0)
        q75 = np.nanquantile(values, 0.75, axis=0)

        color = colors.get(variable, None)
        ax.plot(
            median,
            y,
            linewidth=linewidth,
            color=color,
            label=variable,
        )
        if values.shape[0] > 1:
            ax.fill_betweenx(
                y,
                q25,
                q75,
                color=color,
                alpha=alpha_band,
                linewidth=0,
            )

        finite = np.concatenate([median[np.isfinite(median)], q25[np.isfinite(q25)], q75[np.isfinite(q75)]])
        if finite.size:
            x_limits.append(float(np.nanmax(np.abs(finite))))

    zmin, zmax = layer
    title_time, file_time = _time_window_text(target_datetime)
    ax.set_title(
        f"Layer {zmin/1000:.1f}-{zmax/1000:.1f} km | {title_time}",
        fontsize=title_fs,
        pad=12,
    )
    n_points = int(data.sizes.get("time", 0))
    info_text = f"event | n={n_points}" if process is None else f"{process} | n={n_points}"
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=info_fs,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
    )

    limit = max(x_limits) if x_limits else 1.0
    limit = max(limit * 1.1, 1.0)
    ax.set_xlim(-limit, limit)
    y_limits = _resolve_height_limits_km(subject, kwargs.get("y_limits"))
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        ax.set_ylim(y.min(), y.max())
    ax.set_xlabel("Relative change from top of layer (%)", fontsize=label_fs)
    ax.set_ylabel("Height (km)", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.legend(loc=kwargs.get("legend_loc", "best"), fontsize=kwargs.get("legend_fs", 11))

    output_path: Path | None = None
    if savefig:
        output_dir = Path(kwargs.get("output_dir", Path.cwd()))
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_process = (
            ""
            if process is None
            else f"_{process.replace(' ', '_').replace('/', '_').lower()}"
        )
        joined_vars = "_".join(variables)
        output_path = output_dir / (
            f"{filename_prefix}{safe_process}_{joined_vars}_{file_time}_{zmin}-{zmax}m.png"
        )
        fig.savefig(output_path)

    return fig, ax, output_path


def plot_rain_process_in_layer_2d(
    subject: SupportsProcessPlotting,
    target_datetime: datetime | tuple[datetime, datetime],
    layer: tuple[float, float],
    x: str = "Dm",
    y: str = "LwC",
    z: str = "Nw",
    use_relative_difference: bool = True,
    savefig: bool = False,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """Plot the rain-process evolution in a selected layer as a 2D scatter."""
    return _plot_layer_scatter(
        subject,
        target_datetime=target_datetime,
        layer=layer,
        x=x,
        y=y,
        color=z,
        use_relative_difference=use_relative_difference,
        classified=None,
        process=None,
        processes=None,
        savefig=savefig,
        filename_prefix="rain_process_2D",
        **kwargs,
    )


def plot_event_scatter(
    subject: SupportsProcessPlotting,
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
    """Plot a single scatter figure for one event window and one layer."""
    return _plot_layer_scatter(
        subject,
        target_datetime=target_datetime,
        layer=layer,
        x=x,
        y=y,
        color=color,
        use_relative_difference=use_relative_difference,
        classified=None,
        process=None,
        processes=None,
        savefig=savefig,
        filename_prefix="event_scatter",
        **kwargs,
    )


def plot_region_scatter(
    subject: SupportsProcessPlotting,
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
    """Plot a scatter for one selected time-height region of the quicklook."""
    resolved_layer = layer
    if resolved_layer is None:
        if z_bottom_m is None or z_top_m is None:
            raise ValueError("plot_region_scatter requires either layer or z_bottom_m/z_top_m.")
        resolved_layer = (float(z_bottom_m), float(z_top_m))

    return _plot_layer_scatter(
        subject,
        target_datetime=target_datetime,
        layer=resolved_layer,
        x=x,
        y=y,
        color=color,
        use_relative_difference=use_relative_difference,
        classified=classified,
        process=None,
        processes=processes,
        savefig=savefig,
        filename_prefix="region_scatter",
        **kwargs,
    )


def plot_process_scatter(
    subject: SupportsProcessPlotting,
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
    """Plot a single scatter figure filtered to one classified rain process."""
    return _plot_layer_scatter(
        subject,
        target_datetime=target_datetime,
        layer=layer,
        x=x,
        y=y,
        color=color,
        use_relative_difference=use_relative_difference,
        classified=classified,
        process=process,
        processes=None,
        savefig=savefig,
        filename_prefix="process_scatter",
        **kwargs,
    )


def plot_event_vertical_percent_profiles(
    subject: SupportsProcessPlotting,
    *,
    target_datetime: datetime | tuple[datetime, datetime],
    layer: tuple[float, float],
    variables: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
    use_relative_difference: bool = True,
    savefig: bool = False,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """Plot one vertical percent-profile figure for an event window."""
    return _plot_vertical_percent_profiles(
        subject,
        target_datetime=target_datetime,
        layer=layer,
        variables=variables,
        use_relative_difference=use_relative_difference,
        classified=None,
        process=None,
        savefig=savefig,
        filename_prefix="event_vertical_percent_profiles",
        **kwargs,
    )


def plot_process_vertical_percent_profiles(
    subject: SupportsProcessPlotting,
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
    """Plot one vertical percent-profile figure filtered to one process."""
    return _plot_vertical_percent_profiles(
        subject,
        target_datetime=target_datetime,
        layer=layer,
        variables=variables,
        use_relative_difference=use_relative_difference,
        classified=classified,
        process=process,
        savefig=savefig,
        filename_prefix="process_vertical_percent_profiles",
        **kwargs,
    )

def plot_za_range_histogram(
            ds_za, 
            za_bins=None, 
            range_bins=None,
            cmap: str | None = 'jet', 
            fig_title: str | None = "Za vs Range — 2D histogram", 
            output_dir: Path | None = None):
        """
        2D histogram of Za (reflectivity) vs range, colored by log10(count).
        
        Parameters
        ----------
        ds_za : xr.DataArray
            DataArray with dims (time, range), coords 'range' in meters.
        za_bins : array-like, optional
            Bin edges for Za (dBZ). Defaults to -5 to 40 in 1 dBZ steps.
        range_bins : array-like, optional
            Bin edges for range (m). Defaults to full extent in 50 m steps.
        """
        # flatten time × range
        za_vals = ds_za.values.ravel()
        range_coord = ds_za.coords['range'].values
        range_vals = np.tile(range_coord, ds_za.sizes['time'])

        # drop NaNs together
        mask = np.isfinite(za_vals) & np.isfinite(range_vals)
        za_vals = za_vals[mask]
        range_vals = range_vals[mask]

        if za_bins is None:
            za_bins = np.arange(-5, 41, 1)          # dBZ
        if range_bins is None:
            r0, r1 = range_coord.min(), range_coord.max()
            range_bins = np.arange(r0, r1 + 50, 50)  # 50 m steps

        H, xedges, yedges = np.histogram2d(
            za_vals, range_vals,
            bins=[za_bins, range_bins]
        )

        # log10 of count; mask zeros
        H_log = np.where(H > 0, np.log10(H), np.nan)

        fig, ax = plt.subplots(figsize=(7, 8))

        pcm = ax.pcolormesh(
            xedges, yedges, H_log.T,   # transpose: rows=range, cols=Za
            cmap=cmap,
            vmin=-2, vmax=3,
            shading='flat'
        )

        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label('log₁₀(m⁻³ mm⁻¹)', fontsize=11)
        cbar.set_ticks([-2, -1, 0, 1, 2, 3])

        ax.set_xlabel('Za reflectivity (dBZ)', fontsize=12)
        ax.set_ylabel('Range / Height (m)', fontsize=12)
        ax.set_xlim(za_bins[0], za_bins[-1])
        ax.set_ylim(range_bins[0], range_bins[-1])

        if fig_title:
            ax.set_title(fig_title, fontsize=13)

        plt.tight_layout()
        return fig, ax


def plot_rain_process_in_layer_hexagram(
    subject: SupportsProcessPlotting,
    *,
    analysis: xr.Dataset,
    use_snapped_colors: bool = True,
    savefig: bool = False,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """Overlay an analysed rain-process trajectory on the RGB hexagram."""
    pcfg = subject.plot_cfg
    figsize = kwargs.get("figsize", pcfg.figsize_multipanel)
    markersize = kwargs.get("markersize", pcfg.markersize)
    dpi = kwargs.get("dpi", pcfg.dpi)
    alpha = kwargs.get("alpha", pcfg.alpha_points)

    if analysis is None or not isinstance(analysis, xr.Dataset):
        raise TypeError("analysis debe ser un xr.Dataset (salida de rain_process_analyze).")

    required = ("hex_x", "hex_y", "minutes", "R", "G", "B")
    missing = [v for v in required if v not in analysis]
    if missing:
        raise KeyError(f"analysis no contiene variables requeridas: {missing}")

    k = analysis.attrs["k"]
    hex_assets = get_hexagram_assets(k=k)
    img = hex_assets["img"]
    ny, nx = img.shape[:2]

    hx = analysis["hex_x"].values.astype(float)
    hy = analysis["hex_y"].values.astype(float)
    minutes = analysis["minutes"].values.astype(float)

    if use_snapped_colors and all(v in analysis for v in ("snap_R", "snap_G", "snap_B")):
        colors = np.stack(
            [
                analysis["snap_R"].values,
                analysis["snap_G"].values,
                analysis["snap_B"].values,
            ],
            axis=1,
        ).astype(float)
    else:
        colors = np.stack(
            [analysis["R"].values, analysis["G"].values, analysis["B"].values],
            axis=1,
        ).astype(float)

    ok = (
        np.isfinite(hx)
        & np.isfinite(hy)
        & np.isfinite(minutes)
        & np.isfinite(colors).all(axis=1)
        & (hx >= 0)
        & (hy >= 0)
        & (hx <= (nx - 1))
        & (hy <= (ny - 1))
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        img,
        origin="lower",
        interpolation="nearest",
        alpha=kwargs.get("alpha_hexagram", 0.25),
    )

    scatter = None
    if np.any(ok):
        scatter = ax.scatter(
            hx[ok],
            hy[ok],
            s=markersize,
            c=minutes[ok],
            alpha=alpha,
            cmap=kwargs.get("cmap", "viridis"),
            edgecolors=kwargs.get("edgecolors", "black"),
            linewidths=kwargs.get("linewidths", 0.1),
        )

    z_bottom_m, z_top_m = _layer_bounds_from_attrs(analysis.attrs)
    selection_mode = str(analysis.attrs.get("selection_mode", "fixed_layer"))
    t0s = analysis.attrs.get("period_start", None)
    t1s = analysis.attrs.get("period_end", None)
    layer_txt = (
        f"{'Scan window' if selection_mode == 'scan' else 'Fixed layer'} "
        f"{float(z_bottom_m):.0f}-{float(z_top_m):.0f} m"
        if (z_bottom_m is not None and z_top_m is not None)
        else "Capa (desconocida)"
    )
    period_txt = f"{t0s} → {t1s}" if (t0s is not None and t1s is not None) else ""
    ax.set_title(f"Hexagrama RGB (k={k}) | {layer_txt}\n{period_txt}".rstrip())
    ax.set_xlabel("hex_x (índice rejilla)")
    ax.set_ylabel("hex_y (índice rejilla)")
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.grid(False)

    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Minutes since start")

    fig.tight_layout()

    filepath: Path | None = None
    if savefig:
        outdir = Path.cwd() if output_dir is None else Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        safe_t0 = str(t0s or "t0").replace(":", "").replace("-", "").replace(" ", "_")
        safe_t1 = str(t1s or "t1").replace(":", "").replace("-", "").replace(" ", "_")
        safe_layer = (
            f"{selection_mode}_{float(z_bottom_m):.0f}-{float(z_top_m):.0f}m"
            if (z_bottom_m is not None and z_top_m is not None)
            else "layer"
        )
        filepath = outdir / f"rain_process_hex_{safe_t0}_{safe_t1}_{safe_layer}_k{k}.png"
        fig.savefig(filepath, dpi=dpi)

    return fig, ax, filepath


def plot_processes_evolution(
    subject: SupportsProcessPlotting,
    *,
    classified: xr.Dataset,
    analysis: xr.Dataset | None = None,
    savefig: bool = False,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """Plot a temporal summary of classified rain-process evolution."""
    pcfg = subject.plot_cfg
    cmap = kwargs.get("cmap", pcfg.cmap)
    figsize = kwargs.get("figsize", getattr(pcfg, "figsize_multipanel", (10, 5)))
    dpi = kwargs.get("dpi", pcfg.dpi)

    markersize_timeline = kwargs.get("markersize_timeline", 28.0)
    heatmap_cmap = kwargs.get("heatmap_cmap", "coolwarm")
    title_fs = kwargs.get("title_fs", 18)
    label_fs = kwargs.get("label_fs", 16)
    tick_fs = kwargs.get("tick_fs", 14)

    process_order = kwargs.get(
        "process_order",
        [
            "unknown",
            "evaporation",
            "breakup",
            "growth_depletion",
            "growth_depletion_gain",
            "growth_depletion_loss",
            "growth",
            "activation",
            "autoconversion",
            "no_data",
        ],
    )

    if not isinstance(classified, xr.Dataset):
        raise TypeError("classified debe ser un xr.Dataset.")
    if "time" not in classified.coords:
        raise KeyError("classified debe tener coord 'time'.")
    for variable in ("proc_label", "strength", "sign_R", "sign_G", "sign_B"):
        if variable not in classified:
            raise KeyError(f"classified debe contener '{variable}'.")

    if analysis is not None:
        if not isinstance(analysis, xr.Dataset):
            raise TypeError("analysis debe ser xr.Dataset si se proporciona.")
        classified, analysis = xr.align(classified, analysis, join="inner")

    time_values = classified["time"].values
    if time_values.size == 0:
        raise ValueError("classified no contiene tiempos tras el alineamiento.")

    row_labels = ["R", "G", "B"]
    rgb_map = analysis.attrs.get("rgb_mapping", None) if analysis is not None else classified.attrs.get("rgb_mapping", None)
    if isinstance(rgb_map, dict) and all(k in rgb_map for k in ("R", "G", "B")):
        row_labels = [
            f"R ({rgb_map['R']})",
            f"G ({rgb_map['G']})",
            f"B ({rgb_map['B']})",
        ]

    df = classified[["proc_label", "strength"]].to_dataframe().reset_index()
    df = df[df["proc_label"].astype(str) != "steady_or_weak"].copy()
    labels = df["proc_label"].astype(str).fillna("unknown").to_numpy()
    present_labels = list(pd.unique(labels))
    ordered_labels = [label for label in process_order if label in present_labels]
    extra_labels = [label for label in present_labels if label not in ordered_labels]
    unique_labels = ordered_labels + extra_labels
    if len(unique_labels) > 26:
        raise ValueError("Hay más de 26 categorías de proceso; A,B,C... ya no basta.")

    map_code = {label: PROCESS_CODES.get(label, label.upper()) for label in unique_labels}
    y_index = np.array([unique_labels.index(label) for label in labels], dtype=int)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[40, 1.2],
        height_ratios=[1.35, 1.0],
        wspace=0.08,
        hspace=0.50,
    )

    ax_timeline = fig.add_subplot(gs[0, 0])
    cax_timeline = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[1, 0], sharex=ax_timeline)
    cax_heatmap = fig.add_subplot(gs[1, 1])
    plt.setp(ax_timeline.get_xticklabels(), visible=False)

    scatter = ax_timeline.scatter(
        df["time"].to_numpy(),
        y_index,
        c=df["strength"].to_numpy(dtype=float),
        cmap=cmap,
        s=markersize_timeline,
        vmin=0.0,
        vmax=1.0,
        edgecolors="none",
    )
    ax_timeline.set_title("(a) Process timeline (color = strength)", fontsize=title_fs)
    ax_timeline.set_ylabel("Process", fontsize=label_fs)
    ax_timeline.set_yticks(range(len(unique_labels)))
    ax_timeline.set_yticklabels([map_code[label] for label in unique_labels], fontsize=tick_fs)
    ax_timeline.tick_params(labelsize=tick_fs)

    cb1 = fig.colorbar(scatter, cax=cax_timeline)
    cb1.set_label("strength (0–1)", fontsize=label_fs)
    cax_timeline.tick_params(labelsize=tick_fs)

    sign_r = classified["sign_R"].values.astype(float)
    sign_g = classified["sign_G"].values.astype(float)
    sign_b = classified["sign_B"].values.astype(float)
    signs = np.vstack([sign_r, sign_g, sign_b])
    signs = np.where(np.isfinite(signs), signs, 0.0)

    time_numeric = mdates.date2num(pd.to_datetime(time_values).to_pydatetime())
    if time_numeric.size == 1:
        delta = 1.0 / (24 * 60)
        time_numeric = np.array([time_numeric[0] - delta, time_numeric[0] + delta])

    heatmap = ax_heatmap.imshow(
        signs,
        aspect="auto",
        interpolation="nearest",
        cmap=plt.get_cmap(heatmap_cmap, 3),
        vmin=-1,
        vmax=1,
        extent=(time_numeric[0], time_numeric[-1], -0.5, 2.5),
        origin="lower",
    )
    ax_heatmap.set_title("(b) Signs heatmap (-1 / 0 / +1)", fontsize=title_fs)
    ax_heatmap.set_yticks([0, 1, 2])
    ax_heatmap.set_yticklabels(row_labels, fontsize=tick_fs)
    ax_heatmap.set_ylabel("RGB component", fontsize=label_fs)
    ax_heatmap.tick_params(labelsize=tick_fs)

    cb2 = fig.colorbar(heatmap, cax=cax_heatmap)
    cb2.set_label("sign", fontsize=label_fs)
    cb2.set_ticks([-1, 0, 1])
    cb2.set_ticklabels(["-1", "0", "+1"])
    cax_heatmap.tick_params(labelsize=tick_fs)

    ax_heatmap.xaxis_date()
    ax_heatmap.set_xlabel("Time", fontsize=label_fs)
    ax_heatmap.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    filepath: Path | None = None
    if savefig:
        outdir = Path.cwd() if output_dir is None else Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        t0s = str(np.datetime_as_string(time_values[0], unit="s")).replace(":", "")
        t1s = str(np.datetime_as_string(time_values[-1], unit="s")).replace(":", "")
        if analysis is not None:
            z_bottom_m, z_top_m = _layer_bounds_from_attrs(analysis.attrs)
            selection_mode = str(analysis.attrs.get("selection_mode", "fixed_layer"))
        else:
            z_bottom_m, z_top_m = _layer_bounds_from_attrs(classified.attrs)
            selection_mode = str(classified.attrs.get("selection_mode", "fixed_layer"))
        z_bottom_tag = "zb" if z_bottom_m is None else f"{z_bottom_m:.0f}"
        z_top_tag = "zt" if z_top_m is None else f"{z_top_m:.0f}"
        filepath = outdir / (
            f"processes_evolution_{selection_mode}_{z_bottom_tag}-{z_top_tag}_{t0s}-{t1s}.png"
        )
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    return fig, ax_heatmap, filepath


def plot_column_process_scan(
    subject: SupportsProcessPlotting,
    *,
    scan_df: pd.DataFrame,
    processes: list[str] | None = None,
    savefig: bool = False,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """
    Plot a time-height curtain of process labels from a column scan dataframe.

    Marker selection is controlled with ``marker_mode``:
    - ``"process"`` uses ``PROCESS_MARKERS`` for one marker per process label.
    - ``"square"`` uses square markers for every process label.
    - ``"single"`` uses the value passed in ``marker`` for every process label.

    The older ``pm`` keyword is still accepted for compatibility. ``pm=0`` maps
    to ``marker_mode="process"`` and any other value maps to ``"single"``.
    """
    pcfg = subject.plot_cfg
    figsize = kwargs.get("figsize", getattr(pcfg, "figsize_profiles", (14, 8)))
    dpi = kwargs.get("dpi", pcfg.dpi)
    title_fs = kwargs.get("title_fs", 16)
    label_fs = kwargs.get("label_fs", 15)
    tick_fs = kwargs.get("tick_fs", 12)
    alpha = float(kwargs.get("alpha", 0.85))
    marker = kwargs.get("marker", "s")
    markersize = float(kwargs.get("markersize", 52.0))
    scale_by_strength = bool(kwargs.get("scale_by_strength", True))
    color_mode = str(kwargs.get("color_mode", "process")).lower()
    marker_mode = kwargs.get("marker_mode", None)

    if not isinstance(scan_df, pd.DataFrame):
        raise TypeError("scan_df must be a pandas DataFrame.")
    required = {"time", "z_center_m", "proc_label"}
    missing = sorted(required.difference(scan_df.columns))
    if missing:
        raise KeyError(f"scan_df must contain columns: {missing}")
    if scan_df.empty:
        raise ValueError("scan_df is empty.")
    if color_mode not in {"process", "hexagram"}:
        raise ValueError("color_mode must be 'process' or 'hexagram'.")
    if marker_mode not in {"process", "single", "square"}:
        raise ValueError("marker_mode must be 'process', 'single', or 'square'.")

    process_colors = kwargs.get(
        "process_colors",
        {
            "breakup": "#12af54",
            "growth_depletion": "#1b9e77",
            "growth_depletion_gain": "#f808d0",
            "growth_depletion_loss": "#ff0000",
            "evaporation": "#000000",
            "growth": "#91209b",
            "activation": "#66a61e",
            "steady_or_weak": "#bdbdbd",
            "unknown": "#666666",
            "no_data": "#f0f0f0",
        },
    )

    df = scan_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["proc_label"] = df["proc_label"].astype(str)
    if "proc_strength" in df.columns:
        df["proc_strength"] = pd.to_numeric(df["proc_strength"], errors="coerce")
    else:
        df["proc_strength"] = np.nan
    if processes is not None:
        selected_processes = {str(process) for process in processes if process is not None}
        df = df[df["proc_label"].isin(selected_processes)].copy()

    hex_colors: np.ndarray | None = None
    if color_mode == "hexagram":
        if {"hex_x", "hex_y"}.issubset(df.columns):
            k_attr = df.attrs.get("k", scan_df.attrs.get("k", None))
            if k_attr is not None:
                hex_assets = get_hexagram_assets(k=int(k_attr))
                rgb_cells = np.asarray(hex_assets["rgb_cells"], float)
                yx_cells = np.asarray(hex_assets["yx_cells"], int)
                lut = {tuple(yx): rgb for yx, rgb in zip(yx_cells, rgb_cells)}
                hex_colors = np.full((len(df), 3), np.nan, dtype=float)
                hx = pd.to_numeric(df["hex_x"], errors="coerce").to_numpy()
                hy = pd.to_numeric(df["hex_y"], errors="coerce").to_numpy()
                for index, (xv, yv) in enumerate(zip(hx, hy)):
                    if np.isfinite(xv) and np.isfinite(yv):
                        rgb = lut.get((int(round(yv)), int(round(xv))), None)
                        if rgb is not None:
                            hex_colors[index] = rgb
        if hex_colors is None and {"R", "G", "B"}.issubset(df.columns):
            hex_colors = np.column_stack(
                [
                    pd.to_numeric(df["R"], errors="coerce").to_numpy(),
                    pd.to_numeric(df["G"], errors="coerce").to_numpy(),
                    pd.to_numeric(df["B"], errors="coerce").to_numpy(),
                ]
            )

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    excluded_labels = set() if processes is not None else {"unknown", "no_data"}
    present_labels = [
        label
        for label in pd.unique(df["proc_label"])
        if label and label not in excluded_labels
    ]

    handles: list[Any] = []
    for label in present_labels:
        mask = df["proc_label"] == label
        if not mask.any():
            continue
        if marker_mode == "process":
            process_marker = PROCESS_MARKERS.get(label, marker)
        elif marker_mode == "square":
            process_marker = "s"
        else:
            process_marker = marker
        size = markersize
        if scale_by_strength and np.isfinite(df.loc[mask, "proc_strength"]).any():
            strength = df.loc[mask, "proc_strength"].fillna(0.0).clip(0.0, 1.0)
            size = 18.0 + markersize * strength.to_numpy()

        scatter_kwargs: dict[str, Any] = {
            "s": size,
            "alpha": alpha,
            "marker": process_marker,
            "edgecolors": "none",
            "label": label,
        }
        if color_mode == "hexagram" and hex_colors is not None:
            colors = hex_colors[np.flatnonzero(mask.to_numpy())]
            finite_rgb = np.isfinite(colors).all(axis=1)
            if not np.any(finite_rgb):
                scatter_kwargs["c"] = process_colors.get(label, "#333333")
            else:
                scatter_kwargs["c"] = colors[finite_rgb]
                time_values = df.loc[mask, "time"].to_numpy()[finite_rgb]
                height_values = (df.loc[mask, "z_center_m"].to_numpy()[finite_rgb]) / 1000.0
                scatter = ax.scatter(
                    time_values,
                    height_values,
                    **scatter_kwargs,
                )
                handles.append(scatter)
                continue
        else:
            scatter_kwargs["c"] = process_colors.get(label, "#333333")

        scatter = ax.scatter(
            df.loc[mask, "time"],
            df.loc[mask, "z_center_m"] / 1000.0,
            **scatter_kwargs,
        )
        handles.append(scatter)

    ax.set_xlabel("Time", fontsize=label_fs)
    ax.set_ylabel("Height (km)", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.grid(True, which="both", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    period_start = scan_df.attrs.get("period_start", None)
    period_end = scan_df.attrs.get("period_end", None)
    thickness = scan_df.attrs.get("window_thickness_m", None)
    step = scan_df.attrs.get("window_step_m", None)
    min_tau_strength = scan_df.attrs.get("min_tau_strength", None)
    subtitle_parts = []
    if thickness is not None:
        subtitle_parts.append(f"window={float(thickness):.0f} m")
    if step is not None:
        subtitle_parts.append(f"step={float(step):.0f} m")
    if min_tau_strength is not None:
        subtitle_parts.append(f"min_tau_strength={float(min_tau_strength):.2f}")
    subtitle = " | ".join(subtitle_parts)
    title = "Column Process Scan"
    if period_start and period_end:
        title = f"{title} | {period_start[0:16].replace('T', ' ')} - {period_end[11:16]}"
    if subtitle:
        title = f"{title}\n{subtitle}"
    ax.set_title(title, fontsize=title_fs, pad=12)
    y_limits = _resolve_height_limits_km(subject, kwargs.get("y_limits"))
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    if handles:
        ax.legend(
            handles=handles,
            loc=kwargs.get("legend_loc", "upper left"),
            fontsize=kwargs.get("legend_fs", 10),
            ncol=kwargs.get("legend_ncol", 2),
            frameon=True,
        )

    filepath: Path | None = None
    if savefig:
        outdir = Path.cwd() if output_dir is None else Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        safe_t0 = str(period_start or "t0").replace(":", "").replace("-", "").replace(" ", "_")
        safe_t1 = str(period_end or "t1").replace(":", "").replace("-", "").replace(" ", "_")
        filepath = outdir / f"column_process_scan_{color_mode}_{safe_t0}_{safe_t1}.png"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    return fig, ax, filepath



def plot_fused_process_quicklook(
    scan_df: pd.DataFrame,
    fused_df: pd.DataFrame,
    *,
    processes: list[str] | None = None,
    time_col: str = "time",
    z_top_col: str = "z_top",
    z_bottom_col: str = "z_bottom",
    process_col: str = "proc_label",
    z_top_fused_col: str = "z_top_fused",
    z_bottom_fused_col: str = "z_bottom_fused",
    process_fused_col: str = "proc_label_fused",
    figsize: tuple[float, float] = (10, 6),
    alpha_scan: float = 0.2,
    alpha_fused: float = 0.6,
    savefig: bool = False,
    output_dir: Path | None = None,
    dpi: int = 200,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """
    Quicklook plot to visually validate fused vertical process events.

    The plot overlays:
    - the original scan detections as semi-transparent points (context),
    - fused events as vertical rectangles at each time step.

    This is a lightweight diagnostic plot intended for exploratory validation.

    The layout, defaults, and ``savefig`` behaviour are intentionally aligned
    with :func:`plot_column_process_scan` to make quicklooks consistent across
    the package.

    If ``processes`` is provided, both the scan background and fused rectangles
    are filtered to show only those process labels (same behaviour as
    :func:`plot_column_process_scan`).
    """
    if not isinstance(scan_df, pd.DataFrame):
        raise TypeError("scan_df must be a pandas DataFrame.")
    if not isinstance(fused_df, pd.DataFrame):
        raise TypeError("fused_df must be a pandas DataFrame.")
    if scan_df.empty and fused_df.empty:
        raise ValueError("scan_df and fused_df are both empty.")

    process_colors: dict[str, Any] = {
        "breakup": "#12af54",
        "growth_depletion": "#1b9e77",
        "growth_depletion_gain": "#f808d0",
        "growth_depletion_loss": "#ff0000",
        "evaporation": "#000000",
        "growth": "#91209b",
        "activation": "#66a61e",
        "steady_or_weak": "#bdbdbd",
        "unknown": "#666666",
        "no_data": "#f0f0f0",
    }

    def _resolve_column(df: pd.DataFrame, requested: str, alternatives: tuple[str, ...]) -> str:
        if requested in df.columns:
            return requested
        for alt in alternatives:
            if alt in df.columns:
                return alt
        raise KeyError(f"Missing column {requested!r} in dataframe.")

    scan_time_col = _resolve_column(scan_df, time_col, ("time",))
    scan_proc_col = _resolve_column(scan_df, process_col, ("proc_label",))
    scan_top_col = _resolve_column(scan_df, z_top_col, ("z_top_m", "z_max_m"))
    scan_bottom_col = _resolve_column(scan_df, z_bottom_col, ("z_bottom_m", "z_min_m"))

    fused_time_col = _resolve_column(fused_df, time_col, ("time",))
    fused_proc_col = _resolve_column(fused_df, process_fused_col, ("proc_label_fused", "proc_label"))
    fused_top_col = _resolve_column(fused_df, z_top_fused_col, ("z_top_fused",))
    fused_bottom_col = _resolve_column(fused_df, z_bottom_fused_col, ("z_bottom_fused",))

    df_scan = scan_df.copy()
    df_fused = fused_df.copy()

    df_scan[scan_time_col] = pd.to_datetime(df_scan[scan_time_col])
    df_scan[scan_proc_col] = df_scan[scan_proc_col].astype(str)
    df_scan[scan_top_col] = pd.to_numeric(df_scan[scan_top_col], errors="coerce")
    df_scan[scan_bottom_col] = pd.to_numeric(df_scan[scan_bottom_col], errors="coerce")

    df_fused[fused_time_col] = pd.to_datetime(df_fused[fused_time_col])
    df_fused[fused_proc_col] = df_fused[fused_proc_col].astype(str)
    df_fused[fused_top_col] = pd.to_numeric(df_fused[fused_top_col], errors="coerce")
    df_fused[fused_bottom_col] = pd.to_numeric(df_fused[fused_bottom_col], errors="coerce")

    if processes is not None:
        selected_processes = {str(process) for process in processes if process is not None}
        df_scan = df_scan[df_scan[scan_proc_col].isin(selected_processes)].copy()
        df_fused = df_fused[df_fused[fused_proc_col].isin(selected_processes)].copy()

    # Build a deterministic color map across all labels present.
    all_labels = pd.unique(
        pd.concat(
            [
                df_scan[scan_proc_col].dropna().astype(str),
                df_fused[fused_proc_col].dropna().astype(str),
            ],
            ignore_index=True,
        )
    ).tolist()
    all_labels = [label for label in all_labels if label]
    unknown_labels = sorted([label for label in all_labels if label not in process_colors])
    if unknown_labels:
        cmap = plt.get_cmap("tab20")
        for idx, label in enumerate(unknown_labels):
            process_colors[label] = cmap(idx % cmap.N)

    # Infer a reasonable rectangle width from the time sampling (in matplotlib date units).
    times = pd.to_datetime(
        pd.concat(
            [
                df_scan[scan_time_col] if not df_scan.empty else pd.Series([], dtype="datetime64[ns]"),
                df_fused[fused_time_col] if not df_fused.empty else pd.Series([], dtype="datetime64[ns]"),
            ],
            ignore_index=True,
        )
    )
    unique_times = pd.to_datetime(pd.unique(times)).sort_values()
    width_days = 1.0 / (24.0 * 60.0)  # 1 minute default
    if len(unique_times) >= 2:
        dt = np.diff(mdates.date2num(unique_times.to_numpy()))
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            width_days = float(np.median(dt)) * 0.85
            width_days = max(width_days, 1.0 / (24.0 * 3600.0))  # at least 1 second

    title_fs = kwargs.get("title_fs", 16)
    label_fs = kwargs.get("label_fs", 15)
    tick_fs = kwargs.get("tick_fs", 12)

    def _resolve_height_limits_km_from_frames(
        explicit_limits: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if explicit_limits is not None:
            return explicit_limits

        heights_m: list[np.ndarray] = []
        if not df_scan.empty:
            if "z_center_m" in df_scan.columns:
                heights_m.append(pd.to_numeric(df_scan["z_center_m"], errors="coerce").to_numpy(dtype=float))
            heights_m.append(pd.to_numeric(df_scan[scan_top_col], errors="coerce").to_numpy(dtype=float))
            heights_m.append(pd.to_numeric(df_scan[scan_bottom_col], errors="coerce").to_numpy(dtype=float))
        if not df_fused.empty:
            heights_m.append(pd.to_numeric(df_fused[fused_top_col], errors="coerce").to_numpy(dtype=float))
            heights_m.append(pd.to_numeric(df_fused[fused_bottom_col], errors="coerce").to_numpy(dtype=float))

        if not heights_m:
            return None
        all_vals = np.concatenate(heights_m)
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size == 0:
            return None
        return float(np.min(finite)) / 1000.0, float(np.max(finite)) / 1000.0

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Background scan points at layer center.
    if not df_scan.empty:
        center_m = 0.5 * (df_scan[scan_top_col].to_numpy(dtype=float) + df_scan[scan_bottom_col].to_numpy(dtype=float))
        finite = np.isfinite(center_m) & (~pd.isna(df_scan[scan_time_col]).to_numpy())
        if np.any(finite):
            for label in sorted(pd.unique(df_scan.loc[finite, scan_proc_col])):
                mask = finite & (df_scan[scan_proc_col].to_numpy(dtype=str) == str(label))
                if not np.any(mask):
                    continue
                ax.scatter(
                    df_scan.loc[mask, scan_time_col],
                    center_m[mask] / 1000.0,
                    s=10.0,
                    alpha=float(alpha_scan),
                    c=[process_colors.get(str(label), "#333333")],
                    edgecolors="none",
                )

    # Fused rectangles.
    if not df_fused.empty:
        for _, row in df_fused.iterrows():
            t = row.get(fused_time_col, None)
            z_top = row.get(fused_top_col, np.nan)
            z_bottom = row.get(fused_bottom_col, np.nan)
            label = str(row.get(fused_proc_col, ""))
            if t is None or not pd.notna(t):
                continue
            if not (np.isfinite(z_top) and np.isfinite(z_bottom) and z_top > z_bottom):
                continue

            t_center = pd.Timestamp(t).round("us").to_pydatetime()
            x_center = mdates.date2num(t_center)
            rect = Rectangle(
                (x_center - 0.5 * width_days, float(z_bottom) / 1000.0),
                width_days,
                float(z_top - z_bottom) / 1000.0,
                facecolor=process_colors.get(label, "#333333"),
                edgecolor="none",
                alpha=float(alpha_fused),
            )
            ax.add_patch(rect)

    ax.set_xlabel("Time", fontsize=label_fs)
    ax.set_ylabel("Height (km)", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.grid(True, which="both", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    period_start = scan_df.attrs.get("period_start", fused_df.attrs.get("period_start", None))
    period_end = scan_df.attrs.get("period_end", fused_df.attrs.get("period_end", None))
    thickness = scan_df.attrs.get("window_thickness_m", None)
    step = scan_df.attrs.get("window_step_m", None)
    min_tau_strength = scan_df.attrs.get("min_tau_strength", None)
    min_consecutive = fused_df.attrs.get("min_consecutive", None)

    subtitle_parts = []
    if thickness is not None:
        subtitle_parts.append(f"window={float(thickness):.0f} m")
    if step is not None:
        subtitle_parts.append(f"step={float(step):.0f} m")
    if min_consecutive is not None:
        subtitle_parts.append(f"min_consecutive={int(min_consecutive)}")
    if min_tau_strength is not None:
        subtitle_parts.append(f"min_tau_strength={float(min_tau_strength):.2f}")
    subtitle = " | ".join(subtitle_parts)

    title = "Fused Process Quicklook"
    if period_start and period_end:
        title = f"{title} | {str(period_start)[0:16].replace('T', ' ')} - {str(period_end)[11:16]}"
    if subtitle:
        title = f"{title}\n{subtitle}"
    ax.set_title(title, fontsize=title_fs, pad=12)

    y_limits = _resolve_height_limits_km_from_frames(kwargs.get("y_limits"))
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    # Legend based on fused labels (usually fewer than scan labels).
    fused_labels = []
    if not df_fused.empty:
        fused_labels = [label for label in pd.unique(df_fused[fused_proc_col]) if label]
    if fused_labels and len(fused_labels) <= 12:
        handles = [
            Rectangle((0, 0), 1, 1, facecolor=process_colors.get(str(label), "#333333"), alpha=float(alpha_fused))
            for label in fused_labels
        ]
        ax.legend(
            handles,
            [str(label) for label in fused_labels],
            loc=kwargs.get("legend_loc", "upper left"),
            fontsize=kwargs.get("legend_fs", 10),
            ncol=kwargs.get("legend_ncol", 2),
            frameon=True,
        )

    filepath: Path | None = None
    if savefig:
        outdir = Path(".") if output_dir is None else Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        safe_t0 = str(period_start or "t0").replace(":", "").replace("-", "").replace(" ", "_")
        safe_t1 = str(period_end or "t1").replace(":", "").replace("-", "").replace(" ", "_")
        filepath = outdir / f"fused_process_quicklook_{safe_t0}_{safe_t1}.png"
        fig.savefig(filepath, dpi=int(dpi), bbox_inches="tight")

    return fig, ax, filepath


def plot_scan_process_scatter_compare(
    subject: SupportsProcessPlotting,
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
    """Compare several classified scan processes in a shared microphysical scatter."""
    if not isinstance(scan_df, pd.DataFrame):
        raise TypeError("scan_df must be a pandas DataFrame.")
    selected_processes = [str(process) for process in processes if process is not None]
    if not selected_processes:
        raise ValueError("processes must contain at least one process name.")

    required = {"time", "proc_label", x, y, color}
    missing = sorted(required.difference(scan_df.columns))
    if missing:
        raise KeyError(f"scan_df must contain columns: {missing}")

    process_colors = kwargs.get(
        "process_colors",
        {
            "breakup": "#d95f02",
            "growth_depletion": "#1b9e77",
            "growth_depletion_gain": "#7570b3",
            "growth_depletion_loss": "#6a3d9a",
            "evaporation": "#7570b3",
            "growth": "#e7298a",
            "activation": "#66a61e",
            "steady_or_weak": "#bdbdbd",
            "unknown": "#666666",
            "no_data": "#f0f0f0",
        },
    )
    figsize = kwargs.get("figsize", getattr(subject.plot_cfg, "figsize", (10, 8)))
    dpi = kwargs.get("dpi", subject.plot_cfg.dpi)
    label_fs = kwargs.get("label_fs", 15)
    tick_fs = kwargs.get("tick_fs", 12)
    title_fs = kwargs.get("title_fs", 16)
    legend_fs = kwargs.get("legend_fs", 10)
    markersize = float(kwargs.get("markersize", 55.0))
    alpha = float(kwargs.get("alpha", 0.85))

    df = scan_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["proc_label"] = df["proc_label"].astype(str)
    df = df[df["proc_label"].isin(selected_processes)].copy()
    if period is not None:
        df = df[(df["time"] >= pd.Timestamp(period[0])) & (df["time"] <= pd.Timestamp(period[1]))]
    if z_bottom_m is not None:
        if "z_bottom_m" in df.columns:
            df = df[df["z_bottom_m"] >= float(z_bottom_m)]
        elif "z_center_m" in df.columns:
            df = df[df["z_center_m"] >= float(z_bottom_m)]
    if z_top_m is not None:
        if "z_top_m" in df.columns:
            df = df[df["z_top_m"] <= float(z_top_m)]
        elif "z_center_m" in df.columns:
            df = df[df["z_center_m"] <= float(z_top_m)]
    if df.empty:
        raise ValueError("No scan samples remain after filtering.")

    for column in (x, y, color):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    scatter_for_cbar = None
    legend_handles: list[Line2D] = []
    color_limits = pd.to_numeric(df[color], errors="coerce")
    finite_color = color_limits[np.isfinite(color_limits)]
    vmin = float(np.nanmin(finite_color)) if not finite_color.empty else None
    vmax = float(np.nanmax(finite_color)) if not finite_color.empty else None

    plotted_labels: list[str] = []
    for label in pd.unique(df["proc_label"]):
        group = df[df["proc_label"] == label].copy()
        finite_mask = (
            np.isfinite(group[x].to_numpy(dtype=float))
            & np.isfinite(group[y].to_numpy(dtype=float))
            & np.isfinite(group[color].to_numpy(dtype=float))
        )
        group = group.loc[finite_mask]
        if group.empty:
            continue

        scatter = ax.scatter(
            group[x].to_numpy(dtype=float),
            group[y].to_numpy(dtype=float),
            c=group[color].to_numpy(dtype=float),
            cmap=kwargs.get("cmap", "viridis"),
            vmin=vmin,
            vmax=vmax,
            s=markersize,
            alpha=alpha,
            marker=PROCESS_MARKERS.get(label, "o"),
            edgecolors=process_colors.get(label, "#333333"),
            linewidths=0.5,
        )
        scatter_for_cbar = scatter
        plotted_labels.append(label)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=PROCESS_MARKERS.get(label, "o"),
                color="none",
                markerfacecolor="white",
                markeredgecolor=process_colors.get(label, "#333333"),
                markeredgewidth=1.2,
                markersize=8,
                label=label,
            )
        )

        if show_centroids:
            ax.scatter(
                [float(np.nanmedian(group[x].to_numpy(dtype=float)))],
                [float(np.nanmedian(group[y].to_numpy(dtype=float)))],
                marker="X",
                s=markersize * 1.8,
                c=process_colors.get(label, "#333333"),
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )

        if show_density and len(group) >= 5:
            x_values = group[x].to_numpy(dtype=float)
            y_values = group[y].to_numpy(dtype=float)
            if np.nanmax(x_values) > np.nanmin(x_values) and np.nanmax(y_values) > np.nanmin(y_values):
                hist, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=12)
                if np.any(hist > 0):
                    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
                    level = float(np.nanmax(hist)) * 0.5
                    if np.isfinite(level) and level > 0.0:
                        ax.contour(
                            xx,
                            yy,
                            hist,
                            levels=[level],
                            colors=[process_colors.get(label, "#333333")],
                            linewidths=1.0,
                            alpha=0.8,
                        )

    if scatter_for_cbar is None:
        raise ValueError("No finite scan samples are available for plotting.")

    cbar = fig.colorbar(scatter_for_cbar, ax=ax)
    cbar.set_label(color, fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)

    ax.set_xlabel(x, fontsize=label_fs)
    ax.set_ylabel(y, fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)

    title = "Scan Process Scatter Compare"
    period_start = scan_df.attrs.get("period_start", None)
    period_end = scan_df.attrs.get("period_end", None)
    if period is not None:
        title = (
            f"{title} | {pd.Timestamp(period[0]).strftime('%Y-%m-%d %H:%M')} - "
            f"{pd.Timestamp(period[1]).strftime('%H:%M')}"
        )
    elif period_start and period_end:
        title = f"{title} | {str(period_start)[0:16].replace('T', ' ')} - {str(period_end)[11:16]}"
    ax.set_title(title, fontsize=title_fs, pad=12)

    if legend_handles:
        ax.legend(handles=legend_handles, loc=kwargs.get("legend_loc", "best"), fontsize=legend_fs, frameon=True)

    filepath: Path | None = None
    if savefig:
        outdir = Path.cwd() if output_dir is None else Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        safe_processes = "_vs_".join(label.replace(" ", "_") for label in plotted_labels) or "processes"
        safe_t0 = str(period_start or "t0").replace(":", "").replace("-", "").replace(" ", "_")
        safe_t1 = str(period_end or "t1").replace(":", "").replace("-", "").replace(" ", "_")
        filepath = outdir / (
            f"scan_process_scatter_compare_{safe_processes}_{x}_vs_{y}_color_{color}_{safe_t0}_{safe_t1}.png"
        )
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    return fig, ax, filepath


def plot_classified_processes_on_hexagram(
    subject: SupportsProcessPlotting,
    *,
    classified: xr.Dataset,
    analysis: xr.Dataset | None = None,
    processes: list[str] | None = None,
    show_background: bool = False,
    show_process_masks: bool = True,
    savefig: bool = False,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """Plot classified samples on the package RGB hexagram."""
    pcfg = subject.plot_cfg
    figsize = kwargs.get("figsize", pcfg.figsize_hex)
    dpi = kwargs.get("dpi", pcfg.dpi)
    alpha_hexagram = kwargs.get("alpha_hexagram", pcfg.alpha_hexagram)
    alpha_mask = kwargs.get("alpha_mask", 0.18)
    markersize = kwargs.get("markersize", 35.0)
    
    #Check processes names are correct with respect to the codes in PROCESS_CODES
    if processes is not None:
        for process in processes:
            if process not in PROCESS_CODES:
                raise ValueError(f"Process '{process}' is not recognized. Valid processes: {list(PROCESS_CODES.keys())}")

    if not isinstance(classified, xr.Dataset):
        raise TypeError("classified must be an xr.Dataset.")
    for variable in ("proc_label", "hex_x", "hex_y"):
        if variable not in classified:
            raise KeyError(f"classified must contain '{variable}'.")

    k = classified.attrs.get("k", None)
    if k is None and analysis is not None:
        k = analysis.attrs.get("k", None)
    if k is None:
        raise KeyError("k not found in classified.attrs nor analysis.attrs.")

    hex_assets = get_hexagram_assets(k=k)
    img = np.asarray(hex_assets["img"], float)

    process_colors = kwargs.get(
        "process_colors",
        {
            "breakup": "#d95f02",
            "growth_depletion": "#1b9e77",
            "growth_depletion_gain": "#7570b3",
            "growth_depletion_loss": "#6a3d9a",
            "evaporation": "#7570b3",
            "growth": "#e7298a",
            "activation": "#66a61e",
            "steady_or_weak": "#bdbdbd",
            "unknown": "#666666",
            "no_data": "#d9d9d9",
        },
    )

    if processes is not None:
        for selected_processes in processes:
            classified = classified.where(classified["proc_label"].isin(list(selected_processes)), drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    if show_background:
        ax.imshow(
            img,
            origin="lower",
            interpolation="nearest",
            alpha=alpha_hexagram,
        )

    if show_process_masks and processes is not None:
        for process in processes:
            mask2d, _ = get_process_hexagram_mask(
                process,
                k=k,
                tol_center=classified.attrs["tol_center"],
            )
            color = process_colors.get(process, "#000000")
            rgba = np.zeros((*mask2d.shape, 4), dtype=float)
            rgb = plt.matplotlib.colors.to_rgb(color)
            rgba[mask2d, 0] = rgb[0]
            rgba[mask2d, 1] = rgb[1]
            rgba[mask2d, 2] = rgb[2]
            rgba[mask2d, 3] = alpha_mask
            ax.imshow(rgba, origin="lower", interpolation="nearest")

    hx = classified["hex_x"].values.astype(float)
    hy = classified["hex_y"].values.astype(float)
    labels = classified["proc_label"].values.astype(str)
    valid = np.isfinite(hx) & np.isfinite(hy) & (hx >= 0) & (hy >= 0)

    present_labels = pd.unique(labels[valid])
    handles: list[Any] = []
    strength = classified["strength"].values.astype(float)
    scatter = None
    for label in present_labels:
        mask = valid & (labels == label)
        if not np.any(mask):
            continue
        marker = PROCESS_MARKERS.get(label, "o")
        scatter = ax.scatter(
            hx[mask],
            hy[mask],
            s=markersize,
            c=strength[mask],
            cmap=kwargs.get("cmap", "viridis"),
            vmin=0.0,
            vmax=1.0,
            marker=marker,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.95,
            label=PROCESS_CODES.get(label, label.upper()),
        )
        handles.append(scatter)

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("strength")

    ax.set_title(f"Classified rain processes on RGB hexagram (k={k})")
    ax.set_xlabel("hex_x")
    ax.set_ylabel("hex_y")
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(-0.5, img.shape[0] - 0.5)
    ax.set_aspect("equal")
    ax.grid(False)

    if handles:
        ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()

    filepath: Path | None = None
    if savefig:
        outdir = Path.cwd() if output_dir is None else Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        t0s = str(np.datetime_as_string(classified["time"].values[0], unit="s")).replace(":", "")
        t1s = str(np.datetime_as_string(classified["time"].values[-1], unit="s")).replace(":", "")
        filepath = outdir / f"classified_processes_hexagram_{t0s}_{t1s}_k{k}.png"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    return fig, ax, filepath

