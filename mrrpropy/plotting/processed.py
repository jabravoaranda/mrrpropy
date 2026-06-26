from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xarray as xr


class SupportsProcessedPlotting(Protocol):
    path: str | Path
    raprompro: xr.Dataset | None
    plot_cfg: Any

    def _is_processed(self) -> bool: ...


def _resolve_height_limits_km(
    subject: SupportsProcessedPlotting,
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


def plot_dsdgram(
    subject: SupportsProcessedPlotting,
    *,
    target_datetime: datetime,
    range_limits: tuple[float, float] | None = None,
    drop_limits: tuple[float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    savefig: bool = False,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """Plot a DSD gram at the nearest requested time."""
    if subject.raprompro is None:
        raise RuntimeError(
            "self.raprompro no estÃ¡ cargado. Usa load_raprompro() o procesa antes."
        )

    pcfg = subject.plot_cfg
    dpi = kwargs.get("dpi", pcfg.dpi)
    cmap = kwargs.get("cmap", pcfg.cmap)
    figsize = kwargs.get("figsize", pcfg.figsize)

    ds = subject.raprompro
    if "dsd_3D" not in ds:
        raise KeyError("self.raprompro no contiene la variable 'dsd_3D'.")

    da = ds["dsd_3D"]
    expected = ("time", "range", "DropSize")
    if tuple(da.dims) != expected:
        raise ValueError(f"dsd_3D.dims esperadas {expected}, pero son {da.dims}")

    da2 = da.sel(time=target_datetime, method="nearest")
    if range_limits is not None:
        da2 = da2.sel(range=slice(range_limits[0], range_limits[1]))
    if drop_limits is not None:
        da2 = da2.sel(DropSize=slice(drop_limits[0], drop_limits[1]))

    da2 = da2.transpose("range", "DropSize")
    drop_values = da2["DropSize"].values
    da2 = da2.isel({"DropSize": np.isfinite(drop_values)})

    x_axis = da2["DropSize"].values
    y_axis = da2["range"].values
    z_values = da2.values.astype(float)

    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(
        x_axis,
        y_axis,
        z_values,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_xlabel("Drop diameter (mm)")
    ax.set_ylabel("Range / Height (m)")
    ax.set_xlim(kwargs.get("x_limits", (0.25, 10)))

    selected_time = da2["time"].values
    if np.issubdtype(np.asarray(selected_time).dtype, np.datetime64):
        time_label = str(np.datetime_as_string(selected_time, unit="s"))
    else:
        time_label = str(selected_time)
    ax.set_title(f"DSD-gram \n {time_label}")

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(da.attrs.get("units", "dsd_3D"))

    filepath: Path | None = None
    if savefig:
        if output_dir is None:
            output_dir = Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        time_tag = str(np.datetime_as_string(selected_time, unit="s")).replace(":", "")
        filepath = output_dir / Path(subject.path).name.replace(
            ".nc", f"_DSDgram_{time_tag}.png"
        )
        fig.savefig(filepath, dpi=dpi)

    return fig, ax, filepath


def plot_dsd_by_range(
    subject: SupportsProcessedPlotting,
    target_datetime: datetime | np.datetime64 | str,
    ranges: list[float] | np.ndarray,
    *,
    use_log10: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    ncol: int = 2,
    savefig: bool = False,
    output_dir: Path | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """Plot several N(D) curves at a fixed time for multiple provided ranges."""
    pcfg = subject.plot_cfg
    dpi = kwargs.get("dpi", pcfg.dpi)
    cmap = kwargs.get("cmap", pcfg.cmap)
    figsize = kwargs.get("figsize", pcfg.figsize)
    marker = kwargs.get("marker", pcfg.marker)
    markersize = kwargs.get("markersize", pcfg.markersize)
    legend_fontsize = kwargs.get("legend_fontsize", pcfg.legendfontsize)

    if subject.raprompro is None:
        raise RuntimeError("raprompro not loaded. Use load_raprompro().")
    ds_rp = subject.raprompro
    if "dsd_3D" not in ds_rp:
        raise KeyError(
            "raprompro missing required variable 'dsd_3D'. Check save_dsd_3d is True."
        )
    da = ds_rp["dsd_3D"]

    for dim in ("time", "range", "DropSize"):
        if dim not in da.dims:
            raise ValueError(f"dsd_3D must have dim '{dim}'. dims={da.dims}")

    t_sel = cast(np.datetime64, ds_rp["time"].sel(time=target_datetime, method="nearest").values)

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif fig is not None and ax is None:
        axes = fig.get_axes()
        ax = axes[0] if axes else fig.add_subplot(111)
    elif fig is None and ax is not None:
        fig = cast(Figure, ax.figure)

    if fig is None or ax is None:
        raise ValueError("A matplotlib figure and axes could not be prepared.")

    ranges_in = np.asarray(ranges, dtype=float)
    if ranges_in.size == 0:
        raise ValueError("ranges must contain at least one value.")

    cm = plt.get_cmap(cmap)
    colors = [cm(i / max(1, ranges_in.size - 1)) for i in range(ranges_in.size)]

    units = (da.attrs.get("units", "") or "").lower()
    data_is_log10 = "log10" in units or "log" in units

    diameters = da["DropSize"].values.astype(float)
    diameter_scale = 1.0
    diameter_units = da["DropSize"].attrs.get("units", "")
    if diameter_units.lower() in ("m", "meter", "metre"):
        diameter_scale = 1000.0
        diameter_units_out = "mm"
    elif np.nanmax(diameters) < 0.05:
        diameter_scale = 1000.0
        diameter_units_out = "mm"
    else:
        diameter_units_out = "mm" if diameter_units == "" else diameter_units

    plotted_any = False
    threshold = float(kwargs.get("N_minimum_threshold", 0.0))

    for idx, requested_range in enumerate(ranges_in):
        selected_range = float(
            ds_rp["range"].sel(range=requested_range, method="nearest").values.item()
        )
        dsd_profile = da.sel(time=t_sel, range=selected_range, method="nearest")
        values = dsd_profile.values.astype(float)

        if data_is_log10:
            if threshold > 0:
                threshold_log = np.log10(threshold)
                values = np.where(values >= threshold_log, values, np.nan)
        else:
            values = np.where(values >= threshold, values, np.nan)

        valid = np.isfinite(diameters) & np.isfinite(values)
        if not np.any(valid):
            continue

        x_values = (diameters[valid] * diameter_scale).astype(float)
        if use_log10:
            y_values = values[valid] if data_is_log10 else np.log10(values[valid])
            ax.set_yscale("linear")
            y_label = r"$\log_{10}(N)\ [\mathrm{m^{-3}\,mm^{-1}}]$"
        else:
            y_values = (10.0 ** values[valid]) if data_is_log10 else values[valid]
            y_values = np.where(y_values > 0, y_values, np.nan)
            valid_linear = np.isfinite(y_values)
            x_values = x_values[valid_linear]
            y_values = y_values[valid_linear]
            if x_values.size == 0:
                continue
            ax.set_yscale("log")
            y_label = r"$N\ [\mathrm{m^{-3}\,mm^{-1}}]$"

        ax.plot(
            x_values,
            y_values,
            color=colors[idx],
            label=f"{selected_range:.1f} m",
            marker=marker,
            markersize=markersize,
        )
        plotted_any = True

    if not plotted_any:
        raise ValueError("No valid DSD curves found for the provided ranges/time.")

    ax.set_xlabel(f"D [{diameter_units_out}]")
    ax.set_ylabel(y_label)

    time_text = str(np.datetime_as_string(t_sel, unit="s"))
    ax.set_title(f"RaProMPro N(D) by range\n{time_text}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(ncol=ncol, loc="best", fontsize=legend_fontsize)

    if vmin is not None or vmax is not None:
        ax.set_ylim(vmin, vmax)
    if kwargs.get("xlimits") is not None:
        ax.set_xlim(kwargs["xlimits"])

    fig.tight_layout()

    filepath: Path | None = None
    if savefig:
        if output_dir is None:
            raise ValueError("output_dir must be provided if savefig=True.")
        output_dir.mkdir(parents=True, exist_ok=True)
        time_tag = str(np.datetime_as_string(t_sel, unit="s")).replace(":", "")
        filepath = output_dir / Path(subject.path).name.replace(
            ".nc", f"_DSD_by_range_{time_tag}.png"
        )
        fig.savefig(filepath, dpi=dpi)

    return fig, ax, filepath

def plot_dsd_by_range_3d(
    subject: SupportsProcessedPlotting,
    target_datetime: datetime | np.datetime64 | str,
    ranges: list[float] | np.ndarray,
    *,
    use_log10: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    ncol: int = 2,
    savefig: bool = False,
    output_dir: Path | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes, Path | None]:
    """Plot several N(D) curves at a fixed time for multiple provided ranges."""
    pcfg = subject.plot_cfg
    dpi = kwargs.get("dpi", pcfg.dpi)
    cmap = kwargs.get("cmap", pcfg.cmap)
    figsize = kwargs.get("figsize", pcfg.figsize)
    marker = kwargs.get("marker", pcfg.marker)
    markersize = kwargs.get("markersize", pcfg.markersize)
    legend_fontsize = kwargs.get("legend_fontsize", pcfg.legendfontsize)

    if subject.raprompro is None:
        raise RuntimeError("raprompro not loaded. Use load_raprompro().")
    ds_rp = subject.raprompro
    if "dsd_3D" not in ds_rp:
        raise KeyError(
            "raprompro missing required variable 'dsd_3D'. Check save_dsd_3d is True."
        )
    da = ds_rp["dsd_3D"]

    for dim in ("time", "range", "DropSize"):
        if dim not in da.dims:
            raise ValueError(f"dsd_3D must have dim '{dim}'. dims={da.dims}")

    t_sel = cast(np.datetime64, ds_rp["time"].sel(time=target_datetime, method="nearest").values)

    if fig is None and ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    elif fig is not None and ax is None:
        axes = fig.get_axes()
        ax = next((a for a in axes if getattr(a, "name", "") == "3d"), None)
        if ax is None:
            ax = fig.add_subplot(111, projection="3d")
    elif fig is None and ax is not None:
        fig = cast(Figure, ax.figure)

    if fig is None or ax is None:
        raise ValueError("A matplotlib figure and axes could not be prepared.")

    ranges_in = np.asarray(ranges, dtype=float)
    if ranges_in.size == 0:
        raise ValueError("ranges must contain at least one value.")

    cm = plt.get_cmap(cmap)
    colors = [cm(i / max(1, ranges_in.size - 1)) for i in range(ranges_in.size)]

    units = (da.attrs.get("units", "") or "").lower()
    data_is_log10 = "log10" in units or "log" in units

    diameters = da["DropSize"].values.astype(float)
    diameter_scale = 1000.0
    diameter_units = da["DropSize"].attrs.get("units", "")
    if diameter_units.lower() in ("m", "meter", "metre") or np.nanmax(diameters) < 0.05:
        diameter_scale = 1000.0
        diameter_units_out = "mm"
    else:
        diameter_units_out = diameter_units or "mm"
    x_axis = diameters * diameter_scale

    selected_ranges = []
    y_grid = []

    for requested_range in ranges_in:
        selected_range = float(
            ds_rp["range"].sel(range=requested_range, method="nearest").values.item()
        )
        dsd_profile = da.sel(time=t_sel, range=selected_range, method="nearest")
        values = dsd_profile.values.astype(float)

        valid = np.isfinite(values)
        if not np.any(valid):
            continue

        if data_is_log10:
            if use_log10:
                y_values = values
            else:
                y_values = 10.0 ** values
        else:
            positive = values > 0
            if not np.any(positive):
                continue
            if use_log10:
                y_values = np.log10(values[positive])
            else:
                y_values = values[positive]

        if y_values.shape != x_axis.shape:
            continue

        selected_ranges.append(selected_range)
        y_grid.append(y_values)

    if not selected_ranges:
        raise ValueError("No valid DSD curves found for the provided ranges/time.")

    if use_log10:
        y_label = r"$\log_{10}(N)\ [\mathrm{m^{-3}\,mm^{-1}}]$"
    else:
        y_label = r"$N\ [\mathrm{m^{-3}\,mm^{-1}}]$"


    X, Z = np.meshgrid(x_axis, np.asarray(selected_ranges))
    Y = np.vstack(y_grid)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=plt.get_cmap(cmap),
        edgecolor="none",
        antialiased=True,
        alpha=0.6,
    )
    fig.colorbar(surf, ax=ax, label=y_label)

    ax.set_xlabel(f"D [{diameter_units_out}]", fontsize=12, labelpad=8)
    ax.set_ylabel(y_label, fontsize=12, labelpad=8)
    ax.set_zlabel("Range (m)", fontsize=12, labelpad=8)
    # rotate the 3D view 130 degrees CW around the z axis
    ax.view_init(elev=20, azim=-130)

    ax.tick_params(labelsize=10)
    ax.title.set_fontsize(16)
    ax.legend(ncol=ncol, loc="best", fontsize=legend_fontsize)

    time_text = str(np.datetime_as_string(t_sel, unit="s"))
    ax.set_title(f"RaProMPro N(D) by range (3D)\n{time_text}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(ncol=ncol, loc="best", fontsize=legend_fontsize)

    if vmin is not None or vmax is not None:
        ax.set_ylim(vmin, vmax)
    if kwargs.get("xlimits") is not None:
        ax.set_xlim(kwargs["xlimits"])

    fig.tight_layout()

    filepath: Path | None = None
    if savefig:
        if output_dir is None:
            raise ValueError("output_dir must be provided if savefig=True.")
        output_dir.mkdir(parents=True, exist_ok=True)
        time_tag = str(np.datetime_as_string(t_sel, unit="s")).replace(":", "")
        filepath = output_dir / Path(subject.path).name.replace(
            ".nc", f"_DSD_by_range_{time_tag}.png"
        )
        fig.savefig(filepath, dpi=dpi)

    return fig, ax, filepath


def plot_microphysical_properties_profiles(
    subject: SupportsProcessedPlotting,
    target_datetime: datetime,
    savefig: bool = False,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> tuple[Figure, np.ndarray, Path | None]:
    """Plot a four-panel processed microphysical profile view."""
    if subject._is_processed():
        preprocessed_status = "RaProMPro-preprocessed"
    else:
        raise RuntimeError(
            "Dataset does not appear to be RaProMPro-preprocessed. Missing expected variables or attributes."
        )

    pcfg = subject.plot_cfg
    figsize = kwargs.get("figsize", pcfg.figsize_profiles)
    dpi = kwargs.get("dpi", pcfg.dpi)

    ds = subject.raprompro
    if ds is None:
        raise RuntimeError("raprompro not loaded. Use load_raprompro().")
    if "time" not in ds.coords:
        raise RuntimeError("No 'time' coordinate found in dataset.")

    profile = ds.sel(time=np.datetime64(target_datetime), method="nearest")
    selected_time = profile["time"].values
    try:
        selected_time_str = str(np.datetime_as_string(selected_time, unit="s"))
    except Exception:
        selected_time_str = str(selected_time)

    heights_km = profile["range"].values.astype(float) / 1000.0

    fig, axs = plt.subplots(
        ncols=4,
        figsize=figsize,
        sharey=True,
        constrained_layout=True,
    )

    ax = axs[0]
    reflectivity_variables = ["Ze", "Za", "Zea", "Z_all"]
    markers = {"Ze": "x", "Za": "v", "Zea": "o", "Z_all": "^"}
    for variable in reflectivity_variables:
        if variable not in profile.data_vars:
            continue
        ax.plot(
            profile[variable].values,
            heights_km,
            label=variable,
            linewidth=1,
            marker=markers[variable],
            markersize=4,
        )
    ax.set_xlabel("Reflectivities, dBZ")
    ax.set_ylabel("range (km)")
    ax.set_xlim(kwargs.get("x_limits", (0, 45)))
    ax.grid(True)
    ax.legend(loc="best")

    ax = axs[1]
    ax.plot(profile["Dm"].values, heights_km, linewidth=1, marker="o", markersize=4)
    ax.set_xlabel(r"$D_m$, mm")
    ax.set_xlim(kwargs.get("Dm_limits", (0.0, 4)))
    ax.grid(True)

    ax = axs[2]
    ax.plot(profile["Nw"].values, heights_km, linewidth=1, marker="o", markersize=4)
    ax.set_xlabel(r"$log_{10}(N_w \, mm^{-1} m^{-3})$")
    ax.set_xlim(kwargs.get("Nw_limits", (0.0, 6.0)))
    ax.grid(True)

    ax = axs[3]
    ax.plot(
        profile["LWC_all"].values,
        heights_km,
        linewidth=kwargs.get("LWC_all_linewidth", 5),
        marker=kwargs.get("LWC_all_marker", "o"),
        markersize=kwargs.get("LWC_all_markersize", 8),
        label="LWC_all",
        color=kwargs.get("LWC_all_color", "tab:blue"),
    )
    ax.plot(
        profile["LWC"].values,
        heights_km,
        linewidth=kwargs.get("LWC_linewidth", 1.25),
        marker=kwargs.get("LWC_marker", "."),
        markersize=kwargs.get("LWC_markersize", 7),
        label="LWC",
        color=kwargs.get("LWC_color", "tab:orange"),
    )
    ax.legend(loc="best")
    ax.set_xlabel(r"LWC, g m_^{-3}")
    ax.set_xlim(kwargs.get("LWC_limits", (0, 3.0)))
    ax.grid(True)

    y_limits = _resolve_height_limits_km(subject, kwargs.get("y_limits"))
    if y_limits is not None:
        for axis in axs:
            axis.set_ylim(*y_limits)

    fig.suptitle(f"{preprocessed_status} MRR-Pro \n {selected_time_str}", fontsize=30)

    output_path: Path | None = None
    if savefig:
        if output_dir is None:
            output_dir = Path.cwd()
        datestr = target_datetime.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / (
            f"{Path(subject.path).stem}_{datestr}_{preprocessed_status}_profiles.png"
        )
        fig.savefig(output_path, dpi=dpi)

    return fig, axs, output_path
