from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import xarray as xr


class SupportsProfilePlotting(Protocol):
    path: str | Path
    raprompro: xr.Dataset | None
    plot_cfg: Any

    def _is_processed(self) -> bool: ...


def _resolve_height_limits_km(
    subject: SupportsProfilePlotting,
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


def plot_microphysical_properties_profiles(
    subject: SupportsProfilePlotting,
    target_datetime: datetime,
    savefig: bool = False,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> tuple[Figure, np.ndarray, Path | None]:
    """Plot a four-panel RaProMPro microphysical profile view."""
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
