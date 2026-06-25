from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import xarray as xr


class SupportsQuicklookPlotting(Protocol):
    path: str | Path
    ds: xr.Dataset
    raprompro: xr.Dataset | None
    plot_cfg: Any


def quicklook(
    subject: SupportsQuicklookPlotting,
    variable: str = "Ze",
    source: str = "raprompro",
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a quick time-height view of a raw or RaProMPro 2D field."""
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
    return fig, ax
