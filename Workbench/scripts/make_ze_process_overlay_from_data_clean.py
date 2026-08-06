from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D


PRODUCT_PATH = Path("workbench/output/raprompro/2025/10/29/20251029_190000_raprompro.nc")
PROCESS_CSV = Path("workbench/output/poster/poster_column_process_recomputed_minTau03_20251029_190000.csv")
OUTPUT_DIR = Path("workbench/output/poster/clean_from_data")


PROCESS_ORDER = [
    "steady_or_weak",
    "coalescence_loss",
    "breakup_gain",
    "breakup_loss",
    "coalescence_gain",
    "activation",
    "evaporation",
]

PROCESS_STYLE = {
    "steady_or_weak": {"label": "steady or weak", "color": "#8f8f8f", "marker": "s", "size": 8},
    "coalescence_loss": {"label": "coalescence loss", "color": "#ff2a2a", "marker": "d", "size": 12},
    "breakup_gain": {"label": "breakup gain", "color": "#00d6d6", "marker": "P", "size": 14},
    "breakup_loss": {"label": "breakup loss", "color": "#22c95a", "marker": "o", "size": 14},
    "coalescence_gain": {"label": "coalescence gain", "color": "#ec31d9", "marker": "D", "size": 16},
    "activation": {"label": "activation", "color": "#62cc32", "marker": "*", "size": 18},
    "evaporation": {"label": "evaporation", "color": "#101010", "marker": "x", "size": 18},
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / (
        "ze_process_overlay_from_data_viridis_zeAlpha0.20_procAlpha1.00_"
        "20251029_190000_A0_600dpi_panelwhite_canvastransparent_11of14_labels24.png"
    )

    ds = xr.open_dataset(PRODUCT_PATH)
    ze = ds["Ze"]
    if tuple(ze.dims) != ("time", "range"):
        ze = ze.transpose("time", "range")

    times = pd.to_datetime(ze["time"].values)
    ranges_km = np.asarray(ze["range"].values, dtype=float) / 1000.0
    x = mdates.date2num(times.to_pydatetime())
    y = ranges_km
    ze_values = np.asarray(ze.values, dtype=float).T

    frame = pd.read_csv(PROCESS_CSV)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["range_km"] = pd.to_numeric(frame["range"], errors="coerce") / 1000.0
    frame["proc_label"] = frame["proc_label"].replace(
        {"evaporation_strong": "evaporation", "evaporation_weak": "evaporation"}
    )
    frame = frame[
        frame["proc_label"].isin(PROCESS_ORDER)
        & frame["time"].notna()
        & np.isfinite(frame["range_km"])
    ].copy()

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 18,
        }
    )

    figsize = (22.0 * 11.0 / 14.0, 10.0)
    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.14, 0.18, 0.78, 0.74])
    ax.set_facecolor("white")

    ax.pcolormesh(
        x,
        y,
        ze_values,
        shading="auto",
        cmap="viridis",
        vmin=-10.0,
        vmax=40.0,
        alpha=0.20,
        rasterized=True,
    )

    for process in PROCESS_ORDER:
        style = PROCESS_STYLE[process]
        subset = frame[frame["proc_label"] == process]
        if subset.empty:
            continue
        ax.scatter(
            subset["time"],
            subset["range_km"],
            s=style["size"],
            marker=style["marker"],
            c=style["color"],
            alpha=1.0,
            linewidths=0.8 if style["marker"] == "x" else 0.0,
            edgecolors=style["color"] if style["marker"] == "x" else "none",
            rasterized=True,
        )

    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(0.75, 3.35)
    ax.set_xlabel("Time, UTC", fontsize=24, labelpad=10)
    ax.set_ylabel("Range, [km agl]", fontsize=24, labelpad=12)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(colors="#262626", labelsize=24, width=1.6, length=6)
    for spine in ax.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(1.8)
    ax.grid(True, which="major", color="#d8d8d8", linestyle="-", linewidth=0.8, alpha=0.55)
    ax.grid(False, which="minor")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=PROCESS_STYLE[process]["marker"],
            color="none",
            markerfacecolor=PROCESS_STYLE[process]["color"],
            markeredgecolor=PROCESS_STYLE[process]["color"],
            markersize=8,
            label=PROCESS_STYLE[process]["label"],
        )
        for process in PROCESS_ORDER
        if process in set(frame["proc_label"])
    ]
    legend = ax.legend(
        handles=legend_handles,
        ncol=4,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="white",
        columnspacing=1.5,
        handletextpad=0.5,
        borderpad=0.5,
    )
    for text in legend.get_texts():
        text.set_color("#303030")

    fig.savefig(output_path, dpi=600, facecolor=fig.get_facecolor())
    plt.close(fig)
    ds.close()
    print(output_path.resolve())


if __name__ == "__main__":
    main()
