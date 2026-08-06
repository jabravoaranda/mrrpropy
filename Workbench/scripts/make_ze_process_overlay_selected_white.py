from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from mrrpropy.plotting.processes import plot_sliding_column_process


def main() -> None:
    output_dir = Path("workbench/output/poster/ze_process_overlay_viridis_ze_alpha_series_white")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        "ze_process_overlay_viridis_zeAlpha0.20_procAlpha1.00_"
        "20251029_190000_A0_600dpi_panelwhite_canvastransparent_11of14_labels24.png"
    )

    ds = xr.open_dataset(
        "workbench/output/raprompro/2025/10/29/20251029_190000_raprompro.nc"
    )
    frame = pd.read_csv(
        "workbench/output/poster/poster_column_process_recomputed_minTau03_20251029_190000.csv"
    )
    frame["time"] = pd.to_datetime(frame["time"])
    frame.attrs.update(
        {
            "period_start": frame["time"].min().isoformat(),
            "period_end": frame["time"].max().isoformat(),
            "window_thickness_m": 500.0,
            "window_step_m": 35.0,
            "min_tau_strength": 0.3,
            "k": 11,
        }
    )

    ze = ds["Ze"]
    if tuple(ze.dims) != ("time", "range"):
        ze = ze.transpose("time", "range")
    times = pd.to_datetime(ze["time"].values)
    ranges_km = np.asarray(ze["range"].values, dtype=float) / 1000.0
    x = mdates.date2num(times.to_pydatetime())
    y = ranges_km
    values = np.asarray(ze.values, dtype=float).T

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 20,
        }
    )
    figsize = (22.0 * 11.0 / 14.0, 10.0)
    subject = SimpleNamespace(
        plot_cfg=SimpleNamespace(dpi=600, figsize_profiles=figsize),
        raprompro=ds,
    )

    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.14, 0.18, 0.78, 0.74])
    ax.set_facecolor("white")
    ax.pcolormesh(
        x,
        y,
        values,
        shading="auto",
        cmap="viridis",
        vmin=-10.0,
        vmax=40.0,
        alpha=0.20,
        rasterized=True,
    )

    plot_frame = frame.copy()
    plot_frame["proc_label"] = plot_frame["proc_label"].replace(
        {"evaporation_strong": "evaporation", "evaporation_weak": "evaporation"}
    )
    plot_sliding_column_process(
        subject,
        sliding_df=plot_frame,
        processes=[
            "steady_or_weak",
            "coalescence_loss",
            "breakup_gain",
            "breakup_loss",
            "coalescence_gain",
            "activation",
            "evaporation",
        ],
        process_colors={
            "steady_or_weak": "#8f8f8f",
            "coalescence_loss": "#ff2a2a",
            "breakup_gain": "#00d6d6",
            "breakup_loss": "#22c95a",
            "coalescence_gain": "#ec31d9",
            "activation": "#62cc32",
            "evaporation": "#101010",
        },
        color_mode="rain_signature",
        marker_mode="process",
        render_mode="markers",
        alpha=1.0,
        markersize=38,
        scale_by_strength=False,
        figsize=(17.6, 10),
        dpi=600,
        title_fs=24,
        label_fs=24,
        tick_fs=24,
        legend_fs=20,
        legend_ncol=4,
        legend_loc="upper left",
        y_limits=(0.75, 3.35),
        show_legend=True,
        savefig=False,
        ax=ax,
    )

    ax.set_title("")
    ax.set_xlabel("Time, UTC", fontsize=24, labelpad=10)
    ax.set_ylabel("Range, [km agl]", fontsize=24, labelpad=12)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(colors="#262626", labelsize=24, width=1.6, length=6)
    for spine in ax.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(1.8)
    ax.grid(True, which="major", color="#d8d8d8", linestyle="-", linewidth=0.8, alpha=0.55)
    ax.grid(False, which="minor")

    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("white")
        for text in legend.get_texts():
            text.set_text(text.get_text().replace("_", " "))
            text.set_color("#303030")

    fig.savefig(output_path, dpi=600, facecolor=fig.get_facecolor())
    plt.close(fig)
    ds.close()
    print(output_path.resolve())


if __name__ == "__main__":
    main()
