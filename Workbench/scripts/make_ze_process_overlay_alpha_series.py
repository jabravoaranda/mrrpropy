from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from mrrpropy.plotting.processes import plot_sliding_column_process


def _plot_one(
    *,
    ds: xr.Dataset,
    frame: pd.DataFrame,
    output_path: Path,
    ze_alpha: float,
) -> None:
    subject = SimpleNamespace(
        plot_cfg=SimpleNamespace(dpi=600, figsize_profiles=(22, 10)),
        raprompro=ds,
    )

    fig, ax = plt.subplots(figsize=(22, 10), constrained_layout=True)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    ze = ds["Ze"]
    if tuple(ze.dims) != ("time", "range"):
        ze = ze.transpose("time", "range")
    times = pd.to_datetime(ze["time"].values)
    ranges_km = np.asarray(ze["range"].values, dtype=float) / 1000.0
    x = mdates.date2num(times.to_pydatetime())
    y = ranges_km
    values = np.asarray(ze.values, dtype=float).T
    ax.pcolormesh(
        x,
        y,
        values,
        shading="auto",
        cmap="viridis",
        vmin=-10.0,
        vmax=40.0,
        alpha=ze_alpha,
        rasterized=True,
    )

    plot_frame = frame.copy()
    plot_frame["proc_label"] = plot_frame["proc_label"].replace(
        {"evaporation_strong": "evaporation", "evaporation_weak": "evaporation"}
    )
    processes = [
        "steady_or_weak",
        "coalescence_loss",
        "breakup_gain",
        "breakup_loss",
        "coalescence_gain",
        "activation",
        "evaporation",
    ]
    process_colors = {
        "steady_or_weak": "#8f8f8f",
        "coalescence_loss": "#ff2a2a",
        "breakup_gain": "#00d6d6",
        "breakup_loss": "#22c95a",
        "coalescence_gain": "#ec31d9",
        "activation": "#62cc32",
        "evaporation": "#101010",
    }
    plot_sliding_column_process(
        subject,
        sliding_df=plot_frame,
        processes=processes,
        process_colors=process_colors,
        color_mode="rain_signature",
        marker_mode="process",
        render_mode="markers",
        alpha=1.0,
        markersize=38,
        scale_by_strength=False,
        figsize=(22, 10),
        dpi=600,
        title_fs=24,
        label_fs=24,
        tick_fs=24,
        legend_fs=20,
        legend_ncol=2,
        legend_loc="upper left",
        y_limits=(0.75, 3.35),
        show_legend=True,
        savefig=False,
        ax=ax,
    )

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(colors="#262626", labelsize=24)
    for spine in ax.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(1.8)
    ax.grid(True, which="major", color="#d8d8d8", linestyle="-", linewidth=0.8, alpha=0.55)
    ax.grid(False, which="minor")

    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_alpha(0.0)
        legend.get_frame().set_edgecolor("none")
        for text in legend.get_texts():
            text.set_color("#303030")

    fig.savefig(output_path, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    output_dir = Path("workbench/output/poster/ze_process_overlay_viridis_ze_alpha_series")
    output_dir.mkdir(parents=True, exist_ok=True)
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

    variants = [0.60, 0.50, 0.40, 0.30, 0.20]
    for ze_alpha in variants:
        path = output_dir / (
            "ze_process_overlay_"
            f"viridis_zeAlpha{ze_alpha:.2f}_procAlpha1.00_"
            "20251029_190000_A0_600dpi_transparent.png"
        )
        print(f"[plot] {path}", flush=True)
        _plot_one(
            ds=ds,
            frame=frame,
            output_path=path,
            ze_alpha=ze_alpha,
        )
    ds.close()
    print(output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
