from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from mrrpropy.raw_class import MRRProData
from mrrpropy.rain_process_classification.rain_process_algorithm import (
    sliding_rain_classification_to_dataframe,
)


def main() -> None:
    product_path = Path(
        "workbench/output/raprompro/2025/10/29/20251029_190000_raprompro.nc"
    )
    output_dir = Path("workbench/output/poster")
    output_dir.mkdir(parents=True, exist_ok=True)

    k = 11
    window_thickness_m = 500.0
    window_step_m = 35.0
    min_tau_strength = 0.3
    ze_th = -5.0
    min_points_trend = 10

    print(f"[load] {product_path}", flush=True)
    ds = xr.open_dataset(product_path)
    mrr = MRRProData(product_path, ds)
    mrr.raprompro = ds

    period = (
        pd.Timestamp(ds["time"].values[0]).floor("s").to_pydatetime(),
        pd.Timestamp(ds["time"].values[-1]).floor("s").to_pydatetime(),
    )
    print(
        "[classify] "
        f"k={k}, window={window_thickness_m:g} m, step={window_step_m:g} m, "
        f"min_tau_strength={min_tau_strength:g}, ze_th={ze_th:g}, "
        f"min_points_trend={min_points_trend}",
        flush=True,
    )
    sliding = mrr.sliding_rain_classification(
        period=period,
        k=k,
        window_thickness_m=window_thickness_m,
        window_step_m=window_step_m,
        min_tau_strength=min_tau_strength,
        ze_th=ze_th,
        min_points_trend=min_points_trend,
        vars_trend=("Dm", "Nw", "LWC"),
    )
    frame = sliding_rain_classification_to_dataframe(sliding)
    frame["time"] = pd.to_datetime(frame["time"])
    frame.attrs.update(getattr(sliding, "attrs", {}))
    print(f"[classify] rows={len(frame)}", flush=True)

    csv_path = output_dir / "poster_column_process_recomputed_minTau03_20251029_190000.csv"
    frame.to_csv(csv_path, index=False)

    counts = frame.groupby("proc_label", dropna=False).size().sort_values(ascending=False)
    counts_path = (
        output_dir / "poster_column_process_recomputed_minTau03_counts_20251029_190000.csv"
    )
    counts.rename("count").to_csv(counts_path)
    print("[counts]", flush=True)
    print(counts.to_string(), flush=True)

    preferred_processes = [
        "steady_or_weak",
        "coalescence_loss",
        "breakup_gain",
        "breakup_loss",
        "coalescence_gain",
        "activation",
        "evaporation_weak",
        "evaporation_strong",
        "coalescence",
        "breakup",
        "growth",
    ]
    present = set(frame["proc_label"].astype(str).unique())
    processes = [process for process in preferred_processes if process in present]

    process_colors = {
        "steady_or_weak": "#8f8f8f",
        "coalescence_loss": "#ff2a2a",
        "breakup_gain": "#00d6d6",
        "breakup_loss": "#22c95a",
        "coalescence_gain": "#ec31d9",
        "activation": "#62cc32",
        "evaporation_weak": "#101010",
        "evaporation_strong": "#101010",
        "coalescence": "#e31a1c",
        "breakup": "#12af54",
        "growth": "#91209b",
    }

    fig, ax, _ = mrr.plot_sliding_column_process(
        sliding_df=frame,
        processes=processes,
        process_colors=process_colors,
        color_mode="rain_signature",
        marker_mode="process",
        render_mode="markers",
        alpha=0.95,
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
    )
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors="#262626", labelsize=24)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for spine in ax.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(1.8)
    ax.grid(True, which="major", color="#d8d8d8", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.grid(False, which="minor")

    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_alpha(0.0)
        legend.get_frame().set_edgecolor("none")
        for text in legend.get_texts():
            text.set_color("#303030")

    png_path = (
        output_dir
        / "poster_column_process_recomputed_minTau03_20251029_190000_A0_600dpi_transparent.png"
    )
    fig.savefig(png_path, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    ds.close()
    print(png_path.resolve(), flush=True)
    print(csv_path.resolve(), flush=True)
    print(counts_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
