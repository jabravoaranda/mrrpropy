from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from mrrpropy.plotting.processes import plot_sliding_column_process


def main() -> None:
    output_dir = Path("workbench/output/poster")
    source_csv = output_dir / "poster_column_process_native35_20251029_190000.csv"
    frame = pd.read_csv(source_csv)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["proc_label"] = frame["proc_label"].replace(
        {
            "condensation": "activation",
            "evaporation_strong": "evaporation",
            "evaporation_weak": "evaporation",
        }
    )
    frame.attrs.update(
        {
            "period_start": frame["time"].min().isoformat(),
            "period_end": frame["time"].max().isoformat(),
            "window_thickness_m": 500.0,
            "window_step_m": 35.0,
            "min_tau_strength": 0.5,
            "k": 11,
        }
    )

    updated_csv = output_dir / "poster_column_process_marker_new_names_20251029_190000.csv"
    frame.to_csv(updated_csv, index=False)

    subject = SimpleNamespace(
        plot_cfg=SimpleNamespace(dpi=600, figsize_profiles=(22, 10)),
        raprompro=None,
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
    fig, ax, _ = plot_sliding_column_process(
        subject,
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
        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_edgecolor("black")
        for text in legend.get_texts():
            text.set_color("#303030")

    png_path = output_dir / "poster_column_process_marker_quicklook_new_scheme_20251029_190000_A0_600dpi_transparent.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)

    counts_path = output_dir / "poster_column_process_marker_counts_new_names_20251029_190000.csv"
    (
        frame.groupby("proc_label", dropna=False)
        .size()
        .sort_values(ascending=False)
        .rename("count")
        .to_csv(counts_path)
    )
    print(png_path.resolve())
    print(updated_csv.resolve())
    print(counts_path.resolve())


if __name__ == "__main__":
    main()
