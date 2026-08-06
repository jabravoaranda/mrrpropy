from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

from mrrpropy.plotting.processes import plot_sliding_column_process


def main() -> None:
    output_dir = Path("workbench/output/poster")
    source_csv = output_dir / "poster_column_process_native35_20251029_190000.csv"
    frame = pd.read_csv(source_csv)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["proc_label"] = frame["proc_label"].replace({"condensation": "activation"})
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

    updated_csv = output_dir / "poster_column_process_native35_new_names_20251029_190000.csv"
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
        "evaporation_strong",
        "evaporation_weak",
        "coalescence",
        "breakup",
        "growth",
    ]
    present = set(frame["proc_label"].astype(str).unique())
    selected = [label for label in processes if label in present]

    fig, ax, _ = plot_sliding_column_process(
        subject,
        sliding_df=frame,
        processes=selected,
        color_mode="rain_signature",
        marker_mode="square",
        render_mode="cells",
        cell_gap=0.12,
        alpha=0.96,
        scale_by_strength=False,
        figsize=(22, 10),
        dpi=600,
        title_fs=24,
        label_fs=24,
        tick_fs=22,
        legend_fs=19,
        legend_ncol=4,
        legend_loc="upper left",
        y_limits=(0.75, 3.35),
        show_legend=True,
        savefig=False,
    )

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors="#303030", labelsize=24)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.8)
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.75, alpha=0.85)
    ax.grid(False, which="minor")
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_edgecolor("black")
        for text in legend.get_texts():
            text.set_color("#303030")

    png_path = output_dir / "poster_column_process_quicklook_new_scheme_20251029_190000_A0_600dpi.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    counts_path = output_dir / "poster_column_process_counts_new_names_20251029_190000.csv"
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
