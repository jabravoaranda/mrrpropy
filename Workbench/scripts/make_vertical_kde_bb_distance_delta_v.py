from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from mrrpropy.plotting.paper import _approximate_kde_domain, _paper_grid


def main() -> None:
    output_dir = Path("workbench/output/bimonthly_process_stats_2025_03_04")
    input_csv = output_dir / "combined_column_process_scan.csv"
    output_png = (
        output_dir
        / "kde_bb_distance_delta_v_by_process_vertical_A0_font24_transparent.png"
    )

    usecols = ["proc_label", "bb_distance_m", "delta_v_mean"]
    frame = pd.read_csv(input_csv, usecols=usecols)
    frame["proc_label"] = frame["proc_label"].astype(str).replace(
        {"condensation": "activation"}
    )
    for column in ("bb_distance_m", "delta_v_mean"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    process_order = [
        "breakup_loss",
        "activation",
        "evaporation_strong",
        "coalescence_gain",
        "coalescence_loss",
        "breakup_gain",
    ]
    labels = {
        "breakup_loss": "BU-LOSS",
        "activation": "ACTIV.",
        "evaporation_strong": "EVAP.-STRONG",
        "coalescence_gain": "COAL.-GAIN",
        "coalescence_loss": "COAL.-LOSS",
        "breakup_gain": "BU-GAIN",
    }
    colors = {
        "breakup_loss": "#24ca24",
        "activation": "#ff7f00",
        "evaporation_strong": "#000000",
        "coalescence_gain": "#fb9a99",
        "coalescence_loss": "#a50f15",
        "breakup_gain": "#13d7d7",
    }

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 18,
        }
    )
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10.5, 16.5), constrained_layout=False)
    fig.patch.set_alpha(0.0)

    plot_specs = [
        ("bb_distance_m", "BB distance [m]"),
        ("delta_v_mean", r"$\Delta v$ [m s$^{-1}$]"),
    ]
    for ax, (column, xlabel) in zip(axes, plot_specs):
        ax.patch.set_alpha(0.0)
        plot_df = frame[["proc_label", column]].copy()
        plot_df = plot_df[np.isfinite(plot_df[column])]
        lower, upper = _approximate_kde_domain(
            plot_df,
            value_col=column,
            process_col="proc_label",
            sigma=3.0,
            fallback_quantiles=(0.005, 0.995),
        )
        for process in process_order:
            values = plot_df.loc[plot_df["proc_label"] == process, column]
            if values.nunique(dropna=True) < 2:
                continue
            sns.kdeplot(
                x=values,
                color=colors[process],
                clip=(lower, upper),
                fill=False,
                linewidth=3.2,
                ax=ax,
                legend=False,
                warn_singular=False,
            )
        ax.set_xlim(lower, upper)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.tick_params(axis="both", labelsize=24, width=1.8, length=7)
        ax.tick_params(axis="both", which="minor", width=1.2, length=4)
        _paper_grid(ax)
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
            spine.set_color("#333333")

    handles = [
        Line2D([0], [0], color=colors[process], lw=3.2, label=labels[process])
        for process in process_order
        if process in set(frame["proc_label"])
    ]
    axes[0].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.04, 0.98),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="none",
        facecolor="none",
        fontsize=18,
        handlelength=2.2,
        columnspacing=1.6,
        borderpad=0.7,
    )

    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.07, hspace=0.24)
    fig.savefig(output_png, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(output_png.resolve())


if __name__ == "__main__":
    main()
