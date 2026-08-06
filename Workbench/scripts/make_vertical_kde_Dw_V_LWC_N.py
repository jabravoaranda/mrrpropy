from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrrpropy.plotting.paper import VARIABLE_LABELS, _approximate_kde_domain, _paper_grid


def _resolve_var(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise KeyError(f"None of these columns were found: {', '.join(candidates)}")


def main() -> None:
    output_dir = Path("workbench/output/bimonthly_process_stats_2025_03_04")
    input_csv = output_dir / "combined_column_process_scan.csv"
    output_png = (
        output_dir
        / "kde_Dw_V_LWC_N_by_process_vertical_A0_font24_transparent.png"
    )

    aliases = {
        "Dw": ("Dw", "Dm", "Dm_top", "Dm_layer_mean", "Dm_mean"),
        "V": (
            "V",
            "W",
            "VEL",
            "v_mean_top",
            "v_mean_bottom",
            "V_layer_mean",
            "W_layer_mean",
            "VEL_layer_mean",
        ),
        "LWC": ("LWC", "LWC_top", "LWC_bottom", "LWC_layer_mean", "LWC_mean"),
        "N": (
            "N",
            "Nw",
            "Nw_top",
            "Nw_bottom",
            "N_layer_mean",
            "Nw_layer_mean",
            "Nw_mean",
        ),
    }
    variables = ("Dw", "V", "LWC", "N")
    usecols = ["proc_label", *{name for variable in variables for name in aliases[variable]}]
    header = pd.read_csv(input_csv, nrows=0)
    frame = pd.read_csv(input_csv, usecols=[name for name in usecols if name in header.columns])
    frame["proc_label"] = frame["proc_label"].astype(str).replace(
        {"condensation": "activation"}
    )

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
    variable_limits = {"LWC": (0.0, 0.3)}

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 18,
        }
    )
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(10.5, 30.0),
        constrained_layout=False,
    )
    fig.patch.set_alpha(0.0)

    for ax, variable in zip(axes, variables):
        ax.patch.set_alpha(0.0)
        column = _resolve_var(frame, aliases[variable])
        plot_df = frame[["proc_label", column]].copy()
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
        plot_df = plot_df[np.isfinite(plot_df[column])]
        lower, upper = _approximate_kde_domain(
            plot_df,
            value_col=column,
            process_col="proc_label",
            sigma=3.0,
            fallback_quantiles=(0.005, 0.995),
        )
        if variable in variable_limits:
            explicit_lower, explicit_upper = variable_limits[variable]
            lower = float(explicit_lower) if explicit_lower is not None else lower
            upper = float(explicit_upper) if explicit_upper is not None else upper

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
        ax.set_xlabel(VARIABLE_LABELS.get(variable, VARIABLE_LABELS.get(column, column)))
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

    fig.subplots_adjust(left=0.16, right=0.98, top=0.99, bottom=0.045, hspace=0.32)
    fig.savefig(output_png, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(output_png.resolve())


if __name__ == "__main__":
    main()
