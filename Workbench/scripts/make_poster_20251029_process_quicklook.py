from __future__ import annotations

from pathlib import Path

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

    ds = xr.open_dataset(product_path)
    mrr = MRRProData(product_path, ds)
    mrr.raprompro = ds

    period = (
        pd.Timestamp(ds["time"].values[0]).floor("s").to_pydatetime(),
        pd.Timestamp(ds["time"].values[-1]).floor("s").to_pydatetime(),
    )
    sliding = mrr.sliding_rain_classification(
        period=period,
        k=11,
        window_thickness_m=500.0,
        window_step_m=None,
        min_tau_strength=0.5,
        ze_th=-5.0,
        min_points_trend=10,
        vars_trend=("Dm", "Nw", "LWC"),
    )
    frame = sliding_rain_classification_to_dataframe(sliding)
    frame["time"] = pd.to_datetime(frame["time"])

    csv_path = output_dir / "poster_column_process_native35_new_scheme_20251029_190000.csv"
    frame.to_csv(csv_path, index=False)

    processes = [
        "steady_or_weak",
        "coalescence",
        "coalescence_loss",
        "coalescence_gain",
        "breakup",
        "breakup_loss",
        "breakup_gain",
        "activation",
        "evaporation_weak",
        "evaporation_strong",
        "growth",
    ]
    present = set(frame["proc_label"].astype(str).unique())
    selected = [label for label in processes if label in present]

    fig, ax, _ = mrr.plot_sliding_column_process(
        sliding_df=frame,
        processes=selected,
        color_mode="rain_signature",
        marker_mode="square",
        render_mode="cells",
        cell_gap=0.10,
        scale_by_strength=False,
        alpha=0.95,
        figsize=(22, 10),
        dpi=600,
        title_fs=24,
        label_fs=24,
        tick_fs=22,
        legend_fs=18,
        legend_ncol=4,
        legend_loc="upper left",
        y_limits=(0.75, 3.35),
        savefig=False,
    )
    ax.set_title("")
    ax.set_xlabel("Time", fontsize=24)
    ax.set_ylabel("Height [km]", fontsize=24)

    png_path = output_dir / "poster_column_process_quicklook_new_scheme_20251029_190000_A0.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    counts_path = output_dir / "poster_column_process_counts_new_scheme_20251029_190000.csv"
    (
        frame.groupby("proc_label", dropna=False)
        .size()
        .sort_values(ascending=False)
        .rename("count")
        .to_csv(counts_path)
    )

    print(png_path.resolve())
    print(csv_path.resolve())
    print(counts_path.resolve())


if __name__ == "__main__":
    main()
