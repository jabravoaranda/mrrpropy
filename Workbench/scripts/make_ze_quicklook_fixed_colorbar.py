from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def main() -> None:
    product_path = Path(
        "workbench/output/raprompro/2025/10/29/20251029_190000_raprompro.nc"
    )
    output_dir = Path("workbench/output/poster")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        "ze_quicklook_20251029_190000_viridis_fixed_colorbar_A0_600dpi_transparent.png"
    )

    ds = xr.open_dataset(product_path)
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
        }
    )
    fig = plt.figure(figsize=(11.2, 7))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.13, 0.18, 0.60, 0.74])
    cax = fig.add_axes([0.79, 0.18, 0.045, 0.74])
    ax.patch.set_alpha(0.0)
    cax.patch.set_alpha(0.0)

    mesh = ax.pcolormesh(
        x,
        y,
        values,
        shading="auto",
        cmap="viridis",
        vmin=-10.0,
        vmax=35.0,
        rasterized=True,
    )
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.set_label("Ze [dBZ]", rotation=270, labelpad=42, fontsize=24)
    cbar.ax.tick_params(labelsize=24, width=1.6, length=6, colors="#262626")
    cbar.outline.set_edgecolor("#303030")
    cbar.outline.set_linewidth(1.4)

    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(0.65, 3.35)
    ax.set_xlabel("Time, UTC", labelpad=10)
    ax.set_ylabel("Range, [km agl]", labelpad=12)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(colors="#262626", labelsize=24, width=1.6, length=6)
    for spine in ax.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(1.6)
    ax.grid(False)

    fig.savefig(output_path, dpi=600, transparent=True)
    plt.close(fig)
    ds.close()
    print(output_path.resolve())


if __name__ == "__main__":
    main()
