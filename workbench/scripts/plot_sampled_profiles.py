from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import xarray as xr

from mrrpropy import MRRProData

matplotlib.use("Agg")


DEFAULT_INPUT_FILE = Path(
    r"workbench/output/raprompro/2025/10/29/20251029_190000_raprompro.nc"
)
DEFAULT_OUTPUT_DIR = Path(r"workbench/output/profiles/2025/10/29")


def _select_sampled_times(ds: xr.Dataset, *, step_minutes: int) -> xr.DataArray:
    times = ds["time"]
    if times.size == 0:
        raise RuntimeError("The dataset has no time coordinate.")

    start = times.values[0].astype("datetime64[s]")
    end = times.values[-1].astype("datetime64[s]")
    step = np.timedelta64(step_minutes, "m")

    targets = np.arange(start, end + step, step, dtype="datetime64[s]")
    selected_indices: list[int] = []

    for target in targets:
        idx = int(np.argmin(np.abs(times.values - target)))
        if not selected_indices or idx != selected_indices[-1]:
            selected_indices.append(idx)

    return times.isel(time=selected_indices)


def _has_visible_line(values: np.ndarray) -> bool:
    return int(np.isfinite(values).sum()) >= 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RaProMPro profile diagnostics sampled every N minutes."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to the processed *_raprompro.nc file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the PNG figure will be written.",
    )
    parser.add_argument(
        "--step-minutes",
        type=int,
        default=10,
        help="Sampling interval in minutes between plotted profiles.",
    )
    parser.add_argument(
        "--x-limits",
        nargs=2,
        type=float,
        default=(0.0, 45.0),
        help="X limits for the reflectivity panel.",
    )
    parser.add_argument(
        "--lwc-limits",
        nargs=2,
        type=float,
        default=(0.0, 1.0),
        help="X limits for the LWC panel.",
    )
    parser.add_argument(
        "--target-datetime",
        type=str,
        default=None,
        help=(
            "Optional target datetime for DSD/spectrogram products. "
            "If omitted, the script uses the last time available in the file."
        ),
    )
    parser.add_argument(
        "--ranges",
        nargs="*",
        type=float,
        default=[500.0, 1000.0, 1500.0, 2000.0, 2500.0],
        help="Ranges in metres for the DSD_by_range figure.",
    )
    args = parser.parse_args()

    input_file = args.input_file.resolve()
    output_dir = args.output_dir.resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if args.step_minutes < 1:
        raise ValueError("--step-minutes must be at least 1.")

    output_dir.mkdir(parents=True, exist_ok=True)

    mrr = MRRProData.from_file(input_file)
    try:
        # The file is already a processed product; reuse the same dataset as raprompro.
        mrr.raprompro = mrr.ds
        ds = mrr.raprompro
        assert ds is not None

        for required in ("Ze", "Dm", "Nw", "LWC"):
            if required not in ds:
                raise KeyError(f"Required processed variable '{required}' not found.")

        sampled_times = _select_sampled_times(ds, step_minutes=args.step_minutes)
        z_km = ds["range"].values.astype(float) / 1000.0

        fig, axs = plt.subplots(
            ncols=4,
            figsize=(18, 10),
            sharey=True,
        )
        fig.subplots_adjust(top=0.78)

        colors = plt.cm.viridis(np.linspace(0.05, 0.95, sampled_times.size))

        plotted_time_entries: list[tuple[np.ndarray, np.datetime64]] = []

        for color, time_value in zip(colors, sampled_times.values, strict=True):
            prof = ds.sel(time=time_value)

            has_visible_profile = any(
                _has_visible_line(np.asarray(prof[var_name].values, dtype=float))
                for var_name in ("Ze", "Dm", "Nw", "LWC")
                if var_name in prof
            )
            if not has_visible_profile:
                continue

            plotted_time_entries.append((color, time_value))

            axs[0].plot(prof["Ze"].values, z_km, color=color, linewidth=1.2)
            # axs[0].plot(prof["Zea"].values, z_km, color=color, linewidth=1.0, linestyle="--")
            # axs[0].plot(prof["Za"].values, z_km, color=color, linewidth=1.0, linestyle=":")

            # if "Z_all" in prof:
            #     axs[0].plot(
            #         prof["Z_all"].values,
            #         z_km,
            #         color=color,
            #         linewidth=0.8,
            #         linestyle="-.",
            #     )

            axs[1].plot(prof["Dm"].values, z_km, color=color, linewidth=1.2)
            axs[2].plot(prof["Nw"].values, z_km, color=color, linewidth=1.2)
            # axs[3].plot(prof["LWC_all"].values, z_km, color=color, linewidth=2.0)
            axs[3].plot(prof["LWC"].values, z_km, color=color, linewidth=1.0, linestyle="-")

        axs[0].set_xlabel("Reflectivities, dBZ")
        axs[0].set_ylabel("Range (km)")
        axs[0].set_xlim(tuple(args.x_limits))
        axs[0].grid(True)

        style_handles = [
            Line2D([0], [0], color="0.2", linewidth=1.2, linestyle="-", label="Ze")
        ]
        if len(style_handles) > 1:
            style_legend = axs[0].legend(
                handles=style_handles,
                loc="upper left",
                fontsize=9,
                title="Reflectivity",
            )
            axs[0].add_artist(style_legend)

        axs[1].set_xlabel(r"$D_m$, mm")
        axs[1].set_xlim((0.0, 4.0))
        axs[1].grid(True)

        axs[2].set_xlabel(r"$log_{10}(N_w \, mm^{-1} m^{-3})$")
        axs[2].set_xlim((0.0, 6.0))
        axs[2].grid(True)

        axs[3].set_xlabel(r"LWC, g m$^{-3}$")
        axs[3].set_xlim(tuple(args.lwc_limits))
        axs[3].grid(True)

        if not plotted_time_entries:
            raise RuntimeError("No finite profiles were found for the selected sampling.")

        t0 = plotted_time_entries[0][1]
        t1 = plotted_time_entries[-1][1]
        t0s = np.datetime_as_string(t0, unit="s").replace("-", "").replace(":", "").replace("T", "_")
        t1s = np.datetime_as_string(t1, unit="s").replace("-", "").replace(":", "").replace("T", "_")
        title_start = np.datetime_as_string(t0, unit="m").replace("T", " ")
        title_end = np.datetime_as_string(t1, unit="m").replace("T", " ")
        fig.suptitle(
            (
                "RaProMPro preprocessed profiles\n"
                f"{title_start} to {title_end}"
            ),
            fontsize=18,
            y=0.95,
        )

        plotted_colors = np.asarray([color for color, _ in plotted_time_entries], dtype=float)
        plotted_times = [time_value for _, time_value in plotted_time_entries]
        time_cmap = ListedColormap(plotted_colors)
        time_norm = BoundaryNorm(np.arange(-0.5, len(plotted_time_entries) + 0.5, 1.0), time_cmap.N)
        scalar_mappable = plt.cm.ScalarMappable(norm=time_norm, cmap=time_cmap)
        scalar_mappable.set_array(np.arange(len(plotted_time_entries), dtype=float))

        cax = fig.add_axes([0.20, 0.86, 0.60, 0.020])
        cbar = fig.colorbar(
            scalar_mappable,
            cax=cax,
            orientation="horizontal",
        )
        tick_count = min(8, len(plotted_time_entries))
        tick_positions = np.linspace(0, len(plotted_time_entries) - 1, tick_count, dtype=int)
        tick_positions = np.unique(tick_positions)
        cbar.set_ticks(tick_positions.tolist())
        cbar.set_ticklabels(
            [np.datetime_as_string(plotted_times[index], unit="m")[11:16] for index in tick_positions]
        )
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(f"Profile time ({args.step_minutes} min)", fontsize=10)

        output_name = f"{t0s}_{args.step_minutes}min_{t1s}_RaProMPro-preprocessed_profiles.png"
        output_path = output_dir / output_name
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        if args.target_datetime is None:
            target_datetime = ds["time"].values[-1]
        else:
            target_datetime = np.datetime64(args.target_datetime)

        available_ranges = ds["range"].values.astype(float)
        valid_ranges = [r for r in args.ranges if available_ranges.min() <= r <= available_ranges.max()]

        if "dsd_3D" in ds:
            fig_dsd_by_range, path_dsd_by_range = mrr.plot.dsd_by_range(
                target_datetime=target_datetime,
                ranges=valid_ranges,
                savefig=True,
                output_dir=output_dir,
                dpi=180,
            )
            plt.close(fig_dsd_by_range)
            print(f"DSD_by_range: {path_dsd_by_range}")

            fig_dsdgram, path_dsdgram = mrr.plot.dsdgram(
                target_datetime=target_datetime,
                savefig=True,
                output_dir=output_dir,
                dpi=180,
            )
            plt.close(fig_dsdgram)
            print(f"DSDgram     : {path_dsdgram}")
        else:
            print("[skip] dsd_3D not present in the processed file; DSD figures were not generated.")

        if "spe_3D" in ds:
            fig_spectrogram, path_spectrogram = mrr.plot.spectrogram(
                target_datetime=target_datetime,
                spectrum_var="spe_3D",
                savefig=True,
                output_dir=output_dir,
                dpi=180,
            )
            plt.close(fig_spectrogram)
            print(f"Spectrogram : {path_spectrogram}")
        else:
            print("[skip] spe_3D not present in the processed file; the dealiased spectrogram was not generated.")

        print(f"Input file : {input_file}")
        print(f"Profiles    : {len(plotted_time_entries)}")
        print(f"Output file : {output_path}")
    finally:
        mrr.close()


if __name__ == "__main__":
    main()
