from __future__ import annotations

import argparse
import datetime as dt
import io
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mrrpropy.rain_process_classification.hexagram import plot_process_to_hexagram
from mrrpropy.rain_process_classification.rain_process_algorithm import (
    sliding_rain_classification_to_dataframe,
)
from mrrpropy.plotting.processes import plot_fused_process_quicklook
from mrrpropy.raw_class import MRRProData


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "tests/data/RAW/mrrpro81/2025/10/29/20251029_192300_10min.nc"
PRODUCT_PATH = (
    ROOT / "tests/data/PRODUCTS/mrrpro81/2025/10/29/20251029_192300_10min_raprompro.nc"
)
DEFAULT_OUTPUT_DIR = ROOT / "docs/assets/plot-examples"
TARGET_TIME = dt.datetime(2025, 10, 29, 19, 28, 0)
PERIOD = (dt.datetime(2025, 10, 29, 19, 23, 0), dt.datetime(2025, 10, 29, 19, 33, 0))
LAYER = (1000.0, 2000.0)
QUICKLOOK_PROCESSES = [
    "breakup",
    "coalescence",
    "coalescence_loss",
    "coalescence_gain",
    "activation",
    "evaporation_strong",
    "growth",
]


class PlotContext:
    def __init__(self) -> None:
        if not RAW_PATH.exists():
            raise FileNotFoundError(f"Missing RAW fixture: {RAW_PATH}")
        if not PRODUCT_PATH.exists():
            raise FileNotFoundError(f"Missing RaProMPro product: {PRODUCT_PATH}")

        self.mrr = MRRProData.from_file(RAW_PATH)
        self.mrr.load_raprompro(PRODUCT_PATH)
        self._analysis_fixed = None
        self._classified_fixed = None
        self._sliding_df = None

    def close(self) -> None:
        self.mrr.close()

    @property
    def target_range(self) -> float:
        ds = self.mrr.ds
        return float(ds["range"].values[ds.sizes["range"] // 2])

    @property
    def fixed_analysis(self):
        if self._analysis_fixed is None:
            self._analysis_fixed = self.mrr.layer_rain_classification(
                period=PERIOD,
                k=11,
                z_bottom_m=LAYER[0],
                z_top_m=LAYER[1],
                ze_th=-5.0,
                min_points_trend=10,
                vars_trend=("Dm", "Nw", "LWC"),
            )
        return self._analysis_fixed

    @property
    def fixed_classified(self):
        if self._classified_fixed is None:
            self._classified_fixed = self.mrr.classify_rain_process(
                analysis=self.fixed_analysis
            )
        return self._classified_fixed

    @property
    def sliding_df(self) -> pd.DataFrame:
        if self._sliding_df is None:
            self._sliding_df = sliding_rain_classification_to_dataframe(
                self.mrr.sliding_rain_classification(
                    period=PERIOD,
                    k=11,
                    window_thickness_m=500.0,
                    window_step_m=None,
                    min_tau_strength=0.5,
                )
            )
        return self._sliding_df

    @property
    def representative_process(self) -> str:
        labels = self.fixed_classified["proc_label"].values.astype(str)
        process = next(
            (
                label
                for label in labels
                if label not in {"no_data", "unknown", "steady_or_weak"}
            ),
            None,
        )
        if process is not None:
            return str(process)
        fallback = next((label for label in labels if label != "no_data"), None)
        return str(fallback or "steady_or_weak")

    @property
    def representative_processes(self) -> list[str]:
        labels = sorted(
            {
                label
                for label in self.fixed_classified["proc_label"].values.astype(str)
                if label not in {"no_data", "unknown", "steady_or_weak"}
            }
        )
        return labels[:2] if labels else [self.representative_process]


def _save(fig, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _fused_df_from_sliding(sliding_df: pd.DataFrame) -> pd.DataFrame:
    valid = sliding_df.copy()
    valid["z_top_m"] = pd.to_numeric(valid["z_top_m"], errors="coerce")
    valid["z_bottom_m"] = pd.to_numeric(valid["z_bottom_m"], errors="coerce")
    valid = valid[valid["z_top_m"].gt(valid["z_bottom_m"])].copy()
    if valid.empty:
        return pd.DataFrame()

    plottable = valid[valid["proc_label"].astype(str).isin(QUICKLOOK_PROCESSES)].copy()
    source = plottable.iloc[0] if not plottable.empty else valid.iloc[0]
    time0 = pd.Timestamp(pd.to_datetime(source["time"]))
    label = str(source["proc_label"])
    if label not in QUICKLOOK_PROCESSES:
        label = QUICKLOOK_PROCESSES[-1]

    snap = valid[
        (valid["time"] == time0) & (valid["proc_label"].astype(str) == label)
    ].copy()
    if snap.empty:
        snap = pd.DataFrame([source])

    if "window_id" in snap.columns:
        snap = snap.sort_values("window_id", ascending=False)
    take = snap.head(3)

    return pd.DataFrame(
        {
            "time": [time0],
            "proc_label_fused": [label],
            "z_top_fused": [float(take["z_top_m"].max())],
            "z_bottom_fused": [float(take["z_bottom_m"].min())],
        }
    )


def quicklook_raw(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.quicklook(variable="Ze", source="raw", vmin=-10, vmax=40)
    return _save(fig, output_dir, "quicklook_raw_ze.png")


def quicklook_raprompro(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.quicklook(variable="Ze", source="raprompro", vmin=-10, vmax=40)
    return _save(fig, output_dir, "quicklook_raprompro_ze.png")


def plot_spectrum(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_spectrum(
        TARGET_TIME,
        ctx.target_range,
        spectrum_var="spectrum_raw",
    )
    return _save(fig, output_dir, "plot_spectrum.png")


def plot_spectra_by_range(ctx: PlotContext, output_dir: Path) -> Path:
    ranges = (
        ctx.mrr.ds["range"]
        .values[[5, ctx.mrr.ds.sizes["range"] // 2, -5]]
        .astype(float)
    )
    fig, _, _ = ctx.mrr.plot_spectra_by_range(TARGET_TIME, ranges)
    return _save(fig, output_dir, "plot_spectra_by_range.png")


def plot_spectrogram_raw(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_spectrogram(TARGET_TIME, spectrum_var="spectrum_raw")
    return _save(fig, output_dir, "plot_spectrogram_raw.png")


def plot_spectrogram_raprompro(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_spectrogram(TARGET_TIME, spectrum_var="spe_3D")
    return _save(fig, output_dir, "plot_spectrogram_raprompro.png")


def plot_dsdgram(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_DSDgram(target_datetime=TARGET_TIME)
    return _save(fig, output_dir, "plot_dsdgram.png")


def plot_dsd_by_range(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_DSD_by_range(
        TARGET_TIME,
        ranges=np.arange(500, 2500, 250),
    )
    return _save(fig, output_dir, "plot_dsd_by_range.png")


def plot_microphysical_profiles(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_microphysical_properties_profiles(
        target_datetime=TARGET_TIME
    )
    return _save(fig, output_dir, "plot_microphysical_properties_profiles.png")


def plot_rain_process_2d(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_rain_process_in_layer_2D(
        target_datetime=PERIOD,
        layer=LAYER,
        x="Dm",
        y="Nw",
        z="LWC",
        marker_size=100,
        figsize=(12, 10),
        cmap="seismic",
    )
    return _save(fig, output_dir, "plot_rain_process_in_layer_2d.png")


def plot_event_scatter(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_event_scatter(
        target_datetime=PERIOD,
        layer=LAYER,
        x="Dm",
        y="Nw",
        color="LWC",
        figsize=(12, 10),
        cmap="seismic",
    )
    return _save(fig, output_dir, "plot_event_scatter.png")


def plot_region_scatter(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_region_scatter(
        target_datetime=PERIOD,
        z_bottom_m=LAYER[0],
        z_top_m=LAYER[1],
        x="Dm",
        y="Nw",
        color="LWC",
        processes=ctx.representative_processes,
        classified=ctx.fixed_classified,
        figsize=(12, 10),
        cmap="seismic",
    )
    return _save(fig, output_dir, "plot_region_scatter.png")


def plot_process_scatter(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_process_scatter(
        classified=ctx.fixed_classified,
        process=ctx.representative_process,
        target_datetime=PERIOD,
        layer=LAYER,
        x="Dm",
        y="Nw",
        color="LWC",
        figsize=(12, 10),
        cmap="seismic",
    )
    return _save(fig, output_dir, "plot_process_scatter.png")


def plot_event_vertical_profiles(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_event_vertical_percent_profiles(
        target_datetime=PERIOD,
        layer=LAYER,
        variables=("Dm", "Nw", "LWC"),
        figsize=(7, 6),
    )
    return _save(fig, output_dir, "plot_event_vertical_percent_profiles.png")


def plot_process_vertical_profiles(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_process_vertical_percent_profiles(
        classified=ctx.fixed_classified,
        process=ctx.representative_process,
        target_datetime=PERIOD,
        layer=LAYER,
        variables=("Dm", "Nw", "LWC"),
        figsize=(7, 6),
    )
    return _save(fig, output_dir, "plot_process_vertical_percent_profiles.png")


def plot_layer_hexagram(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_rain_process_in_layer_hexagram(
        analysis=ctx.fixed_analysis,
        alpha_hexagram=0.5,
        cmap="viridis",
    )
    return _save(fig, output_dir, "plot_rain_process_in_layer_hexagram.png")


def plot_processes_evolution(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_processes_evolution(
        classified=ctx.fixed_classified,
        analysis=ctx.fixed_analysis,
        figsize=(14, 10),
        cmap="viridis",
        alpha_hexagram=0.5,
        markersize=40.0,
        line_width=0.8,
    )
    return _save(fig, output_dir, "plot_processes_evolution.png")


def plot_classified_hexagram(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_classified_processes_on_hexagram(
        classified=ctx.fixed_classified,
        analysis=ctx.fixed_analysis,
        show_background=True,
        figsize=(14, 10),
        cmap="viridis",
        alpha_hexagram=0.25,
        markersize=70.0,
        line_width=0.8,
        legend_fontsize=14,
    )
    return _save(fig, output_dir, "plot_classified_processes_on_hexagram.png")


def plot_sliding_column_process(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = ctx.mrr.plot_sliding_column_process(
        sliding_df=ctx.sliding_df,
        figsize=(10, 6),
    )
    return _save(fig, output_dir, "plot_sliding_column_process.png")


def plot_sliding_scatter_compare(ctx: PlotContext, output_dir: Path) -> Path:
    selected = sorted(
        {
            label
            for label in pd.unique(ctx.sliding_df["proc_label"].astype(str))
            if label not in {"unknown", "no_data", "steady_or_weak"}
        }
    )[:2]
    fig, _, _ = ctx.mrr.plot_sliding_process_scatter_compare(
        sliding_df=ctx.sliding_df,
        processes=selected or None,
        show_centroids=True,
        figsize=(10, 8),
    )
    return _save(fig, output_dir, "plot_sliding_process_scatter_compare.png")


def plot_fused_quicklook(ctx: PlotContext, output_dir: Path) -> Path:
    fused_df = _fused_df_from_sliding(ctx.sliding_df)
    fig, _, _ = plot_fused_process_quicklook(
        ctx.sliding_df,
        fused_df,
        processes=QUICKLOOK_PROCESSES,
        figsize=(10, 6),
    )
    return _save(fig, output_dir, "plot_fused_process_quicklook.png")


def plot_hexagram_process(ctx: PlotContext, output_dir: Path) -> Path:
    fig, _, _ = plot_process_to_hexagram(
        process="activation",
        k=11,
        tol_center=0.15,
        crop_to_process=False,
    )
    return _save(fig, output_dir, "plot_process_to_hexagram.png")


PLOTS: dict[str, Callable[[PlotContext, Path], Path]] = {
    "quicklook-raw": quicklook_raw,
    "quicklook-raprompro": quicklook_raprompro,
    "plot-spectrum": plot_spectrum,
    "plot-spectra-by-range": plot_spectra_by_range,
    "plot-spectrogram-raw": plot_spectrogram_raw,
    "plot-spectrogram-raprompro": plot_spectrogram_raprompro,
    "plot-dsdgram": plot_dsdgram,
    "plot-dsd-by-range": plot_dsd_by_range,
    "plot-microphysical-profiles": plot_microphysical_profiles,
    "plot-rain-process-2d": plot_rain_process_2d,
    "plot-event-scatter": plot_event_scatter,
    "plot-region-scatter": plot_region_scatter,
    "plot-process-scatter": plot_process_scatter,
    "plot-event-vertical-profiles": plot_event_vertical_profiles,
    "plot-process-vertical-profiles": plot_process_vertical_profiles,
    "plot-layer-hexagram": plot_layer_hexagram,
    "plot-processes-evolution": plot_processes_evolution,
    "plot-classified-hexagram": plot_classified_hexagram,
    "plot-sliding-column-process": plot_sliding_column_process,
    "plot-sliding-scatter-compare": plot_sliding_scatter_compare,
    "plot-fused-quicklook": plot_fused_quicklook,
    "plot-hexagram-process": plot_hexagram_process,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the plot gallery images used by docs/examples.html."
    )
    parser.add_argument(
        "--only",
        choices=sorted(PLOTS),
        action="append",
        help="Generate one plot example. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG examples are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.only or list(PLOTS)
    ctx = PlotContext()
    try:
        for name in selected:
            with redirect_stdout(io.StringIO()):
                path = PLOTS[name](ctx, args.output_dir)
            print(f"{name}: {path.relative_to(ROOT)}")
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
