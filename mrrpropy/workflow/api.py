"""Workflow helpers exposed as ``MRRProData.workflow``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mrrpropy.dataset import MRRProData

matplotlib.use("Agg")

DEFAULT_WINDOW_THICKNESS_M = 600.0
DEFAULT_WINDOW_STEP_M = 200.0
DEFAULT_MIN_TAU_STRENGTH = 0.5


def _to_python_datetime(value: object) -> datetime:
    return pd.Timestamp(value).round("us").to_pydatetime()


class WorkflowAPI:
    """Operational workflow namespace available as ``mrr.workflow``."""

    def __init__(self, mrr: MRRProData) -> None:
        self._mrr = mrr

    def run_file(
        self,
        *,
        output_dir: Path,
        period: tuple[Any, Any] | None = None,
        force_reprocess: bool = False,
        save_spe_3d: bool = True,
        save_dsd_3d: bool = True,
        include_spectral_plots: bool = False,
        enable_layer_analysis: bool = False,
        layer: tuple[float, float] = (1000.0, 2000.0),
        k: int = 11,
        dpi: int = 150,
        window_thickness_m: float = DEFAULT_WINDOW_THICKNESS_M,
        window_step_m: float = DEFAULT_WINDOW_STEP_M,
    ) -> None:
        """Run the complete processing and diagnostic workflow for this raw file."""
        raw_path = Path(self._mrr.path)
        output_dir.mkdir(parents=True, exist_ok=True)

        product_dir = output_dir / "products" / raw_path.parent.name
        plots_raw_dir = output_dir / "plots" / "raw" / raw_path.stem
        plots_processed_dir = output_dir / "plots" / "processed" / raw_path.stem
        rain_layer_dir = (
            output_dir
            / "plots"
            / f"rain_layer_{int(layer[0])}_{int(layer[1])}"
            / raw_path.stem
        )
        column_dir = (
            output_dir
            / "plots"
            / (
                "column_process_events_hexagram_"
                f"w{int(window_thickness_m)}_step{int(window_step_m)}"
            )
            / raw_path.stem
        )

        product_dir.mkdir(parents=True, exist_ok=True)
        product_path = product_dir / f"{raw_path.stem}_raprompro.nc"

        if force_reprocess or not product_path.exists():
            ds = self._mrr.process_raprompro(
                save=True,
                output_dir=product_dir,
                save_spe_3d=save_spe_3d,
                save_dsd_3d=save_dsd_3d,
            )
            ds.close()
        self._mrr.load_raprompro(product_path)

        time_index = self._mrr.time
        if len(time_index) == 0:
            raise ValueError("Input file contains no time samples.")

        if period is None:
            period_dt = (pd.Timestamp(time_index[0]), pd.Timestamp(time_index[-1]))
        else:
            period_dt = (pd.Timestamp(period[0]), pd.Timestamp(period[1]))
            period_dt = (
                max(period_dt[0], pd.Timestamp(time_index[0])),
                min(period_dt[1], pd.Timestamp(time_index[-1])),
            )

        self._save_raw_plots(
            output_dir=plots_raw_dir,
            dpi=dpi,
            include_spectral_plots=include_spectral_plots,
        )
        self._save_processed_plots(
            output_dir=plots_processed_dir,
            dpi=dpi,
            include_spectral_plots=include_spectral_plots,
        )
        self._save_column_event_sliding(
            period=period_dt,
            k=k,
            window_thickness_m=window_thickness_m,
            window_step_m=window_step_m,
            output_dir=column_dir,
            dpi=dpi,
        )
        if enable_layer_analysis:
            self._save_layer_rain_analysis(
                period=period_dt,
                layer=layer,
                k=k,
                min_tau_strength=self._mrr.micro_cfg.min_tau_strength,
                output_dir=rain_layer_dir,
                dpi=dpi,
            )

    def _save_raw_plots(
        self,
        *,
        output_dir: Path,
        dpi: int,
        include_spectral_plots: bool,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, _ = self._mrr.plot.quicklook(variable="Ze", source="raw")
        fig.savefig(
            output_dir / f"{Path(self._mrr.path).stem}_raw_Ze_quicklook.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

        if not include_spectral_plots:
            return

        time_values = self._mrr.ds["time"].values
        target_time = _to_python_datetime(time_values[len(time_values) // 2])
        range_values = self._mrr.ds["range"].values.astype(float)
        center_range = float(range_values[len(range_values) // 2])
        comparison_ranges = range_values[[5, len(range_values) // 2, -5]].astype(float)

        fig, _ = self._mrr.plot.spectrum(
            target_time,
            center_range,
            spectrum_var="spectrum_raw",
            savefig=True,
            output_dir=output_dir,
            dpi=dpi,
        )
        plt.close(fig)

        fig, _ = self._mrr.plot.spectra_by_range(
            target_time,
            comparison_ranges,
            savefig=True,
            output_dir=output_dir,
            dpi=dpi,
        )
        plt.close(fig)

        fig, _ = self._mrr.plot.spectrogram(
            target_time,
            spectrum_var="spectrum_raw",
            savefig=True,
            output_dir=output_dir,
            dpi=dpi,
        )
        plt.close(fig)

    def _save_processed_plots(
        self,
        *,
        output_dir: Path,
        dpi: int,
        include_spectral_plots: bool,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ds = self._mrr.raprompro
        if ds is None:
            return

        fig, _ = self._mrr.plot.quicklook(variable="Ze", source="raprompro")
        fig.savefig(
            output_dir / f"{Path(self._mrr.path).stem}_raprompro_Ze_quicklook.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

        self._save_transmittance_plots(output_dir=output_dir, dpi=dpi)

        time_values = ds["time"].values
        target_time = _to_python_datetime(time_values[len(time_values) // 2])

        fig, _, _ = self._mrr.plot.microphysical_profiles(
            target_datetime=target_time,
            savefig=True,
            output_dir=output_dir,
            dpi=dpi,
        )
        plt.close(fig)

        if not include_spectral_plots:
            return

        fig, _ = self._mrr.plot.spectrogram(
            target_time,
            spectrum_var="spe_3D",
            savefig=True,
            output_dir=output_dir,
            dpi=dpi,
        )
        plt.close(fig)

        fig, _ = self._mrr.plot.dsdgram(
            target_datetime=target_time,
            savefig=True,
            output_dir=output_dir,
            dpi=dpi,
        )
        plt.close(fig)

        fig, _ = self._mrr.plot.dsd_by_range(
            target_time,
            ranges=np.arange(500.0, 2500.0, 250.0),
            savefig=True,
            output_dir=output_dir,
            dpi=dpi,
        )
        plt.close(fig)

    def _save_transmittance_plots(self, *, output_dir: Path, dpi: int) -> None:
        ds = self._mrr.raprompro
        if ds is None:
            return

        ze = ds["Ze"].values.astype(float)
        zea = ds["Zea"].values.astype(float)
        dbpia = ds["DBPIA"].values.astype(float)
        hydrometeor_type = ds["Type"].values.astype(float)

        liquid_mask = np.isin(hydrometeor_type, [5.0, 10.0])
        corrected_mask = (
            liquid_mask
            & np.isfinite(ze)
            & np.isfinite(zea)
            & np.isfinite(dbpia)
            & (dbpia < 0.0)
        )

        if not np.any(corrected_mask):
            return

        delta = ze - zea
        x = (-dbpia[corrected_mask]).ravel()
        y = delta[corrected_mask].ravel()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, s=6, alpha=0.6)
        lo = float(np.nanmin(np.concatenate([x, y])))
        hi = float(np.nanmax(np.concatenate([x, y])))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("-DBPIA [dB]")
        ax.set_ylabel("Ze - Zea [dB]")
        ax.set_title("PIA correction consistency for liquid hydrometeors")
        fig.savefig(
            output_dir / "transmittance_correction_consistency.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

        correction = ds["Ze"] - ds["Zea"]
        correction_plot = correction.where(liquid_mask)
        fig, ax = plt.subplots(figsize=(12, 6))
        correction_plot.plot(  # type: ignore[call-arg]
            ax=ax,
            x="time",
            y="range",
            cmap="viridis",
            vmin=0.0,
            robust=True,
            cbar_kwargs={"label": "Ze - Zea [dB]"},
        )
        ax.set_title("Transmittance correction quicklook (liquid hydrometeors only)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Range [m]")
        fig.savefig(
            output_dir / "transmittance_correction_quicklook.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _save_layer_rain_analysis(
        self,
        *,
        period: tuple[Any, Any],
        layer: tuple[float, float],
        k: int,
        min_tau_strength: float,
        output_dir: Path,
        dpi: int,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        period_dt = (_to_python_datetime(period[0]), _to_python_datetime(period[1]))
        analysis = self._mrr.rain_process_analyze(
            period=period_dt,
            layer=layer,
            k=k,
            trend_method="kendall_theilsen",
        )
        classified = self._mrr.classify_rain_process(
            analysis=analysis,
            min_tau_strength=min_tau_strength,
        )

        dynamics = self._mrr.build_process_dynamics_dataframe(
            analysis=analysis,
            classified=classified,
        )
        summary = self._mrr.summarize_process_dynamics(
            analysis=analysis,
            classified=classified,
        )

        dynamics.to_csv(output_dir / "process_dynamics_samples.csv", index=True)
        summary.to_csv(output_dir / "process_dynamics_summary.csv", index=False)

        fig, _ = self._mrr.plot.rain.layer_2d(
            target_datetime=period_dt,
            layer=layer,
            x="Dm",
            y="Nw",
            z="LWC",
            savefig=True,
            output_dir=output_dir,
            marker_size=70,
            figsize=(10, 8),
            cmap="seismic",
        )
        plt.close(fig)

        fig, _ = self._mrr.plot.rain.layer_hexagram(
            analysis=analysis,
            savefig=True,
            output_dir=output_dir,
            dpi=dpi,
            alpha_hexagram=0.5,
            cmap="viridis",
        )
        plt.close(fig)

        fig, _ = self._mrr.plot.rain.evolution(
            classified=classified,
            analysis=analysis,
            savefig=True,
            output_dir=output_dir,
            figsize=(14, 10),
            cmap="viridis",
            alpha_hexagram=0.5,
            markersize=40.0,
            line_width=0.8,
            dpi=dpi,
        )
        plt.close(fig)

        fig, _ = self._mrr.plot.rain.classified_hexagram(
            classified=classified,
            analysis=analysis,
            savefig=True,
            output_dir=output_dir,
            dpi=dpi,
            show_background=True,
            show_process_masks=True,
        )
        plt.close(fig)

        fig, _ = self._mrr.plot.rain.event_scatter(
            target_datetime=period_dt,
            layer=layer,
            x="Dm",
            y="Nw",
            color="LWC",
            savefig=True,
            output_dir=output_dir,
            figsize=(10, 8),
            cmap="seismic",
        )
        plt.close(fig)

        fig, _ = self._mrr.plot.rain.event_vertical_profiles(
            target_datetime=period_dt,
            layer=layer,
            variables=("Dm", "Nw", "LWC"),
            savefig=True,
            output_dir=output_dir,
            figsize=(7, 6),
        )
        plt.close(fig)

        labels = sorted({str(value) for value in classified["proc_label"].values})
        for label in labels:
            if label == "no_data":
                continue
            try:
                fig, _ = self._mrr.plot.rain.process_scatter(
                    classified=classified,
                    process=label,
                    target_datetime=period_dt,
                    layer=layer,
                    x="Dm",
                    y="Nw",
                    color="LWC",
                    savefig=True,
                    output_dir=output_dir,
                    figsize=(7, 6),
                    cmap="seismic",
                )
                plt.close(fig)
            except ValueError:
                pass

            try:
                fig, _ = self._mrr.plot.rain.process_vertical_profiles(
                    classified=classified,
                    process=label,
                    target_datetime=period_dt,
                    layer=layer,
                    variables=("Dm", "Nw", "LWC"),
                    savefig=True,
                    output_dir=output_dir,
                    figsize=(7, 6),
                )
                plt.close(fig)
            except ValueError:
                pass

    def _save_column_event_sliding(
        self,
        *,
        period: tuple[Any, Any],
        k: int,
        output_dir: Path,
        dpi: int,
        window_thickness_m: float = DEFAULT_WINDOW_THICKNESS_M,
        window_step_m: float = DEFAULT_WINDOW_STEP_M,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        sliding_df = self._mrr.build_sliding_process_dataframe(
            period=(_to_python_datetime(period[0]), _to_python_datetime(period[1])),
            k=k,
            window_thickness_m=window_thickness_m,
            window_step_m=window_step_m,
            min_tau_strength=DEFAULT_MIN_TAU_STRENGTH,
            trend_method="kendall_theilsen",
        )
        episodes_df = self._mrr.detect_sliding_process_episodes(
            sliding_df=sliding_df,
            min_consecutive_profiles=6,
        )

        sliding_df.to_csv(output_dir / "column_process_sliding.csv", index=False)
        episodes_df.to_csv(output_dir / "column_process_episodes.csv", index=False)

        sliding_df_plot = sliding_df[
            ~sliding_df["proc_label"].isin(["unknown", "no_data"])
        ].copy()
        sliding_df_plot.attrs = dict(getattr(sliding_df, "attrs", {}))
        sliding_df_plot.to_csv(
            output_dir / "column_process_sliding_plot_filtered.csv",
            index=False,
        )

        sliding_df["time"] = pd.to_datetime(sliding_df["time"])
        if not episodes_df.empty:
            episodes_df["start_time"] = pd.to_datetime(episodes_df["start_time"])
            episodes_df["end_time"] = pd.to_datetime(episodes_df["end_time"])

        event_frames: list[pd.DataFrame] = []
        for _, event in episodes_df.iterrows():
            mask = (
                (sliding_df["proc_label"] == event["proc_label"])
                & (sliding_df["window_id"] == event["window_id"])
                & (sliding_df["time"] >= event["start_time"])
                & (sliding_df["time"] <= event["end_time"])
            )
            event_frames.append(sliding_df.loc[mask])

        if event_frames:
            sliding_df_events = pd.concat(
                event_frames, ignore_index=True
            ).drop_duplicates()
        else:
            sliding_df_events = sliding_df.iloc[0:0].copy()

        sliding_df_events = sliding_df_events[
            ~sliding_df_events["proc_label"].isin(["steady_or_weak", "unknown", "no_data"])
        ].copy()
        sliding_df_events.attrs = dict(getattr(sliding_df, "attrs", {}))
        sliding_df_events.to_csv(
            output_dir / "column_process_sliding_events_only.csv",
            index=False,
        )

        processes = [
            "breakup",
            "growth_depletion",
            "growth_depletion_loss",
            "growth_depletion_gain",
            "activation",
            "evaporation",
            "growth",
        ]

        if not sliding_df_plot.empty:
            fig, _ = self._mrr.plot.rain.column_scan(
                scan_df=sliding_df_plot,
                color_mode="hexagram",
                processes=processes,
                savefig=True,
                output_dir=output_dir,
                figsize=(14, 7),
                markersize=42,
                alpha=0.92,
                scale_by_strength=True,
            )
            plt.close(fig)

        if not sliding_df_events.empty:
            fig, _ = self._mrr.plot.rain.column_scan(
                scan_df=sliding_df_events,
                color_mode="process",
                processes=processes,
                savefig=False,
                output_dir=output_dir,
                figsize=(14, 7),
                markersize=42,
                alpha=0.92,
                scale_by_strength=True,
            )
            period_start = sliding_df_events.attrs.get("period_start", "t0")
            period_end = sliding_df_events.attrs.get("period_end", "t1")
            safe_t0 = (
                str(period_start).replace(":", "").replace("-", "").replace(" ", "_")
            )
            safe_t1 = (
                str(period_end).replace(":", "").replace("-", "").replace(" ", "_")
            )
            events_path = (
                output_dir / f"column_process_events_hexagram_{safe_t0}_{safe_t1}.png"
            )
            fig.savefig(events_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
