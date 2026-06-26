from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from mrrpropy.raw_class import MRRProData

matplotlib.use("Agg")

WINDOW_THICKNESS_M = 600.0
WINDOW_STEP_M = 200.0
MIN_TAU_STRENGTH = 0.5

def _to_python_datetime(value: object) -> object:
    """Convert a timestamp-like value to `datetime` without nanosecond warnings."""
    return pd.Timestamp(value).round("us").to_pydatetime()


def _save_quicklook(
    mrr: MRRProData,
    *,
    variable: str,
    source: str,
    output_dir: Path,
    prefix: str,
    dpi: int,
) -> Path:
    fig, _, _ = mrr.quicklook(variable=variable, source=source)
    output_path = output_dir / f"{prefix}_{source}_{variable}_quicklook.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_transmittance_plots(mrr: MRRProData, *, output_dir: Path, dpi: int) -> None:
    ds = mrr.raprompro
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

    if np.any(corrected_mask):
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
        correction_plot.plot(
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


def _save_raw_plots(
    mrr: MRRProData,
    *,
    output_dir: Path,
    dpi: int,
    include_spectral_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_quicklook(
        mrr,
        variable="Ze",
        source="raw",
        output_dir=output_dir,
        prefix=mrr.path.stem,
        dpi=dpi,
    )

    if not include_spectral_plots:
        return

    time_values = mrr.ds["time"].values
    target_time = _to_python_datetime(time_values[len(time_values) // 2])
    range_values = mrr.ds["range"].values.astype(float)
    center_range = float(range_values[len(range_values) // 2])
    comparison_ranges = range_values[[5, len(range_values) // 2, -5]].astype(float)

    fig, _, _ = mrr.plot_spectrum(
        target_time,
        center_range,
        spectrum_var="spectrum_raw",
        savefig=True,
        output_dir=output_dir,
        dpi=dpi,
    )
    plt.close(fig)

    fig, _, _ = mrr.plot_spectra_by_range(
        target_time,
        comparison_ranges,
        savefig=True,
        output_dir=output_dir,
        dpi=dpi,
    )
    plt.close(fig)

    fig, _, _ = mrr.plot_spectrogram(
        target_time,
        spectrum_var="spectrum_raw",
        savefig=True,
        output_dir=output_dir,
        dpi=dpi,
    )
    plt.close(fig)


def _save_processed_plots(
    mrr: MRRProData,
    *,
    output_dir: Path,
    dpi: int,
    include_spectral_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = mrr.raprompro
    if ds is None:
        return

    _save_quicklook(
        mrr,
        variable="Ze",
        source="raprompro",
        output_dir=output_dir,
        prefix=Path(mrr.path).stem,
        dpi=dpi,
    )
    _save_transmittance_plots(mrr, output_dir=output_dir, dpi=dpi)

    time_values = ds["time"].values
    target_time = _to_python_datetime(time_values[len(time_values) // 2])

    fig, _, _ = mrr.plot_microphysical_properties_profiles(
        target_datetime=target_time,
        savefig=True,
        output_dir=output_dir,
        dpi=dpi,
    )
    plt.close(fig)

    if not include_spectral_plots:
        return

    fig, _, _ = mrr.plot_spectrogram(
        target_time,
        spectrum_var="spe_3D",
        savefig=True,
        output_dir=output_dir,
        dpi=dpi,
    )
    plt.close(fig)

    fig, _, _ = mrr.plot_DSDgram(
        target_datetime=target_time,
        savefig=True,
        output_dir=output_dir,
        dpi=dpi,
    )
    plt.close(fig)

    fig, _, _ = mrr.plot_DSD_by_range(
        target_time,
        ranges=np.arange(500.0, 2500.0, 250.0),
        savefig=True,
        output_dir=output_dir,
        dpi=dpi,
    )
    plt.close(fig)


def _save_layer_rain_analysis(
    mrr: MRRProData,
    *,
    period: tuple[pd.Timestamp, pd.Timestamp],
    layer: tuple[float, float],
    k: int,
    min_tau_strength: float,
    output_dir: Path,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = mrr.rain_process_analyze(
        period=(_to_python_datetime(period[0]), _to_python_datetime(period[1])),
        layer=layer,
        k=k,
        trend_method="kendall_theilsen",
    )
    classified = mrr.classify_rain_process(
        analysis=analysis,
        min_tau_strength=min_tau_strength,
    )

    dynamics = mrr.build_process_dynamics_dataframe(
        analysis=analysis,
        classified=classified,
    )
    summary = mrr.summarize_process_dynamics(
        analysis=analysis,
        classified=classified,
    )

    dynamics.to_csv(output_dir / "process_dynamics_samples.csv", index=True)
    summary.to_csv(output_dir / "process_dynamics_summary.csv", index=False)

    fig, _, _ = mrr.plot_rain_process_in_layer_2D(
        target_datetime=(
            _to_python_datetime(period[0]),
            _to_python_datetime(period[1]),
        ),
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

    fig, _, _ = mrr.plot_rain_process_in_layer_hexagram(
        analysis=analysis,
        savefig=True,
        output_dir=output_dir,
        dpi=dpi,
        alpha_hexagram=0.5,
        cmap="viridis",
    )
    plt.close(fig)

    fig, _, _ = mrr.plot_processes_evolution(
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

    fig, _, _ = mrr.plot_classified_processes_on_hexagram(
        classified=classified,
        analysis=analysis,
        savefig=True,
        output_dir=output_dir,
        dpi=dpi,
        show_background=True,
        show_process_masks=True,
    )
    plt.close(fig)

    fig, _, _ = mrr.plot_event_scatter(
        target_datetime=(
            _to_python_datetime(period[0]),
            _to_python_datetime(period[1]),
        ),
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

    fig, _, _ = mrr.plot_event_vertical_percent_profiles(
        target_datetime=(
            _to_python_datetime(period[0]),
            _to_python_datetime(period[1]),
        ),
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
            fig, _, _ = mrr.plot_process_scatter(
                classified=classified,
                process=label,
                target_datetime=(
                    _to_python_datetime(period[0]),
                    _to_python_datetime(period[1]),
                ),
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
            fig, _, _ = mrr.plot_process_vertical_percent_profiles(
                classified=classified,
                process=label,
                target_datetime=(
                    _to_python_datetime(period[0]),
                    _to_python_datetime(period[1]),
                ),
                layer=layer,
                variables=("Dm", "Nw", "LWC"),
                savefig=True,
                output_dir=output_dir,
                figsize=(7, 6),
            )
            plt.close(fig)
        except ValueError:
            pass


def _save_column_event_scan(
    mrr: MRRProData,
    *,
    period: tuple[pd.Timestamp, pd.Timestamp],
    k: int,
    output_dir: Path,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_df = mrr.build_column_process_scan_dataframe(
        period=(_to_python_datetime(period[0]), _to_python_datetime(period[1])),
        k=k,
        window_thickness_m=WINDOW_THICKNESS_M,
        window_step_m=WINDOW_STEP_M,
        min_tau_strength=MIN_TAU_STRENGTH,
        trend_method="kendall_theilsen",
    )
    episodes_df = mrr.detect_column_process_episodes(
        scan_df=scan_df,
        min_consecutive_profiles=6,
    )

    scan_df.to_csv(output_dir / "column_process_scan.csv", index=False)
    episodes_df.to_csv(output_dir / "column_process_episodes.csv", index=False)

    scan_df_plot = scan_df[
        ~scan_df["proc_label"].isin(["unknown", "no_data"])
    ].copy()
    scan_df_plot.attrs = dict(getattr(scan_df, "attrs", {}))
    scan_df_plot.to_csv(
        output_dir / "column_process_scan_plot_filtered.csv",
        index=False,
    )

    scan_df["time"] = pd.to_datetime(scan_df["time"])
    if not episodes_df.empty:
        episodes_df["start_time"] = pd.to_datetime(episodes_df["start_time"])
        episodes_df["end_time"] = pd.to_datetime(episodes_df["end_time"])

    event_frames: list[pd.DataFrame] = []
    for _, event in episodes_df.iterrows():
        mask = (
            (scan_df["proc_label"] == event["proc_label"])
            & (scan_df["window_id"] == event["window_id"])
            & (scan_df["time"] >= event["start_time"])
            & (scan_df["time"] <= event["end_time"])
        )
        event_frames.append(scan_df.loc[mask])

    if event_frames:
        scan_df_events = pd.concat(event_frames, ignore_index=True).drop_duplicates()
    else:
        scan_df_events = scan_df.iloc[0:0].copy()

    scan_df_events = scan_df_events[
        ~scan_df_events["proc_label"].isin(
            ["steady_or_weak", "unknown", "no_data"]
        )
    ].copy()
    scan_df_events.attrs = dict(getattr(scan_df, "attrs", {}))
    scan_df_events.to_csv(
        output_dir / "column_process_scan_events_only.csv",
        index=False,
    )

    if not scan_df_plot.empty:
        fig, _, _ = mrr.plot_column_process_scan(
            scan_df=scan_df_plot,
            color_mode="hexagram",
            processes=['breakup', 'growth_depletion', 'growth_depletion_loss', 'growth_depletion_gain', 'activation', 'evaporation', 'growth'],
            savefig=True,
            output_dir=output_dir,
            figsize=(14, 7),
            markersize=42,
            alpha=0.92,
            scale_by_strength=True,
        )
        plt.close(fig)

    if not scan_df_events.empty:
        fig, _, _ = mrr.plot_column_process_scan(
            scan_df=scan_df_events,
            color_mode="hexagram",
            processes=['breakup', 'growth_depletion', 'growth_depletion_loss', 'growth_depletion_gain', 'activation', 'evaporation', 'growth'],
            savefig=False,
            output_dir=output_dir,
            figsize=(14, 7),
            markersize=42,
            alpha=0.92,
            scale_by_strength=True,
        )
        period_start = scan_df_events.attrs.get("period_start", "t0")
        period_end = scan_df_events.attrs.get("period_end", "t1")
        safe_t0 = str(period_start).replace(":", "").replace("-", "").replace(" ", "_")
        safe_t1 = str(period_end).replace(":", "").replace("-", "").replace(" ", "_")
        events_path = output_dir / f"column_process_events_hexagram_{safe_t0}_{safe_t1}.png"
        fig.savefig(events_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _discover_raw_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.nc" if recursive else "*.nc"
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No NetCDF files found under {input_dir}")
    return files


def _analyze_one_file(
    raw_path: Path,
    *,
    output_root: Path,
    save_spe_3d: bool,
    save_dsd_3d: bool,
    include_spectral_plots: bool,
    force_reprocess: bool,
    enable_layer_analysis: bool,
    layer: tuple[float, float],
    k: int,
    dpi: int,
) -> None:
    print(f"\n=== Processing {raw_path.name} ===")
    product_dir = output_root / "products" / raw_path.parent.name
    plots_raw_dir = output_root / "plots" / "raw" / raw_path.stem
    plots_processed_dir = output_root / "plots" / "processed" / raw_path.stem
    rain_layer_dir = (
        output_root
        / "plots"
        / f"rain_layer_{int(layer[0])}_{int(layer[1])}"
        / raw_path.stem
    )
    column_dir = output_root / "plots" / "column_process_events_hexagram_w500_step35" / raw_path.stem

    mrr = MRRProData.from_file(raw_path)
    try:
        product_dir.mkdir(parents=True, exist_ok=True)
        product_path = product_dir / f"{raw_path.stem}_raprompro.nc"

        if force_reprocess or not product_path.exists():
            ds = mrr.process_raprompro(
                save=True,
                output_dir=product_dir,
                save_spe_3d=save_spe_3d,
                save_dsd_3d=save_dsd_3d,
            )
            ds.close()
        mrr.load_raprompro(product_path)

        time_index = mrr.time
        period = (pd.Timestamp(time_index[0]), pd.Timestamp(time_index[-1]))

        _save_raw_plots(
            mrr,
            output_dir=plots_raw_dir,
            dpi=dpi,
            include_spectral_plots=include_spectral_plots,
        )
        _save_processed_plots(
            mrr,
            output_dir=plots_processed_dir,
            dpi=dpi,
            include_spectral_plots=include_spectral_plots,
        )
        _save_column_event_scan(
            mrr,
            period=period,
            k=k,
            output_dir=column_dir,
            dpi=dpi,
        )
        if enable_layer_analysis:
            _save_layer_rain_analysis(
                mrr,
                period=period,
                layer=layer,
                k=k,
                min_tau_strength=mrr.micro_cfg.min_tau_strength,
                output_dir=rain_layer_dir,
                dpi=dpi,
            )
    finally:
        mrr.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full mrrpropy daily chain: RaProMPro processing, raw and "
            "processed plots, layer rain analysis, and column process-event scan."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"Z:\UGR\mrrpro81\2025\03\11"),
        help="Directory containing RAW MRR-Pro NetCDF files for one day.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"./workbench/output/daily_chain/2025/03/11"),
        help="Root directory where products, plots, and tables will be written.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for RAW NetCDF files under input-dir.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Rebuild RaProMPro products even if *_raprompro.nc already exists.",
    )
    parser.add_argument(
        "--skip-spe-3d",
        action="store_true",
        help="Skip spe_3D generation during RaProMPro processing.",
    )
    parser.add_argument(
        "--skip-dsd-3d",
        action="store_true",
        help="Skip dsd_3D generation during RaProMPro processing.",
    )
    parser.add_argument(
        "--include-spectral-plots",
        action="store_true",
        help=(
            "Also generate the heavy raw/processed spectral and DSD figures. "
            "By default the daily chain skips them to keep runtime manageable."
        ),
    )
    parser.add_argument(
        "--enable-layer-analysis",
        action="store_true",
        help=(
            "Also run the legacy fixed-layer rain analysis. By default the daily "
            "chain only runs the automatic whole-column scan."
        ),
    )
    parser.add_argument(
        "--layer-top-m",
        type=float,
        default=1000.0,
        help="Lower edge of the optional fixed-layer rain analysis.",
    )
    parser.add_argument(
        "--layer-base-m",
        type=float,
        default=2000.0,
        help="Upper edge of the optional fixed-layer rain analysis.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=11,
        help="Hexagram resolution parameter.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI used for saved figures.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    raw_files = _discover_raw_files(input_dir, recursive=args.recursive)
    layer = (float(args.layer_top_m), float(args.layer_base_m))

    print(f"Input directory : {input_dir}")
    print(f"Output root     : {output_root}")
    print(f"RAW files found : {len(raw_files)}")
    print("Column analysis : automatic whole-column scan")
    if args.include_spectral_plots:
        print("Spectral plots  : enabled")
    else:
        print("Spectral plots  : skipped")
    if args.enable_layer_analysis:
        print(f"Layer analysis  : enabled at {layer[0]:.1f}-{layer[1]:.1f} m")
    else:
        print("Layer analysis  : disabled")
    print(f"Hexagram k      : {args.k}")

    for raw_path in raw_files:
        _analyze_one_file(
            raw_path,
            output_root=output_root,
            save_spe_3d=not args.skip_spe_3d,
            save_dsd_3d=not args.skip_dsd_3d,
            include_spectral_plots=args.include_spectral_plots,
            force_reprocess=args.force_reprocess,
            enable_layer_analysis=args.enable_layer_analysis,
            layer=layer,
            k=args.k,
            dpi=args.dpi,
        )

    print("\nDaily chain completed.")


if __name__ == "__main__":
    main()


