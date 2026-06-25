from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mrrpropy import MRRProData

from mrrpropy.workflow import (
    DEFAULT_WINDOW_STEP_M,
    DEFAULT_WINDOW_THICKNESS_M,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the mrrpropy chain for one RAW file and a single-hour period: "
            "RaProMPro processing, raw/processed plots, optional fixed-layer "
            "rain analysis, and column process-event scan."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to one RAW MRR-Pro NetCDF file (typically one hour).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"./workbench/output/single_hour"),
        help="Root directory where products, plots, and tables will be written.",
    )
    parser.add_argument(
        "--hour-start",
        type=str,
        default=None,
        help=(
            "Optional hour start timestamp (e.g. '2025-10-29T19:00:00'). "
            "If omitted, the period starts at the first sample in the file."
        ),
    )
    parser.add_argument(
        "--hour-minutes",
        type=int,
        default=60,
        help="Length of the analysis period in minutes (default 60).",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Rebuild RaProMPro product even if *_raprompro.nc already exists.",
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
        help="Also generate the heavy raw/processed spectral and DSD figures.",
    )
    parser.add_argument(
        "--enable-layer-analysis",
        action="store_true",
        help="Also run the legacy fixed-layer rain analysis (optional).",
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
        "--window-thickness-m",
        type=float,
        default=DEFAULT_WINDOW_THICKNESS_M,
        help="Vertical thickness of the column-scan moving window in metres.",
    )
    parser.add_argument(
        "--window-step-m",
        type=float,
        default=DEFAULT_WINDOW_STEP_M,
        help="Vertical step of the column-scan moving window in metres.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI used for saved figures.",
    )
    args = parser.parse_args()

    raw_path = args.input_file.resolve()
    output_dir = args.output_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = (float(args.layer_top_m), float(args.layer_base_m))

    print(f"Input file      : {raw_path}")
    print(f"Output directory: {output_dir}")
    print(f"Column analysis : automatic whole-column scan")
    print(f"Window thickness: {args.window_thickness_m:g} m")
    print(f"Window step     : {args.window_step_m:g} m")
    print(f"Hexagram k      : {args.k}")

    mrr = MRRProData.from_file(raw_path)
    try:
        time_index = mrr.time
        if len(time_index) == 0:
            raise ValueError("Input file contains no time samples.")

        if args.hour_start is None:
            start = pd.Timestamp(time_index[0])
        else:
            start = pd.Timestamp(args.hour_start)
        end = start + pd.Timedelta(minutes=int(args.hour_minutes))
        # Clamp to available data range to avoid empty selections.
        start = max(start, pd.Timestamp(time_index[0]))
        end = min(end, pd.Timestamp(time_index[-1]))
        period = (start, end)

        mrr.workflow.run_file(
            output_dir=output_dir,
            period=period,
            force_reprocess=args.force_reprocess,
            save_spe_3d=not args.skip_spe_3d,
            save_dsd_3d=not args.skip_dsd_3d,
            include_spectral_plots=args.include_spectral_plots,
            enable_layer_analysis=args.enable_layer_analysis,
            layer=layer,
            k=args.k,
            window_thickness_m=float(args.window_thickness_m),
            window_step_m=float(args.window_step_m),
            dpi=args.dpi,
        )
    finally:
        mrr.close()

    print("\nSingle-hour chain completed.")


if __name__ == "__main__":
    main()
