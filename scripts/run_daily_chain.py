from __future__ import annotations

import argparse
from pathlib import Path

from mrrpropy import MRRProData


def _discover_raw_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.nc" if recursive else "*.nc"
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No NetCDF files found under {input_dir}")
    return files


def _analyze_one_file(
    raw_path: Path,
    *,
    output_dir: Path,
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
    mrr = MRRProData.from_file(raw_path)
    try:
        mrr.workflow.run_file(
            output_dir=output_dir,
            force_reprocess=force_reprocess,
            save_spe_3d=save_spe_3d,
            save_dsd_3d=save_dsd_3d,
            include_spectral_plots=include_spectral_plots,
            enable_layer_analysis=enable_layer_analysis,
            layer=layer,
            k=k,
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
    output_dir = args.output_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = _discover_raw_files(input_dir, recursive=args.recursive)
    layer = (float(args.layer_top_m), float(args.layer_base_m))

    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
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
            output_dir=output_dir,
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
