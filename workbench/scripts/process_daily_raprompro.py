from __future__ import annotations

import argparse
from pathlib import Path

from mrrpropy import MRRProData


DEFAULT_INPUT_DIR = Path(r"workbench/data/mrrpro81/2025/10/29")
DEFAULT_OUTPUT_DIR = Path(r"workbench/output/raprompro/2025/10/29")


def process_file(
    file_path: Path,
    *,
    output_dir: Path,
    save_spe_3d: bool,
    save_dsd_3d: bool,
    overwrite: bool,
) -> None:
    output_path = output_dir / f"{file_path.stem}_raprompro.nc"
    if output_path.exists() and not overwrite:
        print(f"[skip] {file_path.name} -> already exists")
        return

    print(f"[run ] {file_path.name}")
    mrr = MRRProData.from_file(file_path)
    try:
        mrr.process_raprompro(
            save=True,
            output_dir=output_dir,
            filename=output_path.name,
            save_spe_3d=save_spe_3d,
            save_dsd_3d=save_dsd_3d,
        )
        print(f"[done] {output_path}")
    finally:
        mrr.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-process all hourly MRR-PRO NetCDF files in a directory with RaProMPro."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing raw hourly NetCDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where *_raprompro.nc products will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess files even if the output NetCDF already exists.",
    )
    parser.add_argument(
        "--save-spe-3d",
        action="store_true",
        help="Also save the dealiased 3D spectra in the output NetCDF.",
    )
    parser.add_argument(
        "--save-dsd-3d",
        action="store_true",
        help="Also save the 3D DSD field in the output NetCDF.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files found     : {len(files)}")

    for file_path in files:
        process_file(
            file_path,
            output_dir=output_dir,
            save_spe_3d=args.save_spe_3d,
            save_dsd_3d=args.save_dsd_3d,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
