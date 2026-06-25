from __future__ import annotations

import argparse
from pathlib import Path

from mrrpropy import MRRProData


DEFAULT_INPUT_FILE = Path(r"workbench/data/mrrpro81/2025/10/29/20251029_130000.nc")
DEFAULT_OUTPUT_DIR = Path(r"workbench/output/raprompro/2025/10/29")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a single MRR-PRO NetCDF file with RaProMPro."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to the raw hourly NetCDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the *_raprompro.nc product will be written.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output filename. Defaults to <input_stem>_raprompro.nc",
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

    input_file = args.input_file.resolve()
    output_dir = args.output_dir.resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = (
        args.output_name
        if args.output_name is not None
        else f"{input_file.stem}_raprompro.nc"
    )

    print(f"Input file : {input_file}")
    print(f"Output dir : {output_dir}")
    print(f"Output file: {output_name}")

    mrr = MRRProData.from_file(input_file)
    try:
        mrr.process_raprompro(
            save=True,
            output_dir=output_dir,
            filename=output_name,
            save_spe_3d=args.save_spe_3d,
            save_dsd_3d=args.save_dsd_3d,
        )
    finally:
        mrr.close()

    print(f"Done: {output_dir / output_name}")


if __name__ == "__main__":
    main()
