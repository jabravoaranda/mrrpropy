from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import xarray as xr

from mrrpropy import MRRProData

matplotlib.use("Agg")


DEFAULT_INPUT_FILE = Path(
    r"workbench/output/raprompro/2025/10/29/20251029_190000_raprompro.nc"
)
DEFAULT_OUTPUT_DIR = Path(r"workbench/output/quicklooks/raprompro/2025/10/29")
EXCLUDED_2D_VARIABLES = {
    "BB_bottom",
    "BB_top",
    "BB_peak",
    "Noise",
}


def discover_quicklook_variables(mrr: MRRProData) -> list[str]:
    variables: list[str] = []
    if mrr.raprompro is None:
        return variables
    for name, da in mrr.raprompro.data_vars.items():
        if da.dims == ("time", "range") and name not in EXCLUDED_2D_VARIABLES:
            variables.append(name)
    return sorted(variables)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate quicklooks for a single processed *_raprompro.nc file."
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
        help="Directory where quicklook PNGs will be written.",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help=(
            "Optional list of variables to plot. If omitted, the script auto-discovers "
            "all (time, range) variables except selected internal fields."
        ),
    )
    args = parser.parse_args()

    input_file = args.input_file.resolve()
    output_dir = args.output_dir.resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Attach the processed NetCDF as the raprompro dataset used by quicklook().
    mrr = MRRProData(path=input_file, ds=xr.Dataset())
    mrr.load_raprompro(input_file)
    try:
        variables = (
            args.variables
            if args.variables is not None
            else discover_quicklook_variables(mrr)
        )
        if not variables:
            raise RuntimeError(f"No (time, range) variables found in: {input_file}")

        print(f"Input file : {input_file}")
        print(f"Output dir : {output_dir}")
        print(f"Variables   : {', '.join(variables)}")

        for variable in variables:
            fig, _ = mrr.plot.quicklook(variable=variable, source="raprompro")
            png_path = output_dir / f"{input_file.stem}_{variable}.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            fig.clf()
            print(f"[ok] {variable} -> {png_path}")
    finally:
        mrr.close()


if __name__ == "__main__":
    main()
