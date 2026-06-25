from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from mrrpropy import MRRProData

matplotlib.use("Agg")


DEFAULT_INPUT_DIR = Path(r"workbench/data/mrrpro81/2025/10/29")
DEFAULT_OUTPUT_DIR = Path(r"workbench/output/quicklooks/2025/10/29")
EXCLUDED_2D_VARIABLES = {"index_spectra"}


def discover_quicklook_variables(mrr: MRRProData) -> list[str]:
    variables: list[str] = []
    for name, da in mrr.ds.data_vars.items():
        if da.dims == ("time", "range") and name not in EXCLUDED_2D_VARIABLES:
            variables.append(name)
    return sorted(variables)


def plot_file_quicklooks(
    file_path: Path,
    *,
    output_dir: Path,
    variables: list[str] | None = None,
) -> None:
    mrr = MRRProData.from_file(file_path)
    try:
        variable_names = variables if variables is not None else discover_quicklook_variables(mrr)
        if not variable_names:
            print(f"[skip] {file_path.name}: no (time, range) variables found")
            return

        print(f"[file] {file_path.name}")
        for variable in variable_names:
            try:
                fig, _ = mrr.plot.quicklook(variable=variable, source="raw")
                png_name = f"{file_path.stem}_{variable}.png"
                png_path = output_dir / png_name
                fig.savefig(png_path, dpi=150, bbox_inches="tight")
                fig.clf()
                print(f"  [ok] {variable} -> {png_path}")
            except Exception as exc:
                print(f"  [fail] {variable}: {exc}")
    finally:
        mrr.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate quicklooks for all hourly MRR-PRO NetCDF files in a directory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing hourly NetCDF files.",
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
            "all (time, range) variables except internal indexing fields."
        ),
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
        plot_file_quicklooks(file_path, output_dir=output_dir, variables=args.variables)


if __name__ == "__main__":
    main()
