from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


DEFAULT_OUTPUT_PATH = Path(
    r"./tests/data/RAW/mrrpro81/2025/03/08/20250308_120000_10min.nc"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the bundled small RAW fixture from an external larger MRR-Pro NetCDF file."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the source RAW NetCDF file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the subset NetCDF will be written.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Inclusive start time index for the subset.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=60,
        help="Number of time steps to keep in the subset.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = args.input_path.resolve()
    output_path = args.output_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"RAW input file not found: {input_path}")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.count < 1:
        raise ValueError("--count must be >= 1")

    ds = xr.open_dataset(input_path)
    try:
        time_size = ds.sizes.get("time")
        if time_size is None:
            raise ValueError("Dataset has no 'time' dimension.")
        end_index = min(args.start_index + args.count, time_size)
        if args.start_index >= time_size:
            raise ValueError(
                f"--start-index {args.start_index} is outside dataset time size {time_size}."
            )

        subset = ds.isel(time=slice(args.start_index, end_index)).load()
    finally:
        ds.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_netcdf(output_path)

    time_values = subset["time"].values
    print(f"input   : {input_path}")
    print(f"output  : {output_path}")
    print(f"time    : {subset.sizes['time']} steps")
    print(f"range   : {subset.sizes.get('range', 'n/a')}")
    print(f"first   : {time_values[0]}")
    print(f"last    : {time_values[-1]}")


if __name__ == "__main__":
    main()


