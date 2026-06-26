from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from mrrpropy.raw_class import MRRProData


DEFAULT_RAW_PATH = Path(
    r"./tests/data/RAW/mrrpro81/2025/03/08/20250308_120000_10min.nc"
)
QUICK_RAW_PATH = DEFAULT_RAW_PATH


def _run_once(
    raw_path: Path,
    method_name: str,
    *,
    save: bool,
    save_spe_3d: bool,
    save_dsd_3d: bool,
) -> float:
    mrr = MRRProData.from_file(raw_path)
    try:
        method = getattr(mrr, method_name)
        t0 = time.perf_counter()
        ds = method(
            save=save,
            save_spe_3d=save_spe_3d,
            save_dsd_3d=save_dsd_3d,
        )
        elapsed = time.perf_counter() - t0
        ds.close()
        return elapsed
    finally:
        mrr.close()


def _summarize(name: str, samples: list[float]) -> dict[str, float]:
    return {
        "name": name,
        "runs": float(len(samples)),
        "min_s": min(samples),
        "mean_s": statistics.mean(samples),
        "median_s": statistics.median(samples),
        "max_s": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the canonical mrrpropy processing path."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Path to the RAW NetCDF file to process.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the bundled 10-minute RAW fixture for fast benchmarking.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed runs per method.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Enable NetCDF writing during benchmarking.",
    )
    parser.add_argument(
        "--save-spe-3d",
        action="store_true",
        help="Enable spe_3D generation during benchmarking.",
    )
    parser.add_argument(
        "--save-dsd-3d",
        action="store_true",
        help="Enable dsd_3D generation during benchmarking.",
    )
    args = parser.parse_args()

    raw_path = (QUICK_RAW_PATH if args.quick else args.raw_path).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"RAW file not found: {raw_path}")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    method_name = "process_raprompro"
    results: list[float] = []

    print(f"RAW file: {raw_path}")
    print(
        "Options:"
        f" quick={args.quick},"
        f" save={args.save},"
        f" save_spe_3d={args.save_spe_3d},"
        f" save_dsd_3d={args.save_dsd_3d},"
        f" repeats={args.repeats}"
    )
    print()

    print(f"Running {method_name} ...")
    for run_idx in range(1, args.repeats + 1):
        elapsed = _run_once(
            raw_path,
            method_name,
            save=args.save,
            save_spe_3d=args.save_spe_3d,
            save_dsd_3d=args.save_dsd_3d,
        )
        results.append(elapsed)
        print(f"  run {run_idx}: {elapsed:.3f} s")

    summary = _summarize(method_name, results)
    print()
    print("Summary:")
    print(f"  min   : {summary['min_s']:.3f} s")
    print(f"  mean  : {summary['mean_s']:.3f} s")
    print(f"  median: {summary['median_s']:.3f} s")
    print(f"  max   : {summary['max_s']:.3f} s")


if __name__ == "__main__":
    main()


