from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
PAPER_SCRIPT_DIR = REPO_DIR / "Workbench" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PAPER_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_SCRIPT_DIR))

from analyze_bimonthly_process_stats import _filter_process_rows  # noqa: E402
from build_paper_figures import (  # noqa: E402
    KDE_VARIABLE_LIMITS,
    _add_convenience_columns,
    _require_any_column,
)
from mrrpropy.plotting.paper import (  # noqa: E402
    plot_process_distance_velocity_kdes,
    plot_process_kde_2x2,
)
from mrrpropy.rain_process_classification.rain_process_info import (  # noqa: E402
    canonical_process_label,
)

matplotlib.use("Agg")


PRODUCT_DIR = Path("workbench/output/bimonthly_process_stats_2025_03_04/products")
OUTPUT_ROOT = Path("workbench/output/bimonthly_process_stats_2025_03_04/monthly_kdes")
SAMPLE_PER_PROCESS = 50_000
CHUNKSIZE = 100_000
EXCLUDE_PROCESSES = ("steady_or_weak", "unknown", "no_data")
USEFUL_COLUMNS = (
    "time",
    "range",
    "proc_label",
    "proc_strength",
    "v_mean_top",
    "v_mean_bottom",
    "v_mean_layer_mean",
    "delta_v_mean",
    "Dm_top",
    "Dm_layer_mean",
    "Nw_top",
    "Nw_layer_mean",
    "LWC_top",
    "LWC_layer_mean",
    "Dw",
    "N",
    "LWC",
    "V",
    "bb_distance_m",
    "BB_distance_m",
    "dist_bb_peak",
    "dist_bb_bottom",
    "range_bottom_m",
    "range_top_m",
    "source_raw_file",
    "source_product_file",
)


def _month_from_checkpoint(path: Path) -> str | None:
    match = re.search(r"(2025)(03|04)\d{2}_\d{6}_raprompro_sliding\.csv$", path.name)
    if not match:
        return None
    return f"{match.group(1)}_{match.group(2)}"


def _normalise_process_names(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["proc_label"] = out["proc_label"].map(canonical_process_label)
    return out


def _checkpoint_paths_by_month() -> dict[str, list[Path]]:
    now = time.time()
    grouped: dict[str, list[Path]] = {"2025_03": [], "2025_04": []}
    for path in sorted(PRODUCT_DIR.rglob("*_raprompro_sliding.csv")):
        month = _month_from_checkpoint(path)
        if month not in grouped:
            continue
        # Avoid reading a file that may still be in the middle of being written.
        if now - path.stat().st_mtime < 30:
            continue
        grouped[month].append(path)
    return grouped


def _sample_month(month: str, paths: list[Path]) -> tuple[pd.DataFrame, pd.Series]:
    if not paths:
        raise FileNotFoundError(f"No checkpoint CSVs found for {month}.")

    sample_by_process: dict[str, pd.DataFrame] = {}
    counts: dict[str, int] = {}

    for index, path in enumerate(paths, start=1):
        if index == 1 or index == len(paths) or index % 25 == 0:
            print(f"[{month}] {index}/{len(paths)}", flush=True)
        try:
            columns = pd.read_csv(path, nrows=0).columns.tolist()
            usecols = [column for column in USEFUL_COLUMNS if column in columns]
            if "proc_label" not in usecols:
                print(f"[warn] {path}: skipped; no proc_label", flush=True)
                continue
            for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNKSIZE):
                if chunk.empty:
                    continue
                chunk = _normalise_process_names(_add_convenience_columns(chunk))
                chunk = _filter_process_rows(
                    chunk,
                    exclude_processes=EXCLUDE_PROCESSES,
                )
                if chunk.empty:
                    continue
                for label, group in chunk.groupby("proc_label", dropna=False):
                    key = str(label)
                    counts[key] = counts.get(key, 0) + len(group)
                    existing = sample_by_process.get(key)
                    candidate = (
                        group.copy()
                        if existing is None
                        else pd.concat(
                            [existing, group],
                            ignore_index=True,
                        )
                    )
                    if len(candidate) > SAMPLE_PER_PROCESS:
                        candidate = candidate.sample(
                            n=SAMPLE_PER_PROCESS,
                            random_state=20260806,
                        )
                    sample_by_process[key] = candidate
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            print(f"[warn] {path}: skipped ({exc})", flush=True)

    if not sample_by_process:
        raise ValueError(f"No useful process rows found for {month}.")
    sample = pd.concat(sample_by_process.values(), ignore_index=True)
    counts_series = pd.Series(counts, name="rows").sort_index()
    return sample, counts_series


def _write_month_outputs(month: str, sample: pd.DataFrame, counts: pd.Series) -> None:
    outdir = OUTPUT_ROOT / month
    outdir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(outdir / "sampled_process_rows.csv", index=False)
    counts.rename_axis("proc_label").reset_index(name="rows").to_csv(
        outdir / "process_counts.csv",
        index=False,
    )

    _require_any_column(sample, ("Dw", "Dm_layer_mean", "Dm_top"), "Dw KDE")
    _require_any_column(
        sample,
        (
            "V",
            "W",
            "VEL",
            "v_mean_top",
            "V_layer_mean",
            "W_layer_mean",
            "VEL_layer_mean",
        ),
        "V KDE",
    )
    _require_any_column(sample, ("LWC", "LWC_layer_mean", "LWC_top"), "LWC KDE")
    _require_any_column(sample, ("N", "Nw_layer_mean", "Nw_top"), "N KDE")

    fig, _, _ = plot_process_kde_2x2(
        sample,
        variables=("Dw", "V", "LWC", "N"),
        savefig=True,
        output_dir=outdir,
        filename=f"{month}_kde_Dw_V_LWC_N_by_process.png",
        domain_sigma=3.0,
        variable_limits=KDE_VARIABLE_LIMITS,
        dpi=600,
    )
    plt.close(fig)

    _require_any_column(
        sample,
        (
            "bb_distance_m",
            "BB_distance_m",
            "dist_bb_peak",
            "dist_bb_bottom",
            "distance_to_bb_m",
            "z_center_minus_bb_m",
        ),
        "bright-band distance KDE",
    )
    _require_any_column(
        sample,
        (
            "delta_v_mean",
            "delta_v_p50",
            "delta_v",
            "V_delta",
            "delta_V",
            "delta_velocity",
            "velocity_difference",
        ),
        "delta-v KDE",
    )
    fig, _, _ = plot_process_distance_velocity_kdes(
        sample,
        savefig=True,
        output_dir=outdir,
        filename=f"{month}_kde_bb_distance_delta_v_by_process.png",
        domain_sigma=3.0,
        dpi=600,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--months",
        nargs="+",
        default=["2025_03", "2025_04"],
        choices=["2025_03", "2025_04"],
    )
    args = parser.parse_args()
    grouped = _checkpoint_paths_by_month()
    for month in args.months:
        print(f"[month] {month}: {len(grouped[month])} checkpoints", flush=True)
        sample, counts = _sample_month(month, grouped[month])
        print(f"[month] {month}: sampled {len(sample)} rows", flush=True)
        _write_month_outputs(month, sample, counts)
    print(OUTPUT_ROOT.resolve(), flush=True)


if __name__ == "__main__":
    main()
