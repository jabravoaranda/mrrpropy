from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
import traceback

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
PAPER_SCRIPT_DIR = REPO_DIR / "Workbench" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PAPER_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_SCRIPT_DIR))

from build_paper_figures import (  # noqa: E402
    KDE_VARIABLE_LIMITS,
    _aggregate_existing_product_csvs,
    _build_samples_for_raw_file,
    _require_any_column,
    _samples_to_xarray,
)
from mrrpropy.rain_process_classification.rain_process_info import (  # noqa: E402
    canonical_process_label,
)
from mrrpropy.plotting.paper import (  # noqa: E402
    plot_process_distance_velocity_kdes,
    plot_process_kde_2x2,
)

matplotlib.use("Agg")


def _parse_date(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


def _timestamp_from_raw_path(path: Path) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(datetime.strptime(path.stem[:15], "%Y%m%d_%H%M%S"))
    except ValueError:
        return None


def _discover_raw_files(
    raw_root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pattern: str,
) -> list[Path]:
    files: list[Path] = []
    for path in sorted(raw_root.rglob(pattern)):
        if not path.is_file():
            continue
        timestamp = _timestamp_from_raw_path(path)
        if timestamp is None:
            continue
        if start <= timestamp < end:
            files.append(path)
    if not files:
        raise FileNotFoundError(
            f"No raw files matched {pattern!r} under {raw_root} for {start} <= t < {end}."
        )
    return files


def _append_manifest(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _filter_process_rows(
    frame: pd.DataFrame,
    *,
    exclude_processes: tuple[str, ...],
) -> pd.DataFrame:
    out = frame.copy()
    out["proc_label"] = out["proc_label"].map(canonical_process_label)
    process_text = out["proc_label"].astype("string").str.strip().str.lower()
    out = out[out["proc_label"].notna() & ~process_text.isin({"none", "nan", ""})]
    if exclude_processes:
        excluded = {value.strip().lower() for value in exclude_processes}
        process_text = out["proc_label"].astype("string").str.strip().str.lower()
        out = out[~process_text.isin(excluded)]
    return out.copy()


def _write_aggregate_outputs(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
    kde_domain_sigma: float,
    skip_distance_velocity: bool,
) -> None:
    if frame.empty:
        raise ValueError("No rows remain after filtering.")

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = output_dir / "combined_column_process_scan.csv"
    combined_nc = output_dir / "combined_process_samples.nc"
    frame.to_csv(combined_csv, index=False)
    _samples_to_xarray(frame).to_netcdf(combined_nc)

    counts = frame["proc_label"].astype(str).value_counts().sort_index()
    counts.rename_axis("proc_label").reset_index(name="rows").to_csv(
        output_dir / "process_counts.csv",
        index=False,
    )

    _require_any_column(frame, ("Dw", "Dm_layer_mean", "Dm_top"), "Dw KDE")
    _require_any_column(
        frame,
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
    _require_any_column(frame, ("LWC", "LWC_layer_mean", "LWC_top"), "LWC KDE")
    _require_any_column(frame, ("N", "Nw_layer_mean", "Nw_top"), "N KDE")

    fig_kde, _, _ = plot_process_kde_2x2(
        frame,
        variables=("Dw", "V", "LWC", "N"),
        savefig=True,
        output_dir=output_dir,
        filename="kde_Dw_V_LWC_N_by_process.png",
        domain_sigma=kde_domain_sigma,
        variable_limits=KDE_VARIABLE_LIMITS,
        dpi=dpi,
    )
    plt.close(fig_kde)

    if not skip_distance_velocity:
        _require_any_column(
            frame,
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
            frame,
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
        fig_dv, _, _ = plot_process_distance_velocity_kdes(
            frame,
            savefig=True,
            output_dir=output_dir,
            filename="kde_bb_distance_delta_v_by_process.png",
            domain_sigma=kde_domain_sigma,
            dpi=dpi,
            show_legend=True,
        )
        plt.close(fig_dv)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reentrant batch process for rain-process statistics. "
            "Each hourly RAW file produces durable product/sliding checkpoints."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=Path(r"Z:\UGR\mrrpro81\2025"))
    parser.add_argument(
        "--start-date", required=True, help="Inclusive start, e.g. 2025-03-01."
    )
    parser.add_argument(
        "--end-date", required=True, help="Exclusive end, e.g. 2025-03-08."
    )
    parser.add_argument("--raw-pattern", default="*.nc")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workbench/output/bimonthly_process_stats"),
    )
    parser.add_argument(
        "--product-dir",
        type=Path,
        default=Path("workbench/output/bimonthly_process_stats/products"),
    )
    parser.add_argument("--archive-product-dir", type=Path, default=None)
    parser.add_argument("--clean-local-products", action="store_true")
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--window-thickness-m", type=float, default=500.0)
    parser.add_argument(
        "--window-step-m",
        type=float,
        default=None,
        help="Use native vertical resolution when omitted.",
    )
    parser.add_argument("--min-tau-strength", type=float, default=0.3)
    parser.add_argument("--ze-th", type=float, default=-5.0)
    parser.add_argument("--min-points-trend", type=int, default=10)
    parser.add_argument("--save-spe-3d", action="store_true")
    parser.add_argument("--save-dsd-3d", action="store_true")
    parser.add_argument("--force-process", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--aggregate-sample-per-process", type=int, default=50000)
    parser.add_argument("--aggregate-chunksize", type=int, default=100000)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--kde-domain-sigma", type=float, default=3.0)
    parser.add_argument("--skip-distance-velocity", action="store_true")
    parser.add_argument(
        "--exclude-processes",
        nargs="*",
        default=("steady_or_weak", "unknown", "no_data"),
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    product_dir = args.product_dir.resolve()
    archive_product_dir = (
        args.archive_product_dir.resolve() if args.archive_product_dir else None
    )
    manifest_path = output_dir / "run_manifest.jsonl"
    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date)

    output_dir.mkdir(parents=True, exist_ok=True)
    product_dir.mkdir(parents=True, exist_ok=True)
    if archive_product_dir is not None:
        archive_product_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        raw_files = _discover_raw_files(
            raw_root,
            start=start,
            end=end,
            pattern=args.raw_pattern,
        )
        if args.max_files is not None:
            raw_files = raw_files[: args.max_files]
        print(f"[discover] {len(raw_files)} files for {start} <= t < {end}")

        for index, raw_path in enumerate(raw_files, start=1):
            print(f"[process] {index}/{len(raw_files)} {raw_path}")
            record: dict[str, object] = {
                "raw_path": str(raw_path),
                "index": index,
                "total": len(raw_files),
                "started_at": pd.Timestamp.utcnow().isoformat(),
            }
            try:
                frame_one = _build_samples_for_raw_file(
                    raw_path,
                    raw_dir=raw_root,
                    product_dir=product_dir,
                    archive_product_dir=archive_product_dir,
                    clean_local_products=bool(args.clean_local_products),
                    force_process=bool(args.force_process),
                    k=args.k,
                    window_thickness_m=args.window_thickness_m,
                    window_step_m=args.window_step_m,
                    min_tau_strength=args.min_tau_strength,
                    ze_th=args.ze_th,
                    min_points_trend=args.min_points_trend,
                    save_spe_3d=bool(args.save_spe_3d),
                    save_dsd_3d=bool(args.save_dsd_3d),
                )
                record.update(
                    {
                        "status": "ok",
                        "rows": int(len(frame_one)),
                        "finished_at": pd.Timestamp.utcnow().isoformat(),
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "status": "error",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "finished_at": pd.Timestamp.utcnow().isoformat(),
                    }
                )
                print(f"[error] {raw_path}: {exc!r}")
            _append_manifest(manifest_path, record)

    if args.process_only:
        print("[ok] process-only complete; aggregation skipped.")
        return

    combined_csv = output_dir / "combined_column_process_scan.csv"
    print(f"[aggregate] product checkpoints under {product_dir}")
    sample = _aggregate_existing_product_csvs(
        product_dir,
        combined_csv,
        sample_per_process=int(args.aggregate_sample_per_process),
        chunksize=int(args.aggregate_chunksize),
        exclude_processes=tuple(args.exclude_processes or ()),
    )
    sample = _filter_process_rows(
        sample,
        exclude_processes=tuple(args.exclude_processes or ()),
    )
    print("[process-counts] retained sample rows by process:")
    for label, count in (
        sample["proc_label"].astype(str).value_counts().sort_index().items()
    ):
        print(f"  {label}: {count}")

    _write_aggregate_outputs(
        sample,
        output_dir=output_dir,
        dpi=int(args.dpi),
        kde_domain_sigma=float(args.kde_domain_sigma),
        skip_distance_velocity=bool(args.skip_distance_velocity),
    )
    print(f"[ok] combined CSV: {output_dir / 'combined_column_process_scan.csv'}")
    print(f"[ok] combined NetCDF: {output_dir / 'combined_process_samples.nc'}")
    print(f"[ok] counts: {output_dir / 'process_counts.csv'}")


if __name__ == "__main__":
    main()
