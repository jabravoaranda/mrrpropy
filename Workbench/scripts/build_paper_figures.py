from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

try:
    from paper_figures_config import *  # noqa: F403,F401
except ImportError:
    from Workbench.scripts.paper_figures_config import *  # noqa: F403,F401

from mrrpropy.raw_class import MRRProData
from mrrpropy.rain_process_classification.rain_process_algorithm import (
    sliding_rain_classification_to_dataframe,
)
from mrrpropy.plotting.paper import (
    plot_column_process_scan,
    plot_microphysical_tau_triple,
    plot_process_distance_velocity_kdes,
    plot_process_kde_2x2,
    plot_quicklook_comparison,
    plot_quicklook_cps_hex,
    plot_single_column_events,
    plot_sliding_column_process_paper,
    plot_microphysical_properties_triple,
)

matplotlib.use("Agg")


DEFAULT_SCAN_GLOB = (
    "Workbench/output/single_hour/plots/"
    "column_process_events_hexagram_w600_step200/**/column_process_scan*.csv"
)
DEFAULT_OUTPUT_DIR = Path("Workbench/output/paper_figures")
DEFAULT_PRODUCT_DIR = Path("Workbench/output/paper_figures/products")


def _discover_raw_files(raw_dir: Path, pattern: str) -> list[Path]:
    files = sorted(path for path in raw_dir.rglob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No raw files matched {pattern!r} under {raw_dir}.")
    return files


def _processed_product_path(raw_path: Path, raw_dir: Path, product_dir: Path) -> Path:
    existing_folder_product = product_dir / raw_path.stem / f"{raw_path.stem}_raprompro.nc"
    if existing_folder_product.exists():
        return existing_folder_product
    try:
        relative = raw_path.resolve().relative_to(raw_dir.resolve())
    except ValueError:
        relative = raw_path.name
    relative_path = Path(relative).with_suffix("")
    return product_dir / relative_path.parent / f"{relative_path.name}_raprompro.nc"


def _load_or_process_product(
    raw_path: Path,
    *,
    raw_dir: Path,
    product_dir: Path,
    force: bool,
    save_spe_3d: bool,
    save_dsd_3d: bool,
) -> tuple[MRRProData, Path]:
    product_path = _processed_product_path(raw_path, raw_dir, product_dir)
    product_path.parent.mkdir(parents=True, exist_ok=True)
    mrr = MRRProData.from_file(raw_path)
    if product_path.exists() and not force:
        print(f"[product] loading cached product: {product_path}")
        # The paper workflow performs many nearest time/range selections.  An
        # eager load avoids rebuilding the same Dask graph for every sample.
        mrr.load_raprompro(product_path, chunks=None)
        return mrr, product_path
    print(f"[product] processing raw file: {raw_path}")
    mrr.process_raprompro(
        save=True,
        save_spe_3d=save_spe_3d,
        save_dsd_3d=save_dsd_3d,
        output_dir=product_path.parent,
        filename=product_path.name,
    )
    print(f"[product] saved processed product: {product_path}")
    return mrr, product_path


def _nearest_values(
    ds: xr.Dataset,
    *,
    variable: str,
    times: pd.Series,
    ranges_m: pd.Series,
) -> np.ndarray:
    if variable not in ds:
        return np.full(len(times), np.nan, dtype=float)
    values: list[float] = []
    for time_value, range_value in zip(times, ranges_m, strict=False):
        try:
            selected = ds[variable].sel(
                time=np.datetime64(pd.Timestamp(time_value).to_datetime64()),
                range=float(range_value),
                method="nearest",
            )
            values.append(float(selected.values))
        except Exception:
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def _build_samples_for_raw_file(
    raw_path: Path,
    *,
    raw_dir: Path,
    product_dir: Path,
    force_process: bool,
    k: int,
    window_thickness_m: float,
    window_step_m: float | None,
    min_tau_strength: float,
    ze_th: float,
    min_points_trend: int,
    save_spe_3d: bool,
    save_dsd_3d: bool,
) -> pd.DataFrame:
    mrr, product_path = _load_or_process_product(
        raw_path,
        raw_dir=raw_dir,
        product_dir=product_dir,
        force=force_process,
        save_spe_3d=save_spe_3d,
        save_dsd_3d=save_dsd_3d,
    )
    try:
        ds = mrr.raprompro
        if ds is None:
            raise RuntimeError(f"No processed dataset available for {raw_path}.")
        sliding_cache = product_path.parent / f"{raw_path.stem}_raprompro_sliding.csv"
        if sliding_cache.exists() and not force_process:
            print(f"[classification] loading cached sliding scan: {sliding_cache}")
            cached_frame = pd.read_csv(sliding_cache)
            cached_frame["time"] = pd.to_datetime(cached_frame["time"])
            center_col = next((name for name in ("z_center_m", "z_center", "range") if name in cached_frame), "range")
            bottom_col = next((name for name in ("z_bottom_m", "z_bottom") if name in cached_frame), center_col)
            top_col = next((name for name in ("z_top_m", "z_top") if name in cached_frame), center_col)
            velocity_var = next((name for name in ("V", "W", "VEL") if name in ds), None)
            if velocity_var is not None and not cached_frame.get("delta_v_mean", pd.Series(dtype=float)).notna().any():
                bottom_v = _nearest_values(ds, variable=velocity_var, times=cached_frame["time"], ranges_m=cached_frame[bottom_col])
                top_v = _nearest_values(ds, variable=velocity_var, times=cached_frame["time"], ranges_m=cached_frame[top_col])
                cached_frame["delta_v_mean"] = bottom_v - top_v
            bb_var = next((name for name in ("BB_peak", "BB_peak_m", "bb_peak_m") if name in ds), None)
            if bb_var is not None and not cached_frame.get("bb_distance_m", pd.Series(dtype=float)).notna().any():
                bb_values = []
                for time_value in cached_frame["time"]:
                    try:
                        selected = ds[bb_var].sel(time=np.datetime64(pd.Timestamp(time_value).to_datetime64()), method="nearest")
                        bb_values.append(float(np.asarray(selected.values).reshape(-1)[0]))
                    except Exception:
                        bb_values.append(np.nan)
                cached_frame["bb_distance_m"] = pd.to_numeric(cached_frame[center_col], errors="coerce") - np.asarray(bb_values)
            return _add_convenience_columns(cached_frame)
        period = (
            pd.Timestamp(ds["time"].values[0]).floor("s").to_pydatetime(),
            pd.Timestamp(ds["time"].values[-1]).floor("s").to_pydatetime(),
        )
        print(f"[classification] calculating sliding scan: {raw_path.stem}")
        sliding = mrr.sliding_rain_classification(
            period=period,
            k=k,
            window_thickness_m=window_thickness_m,
            window_step_m=window_step_m,
            min_tau_strength=min_tau_strength,
            ze_th=ze_th,
            min_points_trend=min_points_trend,
            vars_trend=("Dm", "Nw", "LWC"),
        )
        frame = (
            sliding_rain_classification_to_dataframe(sliding)
            if isinstance(sliding, xr.Dataset)
            else sliding.copy()
        )
        print(f"[classification] completed sliding scan: {len(frame)} rows")
        frame["time"] = pd.to_datetime(frame["time"])
        center_col = "z_center_m" if "z_center_m" in frame.columns else "range"
        z_bottom_col = "z_bottom_m" if "z_bottom_m" in frame.columns else center_col
        z_top_col = "z_top_m" if "z_top_m" in frame.columns else center_col

        frame["source_raw_file"] = str(raw_path)
        frame["source_product_file"] = str(product_path)
        frame["Dw"] = _nearest_values(
            ds,
            variable="Dw" if "Dw" in ds else "Dm",
            times=frame["time"],
            ranges_m=frame[center_col],
        )
        frame["N"] = _nearest_values(
            ds,
            variable="N" if "N" in ds else "Nw",
            times=frame["time"],
            ranges_m=frame[center_col],
        )
        frame["LWC"] = _nearest_values(
            ds,
            variable="LWC",
            times=frame["time"],
            ranges_m=frame[center_col],
        )
        velocity_var = next((name for name in ("V", "W", "VEL") if name in ds), None)
        if velocity_var is not None:
            if "V" not in frame.columns or not pd.to_numeric(frame["V"], errors="coerce").notna().any():
                frame["V"] = _nearest_values(
                    ds,
                    variable=velocity_var,
                    times=frame["time"],
                    ranges_m=frame[center_col],
                )
            if "delta_v_mean" not in frame.columns or not pd.to_numeric(frame["delta_v_mean"], errors="coerce").notna().any():
                bottom_v = _nearest_values(
                    ds,
                    variable=velocity_var,
                    times=frame["time"],
                    ranges_m=frame[z_bottom_col],
                )
                top_v = _nearest_values(
                    ds,
                    variable=velocity_var,
                    times=frame["time"],
                    ranges_m=frame[z_top_col],
                )
                frame["delta_v_mean"] = bottom_v - top_v
        elif "delta_v_mean" not in frame.columns:
            frame["delta_v_mean"] = np.nan

        bb_var = next((name for name in ("BB_peak", "BB_peak_m", "bb_peak_m") if name in ds), None)
        if bb_var is not None:
            bb_values = []
            for time_value in frame["time"]:
                try:
                    selected = ds[bb_var].sel(
                        time=np.datetime64(pd.Timestamp(time_value).to_datetime64()),
                        method="nearest",
                    )
                    bb_values.append(float(np.asarray(selected.values).reshape(-1)[0]))
                except Exception:
                    bb_values.append(np.nan)
            frame["bb_distance_m"] = pd.to_numeric(frame[center_col], errors="coerce") - np.asarray(bb_values)
        else:
            frame["bb_distance_m"] = np.nan
        frame = _add_convenience_columns(frame)
        frame.to_csv(sliding_cache, index=False)
        return frame
    finally:
        mrr.close()


def _target_time(frame: pd.DataFrame, offset_minutes: float) -> pd.Timestamp:
    start = pd.Timestamp(frame["time"].min())
    end = pd.Timestamp(frame["time"].max())
    candidate = start + pd.Timedelta(minutes=offset_minutes)
    return min(max(candidate, start), end)


def _bright_band_y_limits(
    subject: MRRProData | pd.DataFrame | xr.Dataset,
    *,
    target_datetime: pd.Timestamp | np.datetime64 | str,
    range_limits: tuple[float, float] | None,
    pad_m: float = 250.0,
) -> tuple[float, float] | None:
    if range_limits is None:
        return None

    lower_m = float(range_limits[0])

    if isinstance(subject, pd.DataFrame):
        frame = subject.copy()
        if "time" not in frame.columns:
            return None
        frame["time"] = pd.to_datetime(frame["time"])
        target = pd.Timestamp(target_datetime)
        nearest = frame.loc[(frame["time"] - target).abs().idxmin(), "time"]
        column = frame[frame["time"] == nearest]
        if column.empty:
            return None
        if "bb_distance_m" in column.columns:
            bb_values = pd.to_numeric(column["bb_distance_m"], errors="coerce")
            if bb_values.notna().any():
                upper_m = float(np.nanmin(bb_values.to_numpy(dtype=float))) - pad_m
                return (lower_m / 1000.0, upper_m / 1000.0)
        if "range" in column.columns:
            range_vals = pd.to_numeric(column["range"], errors="coerce")
            if range_vals.notna().any():
                upper_m = float(np.nanmin(range_vals.to_numpy(dtype=float))) - pad_m
                return (lower_m / 1000.0, upper_m / 1000.0)
        return None

    if isinstance(subject, xr.Dataset):
        ds = subject
    else:
        ds = getattr(subject, "raprompro", None)
    if ds is None:
        return None

    bb_var = next((name for name in ("BB_peak", "BB_peak_m", "bb_peak_m") if name in ds), None)
    if bb_var is None:
        return None
    try:
        selected = ds[bb_var].sel(
            time=np.datetime64(pd.Timestamp(target_datetime).to_datetime64()),
            method="nearest",
        )
        values = np.asarray(selected.values, dtype=float).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        upper_m = float(np.nanmin(finite)) - pad_m
    except Exception:
        return None
    return (lower_m / 1000.0, upper_m / 1000.0)


def _make_per_file_figures(
    mrr: MRRProData,
    frame: pd.DataFrame,
    *,
    raw_path: Path,
    output_dir: Path,
    target_time_offset_minutes: float,
    spectrum_var: str,
    range_limits: tuple[float, float] | None,
    short_range_limits: tuple[float, float],
    processes: list[str] | None,
    dpi: int,
) -> list[Path]:
    """Write the eight single-file figures; aggregate KDEs are made later."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = _target_time(frame, target_time_offset_minutes)
    stem = raw_path.stem
    written: list[Path] = []

    figure_calls = [
        ("01 quicklook raw/processed", lambda: plot_quicklook_comparison(
            mrr, variable="Ze", vmin=-10, vmax=40, figsize=(12.0, 5.2),
            y_limits=range_limits,
            savefig=True, output_dir=output_dir,
            filename=f"{stem}_01_quicklook_raw_processed.png", dpi=dpi,
        )[-1]),
        ("02 MPP triple", lambda: plot_microphysical_properties_triple(
            mrr, target_datetime=target, figsize=(12.0, 6.0),
            y_limits=_bright_band_y_limits(mrr, target_datetime=target, range_limits=range_limits),
            savefig=True, output_dir=output_dir,
            filename=f"{stem}_02_mpp_triple.png", dpi=dpi,
        )[-1]),
        ("03 MPP triple short range", lambda: plot_microphysical_properties_triple(
            mrr, target_datetime=target, figsize=(12.0, 5.5),
            y_limits=short_range_limits, savefig=True, output_dir=output_dir,
            filename=f"{stem}_03_mpp_triple_short_range.png", dpi=dpi,
        )[-1]),
        ("04 single column events", lambda: plot_single_column_events(
            frame, target_datetime=target, figsize=(4.0, 7.0),
            savefig=True, output_dir=output_dir,
            filename=f"{stem}_04_single_column_events.png", dpi=dpi,
        )[-1]),
        ("05 tau/Thiel MPP triple", lambda: plot_microphysical_tau_triple(
            frame, target_datetime=target, variables=("Dm", "Nw", "LWC"),
            figsize=(12.0, 6.0),
            y_limits=_bright_band_y_limits(mrr, target_datetime=target, range_limits=range_limits),
            savefig=True, output_dir=output_dir,
            filename=f"{stem}_05_mpp_tau_triple.png", dpi=dpi,
        )[-1]),
        ("06 canonical sliding column process", lambda: plot_sliding_column_process_paper(
            mrr, frame, processes=processes, figsize=(14.0, 8.0),
            y_limits=(range_limits[0] / 1000.0, range_limits[1] / 1000.0) if range_limits else None,
            savefig=True, output_dir=output_dir,
            filename=f"{stem}_06_column_process_scan.png", dpi=dpi,
        )[-1]),
        ("07 column process scan hex", lambda: plot_column_process_scan(
            frame, color_mode="hex", processes=processes,
            figsize=(14.0, 8.0), range_limits=range_limits, show_legend=False,
            marker_size=52.0, label_fs=13.0, tick_fs=10.0,
            savefig=True, output_dir=output_dir,
            filename=f"{stem}_07_column_process_scan_hex.png", dpi=dpi,
        )[-1]),
        ("08 quicklook CPS hex triptych", lambda: plot_quicklook_cps_hex(
            mrr, frame, processes=processes, range_limits=range_limits,
            figsize=(18.0, 6.0), savefig=True,
            output_dir=output_dir,
            filename=f"{stem}_08_quicklook_cps_hex.png", dpi=dpi,
        )[-1]),
    ]
    for figure_name, make_figure in figure_calls:
        print(f"[figure] {stem}: {figure_name}")
        try:
            path = make_figure()
        except (KeyError, RuntimeError, ValueError) as exc:
            print(f"[warn] {stem}: skipped {figure_name} ({exc})")
            continue
        if path is not None:
            written.append(path)
    return written


def _samples_to_xarray(frame: pd.DataFrame) -> xr.Dataset:
    df = frame.reset_index(drop=True).copy()
    coords = {"sample": np.arange(len(df), dtype=int)}
    data_vars: dict[str, tuple[tuple[str], np.ndarray]] = {}
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            data_vars[column] = (("sample",), series.to_numpy(dtype="datetime64[ns]"))
        elif pd.api.types.is_numeric_dtype(series):
            data_vars[column] = (("sample",), pd.to_numeric(series, errors="coerce").to_numpy(dtype=float))
        else:
            data_vars[column] = (("sample",), series.astype(str).to_numpy())
    out = xr.Dataset(data_vars=data_vars, coords=coords)
    out.attrs["description"] = "Aggregated sample-level process features for paper figures."
    out.attrs["sample_dimension"] = "Each sample is one time x sliding-window classification row."
    return out


def _read_scan_csvs(patterns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pattern in patterns:
        paths = sorted(Path().glob(pattern))
        if not paths:
            print(f"[warn] no files matched: {pattern}")
            continue
        for path in paths:
            frame = pd.read_csv(path)
            if frame.empty:
                print(f"[skip] {path} (0 rows)")
                continue
            frame["source_file"] = str(path)
            frames.append(frame)
            print(f"[read] {path} ({len(frame)} rows)")
    if not frames:
        raise FileNotFoundError("No scan CSV files were found.")
    return pd.concat(frames, ignore_index=True)


def _add_convenience_columns(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "Dw" not in df.columns:
        for candidate in ("Dm_layer_mean", "Dm_mean", "Dm_top"):
            if candidate in df.columns:
                df["Dw"] = pd.to_numeric(df[candidate], errors="coerce")
                break
    if "N" not in df.columns:
        for candidate in ("Nw_layer_mean", "Nw_mean", "Nw_top"):
            if candidate in df.columns:
                df["N"] = pd.to_numeric(df[candidate], errors="coerce")
                break
    if "LWC" not in df.columns:
        for candidate in ("LWC_layer_mean", "LWC_mean", "LWC_top"):
            if candidate in df.columns:
                df["LWC"] = pd.to_numeric(df[candidate], errors="coerce")
                break
    if "V" not in df.columns:
        for candidate in ("W", "VEL", "v_mean_top", "v_mean_bottom", "V_layer_mean", "W_layer_mean", "VEL_layer_mean"):
            if candidate in df.columns:
                df["V"] = pd.to_numeric(df[candidate], errors="coerce")
                break
    if "bb_distance_m" not in df.columns:
        for bb_col in ("BB_peak", "BB_peak_m", "bb_peak_m"):
            if bb_col in df.columns and "z_center_m" in df.columns:
                df["bb_distance_m"] = (
                    pd.to_numeric(df["z_center_m"], errors="coerce")
                    - pd.to_numeric(df[bb_col], errors="coerce")
                )
                break
    if "bb_distance_m" not in df.columns:
        for candidate in ("dist_bb_peak", "dist_bb_bottom"):
            if candidate in df.columns:
                df["bb_distance_m"] = pd.to_numeric(df[candidate], errors="coerce")
                break
    return df


def _require_any_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> None:
    if any(column in frame.columns for column in candidates):
        return
    joined = ", ".join(candidates)
    raise KeyError(
        f"Cannot make {label}; expected one of these columns: {joined}. "
        "Add those features to the scan CSVs or pass CSVs that already contain them."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the ten paper-ready figures from a folder of raw files."
    )
    parser.add_argument(
        "--scan-glob",
        action="append",
        default=None,
        help=(
            "Optional fallback glob for precomputed scan CSVs. Can be supplied multiple times. "
            f"Default: {DEFAULT_SCAN_GLOB}"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Folder of raw .nc files to process and aggregate.",
    )
    parser.add_argument(
        "--raw-pattern",
        default=RAW_PATTERN,
        help="Recursive filename pattern used with --raw-dir.",
    )
    parser.add_argument(
        "--product-dir",
        type=Path,
        default=PRODUCT_DIR,
        help="Cache directory for processed *_raprompro.nc products.",
    )
    parser.add_argument("--force-process", action="store_true", default=FORCE_PROCESS, help="Reprocess raw files even when cached products exist.")
    parser.add_argument(
        "--reset-cache",
        type=int,
        choices=(0, 1),
        default=RESET_CACHE,
        help="Set to 1 to rebuild cached processed products and sliding scans.",
    )
    parser.add_argument("--k", type=int, default=K, help="Hexagram resolution for sliding classification.")
    parser.add_argument("--window-thickness-m", type=float, default=WINDOW_THICKNESS_M, help="Sliding window thickness.")
    parser.add_argument("--window-step-m", type=float, default=WINDOW_STEP_M, help="Sliding window step. Omit to use native range spacing.")
    parser.add_argument("--min-tau-strength", type=float, default=MIN_TAU_STRENGTH, help="Minimum trend strength for process classification.")
    parser.add_argument("--ze-th", type=float, default=ZE_TH, help="Reflectivity threshold.")
    parser.add_argument("--min-points-trend", type=int, default=MIN_POINTS_TREND, help="Minimum points per trend estimate.")
    parser.add_argument("--save-spe-3d", action="store_true", default=SAVE_SPE_3D, help="Store spe_3D in cached processed products.")
    parser.add_argument("--save-dsd-3d", action="store_true", default=SAVE_DSD_3D, help="Store dsd_3D in cached processed products.")
    parser.add_argument("--spectrum-var", default=SPECTRUM_VAR, help="Processed spectrum variable for the spectrogram.")
    parser.add_argument("--target-time-offset-minutes", type=float, default=TARGET_TIME_OFFSET_MINUTES, help="Target profile time measured from each file start.")
    parser.add_argument("--range-limits", nargs=2, type=float, default=RANGE_LIMITS, metavar=("MIN_M", "MAX_M"), help="Full figure height limits in metres.")
    parser.add_argument("--short-range-limits", nargs=2, type=float, default=SHORT_RANGE_LIMITS, metavar=("MIN_KM", "MAX_KM"), help="Short MPP height limits in kilometres.")
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI, help="Output resolution.")
    parser.add_argument(
        "--kde-domain-sigma",
        type=float,
        default=KDE_DOMAIN_SIGMA,
        help="KDE domain is approximately the union of per-process mean +/- N standard deviations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where figure PNGs and the combined array will be written.",
    )
    parser.add_argument(
        "--processes",
        nargs="*",
        default=PROCESSES,
        help="Optional process labels to keep before plotting.",
    )
    parser.add_argument("--skip-distance-velocity", action="store_true", help="Skip the tenth figure.")
    args = parser.parse_args()
    reset_cache = bool(args.force_process or args.reset_cache)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_dir is not None and not args.scan_glob:
        raw_dir = args.raw_dir.resolve()
        product_dir = args.product_dir.resolve()
        frames = []
        for raw_path in _discover_raw_files(raw_dir, args.raw_pattern):
            print(f"[process] {raw_path}")
            frame_one = _build_samples_for_raw_file(
                raw_path,
                raw_dir=raw_dir,
                product_dir=product_dir,
                force_process=reset_cache,
                k=args.k,
                window_thickness_m=args.window_thickness_m,
                window_step_m=args.window_step_m,
                min_tau_strength=args.min_tau_strength,
                ze_th=args.ze_th,
                min_points_trend=args.min_points_trend,
                save_spe_3d=args.save_spe_3d,
                save_dsd_3d=args.save_dsd_3d,
            )
            per_file_output = output_dir / raw_path.stem
            figure_mrr, _figure_product = _load_or_process_product(
                raw_path,
                raw_dir=raw_dir,
                product_dir=product_dir,
                # The first call above has already refreshed the cache when
                # requested; load that result for figure generation.
                force=False,
                save_spe_3d=args.save_spe_3d,
                save_dsd_3d=args.save_dsd_3d,
            )
            figure_paths = _make_per_file_figures(
                figure_mrr,
                frame_one,
                raw_path=raw_path,
                output_dir=per_file_output,
                target_time_offset_minutes=args.target_time_offset_minutes,
                spectrum_var=args.spectrum_var,
                range_limits=tuple(args.range_limits),
                short_range_limits=tuple(args.short_range_limits),
                processes=args.processes,
                dpi=args.dpi,
            )
            figure_mrr.close()
            print(f"[ok] paper figures: {len(figure_paths)}")
            frames.append(frame_one)
            print(f"[ok] samples: {len(frame_one)}")
        frame = pd.concat(frames, ignore_index=True)
    else:
        scan_globs = args.scan_glob
        if not scan_globs:
            raise ValueError("Set RAW_DIR in paper_figures_config.py or pass --raw-dir.")
        frame = _add_convenience_columns(_read_scan_csvs(scan_globs))

    if args.processes:
        selected = {str(process) for process in args.processes}
        frame = frame[frame["proc_label"].astype(str).isin(selected)].copy()
    if frame.empty:
        raise ValueError("No rows remain after filtering.")

    combined_csv = output_dir / "combined_column_process_scan.csv"
    frame.to_csv(combined_csv, index=False)
    combined_nc = output_dir / "combined_process_samples.nc"
    _samples_to_xarray(frame).to_netcdf(combined_nc)

    _require_any_column(frame, ("Dw", "Dm_layer_mean", "Dm_top"), "Dw KDE")
    _require_any_column(frame, ("V", "W", "VEL", "v_mean_top", "V_layer_mean", "W_layer_mean", "VEL_layer_mean"), "V KDE")
    _require_any_column(frame, ("LWC", "LWC_layer_mean", "LWC_top"), "LWC KDE")
    _require_any_column(frame, ("N", "Nw_layer_mean", "Nw_top"), "N KDE")

    fig_kde, _, path_kde = plot_process_kde_2x2(
        frame,
        variables=("Dw", "V", "LWC", "N"),
        savefig=True,
        output_dir=output_dir,
        filename="kde_Dw_V_LWC_N_by_process.png",
        domain_sigma=args.kde_domain_sigma,
        dpi=args.dpi,
    )
    plt.close(fig_kde)
    print(f"[ok] 2x2 KDE: {path_kde}")

    if not args.skip_distance_velocity:
        _require_any_column(
            frame,
            ("bb_distance_m", "BB_distance_m", "dist_bb_peak", "dist_bb_bottom", "distance_to_bb_m", "z_center_minus_bb_m"),
            "bright-band distance KDE",
        )
        _require_any_column(
            frame,
            ("delta_v_mean", "delta_v_p50", "delta_v", "V_delta", "delta_V", "delta_velocity", "velocity_difference"),
            "delta-v KDE",
        )
        fig_dv, _, path_dv = plot_process_distance_velocity_kdes(
            frame,
            savefig=True,
            output_dir=output_dir,
            filename="kde_bb_distance_delta_v_by_process.png",
            domain_sigma=args.kde_domain_sigma,
            dpi=args.dpi,
        )
        plt.close(fig_dv)
        print(f"[ok] BB/delta-v KDE: {path_dv}")

    finite_rows = int(np.isfinite(pd.to_numeric(frame.get("Dw"), errors="coerce")).sum()) if "Dw" in frame else 0
    print(f"[ok] combined CSV: {combined_csv}")
    print(f"[ok] combined xarray: {combined_nc}")
    print(f"[ok] rows: {len(frame)}; finite Dw rows: {finite_rows}")


if __name__ == "__main__":
    main()
