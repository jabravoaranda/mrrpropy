from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import xarray as xr

from mrrpropy.analysis.classified_rain_process_metrics import build_process_dynamics_dataframe
from mrrpropy.analysis.rain_processes_classification import (
    classify_rain_process,
    rain_process_analyze,
)
from mrrpropy.analysis.trends import compute_layer_trend


class SupportsRainAnalysis(Protocol):
    path: str | Path
    raprompro: xr.Dataset | None

    def _is_processed(self) -> bool: ...


def _float_from_dynamic(value: object, *, name: str) -> float:
    return float(cast(Any, value))


def _int_from_dynamic(value: object, *, name: str) -> int:
    return int(cast(Any, value))


def _optional_micro_cfg_value(
    subject: object,
    name: str,
    default: object,
) -> Any:
    if not hasattr(subject, "micro_cfg"):
        return default
    micro_cfg = cast(Any, subject).micro_cfg
    if not hasattr(micro_cfg, name):
        return default
    return getattr(micro_cfg, name)


# TODO: Resolve micro_cfg defaults in MRRProData methods and pass explicit
# values into analysis functions, so analysis modules do not need subject-level
# dynamic configuration lookups.
def _micro_cfg_value(subject: object, name: str) -> Any:
    if not hasattr(subject, "micro_cfg"):
        raise RuntimeError(
            "subject.micro_cfg is missing; cannot resolve default parameters."
        )
    micro_cfg = cast(Any, subject).micro_cfg
    if not hasattr(micro_cfg, name):
        raise RuntimeError(f"subject.micro_cfg.{name} is missing.")
    return getattr(micro_cfg, name)


def _optional_raprompro_dataset(subject: object) -> xr.Dataset | None:
    if not hasattr(subject, "raprompro"):
        return None
    return cast(Any, subject).raprompro


def _resolve_processed_dataset(subject: SupportsRainAnalysis) -> xr.Dataset:
    if not subject._is_processed():
        raise RuntimeError("MRR-Pro data not processed (raprompro missing).")
    ds = subject.raprompro
    if ds is None:
        raise RuntimeError("raprompro not loaded. Use load_raprompro().")
    return ds


def _build_sliding_layer_windows(
    range_values: np.ndarray,
    *,
    window_thickness_m: float,
    window_step_m: float,
) -> list[tuple[float, float]]:
    finite_ranges = np.sort(np.asarray(range_values, dtype=float)[np.isfinite(range_values)])
    if finite_ranges.size == 0:
        return []

    z_min = float(finite_ranges.min())
    z_max = float(finite_ranges.max())
    if window_thickness_m <= 0.0:
        raise ValueError("window_thickness_m must be positive.")
    if window_step_m <= 0.0:
        raise ValueError("window_step_m must be positive.")
    if z_max - z_min < window_thickness_m:
        return []

    starts = np.arange(
        z_min,
        z_max - float(window_thickness_m) + float(window_step_m) * 0.5,
        float(window_step_m),
        dtype=float,
    )
    windows: list[tuple[float, float]] = []
    for start in starts:
        stop = float(start + window_thickness_m)
        if stop <= z_max + 1e-6:
            windows.append((float(start), stop))
    return windows


def _detect_process_runs(
    sliding_df: pd.DataFrame,
    *,
    min_consecutive_profiles: int,
    ignored_labels: set[str] | None = None,
) -> pd.DataFrame:
    if min_consecutive_profiles <= 0:
        raise ValueError("min_consecutive_profiles must be positive.")
    if sliding_df.empty:
        return pd.DataFrame()

    ignored = (
        {"no_data", "unknown", "steady_or_weak"}
        if ignored_labels is None
        else set(ignored_labels)
    )

    rows: list[dict[str, object]] = []
    df = sliding_df.sort_values(["window_id", "time"]).copy()

    for window_id, group in df.groupby("window_id", sort=True):
        labels = group["proc_label"].astype(str).to_numpy()
        times = pd.to_datetime(group["time"]).to_numpy()
        if labels.size == 0:
            continue

        start = 0
        for idx in range(1, labels.size + 1):
            closed = idx == labels.size or labels[idx] != labels[start]
            if not closed:
                continue

            label = labels[start]
            run_length = idx - start
            if label not in ignored and run_length >= min_consecutive_profiles:
                run = group.iloc[start:idx]
                dt_seconds = np.nan
                if run_length > 1:
                    diffs = np.diff(times[start:idx]) / np.timedelta64(1, "s")
                    if diffs.size:
                        dt_seconds = float(np.nanmedian(diffs))
                if not np.isfinite(dt_seconds):
                    dt_seconds = np.nan
                duration_seconds = (
                    float(run_length * dt_seconds) if np.isfinite(dt_seconds) else np.nan
                )

                row: dict[str, object] = {
                    "window_id": int(window_id),
                    "proc_label": str(label),
                    "start_time": pd.Timestamp(times[start]),
                    "end_time": pd.Timestamp(times[idx - 1]),
                    "n_profiles": int(run_length),
                    "dt_seconds": dt_seconds,
                    "duration_seconds": duration_seconds,
                }

                for meta_field in (
                    "z_min_m",
                    "z_max_m",
                    "z_bottom_m",
                    "z_top_m",
                    "z_center_m",
                    "window_thickness_m",
                    "window_step_m",
                    "trend_method",
                    "selection_mode",
                ):
                    if meta_field in run.columns:
                        row[meta_field] = run.iloc[0][meta_field]

                for metric in (
                    "proc_strength",
                    "Dm_delta_pct",
                    "Nw_delta_pct",
                    "LWC_delta_pct",
                    "Dm_rate_per_km",
                    "Nw_rate_per_km",
                    "LWC_rate_per_km",
                    "tau_Dm",
                    "tau_Nw",
                    "tau_LWC",
                    "trend_strength_Dm",
                    "trend_strength_Nw",
                    "trend_strength_LWC",
                    "trend_score_Dm",
                    "trend_score_Nw",
                    "trend_score_LWC",
                ):
                    if metric in run.columns:
                        values = pd.to_numeric(run[metric], errors="coerce")
                        finite = values[np.isfinite(values)]
                        row[f"{metric}_mean"] = (
                            float(finite.mean()) if not finite.empty else np.nan
                        )
                        row[f"{metric}_median"] = (
                            float(finite.median()) if not finite.empty else np.nan
                        )

                rows.append(row)

            start = idx

    episodes = pd.DataFrame(rows)
    if episodes.empty:
        return episodes
    return episodes.sort_values(
        by=["start_time", "z_min_m", "proc_label"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_sliding_process_dataframe(
    subject: SupportsRainAnalysis,
    *,
    period: tuple[datetime, datetime],
    k: int,
    window_thickness_m: float | None = None,
    window_step_m: float | None = None,
    min_tau_strength: float | None = None,
    ze_th: float = -5.0,
    trend_method: str = "kendall_theilsen",
    tau_zero_tol: float = 0.05,
    min_points_trend: int | None = None,
    min_points_ols: int | None = None,
    eps_q: float = 0.01,
    rgb_q: float = 0.02,
    vars_trend: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
    max_tau_pvalue: float | None = None,
) -> pd.DataFrame:
    """
    Apply the rain-process analysis across the processed column with a sliding
    vertical window.

    For each window, the function runs the standard rain-process analysis and
    classification pipeline, then exports a per-sample dataframe. The output is
    therefore indexed by both time and layer window, and is intended as the
    input for consecutive-profile episode detection.

    When the caller does not provide ``window_thickness_m``, ``window_step_m``
    or ``min_tau_strength``, and ``subject`` exposes a ``micro_cfg`` attribute,
    the corresponding values are taken from that configuration.

    ``window_step_m=None`` means "raw resolution": infer the window step from the
    native range-grid spacing (median of the range-coordinate differences).
    """
    ds = _resolve_processed_dataset(subject)
    if period[0] >= period[1]:
        raise ValueError("period must be increasing (start, end).")

    thickness_m = (
        float(window_thickness_m)
        if window_thickness_m is not None
        else float(_optional_micro_cfg_value(subject, "window_thickness_m", 1000.0))
    )

    if window_step_m is None:
        values = np.asarray(ds["range"].values, dtype=float)
        diffs = np.abs(np.diff(values))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size == 0:
            raise ValueError("Cannot infer raw vertical resolution from ds['range'].")
        step_m = float(np.median(diffs))
    else:
        step_m = float(window_step_m)

    tau_strength = min_tau_strength

    windows = _build_sliding_layer_windows(
        ds["range"].values,
        window_thickness_m=thickness_m,
        window_step_m=step_m,
    )
    if not windows:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for window_id, (z_min_m, z_max_m) in enumerate(windows):
        analysis = rain_process_analyze(
            subject,
            period=period,
            z_bottom_m=z_min_m,
            z_top_m=z_max_m,
            k=k,
            ze_th=ze_th,
            trend_method=trend_method,
            tau_zero_tol=tau_zero_tol,
            min_points_trend=min_points_trend,
            min_points_ols=min_points_ols,
            eps_q=eps_q,
            rgb_q=rgb_q,
            vars_trend=vars_trend,
        )
        classified = classify_rain_process(
            subject,
            analysis=analysis,
            min_tau_strength=(
                None
                if tau_strength is None
                else _float_from_dynamic(tau_strength, name="min_tau_strength")
            ),
            max_tau_pvalue=max_tau_pvalue,
        )
        frame = build_process_dynamics_dataframe(
            subject,
            analysis=analysis,
            classified=classified,
            variables=vars_trend,
        ).reset_index()
        frame["window_id"] = int(window_id)
        frame["z_min_m"] = float(z_min_m)
        frame["z_max_m"] = float(z_max_m)
        frame["z_bottom_m"] = float(z_min_m)
        frame["z_top_m"] = float(z_max_m)
        frame["z_center_m"] = float(0.5 * (z_min_m + z_max_m))
        frame["window_thickness_m"] = float(thickness_m)
        frame["window_step_m"] = float(step_m)
        frame["trend_method"] = str(analysis.attrs.get("trend_method", trend_method))
        frame["selection_mode"] = "sliding"
        frames.append(frame)

    sliding_df = pd.concat(frames, ignore_index=True)
    sliding_df.attrs = {
        "period_start": str(np.datetime_as_string(np.datetime64(period[0]), unit="s")),
        "period_end": str(np.datetime_as_string(np.datetime64(period[1]), unit="s")),
        "window_thickness_m": float(thickness_m),
        "window_step_m": float(step_m),
        "min_tau_strength": (
            None
            if tau_strength is None
            else _float_from_dynamic(tau_strength, name="min_tau_strength")
        ),
        "trend_method": str(trend_method),
        "tau_zero_tol": float(tau_zero_tol),
        "k": int(k),
        "selection_mode": "sliding",
    }
    return sliding_df


def build_fused_column_process_dataframe(
    subject: SupportsRainAnalysis,
    sliding_df: pd.DataFrame,
    *,
    min_consecutive: int = 3,
    allowed_processes: tuple[str, ...] | None = None,
    exclude_processes: tuple[str, ...] = ("unknown", "steady_or_weak"),
    process_col: str = "proc_label",
    time_col: str = "time",
    z_top_col: str = "z_top",
    z_bottom_col: str = "z_bottom",
    variable_threshold: str | None = None,
    threshold_value: float | None = None,
    trend_method: str | None = None,
    tau_zero_tol: float | None = None,
    min_points_trend: int | None = None,
    vars_trend: tuple[str, str, str] | None = None,
) -> pd.DataFrame:
    """
    Exploratory Option B: fuse vertical sliding-window detections into
    consolidated layers.

    The input ``sliding_df`` is expected to be the dataframe returned by
    :func:`build_sliding_process_dataframe`. For each time step, the
    function searches for *vertically adjacent* runs of the same process label,
    fuses each run into one vertical layer, recomputes the microphysical trends
    on the fused layer using ``subject.raprompro``, and reclassifies the fused
    layer with the standard process classifier.

    Grouping logic (per time step)
    ------------------------------
    - Rows are sorted vertically (top-to-bottom) so adjacent rows represent
      adjacent sliding windows in height.
    - A run is a *strictly adjacent* sequence of rows with the same process
      label. Labels separated by other labels are never grouped.
    - By default, labels in ``exclude_processes`` are ignored and also break
      adjacency.
    - If ``allowed_processes`` is provided, only those labels are considered
      (other labels break adjacency).
    - Only runs with ``len(run) >= min_consecutive`` are fused.

    Fused-layer recomputation
    -------------------------
    Trends are recomputed on the actual fused layer bounds, not inferred from
    the individual sliding-window rows. For each fused event, the trend is
    recomputed on a single time step by subsetting ``subject.raprompro`` to the
    event time.

    Any argument left as ``None`` falls back to ``subject.micro_cfg``:
    ``variable_threshold``, ``threshold_value``, ``trend_method``,
    ``tau_zero_tol``, ``min_points_trend``, and ``vars_trend``.

    Robustness
    ----------
    If recomputation fails for a given fused event, the function keeps that
    event and populates recomputed fields with NaNs, while recording a short
    error message in ``recompute_error``.
    """
    if not isinstance(sliding_df, pd.DataFrame):
        raise TypeError("sliding_df must be a pandas DataFrame.")
    if sliding_df.empty:
        return pd.DataFrame()
    if min_consecutive <= 0:
        raise ValueError("min_consecutive must be positive.")

    ds = _optional_raprompro_dataset(subject)
    if ds is None:
        raise RuntimeError("subject.raprompro is missing; load the processed dataset first.")

    resolved_variable_threshold = (
        str(variable_threshold)
        if variable_threshold is not None
        else str(_micro_cfg_value(subject, "variable_threshold"))
    )
    resolved_threshold_value = (
        float(threshold_value)
        if threshold_value is not None
        else float(_micro_cfg_value(subject, "threshold_value"))
    )
    resolved_trend_method = (
        str(trend_method)
        if trend_method is not None
        else str(_micro_cfg_value(subject, "trend_method"))
    )
    resolved_tau_zero_tol = (
        float(tau_zero_tol)
        if tau_zero_tol is not None
        else float(_micro_cfg_value(subject, "tau_zero_tol"))
    )
    resolved_min_points_trend = (
        int(min_points_trend)
        if min_points_trend is not None
        else int(_micro_cfg_value(subject, "min_points_trend"))
    )
    resolved_vars_trend_raw = (
        tuple(vars_trend)
        if vars_trend is not None
        else tuple(cast(Any, _micro_cfg_value(subject, "vars_trend")))
    )
    if len(resolved_vars_trend_raw) != 3:
        raise ValueError("vars_trend must contain exactly three variable names.")
    resolved_vars_trend = cast(
        tuple[str, str, str],
        tuple(str(variable) for variable in resolved_vars_trend_raw),
    )

    exclude_set = set(exclude_processes)
    allowed_set = set(allowed_processes) if allowed_processes is not None else None

    def _resolve_column(df: pd.DataFrame, requested: str, alternatives: tuple[str, ...]) -> str:
        if requested in df.columns:
            return requested
        for alt in alternatives:
            if alt in df.columns:
                return alt
        raise KeyError(
            f"sliding_df is missing column {requested!r}. Available columns: {list(df.columns)!r}"
        )

    # `build_sliding_process_dataframe` uses z_bottom_m/z_top_m; keep the public
    # signature generic, but accept the package-native names transparently.
    resolved_time_col = _resolve_column(sliding_df, time_col, ("time",))
    resolved_process_col = _resolve_column(sliding_df, process_col, ("proc_label",))
    resolved_z_top_col = _resolve_column(sliding_df, z_top_col, ("z_top_m", "z_max_m"))
    resolved_z_bottom_col = _resolve_column(sliding_df, z_bottom_col, ("z_bottom_m", "z_min_m"))

    has_window_id = "window_id" in sliding_df.columns

    def _iter_vertical_runs(df_t: pd.DataFrame) -> list[dict[str, object]]:
        if df_t.empty:
            return []

        df_sorted = df_t.copy()
        if has_window_id:
            # In the sliding-window workflow, window_id increases with height.
            df_sorted = df_sorted.sort_values("window_id", ascending=False)
        else:
            # Generic fallback: sort by top height (highest first), then bottom.
            df_sorted = df_sorted.sort_values(
                [resolved_z_top_col, resolved_z_bottom_col],
                ascending=[False, False],
            )

        labels = df_sorted[resolved_process_col].astype(str).to_numpy()
        z_top_vals = pd.to_numeric(df_sorted[resolved_z_top_col], errors="coerce").to_numpy(dtype=float)
        z_bottom_vals = pd.to_numeric(df_sorted[resolved_z_bottom_col], errors="coerce").to_numpy(dtype=float)

        runs: list[dict[str, object]] = []

        def _eligible(label: str) -> bool:
            if label in exclude_set:
                return False
            if allowed_set is not None and label not in allowed_set:
                return False
            return True

        active_label: str | None = None
        start: int | None = None

        def _close(end: int) -> None:
            nonlocal active_label, start
            if active_label is None or start is None:
                return
            run_len = int(end - start)
            if run_len >= int(min_consecutive):
                top = float(np.nanmax(z_top_vals[start:end]))
                bottom = float(np.nanmin(z_bottom_vals[start:end]))
                if np.isfinite(top) and np.isfinite(bottom) and top > bottom:
                    run: dict[str, object] = {
                        "run_process_label": str(active_label),
                        "z_top_fused": top,
                        "z_bottom_fused": bottom,
                        "thickness_fused": float(top - bottom),
                        "n_windows_merged": int(run_len),
                    }
                    if has_window_id:
                        window_ids = pd.to_numeric(
                            df_sorted["window_id"].iloc[start:end],
                            errors="coerce",
                        ).to_numpy(dtype=float)
                        finite_ids = window_ids[np.isfinite(window_ids)]
                        if finite_ids.size:
                            run["window_id_top"] = int(np.max(finite_ids))
                            run["window_id_bottom"] = int(np.min(finite_ids))
                    runs.append(run)
            active_label = None
            start = None

        for idx, label in enumerate(labels):
            label_str = str(label)
            if not _eligible(label_str):
                _close(idx)
                continue

            if active_label is None:
                active_label = label_str
                start = idx
                continue

            if label_str != active_label:
                _close(idx)
                active_label = label_str
                start = idx

        _close(labels.size)

        return runs

    class _TempSubject:
        path: str | Path
        raprompro: xr.Dataset | None

        def __init__(self, template: SupportsRainAnalysis, ds_one_time: xr.Dataset) -> None:
            self.path = getattr(template, "path", "")
            self.raprompro = ds_one_time

        def _is_processed(self) -> bool:
            return True

    def _select_time(ds_in: xr.Dataset, time_value: pd.Timestamp) -> xr.Dataset:
        # Preserve a length-1 'time' dimension so downstream code remains consistent.
        try:
            return ds_in.sel(time=[np.datetime64(time_value)])
        except Exception:
            return ds_in.sel(time=[np.datetime64(time_value)], method="nearest")

    def _recompute_one_event(
        *,
        time_value: pd.Timestamp,
        z_bottom_m: float,
        z_top_m: float,
    ) -> pd.DataFrame:
        ds_one_time = _select_time(ds, time_value)
        temp_subject = _TempSubject(subject, ds_one_time)

        trends = compute_layer_trend(
            temp_subject,
            z_bottom_m=float(z_bottom_m),
            z_top_m=float(z_top_m),
            variable_threshold=resolved_variable_threshold,
            threshold_value=resolved_threshold_value,
            vars=resolved_vars_trend,
            trend_method=resolved_trend_method,
            tau_zero_tol=resolved_tau_zero_tol,
            min_points_trend=int(resolved_min_points_trend),
            min_points_ols=int(resolved_min_points_trend),
            q=float(_optional_micro_cfg_value(subject, "eps_q", 0.01)),
        )

        classified = classify_rain_process(
            subject,
            analysis=trends,
            min_tau_strength=float(_micro_cfg_value(subject, "min_tau_strength")),
            max_tau_pvalue=cast(
                float | None,
                _optional_micro_cfg_value(subject, "max_tau_pvalue", None),
            ),
        )

        df_one = build_process_dynamics_dataframe(
            subject,
            analysis=trends,
            classified=classified,
            variables=resolved_vars_trend,
        ).reset_index()

        return df_one

    sliding_times = pd.to_datetime(sliding_df[resolved_time_col])
    out_rows: list[pd.DataFrame] = []

    for time_value, df_t in sliding_df.assign(**{resolved_time_col: sliding_times}).groupby(
        resolved_time_col, sort=True
    ):
        time_value = pd.Timestamp(time_value)
        runs = _iter_vertical_runs(df_t)
        if not runs:
            continue

        for run in runs:
            z_top_fused = _float_from_dynamic(run["z_top_fused"], name="z_top_fused")
            z_bottom_fused = _float_from_dynamic(run["z_bottom_fused"], name="z_bottom_fused")
            base_row = {
                "time": time_value,
                "run_process_label": run["run_process_label"],
                "z_top_fused": z_top_fused,
                "z_bottom_fused": z_bottom_fused,
                "thickness_fused": _float_from_dynamic(
                    run["thickness_fused"], name="thickness_fused"
                ),
                "z_center_fused": float(0.5 * (z_top_fused + z_bottom_fused)),
                "n_windows_merged": _int_from_dynamic(
                    run["n_windows_merged"], name="n_windows_merged"
                ),
                "recompute_error": None,
            }
            for extra_key in ("window_id_top", "window_id_bottom"):
                if extra_key in run:
                    base_row[extra_key] = run[extra_key]

            try:
                df_one = _recompute_one_event(
                    time_value=time_value,
                    z_bottom_m=z_bottom_fused,
                    z_top_m=z_top_fused,
                )

                # Enforce a single-row output per fused event.
                if df_one.empty:
                    raise RuntimeError("Recomputation returned an empty dataframe.")
                fused_row = df_one.iloc[0].to_dict()
                time_selected = fused_row.get("time", None)
                fused_row.update(base_row)
                fused_row["time_selected"] = time_selected

                # Rename for clarity: keep the recomputed label separate from the run label.
                if "proc_label" in fused_row:
                    fused_row["proc_label_fused"] = fused_row.pop("proc_label")
                if "proc_strength" in fused_row:
                    fused_row["proc_strength_fused"] = fused_row.pop("proc_strength")

                out_rows.append(pd.DataFrame([fused_row]))
            except Exception as exc:  # pragma: no cover (depends on data quality)
                failed = dict(base_row)
                failed["time_selected"] = None
                failed["proc_label_fused"] = np.nan
                failed["proc_strength_fused"] = np.nan
                failed["recompute_error"] = str(exc)[:200]
                out_rows.append(pd.DataFrame([failed]))

    if not out_rows:
        return pd.DataFrame()

    out = pd.concat(out_rows, ignore_index=True)
    out.attrs = dict(getattr(sliding_df, "attrs", {}))
    out.attrs["selection_mode"] = "sliding_fused"
    out.attrs["min_consecutive"] = int(min_consecutive)
    out.attrs["excluded_processes"] = tuple(exclude_processes)
    out.attrs["allowed_processes"] = tuple(allowed_processes) if allowed_processes is not None else None
    out.attrs["variable_threshold"] = resolved_variable_threshold
    out.attrs["threshold_value"] = float(resolved_threshold_value)
    out.attrs["trend_method"] = resolved_trend_method
    out.attrs["tau_zero_tol"] = float(resolved_tau_zero_tol)
    out.attrs["min_points_trend"] = int(resolved_min_points_trend)
    out.attrs["vars_trend"] = tuple(resolved_vars_trend)
    out.attrs["min_tau_strength"] = float(
        _micro_cfg_value(subject, "min_tau_strength")
    )
    return out


def detect_sliding_process_episodes(
    subject: SupportsRainAnalysis,
    *,
    sliding_df: pd.DataFrame,
    min_consecutive_profiles: int = 6,
) -> pd.DataFrame:
    """
    Detect temporally persistent process episodes from a column sliding dataframe.

    Only named microphysical processes are promoted to episodes; isolated
    ``unknown`` or ``steady_or_weak`` samples are ignored. Episodes are defined
    independently in each sliding vertical window.
    """
    del subject
    if not isinstance(sliding_df, pd.DataFrame):
        raise TypeError("sliding_df must be a pandas DataFrame.")
    episodes = _detect_process_runs(
        sliding_df,
        min_consecutive_profiles=int(min_consecutive_profiles),
    )
    episodes.attrs = dict(getattr(sliding_df, "attrs", {}))
    episodes.attrs["min_consecutive_profiles"] = int(min_consecutive_profiles)
    return episodes
