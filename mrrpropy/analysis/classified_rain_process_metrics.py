from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import xarray as xr


class SupportsRainAnalysis(Protocol):
    path: str | Path
    raprompro: xr.Dataset | None

    def _is_processed(self) -> bool: ...


def _resolve_processed_dataset(subject: SupportsRainAnalysis) -> xr.Dataset:
    if not subject._is_processed():
        raise RuntimeError("MRR-Pro data not processed (raprompro missing).")
    ds = subject.raprompro
    if ds is None:
        raise RuntimeError("raprompro not loaded. Use load_raprompro().")
    return ds


def _layer_bounds_from_attrs(attrs: dict[str, Any]) -> tuple[float, float]:
    if "z_bottom_m" in attrs and "z_top_m" in attrs:
        z_bottom_m = float(attrs["z_bottom_m"])
        z_top_m = float(attrs["z_top_m"])
    elif "z_top" in attrs and "z_base" in attrs:
        z_bottom_m = float(attrs["z_top"])
        z_top_m = float(attrs["z_base"])
    else:
        raise KeyError("Missing layer bounds in attrs.")
    if z_top_m <= z_bottom_m:
        raise ValueError("Layer attrs must satisfy z_top_m > z_bottom_m.")
    return z_bottom_m, z_top_m


def _layer_metadata(
    *,
    z_bottom_m: float,
    z_top_m: float,
    selection_mode: str,
) -> dict[str, Any]:
    return {
        "z_bottom_m": float(z_bottom_m),
        "z_top_m": float(z_top_m),
        "z_top": float(z_bottom_m),
        "z_base": float(z_top_m),
        "selection_mode": str(selection_mode),
    }


def _safe_relative_change(
    bottom: np.ndarray,
    top: np.ndarray,
    scale_fallback: np.ndarray,
) -> np.ndarray:
    scale = np.where(np.abs(top) > 0.0, np.abs(top), np.abs(scale_fallback))
    scale = np.where(scale > 0.0, scale, np.nan)
    return 100.0 * (bottom - top) / scale


def build_process_dynamics_dataframe(
    subject: SupportsRainAnalysis,
    *,
    analysis: xr.Dataset,
    classified: xr.Dataset,
    variables: tuple[str, ...] = ("Dm", "Nw", "LWC"),
) -> pd.DataFrame:
    """
    Build a per-sample dataframe to quantify rain-process behaviour in a layer.

    The returned dataframe follows the physical descending-rain convention used
    by the rain-process pipeline: ``*_delta`` and ``*_rate_per_km`` represent
    the change from the top of the layer (``z_top_m``) down to the bottom
    (``z_bottom_m``).

    For each requested variable, the dataframe includes:

    - values at the top and bottom of the layer,
    - descending top-to-bottom net change,
    - relative change in percent,
    - net rate per kilometre across the layer,
    - trend diagnostics copied from ``analysis`` when available.
    """
    ds = _resolve_processed_dataset(subject)

    if not isinstance(analysis, xr.Dataset):
        raise TypeError("analysis must be an xr.Dataset.")
    if not isinstance(classified, xr.Dataset):
        raise TypeError("classified must be an xr.Dataset.")
    if "time" not in analysis.coords or "time" not in classified.coords:
        raise KeyError("analysis and classified must contain the 'time' coordinate.")

    z_bottom_m, z_top_m = _layer_bounds_from_attrs(analysis.attrs)

    time_values = analysis["time"].values
    if time_values.size == 0:
        raise ValueError("analysis does not contain any time samples.")
    time_start = time_values[0]
    time_end = time_values[-1]

    ds_event = ds.sel(time=slice(time_start, time_end))
    if ds_event.sizes.get("time", 0) == 0:
        raise ValueError("No processed samples fall inside the analysis period.")

    top_level = ds_event.sel(range=z_top_m, method="nearest")
    bottom_level = ds_event.sel(range=z_bottom_m, method="nearest")
    layer_mean = ds_event.sel(range=slice(z_bottom_m, z_top_m)).mean("range", skipna=True)

    base = xr.Dataset(coords={"time": analysis["time"].values})
    for source in (analysis, classified, top_level, bottom_level, layer_mean):
        base, source_aligned = xr.align(base, source, join="inner")
        for name in source_aligned.data_vars:
            if name not in base:
                base[name] = source_aligned[name]

    if base.sizes.get("time", 0) == 0:
        raise ValueError("analysis/classified do not overlap with the processed dataset.")

    index = pd.to_datetime(base["time"].values)
    df = pd.DataFrame(index=index)
    df.index.name = "time"

    df["proc_label"] = base["proc_label"].values.astype(str)
    if "strength" in base:
        df["proc_strength"] = base["strength"].values.astype(float)
    if "minutes" in base:
        df["minutes"] = base["minutes"].values.astype(float)

    df["z_bottom_m"] = float(z_bottom_m)
    df["z_top_m"] = float(z_top_m)
    df["z_base_m"] = float(z_top_m)
    df["dz_m"] = float(z_top_m - z_bottom_m)
    df["dz_km"] = float((z_top_m - z_bottom_m) / 1000.0)
    df["layer_top_range_m"] = float(top_level["range"].values)
    df["layer_bottom_range_m"] = float(bottom_level["range"].values)

    for passthrough_name in (
        "R",
        "G",
        "B",
        "hex_x",
        "hex_y",
        "hex_area",
        "minutes",
        "sign_R",
        "sign_G",
        "sign_B",
    ):
        if passthrough_name in base:
            df[passthrough_name] = pd.to_numeric(
                base[passthrough_name].values,
                errors="coerce",
            )

    for sign_name in ("sign_R", "sign_G", "sign_B"):
        if sign_name in base:
            df[sign_name] = base[sign_name].values.astype(int)

    for variable_name in variables:
        if variable_name not in ds_event:
            raise KeyError(f"Variable '{variable_name}' not found in processed dataset.")

        top_values = top_level[variable_name].sel(time=base["time"]).values.astype(float)
        bottom_values = bottom_level[variable_name].sel(time=base["time"]).values.astype(float)
        mean_values = layer_mean[variable_name].sel(time=base["time"]).values.astype(float)

        delta_values = bottom_values - top_values
        delta_pct_values = _safe_relative_change(
            bottom_values,
            top_values,
            mean_values,
        )
        rate_values = delta_values / float((z_top_m - z_bottom_m) / 1000.0)

        df[f"{variable_name}_top"] = top_values
        df[f"{variable_name}_bottom"] = bottom_values
        df[f"{variable_name}_layer_mean"] = mean_values
        df[f"{variable_name}_delta"] = delta_values
        df[f"{variable_name}_delta_pct"] = delta_pct_values
        df[f"{variable_name}_rate_per_km"] = rate_values

        for prefix in (
            "tau",
            "p",
            "ts",
            "intercept_ts",
            "sign",
            "strength",
            "trend_mag",
            "trend_sign",
            "trend_strength",
            "trend_score",
            "trend_p",
            "b",
            "r2",
        ):
            field = f"{prefix}_{variable_name}"
            if field in base:
                values = base[field].values
                if np.issubdtype(values.dtype, np.integer):
                    df[field] = values.astype(int)
                else:
                    df[field] = values.astype(float)

    df.attrs = {
        "trend_method": analysis.attrs.get("trend_method"),
        "trend_direction": analysis.attrs.get(
            "trend_direction",
            "positive means increase while descending from z_top_m to z_bottom_m",
        ),
        "period_start": analysis.attrs.get("period_start"),
        "period_end": analysis.attrs.get("period_end"),
        **_layer_metadata(
            z_bottom_m=z_bottom_m,
            z_top_m=z_top_m,
            selection_mode=str(analysis.attrs.get("selection_mode", "fixed_layer")),
        ),
    }
    return df


def summarize_process_dynamics(
    subject: SupportsRainAnalysis,
    *,
    analysis: xr.Dataset,
    classified: xr.Dataset,
    variables: tuple[str, ...] = ("Dm", "Nw", "LWC"),
) -> pd.DataFrame:
    """
    Summarize per-process layer dynamics into a compact grouped dataframe.

    The summary reports, for each process label, the sample count plus
    descriptive statistics of the descending top-to-bottom changes and the
    canonical trend diagnostics.
    """
    df = build_process_dynamics_dataframe(
        subject,
        analysis=analysis,
        classified=classified,
        variables=variables,
    )
    if df.empty:
        return pd.DataFrame()

    metrics: list[str] = ["proc_strength"]
    for variable_name in variables:
        metrics.extend(
            [
                f"{variable_name}_delta",
                f"{variable_name}_delta_pct",
                f"{variable_name}_rate_per_km",
                f"trend_strength_{variable_name}",
                f"trend_score_{variable_name}",
            ]
        )
        for optional_field in (f"tau_{variable_name}", f"ts_{variable_name}"):
            if optional_field in df.columns:
                metrics.append(optional_field)

    metrics = [metric for metric in metrics if metric in df.columns]

    rows: list[dict[str, float | str | int]] = []
    grouped = df.groupby("proc_label", dropna=False)
    for proc_label, group in grouped:
        row: dict[str, float | str | int] = {
            "proc_label": str(proc_label),
            "n_samples": int(len(group)),
            "fraction": float(len(group) / len(df)),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            row[f"{metric}_median"] = float(finite.median()) if not finite.empty else np.nan
            row[f"{metric}_q25"] = float(finite.quantile(0.25)) if not finite.empty else np.nan
            row[f"{metric}_q75"] = float(finite.quantile(0.75)) if not finite.empty else np.nan
            row[f"{metric}_mean"] = float(finite.mean()) if not finite.empty else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        by=["n_samples", "proc_label"],
        ascending=[False, True],
    )
    summary.attrs = dict(df.attrs)
    summary.attrs["summary_level"] = "proc_label"
    return summary.reset_index(drop=True)
