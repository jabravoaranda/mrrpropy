from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from mrrpropy.rain_process_classification.rain_process_info import (
    PROCESS_CODES as _PROCESS_CODES,
    PROCESS_MARKERS as _PROCESS_MARKERS,
    PROCESS_SIGNATURES as _PROCESS_SIGNATURES,
    ProcessSignature,
)
from mrrpropy.rain_process_classification.hexagram import (
    build_rgb_from_unit_scores,
    get_hexagram_assets,
    map_rgb_to_hexagram,
)
from mrrpropy.utils import compute_eps, compute_monotonic_trend, ols_slope_intercept_r2


PROCESS_SIGNATURES = _PROCESS_SIGNATURES
PROCESS_CODES = _PROCESS_CODES
PROCESS_MARKERS = _PROCESS_MARKERS


class _UnsetType:
    pass


_UNSET = _UnsetType()


def _float_from_dynamic(value: object, *, name: str) -> float:
    if isinstance(value, _UnsetType):
        raise ValueError(f"{name} is unset.")
    return float(cast(Any, value))


def _int_from_dynamic(value: object, *, name: str) -> int:
    if isinstance(value, _UnsetType):
        raise ValueError(f"{name} is unset.")
    return int(cast(Any, value))


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


def _resolve_min_points(
    *,
    min_points_trend: int | None,
    min_points_ols: int | None,
    default: int = 10,
) -> int:
    if min_points_trend is not None:
        return int(min_points_trend)
    if min_points_ols is not None:
        return int(min_points_ols)
    return int(default)


def _normalize_signatures(signature_definition: Any) -> list[tuple[int, int, int]]:
    if isinstance(signature_definition, tuple) and len(signature_definition) == 3:
        return [
            (
                int(signature_definition[0]),
                int(signature_definition[1]),
                int(signature_definition[2]),
            )
        ]
    if isinstance(signature_definition, (list, tuple)):
        signatures: list[tuple[int, int, int]] = []
        for item in signature_definition:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                signatures.append((int(item[0]), int(item[1]), int(item[2])))
        if signatures:
            return signatures
    raise ValueError(f"Invalid process signature: {signature_definition!r}")


def _resolve_layer_bounds(
    *,
    z_bottom_m: float | None = None,
    z_top_m: float | None = None,
    layer: tuple[float, float] | None = None,
    z_top: float | None = None,
    z_base: float | None = None,
    caller: str,
) -> tuple[float, float]:
    has_new_bounds = z_bottom_m is not None or z_top_m is not None
    has_layer = layer is not None
    has_legacy_bounds = z_top is not None or z_base is not None

    complete_sources = int(z_bottom_m is not None and z_top_m is not None)
    complete_sources += int(layer is not None)
    complete_sources += int(z_top is not None and z_base is not None)
    if complete_sources > 1:
        raise ValueError(
            f"{caller} received multiple layer definitions. Use either "
            "`z_bottom_m`/`z_top_m`, `layer=(z_bottom_m, z_top_m)`, or "
            "legacy `z_top`/`z_base`, but not more than one."
        )

    if has_new_bounds and not (z_bottom_m is not None and z_top_m is not None):
        raise ValueError(f"{caller} requires both z_bottom_m and z_top_m.")
    if has_legacy_bounds and not (z_top is not None and z_base is not None):
        raise ValueError(f"{caller} requires both legacy z_top and z_base together.")

    if layer is not None:
        warnings.warn(
            "The `layer=(z_bottom_m, z_top_m)` argument is legacy. "
            "Use explicit `z_bottom_m` and `z_top_m` instead.",
            FutureWarning,
            stacklevel=2,
        )
        z_bottom_m, z_top_m = float(layer[0]), float(layer[1])
    elif z_top is not None and z_base is not None:
        warnings.warn(
            "The `z_top`/`z_base` arguments are legacy and use ambiguous naming. "
            "Use `z_bottom_m` and `z_top_m` instead.",
            FutureWarning,
            stacklevel=2,
        )
        z_bottom_m, z_top_m = float(z_top), float(z_base)

    if z_bottom_m is None or z_top_m is None:
        raise ValueError(
            f"{caller} requires a layer defined by z_bottom_m and z_top_m."
        )

    z_bottom_m = float(z_bottom_m)
    z_top_m = float(z_top_m)
    if z_top_m <= z_bottom_m:
        raise ValueError("z_top_m must be greater than z_bottom_m (in meters).")
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


def _distance_below_layer_top(z_layer: np.ndarray, *, z_top_m: float) -> np.ndarray:
    """Return downward progress inside the layer, starting from its upper edge."""
    return (float(z_top_m) - np.asarray(z_layer, dtype=float)).astype(float)


def _safe_relative_change(
    bottom: np.ndarray,
    top: np.ndarray,
    scale_fallback: np.ndarray,
) -> np.ndarray:
    scale = np.where(np.abs(top) > 0.0, np.abs(top), np.abs(scale_fallback))
    scale = np.where(scale > 0.0, scale, np.nan)
    return 100.0 * (bottom - top) / scale


def compute_layer_trend_ols(
    subject: SupportsRainAnalysis,
    *,
    z_bottom_m: float | None = None,
    z_top_m: float | None = None,
    z_top: float | None = None,
    z_base: float | None = None,
    time_dim: str = "time",
    variable_threshold: str = "Ze",
    threshold_value: float = -5.0,
    vars: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
    eps_mode: str = "hourly_quantile",
    q: float = 0.01,
    eps_floor_mode: str = "global_min",
    min_points_ols: int = 10,
) -> xr.Dataset:
    """
    Compute layer-wise legacy OLS trends of selected microphysical variables.

    This function is retained for backward compatibility and diagnostic
    comparison only. The recommended microphysical workflow now relies on
    Kendall's tau plus Theil-Sen slope through :func:`compute_layer_trend`.
    """
    if not subject._is_processed():
        raise RuntimeError("MRR-Pro data not processed (raprompro missing).")

    ds = subject.raprompro
    if ds is None:
        raise RuntimeError("raprompro not loaded. Use load_raprompro().")

    z_bottom_m, z_top_m = _resolve_layer_bounds(
        z_bottom_m=z_bottom_m,
        z_top_m=z_top_m,
        z_top=z_top,
        z_base=z_base,
        caller="compute_layer_trend_ols",
    )

    layer = ds.sel({"range": slice(z_bottom_m, z_top_m)})

    if time_dim not in layer.coords:
        raise KeyError(f"Missing coord '{time_dim}' in dataset.")
    if "range" not in layer.coords:
        raise KeyError("Missing coord 'range' in dataset.")
    if variable_threshold not in layer:
        raise KeyError(f"Missing threshold variable '{variable_threshold}' in dataset.")

    for variable_name in vars:
        if variable_name not in layer:
            raise KeyError(f"Missing variable '{variable_name}' in dataset.")

    z_layer = layer["range"].values.astype(float)
    depth = _distance_below_layer_top(z_layer, z_top_m=float(z_top_m))
    dz = float(z_top_m - z_bottom_m)

    ze = layer[variable_threshold]
    ze_mask = xr.where(np.isfinite(ze) & (ze > threshold_value), True, False)

    out = xr.Dataset(
        coords={
            time_dim: layer[time_dim].values,
            "range_layer": layer["range"].values,
        }
    )
    out = out.assign_coords(depth=("range_layer", depth))

    out["dz"] = xr.DataArray(dz)
    out["mask_ze"] = xr.DataArray(
        ze_mask.values.astype(bool),
        dims=(time_dim, "range_layer"),
    )

    out.attrs.update(
        dict(
            **_layer_metadata(
                z_bottom_m=float(z_bottom_m),
                z_top_m=float(z_top_m),
                selection_mode="fixed_layer",
            ),
            dz=float(dz),
            trend_method="ols_legacy",
            variable_threshold=str(variable_threshold),
            threshold_value=float(threshold_value),
            vars=tuple(vars),
            eps_mode=str(eps_mode),
            eps_floor_mode=str(eps_floor_mode),
            q=float(q),
            min_points_ols=int(min_points_ols),
            trend_direction="positive means increase while descending from z_top_m to z_bottom_m",
        )
    )

    global_eps: dict[str, float] = {}
    if eps_mode == "global_quantile" or eps_floor_mode == "global_min":
        for variable_name in vars:
            global_eps[variable_name] = compute_eps(layer[variable_name].values, q=q)

    times = layer[time_dim].values
    ntime = times.size
    nrange = layer.sizes["range"]
    ze_mask_np = ze_mask.values.astype(bool)

    n_valid = np.sum(ze_mask_np, axis=1).astype(int)
    out["n_valid"] = xr.DataArray(n_valid, dims=(time_dim,))

    for variable_name in vars:
        b_arr = np.full(ntime, np.nan, dtype=float)
        a_arr = np.full(ntime, np.nan, dtype=float)
        r2_arr = np.full(ntime, np.nan, dtype=float)
        f_arr = np.full(ntime, np.nan, dtype=float)

        eps_used = np.full(ntime, np.nan, dtype=float)
        n_fit = np.zeros(ntime, dtype=int)
        mask_fit = np.zeros((ntime, nrange), dtype=bool)

        values = layer[variable_name].values.astype(float)

        for index in range(ntime):
            if n_valid[index] < min_points_ols:
                continue

            mask = (
                ze_mask_np[index, :]
                & np.isfinite(values[index, :])
                & (values[index, :] > 0.0)
            )
            nmask = int(np.sum(mask))
            if nmask < min_points_ols:
                continue

            if eps_mode == "hourly_quantile":
                eps_t = compute_eps(values[index, :], q=q)
            elif eps_mode == "global_quantile":
                eps_t = global_eps.get(variable_name, np.nan)
            else:
                raise ValueError(f"Unsupported eps_mode={eps_mode!r}")

            if not np.isfinite(eps_t) or eps_t <= 0:
                continue

            if eps_floor_mode == "global_min":
                eps_g = global_eps.get(variable_name, np.nan)
                if np.isfinite(eps_g) and eps_g > 0:
                    eps_t = max(float(eps_t), float(eps_g))

            x = depth[mask]
            y = np.log(np.maximum(values[index, mask], eps_t))

            b, a, r2 = ols_slope_intercept_r2(x, y)
            if not (np.isfinite(b) and np.isfinite(a) and np.isfinite(r2)):
                continue

            b_arr[index] = float(b)
            a_arr[index] = float(a)
            r2_arr[index] = float(r2)
            f_arr[index] = float(np.exp(b * dz))

            mask_fit[index, :] = mask
            n_fit[index] = nmask
            eps_used[index] = float(eps_t)

        out[f"b_{variable_name}"] = xr.DataArray(b_arr, dims=(time_dim,))
        out[f"a_{variable_name}"] = xr.DataArray(a_arr, dims=(time_dim,))
        out[f"r2_{variable_name}"] = xr.DataArray(r2_arr, dims=(time_dim,))
        out[f"F_{variable_name}"] = xr.DataArray(f_arr, dims=(time_dim,))

        out[f"eps_{variable_name}"] = xr.DataArray(eps_used, dims=(time_dim,))
        out[f"n_fit_{variable_name}"] = xr.DataArray(n_fit, dims=(time_dim,))
        out[f"mask_fit_{variable_name}"] = xr.DataArray(
            mask_fit,
            dims=(time_dim, "range_layer"),
        )

    return out


def compute_layer_trend(
    subject: SupportsRainAnalysis,
    *,
    z_bottom_m: float | None = None,
    z_top_m: float | None = None,
    z_top: float | None = None,
    z_base: float | None = None,
    time_dim: str = "time",
    variable_threshold: str = "Ze",
    threshold_value: float = -5.0,
    vars: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
    trend_method: str = "kendall_theilsen",
    tau_zero_tol: float = 0.05,
    min_points_trend: int | None = None,
    min_points_ols: int | None = None,
    eps_mode: str = "hourly_quantile",
    q: float = 0.01,
    eps_floor_mode: str = "global_min",
) -> xr.Dataset:
    """
    Compute layer-wise monotonic trends for selected microphysical variables.

    The default implementation characterises each vertical profile with:

    - Kendall's tau for monotonic direction and consistency.
    - Theil-Sen slope for robust magnitude.

    ``trend_method="ols"`` falls back to the legacy OLS implementation for
    diagnostic comparison.
    """
    method = trend_method.lower()
    resolved_min_points = _resolve_min_points(
        min_points_trend=min_points_trend,
        min_points_ols=min_points_ols,
    )

    if method in {"ols", "ols_legacy"}:
        out = compute_layer_trend_ols(
            subject,
            z_bottom_m=z_bottom_m,
            z_top_m=z_top_m,
            z_top=z_top,
            z_base=z_base,
            time_dim=time_dim,
            variable_threshold=variable_threshold,
            threshold_value=threshold_value,
            vars=vars,
            eps_mode=eps_mode,
            q=q,
            eps_floor_mode=eps_floor_mode,
            min_points_ols=resolved_min_points,
        )
        out.attrs["trend_method"] = "ols"
        out.attrs["min_points_trend"] = int(resolved_min_points)
        out.attrs["trend_score_definition"] = "trend_sign * trend_strength = sign(b) * r2"
        for variable_name in vars:
            b_values = out[f"b_{variable_name}"].values.astype(float)
            r2_values = out[f"r2_{variable_name}"].values.astype(float)
            sign_values = np.zeros(b_values.shape, dtype=int)
            finite_b = np.isfinite(b_values)
            sign_values[finite_b & (b_values > 0.0)] = 1
            sign_values[finite_b & (b_values < 0.0)] = -1
            strength_values = np.clip(r2_values, 0.0, 1.0)
            score_values = np.full(b_values.shape, np.nan, dtype=float)
            finite_score = finite_b & np.isfinite(strength_values)
            score_values[finite_score] = (
                sign_values[finite_score] * strength_values[finite_score]
            )

            out[f"trend_mag_{variable_name}"] = xr.DataArray(
                b_values.copy(),
                dims=(time_dim,),
            )
            out[f"trend_sign_{variable_name}"] = xr.DataArray(
                sign_values,
                dims=(time_dim,),
            )
            out[f"trend_strength_{variable_name}"] = xr.DataArray(
                strength_values,
                dims=(time_dim,),
            )
            out[f"trend_score_{variable_name}"] = xr.DataArray(
                score_values,
                dims=(time_dim,),
            )
            out[f"trend_p_{variable_name}"] = xr.DataArray(
                np.full(b_values.shape, np.nan, dtype=float),
                dims=(time_dim,),
            )
        return out

    if method not in {"kendall_theilsen", "kendall-theilsen", "kendall", "tau"}:
        raise ValueError(f"Unsupported trend_method={trend_method!r}")

    if not subject._is_processed():
        raise RuntimeError("MRR-Pro data not processed (raprompro missing).")

    ds = subject.raprompro
    if ds is None:
        raise RuntimeError("raprompro not loaded. Use load_raprompro().")

    z_bottom_m, z_top_m = _resolve_layer_bounds(
        z_bottom_m=z_bottom_m,
        z_top_m=z_top_m,
        z_top=z_top,
        z_base=z_base,
        caller="compute_layer_trend",
    )

    layer = ds.sel({"range": slice(z_bottom_m, z_top_m)})

    if time_dim not in layer.coords:
        raise KeyError(f"Missing coord '{time_dim}' in dataset.")
    if "range" not in layer.coords:
        raise KeyError("Missing coord 'range' in dataset.")
    if variable_threshold not in layer:
        raise KeyError(f"Missing threshold variable '{variable_threshold}' in dataset.")

    for variable_name in vars:
        if variable_name not in layer:
            raise KeyError(f"Missing variable '{variable_name}' in dataset.")

    z_layer = layer["range"].values.astype(float)
    depth = _distance_below_layer_top(z_layer, z_top_m=float(z_top_m))
    dz = float(z_top_m - z_bottom_m)

    ze = layer[variable_threshold]
    ze_mask = xr.where(np.isfinite(ze) & (ze > threshold_value), True, False)

    out = xr.Dataset(
        coords={
            time_dim: layer[time_dim].values,
            "range_layer": layer["range"].values,
        }
    )
    out = out.assign_coords(depth=("range_layer", depth))

    out["dz"] = xr.DataArray(dz)
    out["mask_ze"] = xr.DataArray(
        ze_mask.values.astype(bool),
        dims=(time_dim, "range_layer"),
    )

    out.attrs.update(
        dict(
            **_layer_metadata(
                z_bottom_m=float(z_bottom_m),
                z_top_m=float(z_top_m),
                selection_mode="fixed_layer",
            ),
            dz=float(dz),
            trend_method="kendall_theilsen",
            variable_threshold=str(variable_threshold),
            threshold_value=float(threshold_value),
            vars=tuple(vars),
            tau_zero_tol=float(tau_zero_tol),
            min_points_trend=int(resolved_min_points),
            min_points_ols=int(resolved_min_points),
            trend_score_definition="trend_sign * trend_strength = tau outside the zero deadband",
            legacy_b_definition="For nonparametric trends, b_* aliases ts_* and is diagnostic only.",
            trend_direction="positive means increase while descending from z_top_m to z_bottom_m",
        )
    )

    times = layer[time_dim].values
    ntime = times.size
    nrange = layer.sizes["range"]
    ze_mask_np = ze_mask.values.astype(bool)

    n_valid = np.sum(ze_mask_np, axis=1).astype(int)
    out["n_valid"] = xr.DataArray(n_valid, dims=(time_dim,))

    for variable_name in vars:
        tau_arr = np.full(ntime, np.nan, dtype=float)
        p_arr = np.full(ntime, np.nan, dtype=float)
        ts_arr = np.full(ntime, np.nan, dtype=float)
        intercept_arr = np.full(ntime, np.nan, dtype=float)
        sign_arr = np.zeros(ntime, dtype=int)
        strength_arr = np.full(ntime, np.nan, dtype=float)
        n_fit = np.zeros(ntime, dtype=int)
        mask_fit = np.zeros((ntime, nrange), dtype=bool)

        values = layer[variable_name].values.astype(float)

        for index in range(ntime):
            if n_valid[index] < resolved_min_points:
                continue

            mask = ze_mask_np[index, :] & np.isfinite(values[index, :])
            nmask = int(np.sum(mask))
            if nmask < resolved_min_points:
                continue

            trend = compute_monotonic_trend(
                depth[mask],
                values[index, mask],
                tau_zero_tol=tau_zero_tol,
                min_points=resolved_min_points,
            )

            tau_arr[index] = float(trend["tau"])
            p_arr[index] = float(trend["p_value"])
            ts_arr[index] = float(trend["slope_ts"])
            intercept_arr[index] = float(trend["intercept_ts"])
            sign_arr[index] = int(trend["sign_tau"])
            strength_arr[index] = float(trend["strength_tau"])
            n_fit[index] = nmask
            mask_fit[index, :] = mask

        out[f"tau_{variable_name}"] = xr.DataArray(tau_arr, dims=(time_dim,))
        out[f"p_{variable_name}"] = xr.DataArray(p_arr, dims=(time_dim,))
        out[f"ts_{variable_name}"] = xr.DataArray(ts_arr, dims=(time_dim,))
        out[f"intercept_ts_{variable_name}"] = xr.DataArray(
            intercept_arr,
            dims=(time_dim,),
        )
        out[f"sign_{variable_name}"] = xr.DataArray(sign_arr, dims=(time_dim,))
        out[f"strength_{variable_name}"] = xr.DataArray(
            strength_arr,
            dims=(time_dim,),
        )
        score_arr = np.full(ntime, np.nan, dtype=float)
        finite_score = np.isfinite(strength_arr)
        score_arr[finite_score] = sign_arr[finite_score] * strength_arr[finite_score]
        out[f"trend_mag_{variable_name}"] = xr.DataArray(ts_arr.copy(), dims=(time_dim,))
        out[f"trend_sign_{variable_name}"] = xr.DataArray(sign_arr.copy(), dims=(time_dim,))
        out[f"trend_strength_{variable_name}"] = xr.DataArray(
            strength_arr.copy(),
            dims=(time_dim,),
        )
        out[f"trend_score_{variable_name}"] = xr.DataArray(
            score_arr,
            dims=(time_dim,),
        )
        out[f"trend_p_{variable_name}"] = xr.DataArray(p_arr.copy(), dims=(time_dim,))
        out[f"n_fit_{variable_name}"] = xr.DataArray(n_fit, dims=(time_dim,))
        out[f"mask_fit_{variable_name}"] = xr.DataArray(
            mask_fit,
            dims=(time_dim, "range_layer"),
        )

        legacy_slope = xr.DataArray(ts_arr.copy(), dims=(time_dim,))
        legacy_slope.attrs["legacy_alias_for"] = f"ts_{variable_name}"
        legacy_slope.attrs["note"] = (
            "Legacy alias retained for compatibility. This is not an OLS slope."
        )
        out[f"b_{variable_name}"] = legacy_slope

    return out


def rain_process_analyze(
    subject: SupportsRainAnalysis,
    *,
    period: tuple[datetime, datetime],
    layer: tuple[float, float] | None = None,
    z_bottom_m: float | None = None,
    z_top_m: float | None = None,
    k: int,
    ze_th: float = -5.0,
    trend_method: str = "kendall_theilsen",
    tau_zero_tol: float = 0.05,
    min_points_trend: int | None = None,
    min_points_ols: int | None = None,
    eps_q: float = 0.01,
    rgb_q: float = 0.02,
    vars_trend: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
) -> xr.Dataset:
    """
    Analyse rain-process evolution in one fixed layer over a selected period.

    This is the fixed-layer analysis primitive used internally by the public
    sliding workflow. Positive trend/change means increase while descending from
    ``z_top_m`` to ``z_bottom_m``.

    The workflow computes method-neutral canonical trend variables used
    downstream by RGB mapping and process classification:

    - ``trend_mag_*``: physical magnitude for the selected method.
    - ``trend_sign_*``: directional sign in ``{-1, 0, +1}``.
    - ``trend_strength_*``: bounded consistency/confidence in ``[0, 1]``.
    - ``trend_score_*``: signed bounded score in ``[-1, 1]`` used by RGB.
    - ``trend_p_*``: p-value when available.

    By default, the underlying diagnostics are Kendall's tau plus Theil-Sen
    slope. ``trend_method="ols"`` keeps the legacy OLS diagnostics available
    while still feeding the downstream pipeline through the canonical
    ``trend_*`` names.
    """
    if not subject._is_processed():
        raise RuntimeError("Dataset not preprocessed / raprompro not available.")

    ds = subject.raprompro
    if ds is None:
        raise RuntimeError("raprompro not loaded. Use load_raprompro().")

    z_bottom_m, z_top_m = _resolve_layer_bounds(
        z_bottom_m=z_bottom_m,
        z_top_m=z_top_m,
        layer=layer,
        caller="rain_process_analyze",
    )

    t0, t1 = period
    if t0 >= t1:
        raise ValueError("period must be increasing (t0 < t1).")

    ds_sub = ds.sel(time=slice(t0, t1))
    if ds_sub.sizes.get("time", 0) == 0:
        raise ValueError("Empty temporal selection: revise period.")

    resolved_min_points = _resolve_min_points(
        min_points_trend=min_points_trend,
        min_points_ols=min_points_ols,
    )
    method = trend_method.lower()

    trends = compute_layer_trend(
        subject,
        z_bottom_m=z_bottom_m,
        z_top_m=z_top_m,
        variable_threshold="Ze",
        threshold_value=ze_th,
        vars=vars_trend,
        trend_method=trend_method,
        tau_zero_tol=tau_zero_tol,
        min_points_trend=resolved_min_points,
        min_points_ols=resolved_min_points,
        q=eps_q,
    )

    trends = trends.sel(time=slice(ds_sub["time"].values[0], ds_sub["time"].values[-1]))

    rgb = build_rgb_from_unit_scores(
        trends,
        vars=(
            f"trend_score_{vars_trend[0]}",
            f"trend_score_{vars_trend[1]}",
            f"trend_score_{vars_trend[2]}",
        ),
    )

    time_values = rgb["time"].values
    time_start = time_values[0]
    minutes = ((time_values - time_start) / np.timedelta64(1, "m")).astype(float)

    hex_assets = get_hexagram_assets(k=k)
    hex_ds = map_rgb_to_hexagram(rgb, hex_assets=hex_assets)

    out = xr.Dataset(coords={"time": rgb["time"].values})

    for variable_name in trends.data_vars:
        out[variable_name] = trends[variable_name]
    for coord_name in trends.coords:
        if coord_name not in out.coords:
            out = out.assign_coords({coord_name: trends.coords[coord_name]})
    out.attrs.update(trends.attrs)

    out["R"] = rgb["R"]
    out["G"] = rgb["G"]
    out["B"] = rgb["B"]
    out["minutes"] = xr.DataArray(minutes, dims=("time",))

    for variable_name in hex_ds.data_vars:
        out[variable_name] = hex_ds[variable_name]

    out.attrs.update(
        dict(
            period_start=str(np.datetime_as_string(ds_sub["time"].values[0], unit="s")),
            period_end=str(np.datetime_as_string(ds_sub["time"].values[-1], unit="s")),
            **_layer_metadata(
                z_bottom_m=float(z_bottom_m),
                z_top_m=float(z_top_m),
                selection_mode="fixed_layer",
            ),
            k=int(k),
            ze_th=float(ze_th),
            trend_method="ols" if method in {"ols", "ols_legacy"} else "kendall_theilsen",
            tau_zero_tol=float(tau_zero_tol),
            min_points_trend=int(resolved_min_points),
            min_points_ols=int(resolved_min_points),
            eps_q=float(eps_q),
            rgb_q=float(rgb_q),
            vars_trend=tuple(vars_trend),
            rgb_convention=str(
                f"R={vars_trend[0]}, G={vars_trend[1]}, B={vars_trend[2]}"
            ),
        )
    )
    out.attrs["rgb_mapping"] = {
        "R": vars_trend[0],
        "G": vars_trend[1],
        "B": vars_trend[2],
    }
    out.attrs["rgb_method"] = "trend_score"
    out.attrs["strength_definition"] = "min(trend_strength_Dm, trend_strength_Nw, trend_strength_LWC)"
    return out


def classify_rain_process(
    subject: SupportsRainAnalysis,
    *,
    analysis: xr.Dataset,
    tol_center: float = 0.05,
    min_strength: float = 0.10,
    min_tau_strength: float | None = None,
    max_p_value: float | None = None,
    max_tau_pvalue: float | None = None,
) -> xr.Dataset:
    """
    Classify each time sample into a rain-process category.

    When canonical ``trend_*`` diagnostics are available in ``analysis``, the
    classification is based on ``trend_sign_Dm``, ``trend_sign_Nw`` and
    ``trend_sign_LWC`` independently of the underlying trend method. The
    overall process strength is the minimum component ``trend_strength_*``
    across ``Dm``, ``Nw`` and ``LWC``.

    If only RGB channels are available, a legacy RGB-centre classification is
    used for backwards compatibility.
    """
    del subject

    if analysis is None or not isinstance(analysis, xr.Dataset):
        raise TypeError("analysis must be an xr.Dataset produced by rain_process_analyze.")
    if "time" not in analysis.coords:
        raise KeyError("analysis must include the 'time' coordinate.")

    expected = {"R": "Dm", "G": "Nw", "B": "LWC"}
    rgb_map = analysis.attrs.get("rgb_mapping", None)

    variable_names = ("Dm", "Nw", "LWC")
    has_trend_fields = all(
        f"{prefix}_{variable_name}" in analysis
        for prefix in ("trend_sign", "trend_strength", "trend_p")
        for variable_name in variable_names
    )

    if not has_trend_fields:
        if rgb_map != expected:
            raise ValueError(f"rgb_mapping={rgb_map} but this classifier expects {expected}.")
    elif rgb_map is not None and rgb_map != expected:
        warnings.warn(
            f"rgb_mapping={rgb_map} but canonical classification does not depend on rgb_mapping "
            f"(expected {expected} for legacy RGB fallback).",
            UserWarning,
            stacklevel=2,
        )

    tau_strength_threshold = (
        float(min_tau_strength) if min_tau_strength is not None else float(min_strength)
    )
    p_value_threshold = (
        float(max_tau_pvalue) if max_tau_pvalue is not None else max_p_value
    )

    out = xr.Dataset(coords={"time": analysis["time"].values})
    if "layer" in analysis.dims and "layer" in analysis.coords:
        out = out.assign_coords(layer=analysis["layer"].values)

    if has_trend_fields:
        core = _classify_from_microphysical_trends(
            analysis,
            variables=variable_names,
            min_strength=min_strength,
            min_tau_strength=min_tau_strength,
            max_p_value=max_p_value,
            max_tau_pvalue=max_tau_pvalue,
        )
        out["proc_label"] = core["proc_label"]
        out["strength"] = core["strength"]
        out["sign_R"] = core["sign_Dm"]
        out["sign_G"] = core["sign_Nw"]
        out["sign_B"] = core["sign_LWC"]

        for variable_name in variable_names:
            for prefix in ("tau", "p", "sign", "strength", "ts", "intercept_ts"):
                key = f"{prefix}_{variable_name}"
                if key in analysis:
                    out[key] = analysis[key]

        out.attrs["classification_basis"] = "canonical_trend_sign"
        out.attrs["strength_definition"] = "min(trend_strength_Dm, trend_strength_Nw, trend_strength_LWC)"
        out.attrs["min_tau_strength"] = float(tau_strength_threshold)
        out.attrs["min_strength"] = float(tau_strength_threshold)
        out.attrs["max_tau_pvalue"] = (
            float(p_value_threshold) if p_value_threshold is not None else None
        )
        out.attrs["tau_zero_tol"] = float(
            analysis.attrs.get("tau_zero_tol", tol_center)
        )
        out.attrs["tol_center"] = float(analysis.attrs.get("tau_zero_tol", tol_center))
        for variable_name in variable_names:
            for prefix in ("trend_mag", "trend_sign", "trend_strength", "trend_score", "trend_p"):
                key = f"{prefix}_{variable_name}"
                if key in analysis:
                    out[key] = analysis[key]
    else:
        for variable_name in ("R", "G", "B"):
            if variable_name not in analysis:
                raise KeyError(
                    "analysis must include R, G and B channels for legacy classification."
                )

        r_values = analysis["R"].values.astype(float)
        g_values = analysis["G"].values.astype(float)
        b_values = analysis["B"].values.astype(float)
        ok = np.isfinite(r_values) & np.isfinite(g_values) & np.isfinite(b_values)

        def _sign_from_center(values: np.ndarray, tol: float) -> np.ndarray:
            sign = np.zeros(values.shape, dtype=int)
            sign[values > 0.5 + tol] = +1
            sign[values < 0.5 - tol] = -1
            return sign

        def _strength(values: np.ndarray) -> np.ndarray:
            return np.clip(np.abs(values - 0.5) / 0.5, 0.0, 1.0)

        sign_r = np.zeros(r_values.shape, dtype=int)
        sign_g = np.zeros(g_values.shape, dtype=int)
        sign_b = np.zeros(b_values.shape, dtype=int)
        if np.any(ok):
            sign_r[ok] = _sign_from_center(r_values[ok], tol_center)
            sign_g[ok] = _sign_from_center(g_values[ok], tol_center)
            sign_b[ok] = _sign_from_center(b_values[ok], tol_center)

        strength = np.zeros(r_values.shape, dtype=float)
        if np.any(ok):
            strength[ok] = np.minimum.reduce(
                [
                    _strength(r_values[ok]),
                    _strength(g_values[ok]),
                    _strength(b_values[ok]),
                ]
            )

        label = np.full(r_values.shape, "no_data", dtype=object)
        label[ok] = "unknown"

        for process_name, signature_definition in PROCESS_SIGNATURES.items():
            signatures = _normalize_signatures(signature_definition)
            process_mask = np.zeros(r_values.shape, dtype=bool)
            for sign_r_expected, sign_g_expected, sign_b_expected in signatures:
                process_mask |= (
                    ok
                    & (sign_r == sign_r_expected)
                    & (sign_g == sign_g_expected)
                    & (sign_b == sign_b_expected)
                )
            take = process_mask & (label == "unknown")
            label[take] = process_name

        weak = ok & (strength < min_strength)
        label[weak] = "steady_or_weak"

        out["proc_label"] = xr.DataArray(label, dims=("time",))
        out["sign_R"] = xr.DataArray(sign_r, dims=("time",))
        out["sign_G"] = xr.DataArray(sign_g, dims=("time",))
        out["sign_B"] = xr.DataArray(sign_b, dims=("time",))
        out["strength"] = xr.DataArray(strength, dims=("time",))

        out.attrs["classification_basis"] = "legacy_rgb_center"
        out.attrs["strength_definition"] = "min(|RGB-0.5|)/0.5"
        out.attrs["tol_center"] = float(tol_center)
        out.attrs["min_strength"] = float(min_strength)

    for variable_name in ("R", "G", "B"):
        if variable_name in analysis:
            out[variable_name] = analysis[variable_name]

    for variable_name in (
        "hex_x",
        "hex_y",
        "hex_area",
        "minutes",
        "snap_R",
        "snap_G",
        "snap_B",
    ):
        if variable_name in analysis:
            out[variable_name] = analysis[variable_name]

    out.attrs["rgb_mapping"] = rgb_map if rgb_map is not None else expected

    for key in (
        "rgb_convention",
        "period_start",
        "period_end",
        "z_bottom_m",
        "z_top_m",
        "z_top",
        "z_base",
        "selection_mode",
        "k",
        "rgb_q",
        "eps_q",
        "ze_th",
        "trend_method",
        "tau_zero_tol",
        "min_points_trend",
        "min_points_ols",
    ):
        if key in analysis.attrs:
            out.attrs[key] = analysis.attrs[key]

    return out


def _coords_for_sample_dims(ds: xr.Dataset, sample_dims: tuple[str, ...]) -> dict[str, Any]:
    coords: dict[str, Any] = {}
    for dim in sample_dims:
        if dim in ds.coords:
            coords[dim] = ds[dim].values
    return coords


def _classify_from_microphysical_trends(
    ds: xr.Dataset,
    *,
    variables: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
    min_strength: float = 0.10,
    min_tau_strength: float | None = None,
    max_p_value: float | None = None,
    max_tau_pvalue: float | None = None,
) -> xr.Dataset:
    """
    Pure classification core based only on canonical microphysical trends.

    Required fields (for each variable in `variables`):
    - trend_sign_*
    - trend_strength_*
    - trend_p_*
    """
    if "time" not in ds.coords:
        raise KeyError("ds must include the 'time' coordinate.")

    tau_strength_threshold = (
        float(min_tau_strength) if min_tau_strength is not None else float(min_strength)
    )
    p_value_threshold = float(max_tau_pvalue) if max_tau_pvalue is not None else max_p_value

    for variable_name in variables:
        for prefix in ("trend_sign", "trend_strength", "trend_p"):
            key = f"{prefix}_{variable_name}"
            if key not in ds:
                raise KeyError(f"Missing required field '{key}' for classification.")

    strength_ref = ds[f"trend_strength_{variables[0]}"]
    sample_dims = tuple(str(dim) for dim in strength_ref.dims)

    p_data = {
        variable_name: ds[f"trend_p_{variable_name}"].values.astype(float)
        for variable_name in variables
    }
    sign_data = {
        variable_name: ds[f"trend_sign_{variable_name}"].values.astype(int)
        for variable_name in variables
    }
    strength_data = {
        variable_name: ds[f"trend_strength_{variable_name}"].values.astype(float)
        for variable_name in variables
    }

    ok = np.ones_like(strength_data[variables[0]], dtype=bool)
    for variable_name in variables:
        ok &= np.isfinite(strength_data[variable_name])

    strength = np.full(ok.shape, np.nan, dtype=float)
    if np.any(ok):
        strength[ok] = np.minimum.reduce(
            [
                strength_data[variables[0]][ok],
                strength_data[variables[1]][ok],
                strength_data[variables[2]][ok],
            ]
        )

    p_filter = np.ones_like(ok, dtype=bool)
    if p_value_threshold is not None:
        p_filter = ok.copy()
        for variable_name in variables:
            p_filter &= np.isfinite(p_data[variable_name])
            p_filter &= p_data[variable_name] <= float(p_value_threshold)

    label = np.full(ok.shape, "no_data", dtype=object)
    label[ok] = "unknown"

    sign_r = sign_data[variables[0]].copy()
    sign_g = sign_data[variables[1]].copy()
    sign_b = sign_data[variables[2]].copy()

    for process_name, signature_definition in PROCESS_SIGNATURES.items():
        signatures = _normalize_signatures(signature_definition)
        process_mask = np.zeros(ok.shape, dtype=bool)
        for sign_r_expected, sign_g_expected, sign_b_expected in signatures:
            process_mask |= (
                ok
                & (sign_r == sign_r_expected)
                & (sign_g == sign_g_expected)
                & (sign_b == sign_b_expected)
            )
        take = process_mask & (label == "unknown")
        label[take] = process_name

    weak = ok & (strength < tau_strength_threshold)
    if p_value_threshold is not None:
        weak |= ok & ~p_filter
    label[weak] = "steady_or_weak"

    out = xr.Dataset(coords=_coords_for_sample_dims(ds, sample_dims))
    out["proc_label"] = xr.DataArray(label, dims=sample_dims)
    out["strength"] = xr.DataArray(strength, dims=sample_dims)
    out["sign_Dm"] = xr.DataArray(sign_r, dims=sample_dims)
    out["sign_Nw"] = xr.DataArray(sign_g, dims=sample_dims)
    out["sign_LWC"] = xr.DataArray(sign_b, dims=sample_dims)

    out.attrs["classification_basis"] = "canonical_microphysical_trends"
    out.attrs["strength_definition"] = "min(trend_strength_Dm, trend_strength_Nw, trend_strength_LWC)"
    out.attrs["min_tau_strength"] = float(tau_strength_threshold)
    out.attrs["min_strength"] = float(tau_strength_threshold)
    out.attrs["max_tau_pvalue"] = float(p_value_threshold) if p_value_threshold is not None else None
    return out


def classify_process_from_features(
    process_features: xr.Dataset,
    *,
    refiners: list[Any] | None = None,
    min_strength: float = 0.10,
    min_tau_strength: float | None = None,
    max_p_value: float | None = None,
    max_tau_pvalue: float | None = None,
) -> xr.Dataset:
    """
    Phase B wrapper: classify directly from `process_features` (Phase A output).

    This wrapper never depends on RGB/hexagram fields. The baseline label is
    stored as `proc_label_base`. Future refiners may update `proc_label` while
    keeping `proc_label_base` intact.
    """
    if process_features is None or not isinstance(process_features, xr.Dataset):
        raise TypeError("process_features must be an xr.Dataset produced by build_process_features.")
    if "time" not in process_features.coords:
        raise KeyError("process_features must include the 'time' coordinate.")

    variables = ("Dm", "Nw", "LWC")
    core = _classify_from_microphysical_trends(
        process_features,
        variables=variables,
        min_strength=min_strength,
        min_tau_strength=min_tau_strength,
        max_p_value=max_p_value,
        max_tau_pvalue=max_tau_pvalue,
    )

    # Build a minimal classification dataset without RGB/hexagram attachments.
    sample_dims = tuple(str(dim) for dim in core["proc_label"].dims)
    out = xr.Dataset(coords=_coords_for_sample_dims(process_features, sample_dims))
    for name in ("z_top", "z_bottom", "z_center"):
        if name in process_features.coords:
            out = out.assign_coords({name: process_features.coords[name]})
    out["proc_label_base"] = core["proc_label"]
    out["proc_label"] = core["proc_label"].copy()
    out["strength"] = core["strength"]

    out["sign_Dm"] = core["sign_Dm"]
    out["sign_Nw"] = core["sign_Nw"]
    out["sign_LWC"] = core["sign_LWC"]
    out["sign_R"] = out["sign_Dm"]
    out["sign_G"] = out["sign_Nw"]
    out["sign_B"] = out["sign_LWC"]

    out.attrs["classification_stage1"] = "PROCESS_SIGNATURES"
    out.attrs["classification_basis"] = "canonical_microphysical_trends"
    out.attrs["min_tau_strength"] = float(
        float(min_tau_strength) if min_tau_strength is not None else float(min_strength)
    )
    out.attrs["max_tau_pvalue"] = (
        float(max_tau_pvalue) if max_tau_pvalue is not None else max_p_value
    )

    if refiners:
        for refiner in refiners:
            out = refiner(process_features, out)

    return out


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


def _detect_process_runs_from_sliding(
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


def build_sliding_column_process_dataframe(
    subject: SupportsRainAnalysis,
    *,
    period: tuple[datetime, datetime],
    k: int,
    window_thickness_m: float | None = None,
    window_step_m: float | None | _UnsetType = _UNSET,
    min_tau_strength: float | None | _UnsetType = _UNSET,
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
    Slide across the whole processed column with a sliding vertical window.

    For each window, the function runs the standard rain-process analysis and
    classification pipeline, then exports a per-sample dataframe. The output is
    therefore indexed by both time and layer window, and is intended as the
    input for consecutive-profile episode detection.

    When the caller does not provide ``window_thickness_m``, ``window_step_m``
    or ``min_tau_strength``, and ``subject`` exposes a ``micro_cfg`` attribute,
    the corresponding values are taken from that configuration.

    ``window_step_m=None`` means "raw resolution": infer the sliding step from the
    native range-grid spacing (median of the range-coordinate differences).
    """
    ds = _resolve_processed_dataset(subject)
    if period[0] >= period[1]:
        raise ValueError("period must be increasing (start, end).")

    micro_cfg = getattr(subject, "micro_cfg", None)

    thickness_m = (
        float(window_thickness_m)
        if window_thickness_m is not None
        else float(getattr(micro_cfg, "window_thickness_m", 1000.0))
    )

    step_param = (
        window_step_m
        if window_step_m is not _UNSET
        else getattr(micro_cfg, "window_step_m", 100.0)
    )

    if step_param is None:
        values = np.asarray(ds["range"].values, dtype=float)
        diffs = np.abs(np.diff(values))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size == 0:
            raise ValueError("Cannot infer raw vertical resolution from ds['range'].")
        step_m = float(np.median(diffs))
    else:
        step_m = _float_from_dynamic(step_param, name="window_step_m")

    tau_strength = (
        min_tau_strength
        if min_tau_strength is not _UNSET
        else getattr(micro_cfg, "min_tau_strength", 0.10)
    )

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
    Exploratory Option B: fuse vertical sliding detections into consolidated layers.

    The input ``sliding_df`` is expected to be the dataframe returned by
    :func:`build_sliding_column_process_dataframe`. For each time step, the
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

    ds = getattr(subject, "raprompro", None)
    if ds is None:
        raise RuntimeError("subject.raprompro is missing; load the processed dataset first.")

    micro_cfg = getattr(subject, "micro_cfg", None)
    if micro_cfg is None:
        raise RuntimeError("subject.micro_cfg is missing; cannot resolve default parameters.")

    resolved_variable_threshold = (
        str(variable_threshold) if variable_threshold is not None else str(micro_cfg.variable_threshold)
    )
    resolved_threshold_value = (
        float(threshold_value) if threshold_value is not None else float(micro_cfg.threshold_value)
    )
    resolved_trend_method = (
        str(trend_method) if trend_method is not None else str(micro_cfg.trend_method)
    )
    resolved_tau_zero_tol = (
        float(tau_zero_tol) if tau_zero_tol is not None else float(micro_cfg.tau_zero_tol)
    )
    resolved_min_points_trend = (
        int(min_points_trend) if min_points_trend is not None else int(micro_cfg.min_points_trend)
    )
    resolved_vars_trend_raw = (
        tuple(vars_trend) if vars_trend is not None else tuple(micro_cfg.vars_trend)
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

    # `build_sliding_column_process_dataframe` uses z_bottom_m/z_top_m; keep the public
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
            # In the sliding workflow, window_id increases with height.
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
            q=float(getattr(micro_cfg, "eps_q", 0.01)),
        )

        classified = classify_rain_process(
            subject,
            analysis=trends,
            min_tau_strength=float(micro_cfg.min_tau_strength),
            max_tau_pvalue=getattr(micro_cfg, "max_tau_pvalue", None),
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
    out.attrs["selection_mode"] = "sliding_fused_option_b"
    out.attrs["min_consecutive"] = int(min_consecutive)
    out.attrs["excluded_processes"] = tuple(exclude_processes)
    out.attrs["allowed_processes"] = tuple(allowed_processes) if allowed_processes is not None else None
    out.attrs["variable_threshold"] = resolved_variable_threshold
    out.attrs["threshold_value"] = float(resolved_threshold_value)
    out.attrs["trend_method"] = resolved_trend_method
    out.attrs["tau_zero_tol"] = float(resolved_tau_zero_tol)
    out.attrs["min_points_trend"] = int(resolved_min_points_trend)
    out.attrs["vars_trend"] = tuple(resolved_vars_trend)
    out.attrs["min_tau_strength"] = float(micro_cfg.min_tau_strength)
    return out


def detect_column_process_episodes(
    subject: SupportsRainAnalysis,
    *,
    sliding_df: pd.DataFrame,
    min_consecutive_profiles: int = 6,
) -> pd.DataFrame:
    """
    Detect temporally persistent process episodes from a sliding column dataframe.

    Only named microphysical processes are promoted to episodes; isolated
    ``unknown`` or ``steady_or_weak`` samples are ignored. Episodes are defined
    independently in each sliding vertical window.
    """
    del subject
    if not isinstance(sliding_df, pd.DataFrame):
        raise TypeError("sliding_df must be a pandas DataFrame.")
    episodes = _detect_process_runs_from_sliding(
        sliding_df,
        min_consecutive_profiles=int(min_consecutive_profiles),
    )
    episodes.attrs = dict(getattr(sliding_df, "attrs", {}))
    episodes.attrs["min_consecutive_profiles"] = int(min_consecutive_profiles)
    return episodes
