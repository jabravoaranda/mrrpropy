from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
import warnings

import numpy as np
import xarray as xr

from mrrpropy.utils import compute_eps, compute_monotonic_trend, ols_slope_intercept_r2


class SupportsRainAnalysis(Protocol):
    path: str | Path
    raprompro: xr.Dataset | None

    def _is_processed(self) -> bool: ...


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


def _distance_below_layer_top(z_layer: np.ndarray, *, z_top_m: float) -> np.ndarray:
    return (float(z_top_m) - np.asarray(z_layer, dtype=float)).astype(float)


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
