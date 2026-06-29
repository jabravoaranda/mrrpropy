from __future__ import annotations

from typing import Literal, cast

import numpy as np
import xarray as xr


class _UnsetType:
    pass


_UNSET = _UnsetType()

ProcessFeatureMode = Literal["fixed_layer", "sliding"]


def build_process_features(
    ds: xr.Dataset,
    *,
    mode: ProcessFeatureMode,
    range_coord: str = "range",
    window_thickness_m: float | None = None,
    window_step_m: float | None | _UnsetType = _UNSET,
    fixed_layer_top_m: float | None = None,
    fixed_layer_bottom_m: float | None = None,
    bb_bottom_m: float | xr.DataArray,
    bb_peak_m: float | xr.DataArray,
    bb_top_m: float | xr.DataArray,
    micro_cfg: object | None = None,
    Dm_var: str = "Dm",
    Nw_var: str = "Nw",
    LWC_var: str = "LWC",
    RR_var: str = "RR",
    spectrum_var: str = "spectrum",
    velocity_coord: str = "velocity",
) -> xr.Dataset:
    """
    Phase A: build the full `process_features` dataset by merging:
    - microphysical features
    - Doppler spectral features
    - context features

    Modes:
    - fixed_layer: uses explicit (fixed_layer_top_m, fixed_layer_bottom_m) and returns dims (time)
    - sliding: generates windows using (window_thickness_m, window_step_m) and returns dims (time, layer)
      where coord `layer` is the physical layer-center height (m).

    Sliding defaults
    -------------
    In sliding mode, if ``window_thickness_m`` and/or ``window_step_m`` are not
    provided, the function falls back to the values stored in the caller's
    ``micro_cfg`` (typically :class:`mrrpropy.raw_class.MicrophysicsConfig`).

    ``window_step_m=None`` means "raw resolution": infer the sliding step from the
    native range grid spacing (median of the range-coordinate differences).
    """
    mode_value = str(mode).strip().lower()
    if mode_value not in {"fixed_layer", "sliding"}:
        raise ValueError("mode must be 'fixed_layer' or 'sliding'.")
    mode = cast(ProcessFeatureMode, mode_value)
    if "time" not in ds.coords:
        raise KeyError("ds must contain coord 'time'.")
    if range_coord not in ds.coords:
        raise KeyError(f"ds must contain coord '{range_coord}'.")

    def _infer_native_step_m() -> float:
        values = np.asarray(ds[range_coord].values, dtype=float)
        diffs = np.abs(np.diff(values))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size == 0:
            raise ValueError(
                f"Cannot infer raw vertical resolution from ds['{range_coord}']."
            )
        step_m = float(np.median(diffs))
        if not (np.isfinite(step_m) and step_m > 0.0):
            raise ValueError(
                f"Cannot infer a positive sliding step from ds['{range_coord}']."
            )
        return step_m

    if mode == "fixed_layer":
        if fixed_layer_top_m is None or fixed_layer_bottom_m is None:
            raise ValueError("fixed_layer mode requires fixed_layer_top_m and fixed_layer_bottom_m.")
        z_top_vals = float(fixed_layer_top_m)
        z_bottom_vals = float(fixed_layer_bottom_m)
        if not (np.isfinite(z_top_vals) and np.isfinite(z_bottom_vals) and z_top_vals > z_bottom_vals):
            raise ValueError("fixed_layer requires finite fixed_layer_top_m > fixed_layer_bottom_m.")

        z_top = xr.DataArray(z_top_vals)
        z_bottom = xr.DataArray(z_bottom_vals)
        z_center = xr.DataArray(0.5 * (z_top_vals + z_bottom_vals))
    else:
        if window_thickness_m is None and micro_cfg is not None:
            cfg_thickness = getattr(micro_cfg, "window_thickness_m", None)
            if cfg_thickness is not None:
                window_thickness_m = float(cfg_thickness)

        step_param = (
            getattr(micro_cfg, "window_step_m", None)
            if window_step_m is _UNSET
            else window_step_m
        )

        if window_thickness_m is None:
            raise ValueError("sliding mode requires window_thickness_m.")
        thickness = float(window_thickness_m)
        if step_param is None:
            step = _infer_native_step_m()
        elif isinstance(step_param, _UnsetType):
            raise RuntimeError("Internal error: unresolved window_step_m.")
        else:
            step = float(step_param)
        if not (np.isfinite(thickness) and thickness > 0.0):
            raise ValueError("sliding mode requires window_thickness_m > 0.")
        if not (np.isfinite(step) and step > 0.0):
            raise ValueError("sliding mode requires window_step_m > 0.")

        range_values = np.asarray(ds[range_coord].values, dtype=float)
        finite = np.isfinite(range_values)
        if not np.any(finite):
            raise ValueError(f"ds coord '{range_coord}' has no finite values.")
        z_min = float(np.min(range_values[finite]))
        z_max = float(np.max(range_values[finite]))

        start = z_min + 0.5 * thickness
        stop = z_max - 0.5 * thickness
        if not (np.isfinite(start) and np.isfinite(stop) and stop >= start):
            raise ValueError("sliding mode temporal/vertical window selection is empty: revise thickness/range.")

        centers = np.arange(start, np.nextafter(stop, stop + step), step, dtype=float)
        if centers.size == 0:
            raise ValueError("sliding mode produced no layer centers: revise thickness/step/range.")

        z_center = xr.DataArray(centers, dims=("layer",))
        z_top = xr.DataArray(centers + 0.5 * thickness, dims=("layer",))
        z_bottom = xr.DataArray(centers - 0.5 * thickness, dims=("layer",))

    micro = get_microphysical_features(
        ds,
        mode=mode,
        z_top=z_top,
        z_bottom=z_bottom,
        z_center=z_center,
        range_coord=range_coord,
        Dm_var=Dm_var,
        Nw_var=Nw_var,
        LWC_var=LWC_var,
    )
    spectral = get_spectral_features(
        ds,
        mode=mode,
        z_top=z_top,
        z_bottom=z_bottom,
        z_center=z_center,
        range_coord=range_coord,
        spectrum_var=spectrum_var,
        velocity_coord=velocity_coord,
    )
    context = get_context(
        ds,
        mode=mode,
        z_top=z_top,
        z_bottom=z_bottom,
        z_center=z_center,
        bb_bottom_m=bb_bottom_m,
        bb_peak_m=bb_peak_m,
        bb_top_m=bb_top_m,
        range_coord=range_coord,
        RR_var=RR_var,
    )

    out = xr.merge([micro, spectral, context], compat="no_conflicts")
    out.attrs = dict(ds.attrs)
    return out


def _weighted_percentile_1d(
    values: np.ndarray,
    weights: np.ndarray,
    p: float,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not np.any(finite):
        return np.nan
    v = values[finite]
    w = weights[finite]
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return np.nan

    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order] / w_sum
    cdf = np.cumsum(w_sorted)
    cdf = np.clip(cdf, 0.0, 1.0)

    # Weighted quantile as the smallest value where CDF >= p.
    target = float(np.clip(p, 0.0, 1.0))
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = max(0, min(idx, v_sorted.size - 1))
    return float(v_sorted[idx])


def _spectral_moments_1d(
    spectrum: np.ndarray,
    velocity: np.ndarray,
) -> tuple[float, float, float, float, float]:
    spectrum = np.asarray(spectrum, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    finite = np.isfinite(spectrum) & np.isfinite(velocity) & (spectrum >= 0.0)
    if not np.any(finite):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    s = spectrum[finite]
    v = velocity[finite]
    s_sum = float(np.sum(s))
    if not np.isfinite(s_sum) or s_sum <= 0.0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    w = s / s_sum
    v_mean = float(np.sum(w * v))
    v_var = float(np.sum(w * (v - v_mean) ** 2))
    v_std = float(np.sqrt(v_var)) if np.isfinite(v_var) and v_var >= 0.0 else np.nan

    v_p10 = _weighted_percentile_1d(v, s, 0.10)
    v_p50 = _weighted_percentile_1d(v, s, 0.50)
    v_p90 = _weighted_percentile_1d(v, s, 0.90)
    return (v_mean, v_std, v_p10, v_p50, v_p90)


def _spectral_moments_nd(
    spectra: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    out_shape = spectra.shape[:-1]
    out = np.full((*out_shape, 5), np.nan, dtype=float)

    finite_velocity = np.isfinite(velocity)
    if not np.any(finite_velocity):
        return out

    v = velocity[finite_velocity]
    order = np.argsort(v)
    v = v[order]
    weights = spectra[..., finite_velocity][..., order]
    weights = np.where(np.isfinite(weights) & (weights >= 0.0), weights, 0.0)

    w_sum = np.sum(weights, axis=-1)
    valid = np.isfinite(w_sum) & (w_sum > 0.0)
    if not np.any(valid):
        return out

    mean = np.sum(weights * v, axis=-1) / w_sum
    var = np.sum(weights * (v - mean[..., np.newaxis]) ** 2, axis=-1) / w_sum

    out[..., 0] = np.where(valid, mean, np.nan)
    out[..., 1] = np.where(valid & np.isfinite(var) & (var >= 0.0), np.sqrt(var), np.nan)

    cdf = np.cumsum(weights, axis=-1) / w_sum[..., np.newaxis]
    cdf = np.clip(cdf, 0.0, 1.0)
    for out_index, percentile in enumerate((0.10, 0.50, 0.90), start=2):
        idx = np.argmax(cdf >= percentile, axis=-1)
        out[..., out_index] = np.where(valid, np.take(v, idx), np.nan)

    return out


def get_spectral_features(
    ds: xr.Dataset,
    *,
    mode: ProcessFeatureMode,
    z_top: xr.DataArray,
    z_bottom: xr.DataArray,
    z_center: xr.DataArray,
    range_coord: str = "range",
    spectrum_var: str = "spectrum",
    velocity_coord: str = "velocity",
) -> xr.Dataset:
    """
    Build Doppler spectral features for one layer (fixed_layer) or many (sliding).

    ds[spectrum_var] must have dims (time, range, velocity) and be non-negative.
    velocity is in m/s using the public mrrpropy convention: negative downward.
    Therefore ``delta_v_*`` is bottom minus top; stronger downward motion at the
    bottom appears as a more negative delta.

    Inputs `z_top`, `z_bottom`, `z_center`:
    - fixed_layer: scalar DataArray (meters)
    - sliding: DataArray with dim 'layer' (meters)
    """
    if "time" not in ds.coords:
        raise KeyError("ds must contain coord 'time'.")
    if range_coord not in ds.coords:
        raise KeyError(f"ds must contain coord '{range_coord}'.")
    if spectrum_var not in ds:
        raise KeyError(f"ds must contain variable '{spectrum_var}'.")

    mode_value = str(mode).strip().lower()
    if mode_value not in {"fixed_layer", "sliding"}:
        raise ValueError("mode must be 'fixed_layer' or 'sliding'.")
    mode = cast(ProcessFeatureMode, mode_value)

    spectrum = ds[spectrum_var]
    if spectrum.ndim != 3 or spectrum.dims[0] != "time" or spectrum.dims[1] != range_coord:
        raise ValueError(
            f"ds['{spectrum_var}'] must have dims ('time','{range_coord}','{velocity_coord}')."
        )
    if velocity_coord not in ds.coords:
        spectral_velocity_coord = str(spectrum.dims[2])
        if spectral_velocity_coord in ds.coords:
            velocity_coord = spectral_velocity_coord
        elif velocity_coord == "W" and "W" in ds:
            raise KeyError(
                "'W' is a Doppler velocity field, but spectral moments require "
                f"the spectrum velocity coordinate '{spectral_velocity_coord}'."
            )
        else:
            raise KeyError(f"ds must contain coord '{velocity_coord}'.")
    if spectrum.dims[2] != velocity_coord:
        raise ValueError(
            f"ds['{spectrum_var}'] must have velocity dimension '{velocity_coord}'."
        )

    out_dims: tuple[str, ...]
    if mode == "fixed_layer":
        if z_top.dims or z_bottom.dims or z_center.dims:
            raise ValueError("fixed_layer mode requires scalar z_top/z_bottom/z_center.")
        out_dims = ("time",)
    else:
        if z_top.dims != ("layer",) or z_bottom.dims != ("layer",) or z_center.dims != ("layer",):
            raise ValueError("sliding mode requires z_top/z_bottom/z_center with dim ('layer',).")
        out_dims = ("time", "layer")
        # In sliding mode, use the physical center height (m) as the layer coordinate.
        layer_values_m = np.asarray(z_center.values, dtype=float).reshape((-1,))
        z_top = z_top.assign_coords(layer=layer_values_m)
        z_bottom = z_bottom.assign_coords(layer=layer_values_m)
        z_center = z_center.assign_coords(layer=layer_values_m)

    vel = np.asarray(ds[velocity_coord].values, dtype=float)
    spec_np = np.asarray(spectrum.values, dtype=float)
    if np.any(np.isfinite(spec_np) & (spec_np < 0.0)):
        raise ValueError(f"ds['{spectrum_var}'] must be non-negative.")

    n_time = int(ds.sizes.get("time", 0))
    n_layer = 1 if mode == "fixed_layer" else int(z_center.sizes["layer"])

    def _alloc() -> np.ndarray:
        shape = (n_time,) if mode == "fixed_layer" else (n_time, n_layer)
        return np.full(shape, np.nan, dtype=float)

    out = xr.Dataset(coords={"time": ds["time"].values})
    if mode == "sliding":
        out = out.assign_coords(layer=layer_values_m)
    if mode == "fixed_layer":
        # Broadcast scalar layer coordinates to time so outputs are (time,).
        out = out.assign_coords(
            {
                "z_top": xr.DataArray(
                    np.full(out.sizes["time"], float(z_top.values), dtype=float),
                    dims=("time",),
                ),
                "z_bottom": xr.DataArray(
                    np.full(out.sizes["time"], float(z_bottom.values), dtype=float),
                    dims=("time",),
                ),
                "z_center": xr.DataArray(
                    np.full(out.sizes["time"], float(z_center.values), dtype=float),
                    dims=("time",),
                ),
            }
        )
    else:
        out = out.assign_coords({"z_top": z_top, "z_bottom": z_bottom, "z_center": z_center})

    z_top_vals = (
        np.asarray(z_top.values, dtype=float).reshape((-1,))
        if mode == "sliding"
        else np.asarray([float(z_top.values)], dtype=float)
    )
    z_bottom_vals = (
        np.asarray(z_bottom.values, dtype=float).reshape((-1,))
        if mode == "sliding"
        else np.asarray([float(z_bottom.values)], dtype=float)
    )

    range_values_m = np.asarray(ds[range_coord].values, dtype=float)

    def _nearest_range_index(z_m: float) -> int:
        diffs = np.abs(range_values_m - float(z_m))
        idx = int(np.nanargmin(diffs))
        return idx

    v_mean_top = _alloc()
    v_mean_bottom = _alloc()
    v_std_top = _alloc()
    v_std_bottom = _alloc()
    v_p10_top = _alloc()
    v_p10_bottom = _alloc()
    v_p50_top = _alloc()
    v_p50_bottom = _alloc()
    v_p90_top = _alloc()
    v_p90_bottom = _alloc()

    valid_layers = np.isfinite(z_top_vals) & np.isfinite(z_bottom_vals) & (z_top_vals > z_bottom_vals)
    if np.any(valid_layers):
        top_indices = np.array([_nearest_range_index(z) for z in z_top_vals[valid_layers]], dtype=int)
        bottom_indices = np.array([_nearest_range_index(z) for z in z_bottom_vals[valid_layers]], dtype=int)

        top_stats = _spectral_moments_nd(spec_np[:, top_indices, :], vel)
        bottom_stats = _spectral_moments_nd(spec_np[:, bottom_indices, :], vel)

        if mode == "fixed_layer":
            v_mean_top[:] = top_stats[:, 0, 0]
            v_std_top[:] = top_stats[:, 0, 1]
            v_p10_top[:] = top_stats[:, 0, 2]
            v_p50_top[:] = top_stats[:, 0, 3]
            v_p90_top[:] = top_stats[:, 0, 4]

            v_mean_bottom[:] = bottom_stats[:, 0, 0]
            v_std_bottom[:] = bottom_stats[:, 0, 1]
            v_p10_bottom[:] = bottom_stats[:, 0, 2]
            v_p50_bottom[:] = bottom_stats[:, 0, 3]
            v_p90_bottom[:] = bottom_stats[:, 0, 4]
        else:
            v_mean_top[:, valid_layers] = top_stats[..., 0]
            v_std_top[:, valid_layers] = top_stats[..., 1]
            v_p10_top[:, valid_layers] = top_stats[..., 2]
            v_p50_top[:, valid_layers] = top_stats[..., 3]
            v_p90_top[:, valid_layers] = top_stats[..., 4]

            v_mean_bottom[:, valid_layers] = bottom_stats[..., 0]
            v_std_bottom[:, valid_layers] = bottom_stats[..., 1]
            v_p10_bottom[:, valid_layers] = bottom_stats[..., 2]
            v_p50_bottom[:, valid_layers] = bottom_stats[..., 3]
            v_p90_bottom[:, valid_layers] = bottom_stats[..., 4]

    out["v_mean_top"] = xr.DataArray(v_mean_top, dims=out_dims)
    out["v_mean_bottom"] = xr.DataArray(v_mean_bottom, dims=out_dims)
    out["delta_v_mean"] = out["v_mean_bottom"] - out["v_mean_top"]

    out["v_std_top"] = xr.DataArray(v_std_top, dims=out_dims)
    out["v_std_bottom"] = xr.DataArray(v_std_bottom, dims=out_dims)
    out["delta_v_std"] = out["v_std_bottom"] - out["v_std_top"]

    out["v_p10_top"] = xr.DataArray(v_p10_top, dims=out_dims)
    out["v_p10_bottom"] = xr.DataArray(v_p10_bottom, dims=out_dims)
    out["delta_v_p10"] = out["v_p10_bottom"] - out["v_p10_top"]

    out["v_p50_top"] = xr.DataArray(v_p50_top, dims=out_dims)
    out["v_p50_bottom"] = xr.DataArray(v_p50_bottom, dims=out_dims)
    out["delta_v_p50"] = out["v_p50_bottom"] - out["v_p50_top"]

    out["v_p90_top"] = xr.DataArray(v_p90_top, dims=out_dims)
    out["v_p90_bottom"] = xr.DataArray(v_p90_bottom, dims=out_dims)
    out["delta_v_p90"] = out["v_p90_bottom"] - out["v_p90_top"]

    for name in (
        "v_mean_top",
        "v_mean_bottom",
        "v_p10_top",
        "v_p10_bottom",
        "v_p50_top",
        "v_p50_bottom",
        "v_p90_top",
        "v_p90_bottom",
    ):
        out[name].attrs.update(
            {
                "units": "m s-1",
                "positive": "up",
                "convention": "negative values indicate downward hydrometeor motion",
            }
        )
    for name in ("v_std_top", "v_std_bottom", "delta_v_std"):
        out[name].attrs["units"] = "m s-1"
    for name in ("delta_v_mean", "delta_v_p10", "delta_v_p50", "delta_v_p90"):
        out[name].attrs.update(
            {
                "units": "m s-1",
                "positive": "up",
                "description": "Bottom minus top velocity difference; negative values indicate stronger downward motion at the bottom.",
            }
        )

    return out


def get_context(
    ds: xr.Dataset,
    *,
    mode: ProcessFeatureMode,
    z_top: xr.DataArray,
    z_bottom: xr.DataArray,
    z_center: xr.DataArray,
    bb_bottom_m: float | xr.DataArray,
    bb_peak_m: float | xr.DataArray,
    bb_top_m: float | xr.DataArray,
    range_coord: str = "range",
    RR_var: str = "RR",
) -> xr.Dataset:
    """
    Build context features for one layer (fixed_layer) or many (sliding).

    - dist_bb_peak = z_center - bb_peak
    - dist_bb_bottom = z_center - bb_bottom
    - overlaps_bb = (z_top > bb_bottom) & (z_bottom < bb_top)
    - RR_mean = mean(RR) over range within [z_bottom, z_top]

    bb_* inputs may be floats or DataArrays aligned/broadcastable to time.
    """
    if "time" not in ds.coords:
        raise KeyError("ds must contain coord 'time'.")
    if range_coord not in ds.coords:
        raise KeyError(f"ds must contain coord '{range_coord}'.")
    if RR_var not in ds:
        raise KeyError(f"ds must contain variable '{RR_var}'.")

    mode_value = str(mode).strip().lower()
    if mode_value not in {"fixed_layer", "sliding"}:
        raise ValueError("mode must be 'fixed_layer' or 'sliding'.")
    mode = cast(ProcessFeatureMode, mode_value)

    out_dims: tuple[str, ...]
    if mode == "fixed_layer":
        if z_top.dims or z_bottom.dims or z_center.dims:
            raise ValueError("fixed_layer mode requires scalar z_top/z_bottom/z_center.")
        out_dims = ("time",)
    else:
        if z_top.dims != ("layer",) or z_bottom.dims != ("layer",) or z_center.dims != ("layer",):
            raise ValueError("sliding mode requires z_top/z_bottom/z_center with dim ('layer',).")
        out_dims = ("time", "layer")
        # In sliding mode, use the physical center height (m) as the layer coordinate.
        layer_values_m = np.asarray(z_center.values, dtype=float).reshape((-1,))
        z_top = z_top.assign_coords(layer=layer_values_m)
        z_bottom = z_bottom.assign_coords(layer=layer_values_m)
        z_center = z_center.assign_coords(layer=layer_values_m)

    out = xr.Dataset(coords={"time": ds["time"].values})
    if mode == "sliding":
        out = out.assign_coords(layer=layer_values_m)
    out = out.assign_coords({"z_top": z_top, "z_bottom": z_bottom, "z_center": z_center})

    bb_bottom = xr.DataArray(bb_bottom_m) if not isinstance(bb_bottom_m, xr.DataArray) else bb_bottom_m
    bb_peak = xr.DataArray(bb_peak_m) if not isinstance(bb_peak_m, xr.DataArray) else bb_peak_m
    bb_top = xr.DataArray(bb_top_m) if not isinstance(bb_top_m, xr.DataArray) else bb_top_m

    # Align on time if time coordinate exists on bb_* arrays.
    if "time" in bb_bottom.coords:
        bb_bottom = bb_bottom.sel(time=out["time"])
    if "time" in bb_peak.coords:
        bb_peak = bb_peak.sel(time=out["time"])
    if "time" in bb_top.coords:
        bb_top = bb_top.sel(time=out["time"])
    if mode == "fixed_layer":
        # Ensure bb_* are time-varying arrays so outputs keep (time,) shape.
        if not bb_bottom.dims:
            bb_bottom = xr.DataArray(
                np.full(out.sizes["time"], float(bb_bottom.values), dtype=float),
                dims=("time",),
            )
        if not bb_peak.dims:
            bb_peak = xr.DataArray(
                np.full(out.sizes["time"], float(bb_peak.values), dtype=float),
                dims=("time",),
            )
        if not bb_top.dims:
            bb_top = xr.DataArray(
                np.full(out.sizes["time"], float(bb_top.values), dtype=float),
                dims=("time",),
            )

    z_top_used = out.coords["z_top"]
    z_bottom_used = out.coords["z_bottom"]
    z_center_used = out.coords["z_center"]
    if mode == "sliding":
        z_top_used = z_top_used.expand_dims(time=out["time"])
        z_bottom_used = z_bottom_used.expand_dims(time=out["time"])
        z_center_used = z_center_used.expand_dims(time=out["time"])
    out["dist_bb_peak"] = z_center_used - bb_peak
    out["dist_bb_bottom"] = z_center_used - bb_bottom
    out["overlaps_bb"] = (z_top_used > bb_bottom) & (z_bottom_used < bb_top)

    rr = ds[RR_var]
    range_values_m = np.asarray(ds[range_coord].values, dtype=float)
    rr_np = np.asarray(rr.values, dtype=float)

    n_time = int(ds.sizes.get("time", 0))
    n_layer = 1 if mode == "fixed_layer" else int(z_center.sizes["layer"])
    rr_mean = np.full((n_time,) if mode == "fixed_layer" else (n_time, n_layer), np.nan, dtype=float)

    z_top_vals = (
        np.asarray(z_top.values, dtype=float).reshape((-1,))
        if mode == "sliding"
        else np.asarray([float(z_top.values)], dtype=float)
    )
    z_bottom_vals = (
        np.asarray(z_bottom.values, dtype=float).reshape((-1,))
        if mode == "sliding"
        else np.asarray([float(z_bottom.values)], dtype=float)
    )

    for time_index in range(n_time):
        for layer_index in range(n_layer):
            zt = float(z_top_vals[layer_index])
            zb = float(z_bottom_vals[layer_index])
            if not (np.isfinite(zt) and np.isfinite(zb) and zt > zb):
                continue

            inside = (range_values_m >= zb) & (range_values_m <= zt)
            if not np.any(inside):
                continue
            vals = rr_np[time_index, inside]
            finite = vals[np.isfinite(vals)]
            rr_mean_val = float(np.mean(finite)) if finite.size else np.nan
            if mode == "fixed_layer":
                rr_mean[time_index] = rr_mean_val
            else:
                rr_mean[time_index, layer_index] = rr_mean_val

    out["RR_mean"] = xr.DataArray(rr_mean, dims=out_dims)
    return out


def _trend_over_layer(
    range_values_m: np.ndarray,
    profile_values: np.ndarray,
    *,
    z_top_m: float,
    z_bottom_m: float,
) -> tuple[float, int, float, float, float]:
    """
    Compute (tau, sign, strength, p, magnitude) for one vertical profile.

    Convention:
    - z_top_m > z_bottom_m
    - "descending" means moving from z_top_m down to z_bottom_m
    - positive trend means increase while descending
    """
    finite = np.isfinite(range_values_m) & np.isfinite(profile_values)
    if not np.any(finite):
        return (np.nan, 0, np.nan, np.nan, np.nan)

    z = np.asarray(range_values_m[finite], dtype=float)
    y = np.asarray(profile_values[finite], dtype=float)
    inside = (z >= float(z_bottom_m)) & (z <= float(z_top_m))
    if not np.any(inside):
        return (np.nan, 0, np.nan, np.nan, np.nan)

    z = z[inside]
    y = y[inside]

    # Order physically from top to bottom (descending in height).
    order = np.argsort(z)[::-1]
    z = z[order]
    y = y[order]

    if z.size < 2:
        return (np.nan, 0, np.nan, np.nan, np.nan)

    # Depth increases while descending from the top of the layer.
    x_depth = float(z_top_m) - z

    from scipy.stats import kendalltau, theilslopes

    tau, p_value = kendalltau(x_depth, y, nan_policy="omit")
    if not np.isfinite(tau):
        return (np.nan, 0, np.nan, float(p_value) if np.isfinite(p_value) else np.nan, np.nan)

    sign = int(np.sign(tau))
    strength = float(np.clip(abs(tau), 0.0, 1.0))

    slope = np.nan
    try:
        slope, _, _, _ = theilslopes(y, x_depth)
    except Exception:
        slope = np.nan

    return (float(tau), sign, strength, float(p_value), float(slope) if np.isfinite(slope) else np.nan)


def _sign_char(sign: int | float) -> str:
    if not np.isfinite(sign):
        return "?"
    if sign > 0:
        return "+"
    if sign < 0:
        return "-"
    return "0"


def get_microphysical_features(
    ds: xr.Dataset,
    *,
    mode: ProcessFeatureMode,
    z_top: xr.DataArray,
    z_bottom: xr.DataArray,
    z_center: xr.DataArray,
    range_coord: str = "range",
    Dm_var: str = "Dm",
    Nw_var: str = "Nw",
    LWC_var: str = "LWC",
) -> xr.Dataset:
    """
    Build microphysical process features for one layer (fixed_layer) or many (sliding).

    Inputs `z_top`, `z_bottom`, `z_center`:
    - fixed_layer: scalar DataArray (meters)
    - sliding: DataArray with dim 'layer' (meters)
    """
    if "time" not in ds.coords:
        raise KeyError("ds must contain coord 'time'.")
    if range_coord not in ds.coords:
        raise KeyError(f"ds must contain coord '{range_coord}'.")
    for name in (Dm_var, Nw_var, LWC_var):
        if name not in ds:
            raise KeyError(f"ds must contain variable '{name}'.")

    mode_value = str(mode).strip().lower()
    if mode_value not in {"fixed_layer", "sliding"}:
        raise ValueError("mode must be 'fixed_layer' or 'sliding'.")
    mode = cast(ProcessFeatureMode, mode_value)

    out_dims: tuple[str, ...]
    if mode == "fixed_layer":
        if z_top.dims or z_bottom.dims or z_center.dims:
            raise ValueError("fixed_layer mode requires scalar z_top/z_bottom/z_center.")
        out_dims = ("time",)
    else:
        if z_top.dims != ("layer",) or z_bottom.dims != ("layer",) or z_center.dims != ("layer",):
            raise ValueError("sliding mode requires z_top/z_bottom/z_center with dim ('layer',).")
        out_dims = ("time", "layer")
        # In sliding mode, use the physical center height (m) as the layer coordinate.
        layer_values_m = np.asarray(z_center.values, dtype=float).reshape((-1,))
        z_top = z_top.assign_coords(layer=layer_values_m)
        z_bottom = z_bottom.assign_coords(layer=layer_values_m)
        z_center = z_center.assign_coords(layer=layer_values_m)

    range_values_m = np.asarray(ds[range_coord].values, dtype=float)

    top = {
        "Dm": ds[Dm_var].sel({range_coord: z_top}, method="nearest"),
        "Nw": ds[Nw_var].sel({range_coord: z_top}, method="nearest"),
        "LWC": ds[LWC_var].sel({range_coord: z_top}, method="nearest"),
    }
    bottom = {
        "Dm": ds[Dm_var].sel({range_coord: z_bottom}, method="nearest"),
        "Nw": ds[Nw_var].sel({range_coord: z_bottom}, method="nearest"),
        "LWC": ds[LWC_var].sel({range_coord: z_bottom}, method="nearest"),
    }

    out = xr.Dataset(coords={"time": ds["time"].values})
    if mode == "sliding":
        out = out.assign_coords(layer=layer_values_m)

    out = out.assign_coords(
        {
            "z_top": z_top,
            "z_bottom": z_bottom,
            "z_center": z_center,
        }
    )

    variables = ("Dm", "Nw", "LWC")
    for var in variables:
        out[f"{var}_top"] = top[var]
        out[f"{var}_bottom"] = bottom[var]
        out[f"delta_{var}"] = bottom[var] - top[var]
        out[f"rel_change_{var}"] = xr.where(
            top[var] != 0.0,
            100.0 * (bottom[var] - top[var]) / top[var],
            np.nan,
        )

    n_time = int(ds.sizes.get("time", 0))
    n_layer = 1 if mode == "fixed_layer" else int(z_center.sizes["layer"])

    def _alloc() -> np.ndarray:
        shape = (n_time,) if mode == "fixed_layer" else (n_time, n_layer)
        return np.full(shape, np.nan, dtype=float)

    tau = {v: _alloc() for v in variables}
    pval = {v: _alloc() for v in variables}
    mag = {v: _alloc() for v in variables}
    sign = {v: np.zeros(tau[variables[0]].shape, dtype=int) for v in variables}
    strength = {v: _alloc() for v in variables}

    z_top_vals = np.asarray(z_top.values, dtype=float).reshape((-1,)) if mode == "sliding" else np.asarray([float(z_top.values)], dtype=float)
    z_bottom_vals = np.asarray(z_bottom.values, dtype=float).reshape((-1,)) if mode == "sliding" else np.asarray([float(z_bottom.values)], dtype=float)

    data_np = {
        "Dm": np.asarray(ds[Dm_var].values, dtype=float),
        "Nw": np.asarray(ds[Nw_var].values, dtype=float),
        "LWC": np.asarray(ds[LWC_var].values, dtype=float),
    }

    for time_index in range(n_time):
        for layer_index in range(n_layer):
            zt = float(z_top_vals[layer_index])
            zb = float(z_bottom_vals[layer_index])
            if not (np.isfinite(zt) and np.isfinite(zb) and zt > zb):
                continue
            for var in variables:
                t, s, st, p, m = _trend_over_layer(
                    range_values_m,
                    data_np[var][time_index, :],
                    z_top_m=zt,
                    z_bottom_m=zb,
                )
                if mode == "fixed_layer":
                    tau[var][time_index] = t
                    pval[var][time_index] = p
                    mag[var][time_index] = m
                    sign[var][time_index] = s
                    strength[var][time_index] = st
                else:
                    tau[var][time_index, layer_index] = t
                    pval[var][time_index, layer_index] = p
                    mag[var][time_index, layer_index] = m
                    sign[var][time_index, layer_index] = s
                    strength[var][time_index, layer_index] = st

    for var in variables:
        out[f"tau_{var}"] = xr.DataArray(tau[var], dims=out_dims)
        out[f"trend_sign_{var}"] = xr.DataArray(sign[var], dims=out_dims)
        out[f"trend_strength_{var}"] = xr.DataArray(strength[var], dims=out_dims)
        out[f"trend_p_{var}"] = xr.DataArray(pval[var], dims=out_dims)
        out[f"trend_mag_{var}"] = xr.DataArray(mag[var], dims=out_dims)

    # Optional compact signature string like "+,-,-" for [Dm,Nw,LWC].
    sig = np.empty(sign["Dm"].shape, dtype=object)
    it = np.nditer(sign["Dm"], flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        s_dm = _sign_char(sign["Dm"][idx])
        s_nw = _sign_char(sign["Nw"][idx])
        s_lwc = _sign_char(sign["LWC"][idx])
        sig[idx] = f"{s_dm},{s_nw},{s_lwc}"
        it.iternext()

    out["micro_signature_str"] = xr.DataArray(sig, dims=out_dims)
    return out
