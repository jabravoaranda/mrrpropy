from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
import warnings

import numpy as np
import xarray as xr

from mrrpropy.analysis.trends import compute_layer_trend
from mrrpropy.hexagram import (
    build_rgb_from_unit_scores,
    get_hexagram_assets,
    map_rgb_to_hexagram,
)
from mrrpropy.processing.rain_process_info import PROCESS_SIGNATURES


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
    caller: str,
) -> tuple[float, float]:
    complete_sources = int(z_bottom_m is not None and z_top_m is not None)
    complete_sources += int(layer is not None)
    if complete_sources > 1:
        raise ValueError(
            f"{caller} received multiple layer definitions. Use either "
            "`z_bottom_m`/`z_top_m` or `layer=(z_bottom_m, z_top_m)`."
        )

    if (z_bottom_m is None) != (z_top_m is None):
        raise ValueError(f"{caller} requires both z_bottom_m and z_top_m.")

    if layer is not None:
        warnings.warn(
            "The `layer=(z_bottom_m, z_top_m)` argument is legacy. "
            "Use explicit `z_bottom_m` and `z_top_m` instead.",
            FutureWarning,
            stacklevel=2,
        )
        z_bottom_m, z_top_m = float(layer[0]), float(layer[1])

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
    scan workflow. Positive trend/change means increase while descending from
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


def classify_rain_process_features(
    rain_process_features: xr.Dataset,
    *,
    refiners: list[Any] | None = None,
    min_strength: float = 0.10,
    min_tau_strength: float | None = None,
    max_p_value: float | None = None,
    max_tau_pvalue: float | None = None,
) -> xr.Dataset:
    """
    Classify directly from `rain_process_features`.

    This wrapper never depends on RGB/hexagram fields. The baseline label is
    stored as `proc_label_base`. Future refiners may update `proc_label` while
    keeping `proc_label_base` intact.
    """
    if rain_process_features is None or not isinstance(rain_process_features, xr.Dataset):
        raise TypeError("rain_process_features must be an xr.Dataset produced by build_rain_process_features.")
    if "time" not in rain_process_features.coords:
        raise KeyError("rain_process_features must include the 'time' coordinate.")

    variables = ("Dm", "Nw", "LWC")
    core = _classify_from_microphysical_trends(
        rain_process_features,
        variables=variables,
        min_strength=min_strength,
        min_tau_strength=min_tau_strength,
        max_p_value=max_p_value,
        max_tau_pvalue=max_tau_pvalue,
    )

    # Build a minimal classification dataset without RGB/hexagram attachments.
    sample_dims = tuple(str(dim) for dim in core["proc_label"].dims)
    out = xr.Dataset(coords=_coords_for_sample_dims(rain_process_features, sample_dims))
    for name in ("z_top", "z_bottom", "z_center"):
        if name in rain_process_features.coords:
            out = out.assign_coords({name: rain_process_features.coords[name]})
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
            out = refiner(rain_process_features, out)

    return out

