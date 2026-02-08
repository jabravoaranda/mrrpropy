from datetime import datetime
import numpy as np
import xarray as xr

from mrrpropy.generate_rgb_hex import generate_rgb_hex  # ajusta import según tu estructura


def to_time_slice(target: datetime | tuple[datetime, datetime] | slice) -> slice:
    if isinstance(target, slice):
        return target
    if isinstance(target, tuple):
        t0, t1 = target
        if t0 >= t1:
            raise ValueError("target_datetime tuple must be increasing (start, end).")
        return slice(np.datetime64(t0), np.datetime64(t1))
    # single datetime -> a tiny slice around it (nearest semantics later)
    t = np.datetime64(target)
    return slice(t, t)


def ols_slope_intercept_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    OLS for y = a + b x.
    Returns (b, a, r2). Requires len(x) >= 2 and finite values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_mean = x.mean()
    y_mean = y.mean()

    dx = x - x_mean
    dy = y - y_mean

    sxx = np.dot(dx, dx)
    if sxx == 0.0:
        return np.nan, np.nan, np.nan

    b = np.dot(dx, dy) / sxx
    a = y_mean - b * x_mean

    # R^2
    y_hat = a + b * x
    ss_res = np.dot(y - y_hat, y - y_hat)
    ss_tot = np.dot(y - y_mean, y - y_mean)
    if ss_tot == 0.0:
        # y constant -> define r2 as 1 if perfect fit, else nan; here residual also 0 => 1
        return b, a, 1.0 if ss_res == 0.0 else np.nan

    r2 = 1.0 - (ss_res / ss_tot)
    return b, a, r2


def compute_eps(values: np.ndarray, q: float) -> float:
    """Quantile-based epsilon over positive finite values."""
    v = values[np.isfinite(values) & (values > 0)]
    if v.size == 0:
        return np.nan
    return float(np.quantile(v, q))


def build_rgb_from_trends(
    ds: xr.Dataset,
    *,
    time_dim: str = "time",
    vars: tuple[str, str, str] = ("b_Dm", "b_Nw", "b_LWC"),
    q: float = 0.02,
    per_hour: bool = False,
) -> xr.Dataset:
    """
    Convierte tres series (con signo) a canales RGB en [0,1], con 0.5 = 0.
    Normalización robusta por cuantiles simétricos.

    vars: nombres en ds para (R,G,B) en ese orden.
    q: cuantil para escala robusta (ej. 0.02 => recorta 2% extremos).
    per_hour: si True, calcula la escala por cada instante (solo tiene sentido si ds tiene sub-sampling dentro de la hora;
              si cada fichero es 1 hora con múltiples timestamps, funcionará; si es 1 timestamp/hora, per_hour no aporta).
    """
    vR = ds[vars[0]].values.astype(float)
    vG = ds[vars[1]].values.astype(float)
    vB = ds[vars[2]].values.astype(float)

    def _scale_global(v):
        vv = v[np.isfinite(v)]
        if vv.size == 0:
            return np.nan
        lo = np.quantile(vv, q)
        hi = np.quantile(vv, 1 - q)
        s = max(abs(lo), abs(hi))
        return float(s) if s > 0 else 0.0

    def _to_unit(v, s):
        if not np.isfinite(s) or s <= 0:
            # si no hay escala (todo ~0), devolvemos 0.5 cuando v es finito, NaN si no
            out = np.full_like(v, np.nan, dtype=float)
            out[np.isfinite(v)] = 0.5
            return out
        x = np.clip(v / s, -1.0, 1.0)
        return 0.5 * (x + 1.0)

    # En tu caso habitual, per_hour=False (escala global del evento/capa).
    sR = _scale_global(vR)
    sG = _scale_global(vG)
    sB = _scale_global(vB)

    R = _to_unit(vR, sR)
    G = _to_unit(vG, sG)
    B = _to_unit(vB, sB)

    out = xr.Dataset(coords={time_dim: ds[time_dim].values})
    out["R"] = xr.DataArray(R, dims=(time_dim,))
    out["G"] = xr.DataArray(G, dims=(time_dim,))
    out["B"] = xr.DataArray(B, dims=(time_dim,))

    out.attrs["q"] = q
    out.attrs["scale_R"] = sR
    out.attrs["scale_G"] = sG
    out.attrs["scale_B"] = sB
    out.attrs["source_vars"] = ",".join(vars)
    return out


def _sign_from_center(c: np.ndarray, tol: float = 0.05) -> np.ndarray:
    # c en [0,1], centro 0.5
    s = np.zeros_like(c, dtype=int)
    s[c > 0.5 + tol] = +1
    s[c < 0.5 - tol] = -1
    return s

def _strength(c: np.ndarray) -> np.ndarray:
    # 0..1
    return np.clip(np.abs(c - 0.5) / 0.5, 0.0, 1.0)

def get_hexagram_assets(k: int, valid_threshold: float = -0.5):
    r_hex, g_hex, b_hex, num_hex = generate_rgb_hex(k=k)

    r = np.asarray(r_hex, float)
    g = np.asarray(g_hex, float)
    b = np.asarray(b_hex, float)
    n = np.asarray(num_hex)

    valid = (r > valid_threshold) & (g > valid_threshold) & (b > valid_threshold)
    ys, xs = np.where(valid)

    # LUT para mapping
    rgb_cells = np.stack([r[ys, xs], g[ys, xs], b[ys, xs]], axis=1)
    yx_cells = np.stack([ys, xs], axis=1)

    if np.nanmax(n[ys, xs]) <= 1.0:
        area_cells = np.rint(n[ys, xs] * 256).astype(int)
    else:
        area_cells = np.rint(n[ys, xs]).astype(int)

    # Imagen para plotting
    img = np.ones((r.shape[0], r.shape[1], 3))
    img[valid, 0] = r[valid]
    img[valid, 1] = g[valid]
    img[valid, 2] = b[valid]

    return {
        "k": k,
        "img": img,
        "rgb_cells": rgb_cells,
        "yx_cells": yx_cells,
        "area_cells": area_cells,
    }


def map_rgb_to_hexagram(rgb: xr.Dataset, *, hex_assets, time_dim="time"):
    rgb_cells = hex_assets["rgb_cells"]
    yx_cells = hex_assets["yx_cells"]
    area_cells = hex_assets["area_cells"]

    P = np.stack(
        [rgb["R"].values, rgb["G"].values, rgb["B"].values], axis=1
    )

    N = P.shape[0]
    yx = np.full((N, 2), -1, int)
    area = np.full(N, -1, int)

    ok = np.isfinite(P).all(axis=1)
    if np.any(ok):
        diff = P[ok][:, None, :] - rgb_cells[None, :, :]
        idx = np.argmin(np.sum(diff**2, axis=2), axis=1)
        yx[ok] = yx_cells[idx]
        area[ok] = area_cells[idx]

    out = xr.Dataset(coords={time_dim: rgb[time_dim]})
    out["hex_y"] = (time_dim, yx[:, 0])
    out["hex_x"] = (time_dim, yx[:, 1])
    out["hex_area"] = (time_dim, area)
    return out

