"""Hexagram mapping and classification helpers for rain-process diagnostics."""

import csv
from pathlib import Path
from typing import TypeAlias, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import xarray as xr

from mrrpropy.processes import PROCESS_CODES, PROCESS_MARKERS, PROCESS_SIGNATURES, ProcessSignature

FloatArray: TypeAlias = NDArray[np.float64]
Float32Array: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int_]
BoolArray: TypeAlias = NDArray[np.bool_]


class HexagramAssets(TypedDict):
    k: int
    img: FloatArray
    rgb_cells: FloatArray
    yx_cells: IntArray
    area_cells: IntArray


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


def build_rgb_from_tau(
    ds: xr.Dataset,
    *,
    time_dim: str = "time",
    vars: tuple[str, str, str] = ("tau_Dm", "tau_Nw", "tau_LWC"),
) -> xr.Dataset:
    """
    Build RGB channels from bounded signed trend scores.

    Each component is mapped naturally from ``[-1, 1]`` to ``[0, 1]`` so that
    a score of ``0`` sits at the hexagram centre (``0.5``).
    """

    def _tau_to_unit(values: np.ndarray) -> np.ndarray:
        out = np.full_like(values, np.nan, dtype=float)
        finite = np.isfinite(values)
        out[finite] = 0.5 * (np.clip(values[finite], -1.0, 1.0) + 1.0)
        return out

    out = xr.Dataset(coords={time_dim: ds[time_dim].values})
    out["R"] = xr.DataArray(_tau_to_unit(ds[vars[0]].values.astype(float)), dims=(time_dim,))
    out["G"] = xr.DataArray(_tau_to_unit(ds[vars[1]].values.astype(float)), dims=(time_dim,))
    out["B"] = xr.DataArray(_tau_to_unit(ds[vars[2]].values.astype(float)), dims=(time_dim,))
    out.attrs["method"] = "tau"
    out.attrs["source_vars"] = ",".join(vars)
    out.attrs["mapping"] = "natural: tau in [-1,1] -> RGB in [0,1]"
    return out


def build_rgb_from_unit_scores(
    ds: xr.Dataset,
    *,
    time_dim: str = "time",
    vars: tuple[str, str, str] = (
        "trend_score_Dm",
        "trend_score_Nw",
        "trend_score_LWC",
    ),
) -> xr.Dataset:
    """
    Build RGB channels from method-neutral signed trend scores in ``[-1, 1]``.

    This is the preferred helper when the analysis pipeline exposes canonical
    ``trend_score_*`` variables independently of the underlying trend method.
    """
    out = build_rgb_from_tau(ds, time_dim=time_dim, vars=vars)
    out.attrs["method"] = "unit_score"
    out.attrs["source_vars"] = ",".join(vars)
    out.attrs["mapping"] = "natural: trend_score in [-1,1] -> RGB in [0,1]"
    return out


def generate_rgb_hex(
    k: int,
    save: bool = False,
    r_file: str | Path = "rw_hex_test_d.csv",
    g_file: str | Path = "gw_hex_test_d.csv",
    b_file: str | Path = "bw_hex_test_d.csv",
    n_file: str | Path = "nw_hex_test_d.csv",
) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array]:
    m = k * 2 + 1
    Q = m * 2 + 1
    N = 4 * m + 4  # Q + 2*(m+1)

    # grid initialization
    rgb_grid = np.full((N, N), -999, dtype=int)
    r_hex = np.full((N, N), -300.0 / 256.0, dtype=np.float32)
    g_hex = np.full((N, N), -300.0 / 256.0, dtype=np.float32)
    b_hex = np.full((N, N), -999.0 / 256.0, dtype=np.float32)
    num_hex = np.full((N, N), -999.0 / 256.0, dtype=np.float32)
    
    p = 1
    c = (N - 1) // 2  # Python index adjustment
    rgb_grid[c, c] = 0
    r_hex[c, c] = 1.0
    g_hex[c, c] = 1.0
    b_hex[c, c] = 1.0
    num_hex[c, c] = 0.0

    # File output (stub - replace with actual saving as needed)
    print(f"grid size : {N}")
    print(f"center pos: {m + 1}")
    print("RGB pos for Green:")
    print(f"R: {m + 1}, {N - k}")
    print(f"G: {k + 1}, {m + 1}")
    print(f"B: {N - k}, {k + 1}")
    print("CYM pos:")
    print(f"C: {m + 1}, {k + 1}")
    print(f"Y: {k + 1}, {N - k}")
    print(f"M: {N - k}, {m + 1}")
    print("----")
    print(f"basic RGB-W grid num: {k}")
    print(f"one-cycle grid num  : {(k + 1) * 6}")
    print(f"max-cycle grid num  : {(2 * k + 1) * 6}")
    print(f"changing ratio      : {255.0 / (k + 1)}")
    print(f"min changing ratio  : {255.0 / (2 * k + 1)}")
    print("----")
    print("sample grid cycle")
    print("R->Y->G->C->B->M->...")

    
    # main loop of r_hex
    for t in range(1, 2 * m + 2):
        for i in range(1, 7):
            for j in range(1, t + 1):
                def set_cell(y: int, x: int, value: float) -> None:
                    nonlocal p
                    rgb_grid[y, x] = p
                    p += 1
                    r_hex[y, x] = value

                if i == 1:
                    y = c - (j - 1)
                    x = c + t
                    if t <= k + 1:
                        value = 1.0
                    elif t <= m:
                        value = float(m - (t - 1)) / (m - k)
                    elif t == m + 1:
                        value = 0.0
                    elif j == 1:
                        if t <= m + k + 1:
                            value = float(t - (m + 1)) / ((k + 1) * (Q - t + 1))
                        else:
                            value = float(t - (m + k + 1)) / (k + 1)
                    elif 1 < j <= 2 * m + 3 - t - 1:
                        tt = float(t - (m + 1)) / (k + 1)
                        value = tt if tt <= 1.0 else 1.0
                    else:
                        value = -999.0 / 256.0
                    set_cell(y, x, value)

                elif i == 2:
                    y = c - t
                    x = c + t - (j - 1)
                    if t <= k + 1:
                        value = float(k + 1 - (j - 1)) / (k + 1)
                    elif t < m + 1:
                        value = (float(m - (t - 1)) / (m - k)) * float(t - (j - 1)) / t
                    elif t == m + 1:
                        value = 0.0
                    else:
                        if j >= 2 * (t - m):
                            if j == 2 * (t - m) - 1:
                                tt = float(t - (m + 1)) / (k + 1)
                                value = tt if tt <= 1.0 else 1.0
                            else:
                                if Q - t > k:
                                    value = float(t - (j - 1)) * float(k + 1 - (m + 2 + k - t)) / ((k + 1) * (2 * m + 2 - t))
                                else:
                                    value = float(k + 1 - (j - 2 * (t - m) + 1)) / (k + 1)
                        else:
                            value = -999.0 / 256.0
                    set_cell(y, x, value)

                elif i == 3:
                    y = c - t + (j - 1)
                    x = c - (j - 1)
                    if t < k + 1:
                        value = float(k + 1 - t) / (m - k)
                    elif t < m + 1:
                        value = float(k + 2 - (t - k)) / ((k + 1) * (t + 1))
                    elif t == m + 1 or j == 1:
                        value = 0.0
                    elif t - m + 1 <= j <= m + 1:
                        value = 0.0
                    else:
                        value = -999.0 / 256.0
                    set_cell(y, x, value)

                elif i == 4:
                    y = c + (j - 1)
                    x = c - t
                    if t < k + 1:
                        value = float(k + 1 - t) / (m - k)
                    elif t < m + 1:
                        value = float(k + 2 - (t - k)) / ((k + 1) * (t + 1))
                    elif t == m + 1:
                        value = 0.0
                    elif 1 < j <= 2 * (m + 1) - t:
                        value = 0.0
                    else:
                        value = -999.0 / 256.0
                    set_cell(y, x, value)

                elif i == 5:
                    y = c + t
                    x = c - t + (j - 1)
                    if t <= k:
                        value = float(k + 1 - t + (j - 1)) / (k + 1)
                    elif t < m + 1:
                        if j == 1:
                            value = float(j) * float(k + 1 - (t - k)) / ((k + 1) * (t + 1))
                        else:
                            value = float(2 * k + 2 - t) * float(j - 1) / ((k + 1) * t)
                    elif t == m + 1 or j == 1:
                        value = 0.0
                    elif j >= 2 * (t - m):
                        if j == 2 * (t - m) - 1:
                            if Q - t >= k:
                                value = float(t - (m + 2)) / ((k + 1) * (m - (t - (m + 3))))
                            else:
                                value = float(t - (m + 2) - k) / (k + 1)
                        else:
                            jt = j - 2 * (t - m) - 1
                            if Q - t >= k:
                                value = float(jt + 2) * float(t - (m + 1)) / ((k + 1) * (m - (t - (m + 2))))
                            else:
                                value = float(jt + k - (Q - t) + 2) / (k + 1)
                    else:
                        value = -999.0 / 256.0
                    set_cell(y, x, value)

                elif i == 6:
                    y = c + t - (j - 1)
                    x = c + (j - 1)
                    if t <= k + 1:
                        value = 1.0
                    elif t <= m + 1:
                        value = float(m - (t - 1)) / (m - k)
                    elif t - m + 1 <= j <= m + 1:
                        tt = -1.0 * float(m - (t - 1)) / (m - k)
                        value = tt if tt <= 1.0 else 1.0
                    else:
                        value = -999.0 / 256.0
                    set_cell(y, x, value)

    # Save to CSV
    if save:
        with open(r_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(N):
                writer.writerow(r_hex[:, i].T)
    r_hex = r_hex.T
    
    # main loop of g_hex
    for t in range(1, 2 * m + 2):
        for i in range(1, 7):
            for j in range(1, t + 1):
                #if j == 1:
                #    print("R-Y-G-C-B-M")
                #    print("RGB loop:", t)
                
                # 座標変換（Fortran → Python）
                def set_val(ix: int, iy: int, val: float) -> None:
                    rgb_grid[iy, ix] = p
                    g_hex[iy, ix] = val

                if i == 1:  # R->Y
                    ix, iy = c - (j - 1), c + t
                    if t <= k:
                        val = (k + 1 - t + (j - 1)) / (k + 1)
                    elif t <= m:
                        if j == 1:
                            val = (k + 2 - (t - k)) / ((k + 1) * (t + 1))
                        else:
                            val = ((m - (t - 1)) / (m - k)) * ((j - 1) / t)
                    elif t == m + 1 or j == 1:
                        val = 0.0
                    elif j <= 2 * m + 3 - t - 1:
                        if Q - t > k:
                            val = (j - 1) * (k + 1 - (m + 2 + k - t)) / ((k + 1) * (2 * m + 2 - t))
                        else:
                            val = (k - (Q - t) + j - 1) / (k + 1)
                    else:
                        val = -999.0 / 256.0
                    set_val(ix, iy, val)

                elif i == 2:  # Y->G
                    ix, iy = c - t, c + t - (j - 1)
                    if t <= k + 1:
                        val = 1.0
                    elif t < m + 1:
                        val = (m - (t - 1)) / (m - k)
                    elif t == m + 1:
                        val = 0.0
                    elif j >= 2 * (t - m):
                        tt = (t - (m + 1)) / (k + 1)
                        val = min(tt, 1.0)
                    else:
                        val = -999.0 / 256.0
                    set_val(ix, iy, val)

                elif i == 3:  # G->C
                    ix, iy = c - t + (j - 1), c - (j - 1)
                    if t <= k + 1:
                        val = 1.0
                    elif t < m + 1:
                        val = (m - (t - 1)) / (m - k)
                    elif t == m + 1:
                        val = 0.0
                    elif j == 1:
                        if t <= m + k + 1:
                            val = (t - (m + 1)) / ((k + 1) * (Q - t + 1))
                        else:
                            val = (t - (m + k + 1)) / (k + 1)
                    elif (t - m + 1) <= j <= m + 1:
                        tt = -1.0 * (m - (t - 1)) / (m - k)
                        val = min(tt, 1.0)
                    else:
                        val = -999.0 / 256.0
                    set_val(ix, iy, val)

                elif i == 4:  # C->B
                    ix, iy = c + (j - 1), c - t
                    if t <= k + 1:
                        val = (k + 1 - (j - 1)) / (k + 1)
                    elif t < m + 1:
                        val = ((m - (t - 1)) / (m - k)) * ((t - (j - 1)) / t)
                    elif t == m + 1:
                        val = 0.0
                    elif 1 < j <= 2 * (m + 1) - t:
                        if Q - t >= k:
                            jt = 2 * (m + 1) + 1 - t - j
                            val = (jt * (t - (m + 1))) / ((k + 1) * (m - (t - (m + 2)))) if jt > 0 else \
                                  ((t - 1 - (m + 1)) / ((k + 1) * (m - (t - 1 - (m + 2)))))
                        else:
                            val = (k + 1 - (j - 1)) / (k + 1)
                    else:
                        val = -999.0 / 256.0
                    set_val(ix, iy, val)

                elif i == 5:  # B->M
                    ix, iy = c + t, c - t + (j - 1)
                    if t < k + 1:
                        val = (k + 1 - t) / (m - k)
                    elif t < m + 1:
                        if j == 1:
                            val = (k + 1 - (t - k)) / ((k + 1) * (t + 1))
                        else:
                            val = (k + 2 - (t - k)) / ((k + 1) * (t + 1)) #5.0 #(k + 1 - (t - k)) / ((k + 1) * (t + 1))
                    elif t == m + 1:
                        val = 0.0
                    elif j == 1:
                        val = 0.0
                    else:
                        if j >= 2 * (t - m):
                            if j == 2 * (t - m) - 1:
                                if Q - t >= k:
                                    val = 0.0  # (t-(m+2)) / ((k+1)*(m-(t-(m+3))))
                                else:
                                    val = 0.0  # (t-(m+2)-k)/(k+1)
                            else:
                                jt = j - 2 * (t - m) - 1
                                if Q - t >= k:
                                    val = 0.0  # (jt+2)*(t-(m+1)) / ((k+1)*(m-(t-(m+2))))
                                else:
                                    val = 0.0  # (jt+k-(Q-t)+2)/(k+1)
                        else:
                            val = -999.0 / 256.0
                    set_val(ix, iy, val)

                elif i == 6:  # M->R
                    ix, iy = c + t - (j - 1), c + (j - 1)
                    if t < k + 1:
                        val = (k + 1 - t) / (m - k)
                    elif t < m + 1:
                        val = (k + 2 - (t - k)) / ((k + 1) * (t + 1))
                    elif t == m + 1:
                        val = 0.0
                    elif t > m + 1:
                        if t - m + 1 <= j <= m + 1:
                            tt = -1.0 * (m - (t - 1)) / (m - k)
                            if tt <= 1.0:
                                val = 0.0  # tt
                            else:
                                val = 0.0  # 1.0
                        else:
                            val = -999.0 / 256.0
                    set_val(ix, iy, val)
                p += 1
    
    
    # Save to CSV
    g_hex = g_hex.T
    if  save:
        with open(g_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(N):
                writer.writerow(g_hex[:, i])
    g_hex = g_hex.T                 
    
    # main loop of b_hex
    for t in range(1, 2 * m + 2):
        for i in range(1, 7):
            for j in range(1, t + 1):
                if i == 1:
                    x, y = c - (j - 1), c + t
                elif i == 2:
                    x, y = c - t, c + t - (j - 1)
                elif i == 3:
                    x, y = c - t + (j - 1), c - (j - 1)
                elif i == 4:
                    x, y = c + (j - 1), c - t
                elif i == 5:
                    x, y = c + t, c - t + (j - 1)
                elif i == 6:
                    x, y = c + t - (j - 1), c + (j - 1)
                else:
                    continue
    
                if 0 <= x < N and 0 <= y < N:
                    rgb_grid[x, y] = p
                    p += 1
                    val = -999.0 / 256.0
    
                    if i == 1:
                        if t < k + 1:
                            val = float(k + 1 - t) / (m - k)
                        elif t <= m:
                            val = float(k + 2 - (t - k)) / ((k + 1) * (t + 1))
                        elif t == m + 1 or j == 1:
                            val = 0.0
                        elif 1 < j <= 2 * m + 3 - t - 1:
                            tt = float(t - (m + 1)) / (k + 1)
                            val = 0.0 if tt <= 1.0 else 0.0
                    elif i == 2:
                        if t < k + 1:
                            val = float(k + 1 - t) / (m - k)
                        elif t < m + 1:
                            val = float(k + 2 - (t - k)) / ((k + 1) * (t + 1))
                        elif t == m + 1:
                            val = 0.0
                        elif j >= 2 * (t - m):
                            if j == 2 * (t - m) - 1:
                                tt = float(t - (m + 1)) / (k + 1)
                                val = 0.0 if tt <= 1.0 else 0.0
                            else:
                                val = 0.0
                    elif i == 3:
                        if t <= k:
                            val = float(k + 1 - t + (j - 1)) / (k + 1)
                        elif t < m + 1:
                            if j == 1:
                                val = float(m - t) / (m - k) / (t + 1)
                            else:
                                val = (float(m - (t - 1)) / (m - k)) * float(j - 1) / t
                        elif t == m + 1 or j == 1:
                            val = 0.0
                        elif t - m + 1 <= j <= m + 1:
                            jt = j - (t - m + 1) + 2
                            if Q - t >= k:
                                val = (float(m - (2 * m - t + 1)) / (m - k)) * float(jt - 1) / (2 * m + 2 - t)
                            else:
                                val = float(k - (Q - t) + jt - 1) / (k + 1)
                    elif i == 4:
                        if t <= k + 1:
                            val = 1.0
                        elif t < m + 1:
                            val = float(m - (t - 1)) / (m - k)
                        elif t == m + 1:
                            val = 0.0
                        elif 1 < j <= 2 * (m + 1) - t:
                            tt = float(t - (m + 1)) / (k + 1)
                            val = tt if tt <= 1.0 else 1.0
                    elif i == 5:
                        if t <= k + 1:
                            val = 1.0
                        elif t < m + 1:
                            val = float(m - (t - 1)) / (m - k)
                        elif t == m + 1:
                            val = 0.0
                        elif j == 1:
                            if t <= m + k + 1:
                                val = float(t - (m + 1)) / ((k + 1) * (Q - t + 1))
                            else:
                                val = float(t - (m + k + 1)) / (k + 1)
                        elif j >= 2 * (t - m):
                            tt = float(t - (m + 1)) / (k + 1)
                            val = tt if tt <= 1.0 else 1.0
                    elif i == 6:
                        if t <= k + 1:
                            val = float(k + 1 - (j - 1)) / (k + 1)
                        elif t <= m + 1:
                            val = (float(m - (t - 1)) / (m - k)) * float(t - (j - 1)) / t
                        elif t - m + 1 <= j <= m + 1:
                            if Q - t >= k:
                                tj = 2 * m + 2 - t
                                val = float(t - (m + 1)) * float(t - j - (t - (m + 2))) / ((k + 1) * tj)
                            else:
                                val = float(t + k - m - (j - 1)) / (k + 1)
    
                    b_hex[x, y] = val
    
    # Save to CSV
    if save:
        with open(b_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(N):
                writer.writerow(b_hex[:, i])
    b_hex = b_hex.T
            
    # main loop of num
    for t in range(1, 2 * m + 2):
        for i in range(1, 7):
            for j in range(1, t + 1):
                if i == 1:  # R->Y
                    y, x = c - (j - 1), c + t
                    rgb_grid[y, x] = p
                    p += 1
                    if t <= k:
                        num_hex[y, x] = 1.0/256.0
                    elif t <= m:
                        if j == 1:
                            num_hex[y, x] = 1.0/256.0
                        else:
                            num_hex[y, x] = 1.0/256.0
                    elif t == m + 1:
                        num_hex[y, x] = 0.0/256.0
                    elif j == 1:
                        num_hex[y, x] = 13.0/256.0
                    elif 1 < j <= 2 * m + 3 - t - 1:
                        num_hex[y, x] = 7.0/256.0
    
                elif i == 2:  # Y->G
                    y, x = c - t, c + t - (j - 1)
                    rgb_grid[y, x] = p
                    p += 1
                    if t <= k + 1:
                        num_hex[y, x] = 2.0/256.0
                    elif t < m + 1:
                        num_hex[y, x] = 2.0/256.0
                    elif t == m + 1:
                        num_hex[y, x] = 0.0/256.0
                    elif j >= 2 * (t - m):
                        tt = (t - (m + 1)) / (k + 1)
                        num_hex[y, x] = 8.0/256.0
    
                elif i == 3:  # G->C
                    y, x = c - t + (j - 1), c - (j - 1)
                    rgb_grid[y, x] = p
                    p += 1
                    if t <= k + 1:
                        num_hex[y, x] = 3.0/256.0
                    elif t < m + 1:
                        num_hex[y, x] = 3.0/256.0
                    elif t == m + 1:
                        num_hex[y, x] = 0.0
                    elif j == 1:
                        if t <= m + k + 1:
                            num_hex[y, x] = 14.0/256.0
                        else:
                            num_hex[y, x] = 14.0/256.0
                    elif t - m + 1 <= j <= m + 1:
                        tt = -1.0 * (m - (t - 1)) / (m - k)
                        num_hex[y, x] = 9.0/256.0
    
                elif i == 4:  # C->B
                    y, x = c + (j - 1), c - t
                    rgb_grid[y, x] = p
                    p += 1
                    if t <= k + 1:
                        num_hex[y, x] = 4.0/256.0
                    elif t < m + 1:
                        num_hex[y, x] = 4.0/256.0
                    elif t == m + 1:
                        num_hex[y, x] = 0.0
                    elif 1 < j <= 2 * (m + 1) - t:
                        if Q - t >= k:
                            jt = 2 * (m + 1) + 1 - t - j
                            if jt > 0:
                                num_hex[y, x] = 10.0/256.0
                            else:
                                num_hex[y, x] = 10.0/256.0
                        else:
                            jt = 2 * (m + 1) + 1 - t - j
                            num_hex[y, x] = 10.0/256.0
    
                elif i == 5:  # B->M
                    y, x = c + t, c - t + (j - 1)
                    rgb_grid[y, x] = p
                    p += 1
                    if t < k + 1:
                        num_hex[y, x] = 5.0/256.0
                    elif t < m + 1:
                        num_hex[y, x] = 5.0/256.0
                    elif t == m + 1 or j == 1:
                        num_hex[y, x] = 0.0
                    elif j >= 2 * (t - m):
                        num_hex[y, x] = 11.0/256.0
                    
                    if j == 1:
                        if t > m + 1:
                            num_hex[y, x] = 15.0/256.0
    
    
                elif i == 6:  # M->R
                    y, x = c + t - (j - 1), c + (j - 1)
                    rgb_grid[y, x] = p
                    p += 1
                    if t < k + 1:
                        num_hex[y, x] = 6.0/256.0
                    elif t < m + 1:
                        num_hex[y, x] = 6.0/256.0
                    elif t == m + 1:
                        num_hex[y, x] = 0.0
                        
                    elif t - m + 1 <= j <= m + 1:
                        if Q - t >= k:
                            tj = 2 * m + 2 - t
                            num_hex[y, x] = 12.0/256.0
                        else:
                            num_hex[y, x] = 12.0/256.0                  
                    
    # Save to CSV
    if save:
        with open(n_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(N):
                writer.writerow(num_hex[:, i])
    num_hex = num_hex.T

    print("rgb_hex shape:",g_hex.shape)
    print("----")
    print("output_filename-> r_file:",r_file,", g_file:",g_file,", b_file:",b_file,"n_file",n_file)
    
    return r_hex, g_hex, b_hex, num_hex


def get_hexagram_assets(k: int, valid_threshold: float = -0.5) -> HexagramAssets:
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


def map_rgb_to_hexagram(
    rgb: xr.Dataset,
    *,
    hex_assets: HexagramAssets,
    time_dim: str = "time",
) -> xr.Dataset:
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


def _component_mask(values: FloatArray, sign: int, tol_center: float) -> BoolArray:
    """
    Devuelve máscara booleana para una componente RGB según el signo:
      -1 -> baja     : value < 0.5 - tol_center
       0 -> central  : |value - 0.5| <= tol_center
      +1 -> alta     : value > 0.5 + tol_center
    """
    if sign == -1:
        return values < (0.5 - tol_center)
    if sign == 0:
        return np.abs(values - 0.5) <= tol_center
    if sign == +1:
        return values > (0.5 + tol_center)

    raise ValueError(f"sign debe ser -1, 0 o +1; recibido: {sign}")


def get_process_hexagram_mask(
    process: str,
    *,
    k: int,
    tol_center: float = 0.05,
    valid_threshold: float = -0.5,
) -> tuple[BoolArray, HexagramAssets]:
    """
    Devuelve la máscara 2D del hexagrama correspondiente a un proceso.

    Esta versión admite que PROCESS_SIGNATURES[process] sea:
      - una única firma:        (-1, +1, 0)
      - varias firmas válidas: [(-1, -1, -1), (-1, -1, 0), (-1, 0, -1)]

    Parameters
    ----------
    process : str
        Nombre del proceso ('breakup', 'growth_depletion', 'evaporation',
        'growth', 'activation', ...).
    k : int
        Parámetro del hexagrama.
    tol_center : float, optional
        Tolerancia alrededor de 0.5 para la banda central.
    valid_threshold : float, optional
        Umbral que se pasa a get_hexagram_assets.

    Returns
    -------
    mask2d : np.ndarray
        Máscara booleana 2D con True en las celdas del proceso.
    hex_assets : dict
        Diccionario devuelto por get_hexagram_assets.
    """
    if process not in PROCESS_SIGNATURES:
        valid = ", ".join(PROCESS_SIGNATURES.keys())
        raise ValueError(f"Proceso desconocido: {process!r}. Válidos: {valid}")

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k debe ser un entero positivo.")

    hex_assets = get_hexagram_assets(k=k, valid_threshold=valid_threshold)

    rgb_cells = np.asarray(hex_assets["rgb_cells"], float)
    yx_cells = np.asarray(hex_assets["yx_cells"], int)
    img = np.asarray(hex_assets["img"], float)

    if rgb_cells.ndim != 2 or rgb_cells.shape[1] != 3:
        raise ValueError(
            f"hex_assets['rgb_cells'] debe tener shape (n, 3), no {rgb_cells.shape}"
        )
    if yx_cells.ndim != 2 or yx_cells.shape[1] != 2:
        raise ValueError(
            f"hex_assets['yx_cells'] debe tener shape (n, 2), no {yx_cells.shape}"
        )

    R = rgb_cells[:, 0]
    G = rgb_cells[:, 1]
    B = rgb_cells[:, 2]

    signatures = PROCESS_SIGNATURES[process]
    sig_def = signatures

    # Normalizar a lista de firmas [(sR,sG,sB), ...]
    if isinstance(sig_def, tuple) and len(sig_def) == 3:
        signatures = [tuple(int(v) for v in sig_def)]
    elif isinstance(sig_def, (list, tuple)):
        signatures = []
        for item in sig_def:
            if not (isinstance(item, (list, tuple)) and len(item) == 3):
                raise ValueError(
                    f"Firma inválida para proceso {process!r}: {item!r}. "
                    "Cada firma debe ser una tupla/lista de 3 enteros."
                )
            signatures.append((int(item[0]), int(item[1]), int(item[2])))
    else:
        raise ValueError(
            f"PROCESS_SIGNATURES[{process!r}] no tiene un formato válido: {sig_def!r}"
        )

    mask_total = np.zeros_like(R, dtype=bool)

    for sR, sG, sB in signatures:
        m = (
            _component_mask(R, sR, tol_center)
            & _component_mask(G, sG, tol_center)
            & _component_mask(B, sB, tol_center)
        )
        mask_total |= m

    mask2d = np.zeros(img.shape[:2], dtype=bool)
    if np.any(mask_total):
        ys = yx_cells[mask_total, 0]
        xs = yx_cells[mask_total, 1]
        mask2d[ys, xs] = True

    return mask2d, hex_assets

def plot_process_to_hexagram(
    process: str,
    *,
    k: int,
    tol_center: float = 0.05,
    valid_threshold: float = -0.5,
    figsize: tuple[float, float] = (8, 8),
    show_background: bool = True,
    alpha_hexagram: float = 0.20,
    alpha_process: float = 0.95,
    process_color: tuple[float, float, float] | None = None,
    show_cell_centers: bool = False,
    crop_to_process: bool = False,
    crop_margin_cells: int = 6,
    title: str | None = None,
    title_fs: float = 16,
    label_fs: float = 13,
    tick_fs: float = 11,
    savefig: bool = False,
    output_dir: str | Path | None = None,
    dpi: int = 200,
) -> tuple[plt.Figure, plt.Axes, Path | None]:
    """
    Pinta el espacio del hexagrama correspondiente a un proceso microfísico.

    Parameters
    ----------
    process : str
        Nombre del proceso.
    k : int
        Parámetro del hexagrama.
    tol_center : float, optional
        Tolerancia de la banda central en RGB.
    valid_threshold : float, optional
        Umbral pasado a get_hexagram_assets.
    figsize : tuple, optional
        Tamaño de figura.
    show_background : bool, optional
        Si True, muestra el hexagrama RGB completo de fondo.
    alpha_hexagram : float, optional
        Transparencia del hexagrama de fondo.
    alpha_process : float, optional
        Transparencia del overlay del proceso.
    process_color : tuple or None, optional
        Color fijo RGB para pintar el proceso. Si es None, se usan los colores
        reales de cada celda del hexagrama.
    show_cell_centers : bool, optional
        Si True, superpone marcadores en los centros de las celdas seleccionadas.
    title : str or None, optional
        Título personalizado.
    savefig : bool, optional
        Si True, guarda la figura.
    output_dir : str | Path | None, optional
        Carpeta de salida si savefig=True.
    dpi : int, optional
        Resolución de guardado.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura creada.
    ax : matplotlib.axes.Axes
        Axes created for the hexagram.
    filepath : pathlib.Path | None
        Path to the saved figure if savefig=True, otherwise None.
    """
    mask2d, hex_assets = get_process_hexagram_mask(
        process,
        k=k,
        tol_center=tol_center,
        valid_threshold=valid_threshold,
    )
    
    img = np.asarray(hex_assets["img"], float)
    rgb_cells = np.asarray(hex_assets["rgb_cells"], float)
    yx_cells = np.asarray(hex_assets["yx_cells"], int)

    ny, nx = img.shape[:2]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if show_background:
        ax.imshow(
            img,
            origin="lower",
            interpolation="nearest",
            alpha=alpha_hexagram,
        )

    # Overlay RGBA del proceso
    overlay = np.zeros((ny, nx, 4), dtype=float)

    ys, xs = np.where(mask2d)

    if len(xs) > 0:
        if process_color is None:
            # usar color real de la LUT para cada celda
            lut = {tuple(yx): rgb for yx, rgb in zip(yx_cells, rgb_cells)}
            for y, x in zip(ys, xs):
                overlay[y, x, :3] = lut[(y, x)]
                overlay[y, x, 3] = alpha_process
        else:
            overlay[ys, xs, 0] = process_color[0]
            overlay[ys, xs, 1] = process_color[1]
            overlay[ys, xs, 2] = process_color[2]
            overlay[ys, xs, 3] = alpha_process

    ax.imshow(
        overlay,
        origin="lower",
        interpolation="nearest",
    )

    if show_cell_centers and len(xs) > 0:
        ax.scatter(xs, ys, s=8, c="k", alpha=0.6)

    if title is None:
        title = f"{process.capitalize()} | k={k} | tol={tol_center:.3f}"
    ax.set_title(title, fontsize=title_fs, pad=8)

    ax.set_xlabel("hex_x", fontsize=label_fs)
    ax.set_ylabel("hex_y", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    if crop_to_process and len(xs) > 0:
        margin = max(int(crop_margin_cells), 0)
        ax.set_xlim(max(xs.min() - margin - 0.5, -0.5), min(xs.max() + margin + 0.5, nx - 0.5))
        ax.set_ylim(max(ys.min() - margin - 0.5, -0.5), min(ys.max() + margin + 0.5, ny - 0.5))
    else:
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)
    ax.set_aspect("equal")
    ax.grid(False)

    filepath = None
    if savefig:
        output_dir = Path.cwd() if output_dir is None else Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_process = process.replace(" ", "_")
        safe_tol = str(tol_center).replace(".", "p")
        filepath = output_dir / f"hexagram_process_{safe_process}_k{k}_tol{safe_tol}.png"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    return fig, ax, filepath

