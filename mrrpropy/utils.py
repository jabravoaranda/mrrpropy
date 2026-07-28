"""Small numerical helpers used by the public analysis API."""

from datetime import datetime
from typing import TypedDict

import numpy as np
from scipy.stats import kendalltau, theilslopes


class MonotonicTrend(TypedDict):
    """Summary of a monotonic vertical trend based on Kendall and Theil-Sen."""

    tau: float
    p_value: float
    slope_ts: float
    intercept_ts: float
    n_valid: int
    sign_tau: int
    strength_tau: float


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


def compute_monotonic_trend(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tau_zero_tol: float = 0.05,
    min_points: int = 2,
) -> MonotonicTrend:
    """
    Compute a robust monotonic trend summary from paired samples.

    Parameters
    ----------
    x, y:
        Paired coordinates. Non-finite values are ignored.
    tau_zero_tol:
        Dead band around zero used to convert Kendall's tau into ``-1 / 0 / +1``.
    min_points:
        Minimum number of valid paired samples required to estimate the trend.

    Returns
    -------
    MonotonicTrend
        Dictionary-like summary with:

        - ``tau``: Kendall's tau, describing monotonic direction and consistency.
        - ``p_value``: p-value associated with tau.
        - ``slope_ts``: Theil-Sen slope.
        - ``intercept_ts``: Theil-Sen intercept.
        - ``n_valid``: number of valid paired samples.
        - ``sign_tau``: monotonic sign in ``{-1, 0, +1}``.
        - ``strength_tau``: ``abs(tau)`` in ``[0, 1]``.

    Notes
    -----
    Constant profiles are treated as ``tau = 0`` and ``p_value = 1`` so they
    behave as weak/steady profiles instead of propagating ``NaN`` signs.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    n_valid = int(np.sum(mask))

    out: MonotonicTrend = {
        "tau": np.nan,
        "p_value": np.nan,
        "slope_ts": np.nan,
        "intercept_ts": np.nan,
        "n_valid": n_valid,
        "sign_tau": 0,
        "strength_tau": np.nan,
    }

    if n_valid < max(2, int(min_points)):
        return out

    x_valid = x_arr[mask]
    y_valid = y_arr[mask]

    order = np.argsort(x_valid)
    x_valid = x_valid[order]
    y_valid = y_valid[order]

    tau_result = kendalltau(x_valid, y_valid)
    tau = float(tau_result.statistic)
    p_value = float(tau_result.pvalue)

    if not np.isfinite(tau):
        if np.allclose(y_valid, y_valid[0], equal_nan=False):
            tau = 0.0
            p_value = 1.0
        else:
            tau = np.nan
            p_value = np.nan

    try:
        slope_result = theilslopes(y_valid, x_valid)
        slope_ts = float(slope_result.slope)
        intercept_ts = float(slope_result.intercept)
    except ValueError:
        slope_ts = np.nan
        intercept_ts = np.nan

    strength_tau = float(np.clip(abs(tau), 0.0, 1.0)) if np.isfinite(tau) else np.nan

    sign_tau = 0
    if np.isfinite(tau):
        if tau > tau_zero_tol:
            sign_tau = 1
        elif tau < -tau_zero_tol:
            sign_tau = -1

    out.update(
        {
            "tau": tau,
            "p_value": p_value,
            "slope_ts": slope_ts,
            "intercept_ts": intercept_ts,
            "sign_tau": sign_tau,
            "strength_tau": strength_tau,
        }
    )
    return out


def sign_from_center(c: np.ndarray, tol: float = 0.05) -> np.ndarray:
    # c en [0,1], centro 0.5
    s = np.zeros_like(c, dtype=int)
    s[c > 0.5 + tol] = +1
    s[c < 0.5 - tol] = -1
    return s


def strength(c: np.ndarray) -> np.ndarray:
    # 0..1
    return np.clip(np.abs(c - 0.5) / 0.5, 0.0, 1.0)
