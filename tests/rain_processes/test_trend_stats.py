import numpy as np
import pytest
import xarray as xr

from mrrpropy.analysis import processes as process_analysis
from mrrpropy.utils import compute_monotonic_trend


def test_compute_monotonic_trend_strictly_increasing():
    x = np.arange(6, dtype=float)
    y = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    trend = compute_monotonic_trend(x, y, tau_zero_tol=0.05)

    assert trend["tau"] > 0.9
    assert trend["slope_ts"] > 0.0
    assert trend["sign_tau"] == 1
    assert trend["strength_tau"] > 0.9


def test_compute_monotonic_trend_strictly_decreasing():
    x = np.arange(6, dtype=float)
    y = np.array([6, 5, 4, 3, 2, 1], dtype=float)

    trend = compute_monotonic_trend(x, y, tau_zero_tol=0.05)

    assert trend["tau"] < -0.9
    assert trend["slope_ts"] < 0.0
    assert trend["sign_tau"] == -1
    assert trend["strength_tau"] > 0.9


def test_compute_monotonic_trend_constant_profile():
    x = np.arange(6, dtype=float)
    y = np.full(6, 3.0)

    trend = compute_monotonic_trend(x, y, tau_zero_tol=0.05)

    assert trend["tau"] == pytest.approx(0.0)
    assert trend["sign_tau"] == 0
    assert trend["strength_tau"] == pytest.approx(0.0)
    assert trend["slope_ts"] == pytest.approx(0.0)


def test_compute_monotonic_trend_nonlinear_but_monotonic():
    x = np.arange(6, dtype=float)
    y = np.exp(x / 3.0)

    trend = compute_monotonic_trend(x, y, tau_zero_tol=0.05)

    assert trend["tau"] > 0.9
    assert trend["slope_ts"] > 0.0
    assert trend["sign_tau"] == 1


def test_compute_monotonic_trend_oscillatory_profile():
    x = np.arange(6, dtype=float)
    y = np.array([1.0, 3.0, 2.0, 4.0, 2.5, 3.0])

    trend = compute_monotonic_trend(x, y, tau_zero_tol=0.05)

    assert abs(trend["tau"]) < 0.6
    assert trend["sign_tau"] in (-1, 0, 1)


def test_compute_monotonic_trend_handles_nans():
    x = np.arange(6, dtype=float)
    y = np.array([1.0, np.nan, 2.0, np.nan, 3.0, 4.0])

    trend = compute_monotonic_trend(x, y, tau_zero_tol=0.05, min_points=3)

    assert trend["n_valid"] == 4
    assert trend["tau"] > 0.9
    assert trend["sign_tau"] == 1


def test_classify_rain_process_uses_tau_signatures():
    analysis = xr.Dataset(
        coords={"time": np.array([0, 1], dtype=int)},
        attrs={
            "rgb_mapping": {"R": "Dm", "G": "Nw", "B": "LWC"},
            "tau_zero_tol": 0.25,
            "trend_method": "kendall_theilsen",
        },
    )
    analysis["R"] = xr.DataArray(np.array([0.15, 0.85]), dims=("time",))
    analysis["G"] = xr.DataArray(np.array([0.95, 0.15]), dims=("time",))
    analysis["B"] = xr.DataArray(np.array([0.60, 0.40]), dims=("time",))

    analysis["tau_Dm"] = xr.DataArray(np.array([-0.7, 0.7]), dims=("time",))
    analysis["tau_Nw"] = xr.DataArray(np.array([0.8, -0.8]), dims=("time",))
    analysis["tau_LWC"] = xr.DataArray(np.array([0.2, -0.2]), dims=("time",))
    analysis["p_Dm"] = xr.DataArray(np.array([0.01, 0.01]), dims=("time",))
    analysis["p_Nw"] = xr.DataArray(np.array([0.01, 0.01]), dims=("time",))
    analysis["p_LWC"] = xr.DataArray(np.array([0.01, 0.01]), dims=("time",))
    analysis["ts_Dm"] = xr.DataArray(np.array([-0.3, 0.3]), dims=("time",))
    analysis["ts_Nw"] = xr.DataArray(np.array([0.4, -0.4]), dims=("time",))
    analysis["ts_LWC"] = xr.DataArray(np.array([0.1, -0.1]), dims=("time",))
    analysis["intercept_ts_Dm"] = xr.DataArray(np.array([1.0, 1.0]), dims=("time",))
    analysis["intercept_ts_Nw"] = xr.DataArray(np.array([1.0, 1.0]), dims=("time",))
    analysis["intercept_ts_LWC"] = xr.DataArray(np.array([1.0, 1.0]), dims=("time",))
    analysis["sign_Dm"] = xr.DataArray(np.array([-1, 1]), dims=("time",))
    analysis["sign_Nw"] = xr.DataArray(np.array([1, -1]), dims=("time",))
    analysis["sign_LWC"] = xr.DataArray(np.array([0, 0]), dims=("time",))
    analysis["strength_Dm"] = xr.DataArray(np.array([0.7, 0.7]), dims=("time",))
    analysis["strength_Nw"] = xr.DataArray(np.array([0.8, 0.8]), dims=("time",))
    analysis["strength_LWC"] = xr.DataArray(np.array([0.2, 0.2]), dims=("time",))
    analysis["trend_mag_Dm"] = xr.DataArray(np.array([-0.3, 0.3]), dims=("time",))
    analysis["trend_mag_Nw"] = xr.DataArray(np.array([0.4, -0.4]), dims=("time",))
    analysis["trend_mag_LWC"] = xr.DataArray(np.array([0.1, -0.1]), dims=("time",))
    analysis["trend_sign_Dm"] = xr.DataArray(np.array([-1, 1]), dims=("time",))
    analysis["trend_sign_Nw"] = xr.DataArray(np.array([1, -1]), dims=("time",))
    analysis["trend_sign_LWC"] = xr.DataArray(np.array([0, 0]), dims=("time",))
    analysis["trend_strength_Dm"] = xr.DataArray(np.array([0.7, 0.7]), dims=("time",))
    analysis["trend_strength_Nw"] = xr.DataArray(np.array([0.8, 0.8]), dims=("time",))
    analysis["trend_strength_LWC"] = xr.DataArray(np.array([0.2, 0.2]), dims=("time",))
    analysis["trend_score_Dm"] = xr.DataArray(np.array([-0.7, 0.7]), dims=("time",))
    analysis["trend_score_Nw"] = xr.DataArray(np.array([0.8, -0.8]), dims=("time",))
    analysis["trend_score_LWC"] = xr.DataArray(np.array([0.0, 0.0]), dims=("time",))
    analysis["trend_p_Dm"] = xr.DataArray(np.array([0.01, 0.01]), dims=("time",))
    analysis["trend_p_Nw"] = xr.DataArray(np.array([0.01, 0.01]), dims=("time",))
    analysis["trend_p_LWC"] = xr.DataArray(np.array([0.01, 0.01]), dims=("time",))

    classified = process_analysis.classify_rain_process(
        None,
        analysis=analysis,
        min_tau_strength=0.1,
    )

    assert list(classified["proc_label"].values) == ["breakup", "growth_depletion"]
    assert classified.attrs["classification_basis"] == "canonical_trend_sign"


def test_classify_rain_process_recognizes_growth_depletion_gain_and_loss():
    analysis = xr.Dataset(
        coords={"time": np.array([0, 1], dtype=int)},
        attrs={
            "rgb_mapping": {"R": "Dm", "G": "Nw", "B": "LWC"},
            "tau_zero_tol": 0.25,
            "trend_method": "kendall_theilsen",
        },
    )
    analysis["R"] = xr.DataArray(np.array([0.85, 0.85]), dims=("time",))
    analysis["G"] = xr.DataArray(np.array([0.15, 0.15]), dims=("time",))
    analysis["B"] = xr.DataArray(np.array([0.15, 0.85]), dims=("time",))

    for name, values in {
        "trend_sign_Dm": np.array([1, 1]),
        "trend_sign_Nw": np.array([-1, -1]),
        "trend_sign_LWC": np.array([-1, 1]),
        "trend_strength_Dm": np.array([0.8, 0.8]),
        "trend_strength_Nw": np.array([0.8, 0.8]),
        "trend_strength_LWC": np.array([0.8, 0.8]),
        "trend_score_Dm": np.array([0.8, 0.8]),
        "trend_score_Nw": np.array([-0.8, -0.8]),
        "trend_score_LWC": np.array([-0.8, 0.8]),
        "trend_p_Dm": np.array([0.01, 0.01]),
        "trend_p_Nw": np.array([0.01, 0.01]),
        "trend_p_LWC": np.array([0.01, 0.01]),
    }.items():
        analysis[name] = xr.DataArray(values, dims=("time",))

    classified = process_analysis.classify_rain_process(
        None,
        analysis=analysis,
        min_tau_strength=0.1,
    )

    assert list(classified["proc_label"].values) == [
        "growth_depletion_loss",
        "growth_depletion_gain",
    ]


def test_classify_rain_process_filters_weak_tau():
    analysis = xr.Dataset(
        coords={"time": np.array([0], dtype=int)},
        attrs={
            "rgb_mapping": {"R": "Dm", "G": "Nw", "B": "LWC"},
            "tau_zero_tol": 0.25,
            "trend_method": "kendall_theilsen",
        },
    )
    analysis["R"] = xr.DataArray(np.array([0.55]), dims=("time",))
    analysis["G"] = xr.DataArray(np.array([0.55]), dims=("time",))
    analysis["B"] = xr.DataArray(np.array([0.55]), dims=("time",))
    analysis["tau_Dm"] = xr.DataArray(np.array([0.04]), dims=("time",))
    analysis["tau_Nw"] = xr.DataArray(np.array([0.06]), dims=("time",))
    analysis["tau_LWC"] = xr.DataArray(np.array([0.03]), dims=("time",))
    analysis["p_Dm"] = xr.DataArray(np.array([0.4]), dims=("time",))
    analysis["p_Nw"] = xr.DataArray(np.array([0.4]), dims=("time",))
    analysis["p_LWC"] = xr.DataArray(np.array([0.4]), dims=("time",))
    analysis["ts_Dm"] = xr.DataArray(np.array([0.01]), dims=("time",))
    analysis["ts_Nw"] = xr.DataArray(np.array([0.01]), dims=("time",))
    analysis["ts_LWC"] = xr.DataArray(np.array([0.01]), dims=("time",))
    analysis["intercept_ts_Dm"] = xr.DataArray(np.array([1.0]), dims=("time",))
    analysis["intercept_ts_Nw"] = xr.DataArray(np.array([1.0]), dims=("time",))
    analysis["intercept_ts_LWC"] = xr.DataArray(np.array([1.0]), dims=("time",))
    analysis["sign_Dm"] = xr.DataArray(np.array([0]), dims=("time",))
    analysis["sign_Nw"] = xr.DataArray(np.array([0]), dims=("time",))
    analysis["sign_LWC"] = xr.DataArray(np.array([0]), dims=("time",))
    analysis["strength_Dm"] = xr.DataArray(np.array([0.04]), dims=("time",))
    analysis["strength_Nw"] = xr.DataArray(np.array([0.06]), dims=("time",))
    analysis["strength_LWC"] = xr.DataArray(np.array([0.03]), dims=("time",))
    analysis["trend_mag_Dm"] = xr.DataArray(np.array([0.01]), dims=("time",))
    analysis["trend_mag_Nw"] = xr.DataArray(np.array([0.01]), dims=("time",))
    analysis["trend_mag_LWC"] = xr.DataArray(np.array([0.01]), dims=("time",))
    analysis["trend_sign_Dm"] = xr.DataArray(np.array([0]), dims=("time",))
    analysis["trend_sign_Nw"] = xr.DataArray(np.array([0]), dims=("time",))
    analysis["trend_sign_LWC"] = xr.DataArray(np.array([0]), dims=("time",))
    analysis["trend_strength_Dm"] = xr.DataArray(np.array([0.04]), dims=("time",))
    analysis["trend_strength_Nw"] = xr.DataArray(np.array([0.06]), dims=("time",))
    analysis["trend_strength_LWC"] = xr.DataArray(np.array([0.03]), dims=("time",))
    analysis["trend_score_Dm"] = xr.DataArray(np.array([0.0]), dims=("time",))
    analysis["trend_score_Nw"] = xr.DataArray(np.array([0.0]), dims=("time",))
    analysis["trend_score_LWC"] = xr.DataArray(np.array([0.0]), dims=("time",))
    analysis["trend_p_Dm"] = xr.DataArray(np.array([0.4]), dims=("time",))
    analysis["trend_p_Nw"] = xr.DataArray(np.array([0.4]), dims=("time",))
    analysis["trend_p_LWC"] = xr.DataArray(np.array([0.4]), dims=("time",))

    classified = process_analysis.classify_rain_process(
        None,
        analysis=analysis,
        min_tau_strength=0.1,
        max_tau_pvalue=0.05,
    )

    assert classified["proc_label"].values[0] == "steady_or_weak"


