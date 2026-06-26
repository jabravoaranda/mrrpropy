from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mrrpropy.analysis import processes as process_analysis

matplotlib.use("Agg")

pytestmark = [pytest.mark.integration]


class _SyntheticProcessedMRR:
    def __init__(self, ds: xr.Dataset):
        self.raprompro = ds

    def _is_processed(self):
        return True


def test_compute_layer_trend_ols(raprompro_subset_10min_loaded_mrr):
    result = raprompro_subset_10min_loaded_mrr.compute_layer_trend_ols(
        z_bottom_m=1000.0,
        z_top_m=2000.0,
        variable_threshold="Ze",
        threshold_value=-5.0,
        eps_mode="hourly_quantile",
        q=0.1,
        time_dim="time",
    )

    assert result.attrs["trend_method"] == "ols_legacy"
    for variable_name in ("Dm", "Nw", "LWC"):
        for prefix in ("b", "a", "r2", "F", "eps", "n_fit", "mask_fit"):
            assert f"{prefix}_{variable_name}" in result


def test_compute_layer_trend_kendall_theilsen(raprompro_subset_10min_loaded_mrr):
    result = raprompro_subset_10min_loaded_mrr.compute_layer_trend(
        z_bottom_m=1000.0,
        z_top_m=2000.0,
        variable_threshold="Ze",
        threshold_value=-5.0,
        time_dim="time",
        tau_zero_tol=0.05,
        min_points_trend=6,
    )

    assert result.attrs["trend_method"] == "kendall_theilsen"
    assert result.attrs["min_points_trend"] == 6
    assert "n_valid" in result

    n_time = raprompro_subset_10min_loaded_mrr.ds.sizes["time"]
    for variable_name in ("Dm", "Nw", "LWC"):
        for field_name in (
            f"tau_{variable_name}",
            f"p_{variable_name}",
            f"ts_{variable_name}",
            f"intercept_ts_{variable_name}",
            f"sign_{variable_name}",
            f"strength_{variable_name}",
            f"trend_mag_{variable_name}",
            f"trend_sign_{variable_name}",
            f"trend_strength_{variable_name}",
            f"trend_score_{variable_name}",
            f"trend_p_{variable_name}",
            f"n_fit_{variable_name}",
            f"mask_fit_{variable_name}",
            f"b_{variable_name}",
        ):
            assert field_name in result
        assert result[f"tau_{variable_name}"].shape == (n_time,)
        assert result[f"ts_{variable_name}"].shape == (n_time,)
        assert result[f"sign_{variable_name}"].dtype.kind in ("i", "u")


def test_compute_layer_trend_uses_descending_rain_evolution():
    ds = xr.Dataset(
        coords={
            "time": np.array(["2025-10-29T19:23:00"], dtype="datetime64[s]"),
            "range": np.array([1000.0, 1500.0, 2000.0], dtype=float),
        }
    )
    ds["Ze"] = xr.DataArray(np.full((1, 3), 10.0), dims=("time", "range"))
    ds["Dm"] = xr.DataArray(np.array([[2.0, 2.0, 2.0]]), dims=("time", "range"))
    ds["Nw"] = xr.DataArray(np.array([[3.0, 2.0, 1.0]]), dims=("time", "range"))
    ds["LWC"] = xr.DataArray(np.array([[6.0, 4.0, 2.0]]), dims=("time", "range"))

    subject = _SyntheticProcessedMRR(ds)
    result = process_analysis.compute_layer_trend(
        subject,
        z_bottom_m=1000.0,
        z_top_m=2000.0,
        min_points_trend=3,
    )

    assert result.attrs["trend_direction"] == (
        "positive means increase while descending from z_top_m to z_bottom_m"
    )
    assert result.attrs["z_bottom_m"] == pytest.approx(1000.0)
    assert result.attrs["z_top_m"] == pytest.approx(2000.0)
    assert result.attrs["selection_mode"] == "fixed_layer"
    assert result["sign_Dm"].values[0] == 0
    assert result["sign_Nw"].values[0] == 1
    assert result["sign_LWC"].values[0] == 1
    assert result["tau_Nw"].values[0] > 0.9
    assert result["tau_LWC"].values[0] > 0.9


def test_compute_layer_trend_ols_exposes_canonical_trend_fields(
    raprompro_subset_10min_loaded_mrr,
):
    result = raprompro_subset_10min_loaded_mrr.compute_layer_trend(
        z_bottom_m=1000.0,
        z_top_m=2000.0,
        trend_method="ols",
        variable_threshold="Ze",
        threshold_value=-5.0,
        time_dim="time",
        min_points_trend=6,
    )

    assert result.attrs["trend_method"] == "ols"
    for variable_name in ("Dm", "Nw", "LWC"):
        for field_name in (
            f"b_{variable_name}",
            f"r2_{variable_name}",
            f"trend_mag_{variable_name}",
            f"trend_sign_{variable_name}",
            f"trend_strength_{variable_name}",
            f"trend_score_{variable_name}",
            f"trend_p_{variable_name}",
        ):
            assert field_name in result


def test_rain_process_analyze_uses_nonparametric_pipeline(
    raprompro_subset_10min_loaded_mrr,
    monkeypatch,
):
    def _raise_if_called(*args, **kwargs):
        raise AssertionError(
            "OLS helper should not be used by the default analysis pipeline."
        )

    monkeypatch.setattr(process_analysis, "ols_slope_intercept_r2", _raise_if_called)

    analysis = raprompro_subset_10min_loaded_mrr.rain_process_analyze(
        period=(datetime(2025, 10, 29, 19, 23, 0), datetime(2025, 10, 29, 19, 33, 0)),
        k=11,
        selection_mode="fixed_layer",
        z_bottom_m=1000.0,
        z_top_m=2000.0,
        ze_th=-5.0,
        min_points_trend=6,
        tau_zero_tol=0.05,
        eps_q=0.01,
        rgb_q=0.02,
        vars_trend=("Dm", "Nw", "LWC"),
    )

    assert isinstance(analysis, xr.Dataset)
    assert "time" in analysis.coords
    assert analysis.sizes["time"] > 0

    for field_name in ("R", "G", "B", "minutes", "hex_x", "hex_y"):
        assert field_name in analysis

    for variable_name in ("Dm", "Nw", "LWC"):
        for prefix in (
            "tau",
            "p",
            "ts",
            "sign",
            "strength",
            "trend_mag",
            "trend_sign",
            "trend_strength",
            "trend_score",
            "trend_p",
        ):
            assert f"{prefix}_{variable_name}" in analysis
        assert f"b_{variable_name}" in analysis

    assert analysis.attrs["trend_method"] == "kendall_theilsen"
    assert analysis.attrs["rgb_method"] == "trend_score"
    assert analysis.attrs["min_points_trend"] == 6
    assert analysis.attrs["vars_trend"] == ("Dm", "Nw", "LWC")
    assert analysis.attrs["z_bottom_m"] == pytest.approx(1000.0)
    assert analysis.attrs["z_top_m"] == pytest.approx(2000.0)
    assert analysis.attrs["selection_mode"] == "fixed_layer"

    rgb_mapping = analysis.attrs.get("rgb_mapping", None)
    assert rgb_mapping == {"R": "Dm", "G": "Nw", "B": "LWC"}

    finite_r = analysis["R"].values[np.isfinite(analysis["R"].values)]
    if finite_r.size:
        assert np.nanmin(finite_r) >= 0.0
        assert np.nanmax(finite_r) <= 1.0

    classification = raprompro_subset_10min_loaded_mrr.classify_rain_process(
        analysis=analysis,
        min_tau_strength=0.10,
    )

    assert isinstance(classification, xr.Dataset)
    for field_name in (
        "proc_label",
        "sign_R",
        "sign_G",
        "sign_B",
        "strength",
        "tau_Dm",
        "tau_Nw",
        "tau_LWC",
        "trend_sign_Dm",
        "trend_strength_Dm",
        "trend_score_Dm",
    ):
        assert field_name in classification
    assert classification.attrs["classification_basis"] == "canonical_trend_sign"


def test_build_process_dynamics_dataframe(raprompro_subset_10min_loaded_mrr):
    analysis = raprompro_subset_10min_loaded_mrr.rain_process_analyze(
        period=(datetime(2025, 10, 29, 19, 23, 0), datetime(2025, 10, 29, 19, 33, 0)),
        k=11,
        selection_mode="fixed_layer",
        z_bottom_m=1000.0,
        z_top_m=2000.0,
        min_points_trend=6,
    )
    classified = raprompro_subset_10min_loaded_mrr.classify_rain_process(
        analysis=analysis,
        min_tau_strength=0.10,
    )

    df = raprompro_subset_10min_loaded_mrr.build_process_dynamics_dataframe(
        analysis=analysis,
        classified=classified,
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "proc_label" in df.columns
    assert "proc_strength" in df.columns
    for variable_name in ("Dm", "Nw", "LWC"):
        for field_name in (
            f"{variable_name}_top",
            f"{variable_name}_bottom",
            f"{variable_name}_delta",
            f"{variable_name}_delta_pct",
            f"{variable_name}_rate_per_km",
            f"trend_strength_{variable_name}",
            f"trend_score_{variable_name}",
        ):
            assert field_name in df.columns
    assert df.attrs["trend_direction"] == (
        "positive means increase while descending from z_top_m to z_bottom_m"
    )
    assert "z_bottom_m" in df.columns
    assert "z_top_m" in df.columns
    assert df.attrs["selection_mode"] == "fixed_layer"


def test_summarize_process_dynamics(raprompro_subset_10min_loaded_mrr):
    analysis = raprompro_subset_10min_loaded_mrr.rain_process_analyze(
        period=(datetime(2025, 10, 29, 19, 23, 0), datetime(2025, 10, 29, 19, 33, 0)),
        k=11,
        selection_mode="fixed_layer",
        z_bottom_m=1000.0,
        z_top_m=2000.0,
        min_points_trend=6,
    )
    classified = raprompro_subset_10min_loaded_mrr.classify_rain_process(
        analysis=analysis,
        min_tau_strength=0.10,
    )

    summary = raprompro_subset_10min_loaded_mrr.summarize_process_dynamics(
        analysis=analysis,
        classified=classified,
    )

    assert isinstance(summary, pd.DataFrame)
    assert not summary.empty
    assert "proc_label" in summary.columns
    assert "n_samples" in summary.columns
    assert "fraction" in summary.columns
    assert "LWC_delta_pct_median" in summary.columns
    assert summary["n_samples"].sum() == len(classified["time"])


def test_build_column_process_scan_dataframe(raprompro_subset_10min_loaded_mrr):
    scan_df = raprompro_subset_10min_loaded_mrr.build_column_process_scan_dataframe(
        period=(datetime(2025, 10, 29, 19, 23, 0), datetime(2025, 10, 29, 19, 33, 0)),
        k=11,
        window_thickness_m=1000.0,
        window_step_m=100.0,
        min_tau_strength=0.10,
    )

    assert isinstance(scan_df, pd.DataFrame)
    assert not scan_df.empty
    for field_name in (
        "time",
        "window_id",
        "z_min_m",
        "z_max_m",
        "z_bottom_m",
        "z_top_m",
        "z_center_m",
        "proc_label",
        "proc_strength",
        "Dm_delta_pct",
        "Nw_delta_pct",
        "LWC_delta_pct",
    ):
        assert field_name in scan_df.columns
    assert scan_df.attrs["window_thickness_m"] == pytest.approx(1000.0)
    assert scan_df.attrs["window_step_m"] == pytest.approx(100.0)
    assert scan_df.attrs["selection_mode"] == "scan"
    assert scan_df["window_id"].nunique() >= 1


def test_public_rain_process_analyze_defaults_to_scan(
    raprompro_subset_10min_loaded_mrr,
):
    scan_df = raprompro_subset_10min_loaded_mrr.rain_process_analyze(
        period=(datetime(2025, 10, 29, 19, 23, 0), datetime(2025, 10, 29, 19, 33, 0)),
        k=11,
        window_thickness_m=1000.0,
        window_step_m=100.0,
        min_tau_strength=0.10,
    )

    assert isinstance(scan_df, pd.DataFrame)
    assert not scan_df.empty
    assert scan_df.attrs["selection_mode"] == "scan"


def test_legacy_layer_argument_still_works_with_warning(
    raprompro_subset_10min_loaded_mrr,
):
    with pytest.warns(FutureWarning):
        analysis = raprompro_subset_10min_loaded_mrr.rain_process_analyze(
            period=(
                datetime(2025, 10, 29, 19, 23, 0),
                datetime(2025, 10, 29, 19, 33, 0),
            ),
            k=11,
            layer=(1000.0, 2000.0),
        )

    assert isinstance(analysis, xr.Dataset)
    assert analysis.attrs["z_bottom_m"] == pytest.approx(1000.0)
    assert analysis.attrs["z_top_m"] == pytest.approx(2000.0)
    assert analysis.attrs["selection_mode"] == "fixed_layer"


def test_detect_column_process_episodes_from_scan():
    times = pd.date_range("2025-10-29T19:23:00", periods=10, freq="10s")
    scan_df = pd.DataFrame(
        {
            "time": times,
            "window_id": [0] * 10,
            "z_min_m": [1000.0] * 10,
            "z_max_m": [2000.0] * 10,
            "z_center_m": [1500.0] * 10,
            "window_thickness_m": [1000.0] * 10,
            "window_step_m": [100.0] * 10,
            "trend_method": ["kendall_theilsen"] * 10,
            "proc_label": [
                "evaporation",
                "evaporation",
                "evaporation",
                "evaporation",
                "evaporation",
                "evaporation",
                "unknown",
                "unknown",
                "activation",
                "activation",
            ],
            "proc_strength": [0.4, 0.45, 0.5, 0.55, 0.5, 0.6, 0.2, 0.2, 0.7, 0.8],
            "Dm_delta_pct": [-10, -11, -9, -12, -10, -8, 0, 0, 4, 5],
            "Nw_delta_pct": [-5, -4, -6, -5, -5, -4, 0, 0, 7, 8],
            "LWC_delta_pct": [-30, -32, -28, -35, -31, -29, 0, 0, 6, 7],
            "Dm_rate_per_km": [-0.1] * 10,
            "Nw_rate_per_km": [-0.2] * 10,
            "LWC_rate_per_km": [-0.3] * 10,
            "tau_Dm": [-0.8] * 10,
            "tau_Nw": [-0.7] * 10,
            "tau_LWC": [-0.9] * 10,
            "trend_strength_Dm": [0.4] * 10,
            "trend_strength_Nw": [0.5] * 10,
            "trend_strength_LWC": [0.6] * 10,
            "trend_score_Dm": [-0.4] * 10,
            "trend_score_Nw": [-0.5] * 10,
            "trend_score_LWC": [-0.6] * 10,
        }
    )

    episodes = process_analysis.detect_column_process_episodes(
        None,
        scan_df=scan_df,
        min_consecutive_profiles=6,
    )

    assert isinstance(episodes, pd.DataFrame)
    assert len(episodes) == 1
    assert episodes.loc[0, "proc_label"] == "evaporation"
    assert episodes.loc[0, "n_profiles"] == 6
    assert episodes.loc[0, "duration_seconds"] == pytest.approx(60.0)


