import numpy as np
import xarray as xr

from mrrpropy.analysis.processes import (
    classify_process_from_features,
    classify_rain_process,
)
from mrrpropy.analysis.process_features import (
    build_process_features,
    get_microphysical_features,
)


def test_get_microphysical_features_fixed_layer_simple_profile():
    ds = xr.Dataset(
        coords={
            "time": np.array(["2025-10-29T19:23:00"], dtype="datetime64[s]"),
            "range": np.array([0.0, 100.0, 200.0], dtype=float),
        }
    )
    ds["Dm"] = xr.DataArray(np.array([[3.0, 2.0, 1.0]]), dims=("time", "range"))
    ds["Nw"] = xr.DataArray(np.array([[30.0, 20.0, 10.0]]), dims=("time", "range"))
    ds["LWC"] = xr.DataArray(np.array([[0.3, 0.2, 0.1]]), dims=("time", "range"))

    z_top = xr.DataArray(200.0)
    z_bottom = xr.DataArray(0.0)
    z_center = xr.DataArray(100.0)

    features = get_microphysical_features(
        ds,
        mode="fixed_layer",
        z_top=z_top,
        z_bottom=z_bottom,
        z_center=z_center,
    )

    assert dict(features.sizes) == {"time": 1}
    assert float(features["Dm_top"].values[0]) == 1.0
    assert float(features["Dm_bottom"].values[0]) == 3.0
    assert float(features["delta_Dm"].values[0]) == 2.0
    assert float(features["rel_change_Dm"].values[0]) == 200.0

    assert float(features["tau_Dm"].values[0]) == 1.0
    assert int(features["trend_sign_Dm"].values[0]) == 1
    assert float(features["trend_strength_Dm"].values[0]) == 1.0

    assert str(features["micro_signature_str"].values[0]) == "+,+,+"


def test_get_microphysical_features_scan_uses_z_center_as_layer_coord():
    ds = xr.Dataset(
        coords={
            "time": np.array(["2025-10-29T19:23:00"], dtype="datetime64[s]"),
            "range": np.array([0.0, 100.0, 200.0, 300.0], dtype=float),
        }
    )
    ds["Dm"] = xr.DataArray(np.array([[4.0, 3.0, 2.0, 1.0]]), dims=("time", "range"))
    ds["Nw"] = xr.DataArray(
        np.array([[40.0, 30.0, 20.0, 10.0]]), dims=("time", "range")
    )
    ds["LWC"] = xr.DataArray(np.array([[0.4, 0.3, 0.2, 0.1]]), dims=("time", "range"))

    z_top = xr.DataArray(np.array([100.0, 300.0], dtype=float), dims=("layer",))
    z_bottom = xr.DataArray(np.array([0.0, 200.0], dtype=float), dims=("layer",))
    z_center = xr.DataArray(np.array([50.0, 250.0], dtype=float), dims=("layer",))

    features = get_microphysical_features(
        ds,
        mode="scan",
        z_top=z_top,
        z_bottom=z_bottom,
        z_center=z_center,
    )

    assert dict(features.sizes) == {"time": 1, "layer": 2}
    assert np.allclose(features["layer"].values, np.array([50.0, 250.0], dtype=float))
    assert float(features["delta_Dm"].values[0, 0]) == 1.0
    assert float(features["delta_Dm"].values[0, 1]) == 1.0
    assert str(features["micro_signature_str"].values[0, 0]) == "+,+,+"
    assert str(features["micro_signature_str"].values[0, 1]) == "+,+,+"


def _synthetic_full_ds():
    ds = xr.Dataset(
        coords={
            "time": np.array(["2025-10-29T19:23:00"], dtype="datetime64[s]"),
            "range": np.array([0.0, 100.0, 200.0, 300.0], dtype=float),
            "velocity": np.array([-2.0, 0.0], dtype=float),
        }
    )
    ds["Dm"] = xr.DataArray(np.array([[4.0, 3.0, 2.0, 1.0]]), dims=("time", "range"))
    ds["Nw"] = xr.DataArray(
        np.array([[40.0, 30.0, 20.0, 10.0]]), dims=("time", "range")
    )
    ds["LWC"] = xr.DataArray(np.array([[0.4, 0.3, 0.2, 0.1]]), dims=("time", "range"))
    ds["RR"] = xr.DataArray(np.array([[1.0, 2.0, 3.0, 4.0]]), dims=("time", "range"))

    spectrum = np.zeros((1, 4, 2), dtype=float)
    spectrum[0, 0, :] = np.array([1.0, 0.0])
    spectrum[0, 1, :] = np.array([1.0, 0.0])
    spectrum[0, 2, :] = np.array([0.0, 1.0])
    spectrum[0, 3, :] = np.array([0.0, 1.0])
    ds["spectrum"] = xr.DataArray(spectrum, dims=("time", "range", "velocity"))
    return ds


def test_build_process_features_fixed_layer_integration():
    ds = _synthetic_full_ds()
    pf = build_process_features(
        ds,
        mode="fixed_layer",
        fixed_layer_top_m=300.0,
        fixed_layer_bottom_m=0.0,
        bb_bottom_m=10.0,
        bb_peak_m=50.0,
        bb_top_m=90.0,
    )

    assert dict(pf.sizes) == {"time": 1}
    assert "layer" not in pf.dims
    assert "proc_label" not in pf

    for v in (
        "Dm_top",
        "Dm_bottom",
        "delta_Dm",
        "tau_Dm",
        "v_mean_top",
        "v_mean_bottom",
        "delta_v_mean",
        "dist_bb_peak",
        "overlaps_bb",
        "RR_mean",
    ):
        assert v in pf

    assert float((pf["Dm_bottom"] - pf["Dm_top"]).values[0]) == float(
        pf["delta_Dm"].values[0]
    )
    assert float((pf["v_mean_bottom"] - pf["v_mean_top"]).values[0]) == float(
        pf["delta_v_mean"].values[0]
    )
    assert pf["overlaps_bb"].dtype == bool


def test_build_process_features_scan_integration():
    ds = _synthetic_full_ds()
    pf = build_process_features(
        ds,
        mode="scan",
        window_thickness_m=200.0,
        window_step_m=100.0,
        bb_bottom_m=10.0,
        bb_peak_m=50.0,
        bb_top_m=90.0,
    )

    assert dict(pf.sizes) == {"time": 1, "layer": 2}
    assert np.allclose(pf["layer"].values, np.array([100.0, 200.0], dtype=float))
    assert "proc_label" not in pf

    for v in (
        "Dm_top",
        "Dm_bottom",
        "delta_Dm",
        "tau_Dm",
        "v_mean_top",
        "v_mean_bottom",
        "delta_v_mean",
        "dist_bb_peak",
        "overlaps_bb",
        "RR_mean",
    ):
        assert v in pf

    assert np.allclose(
        (pf["Dm_bottom"] - pf["Dm_top"]).values, pf["delta_Dm"].values, equal_nan=True
    )
    assert np.allclose(
        (pf["v_mean_bottom"] - pf["v_mean_top"]).values,
        pf["delta_v_mean"].values,
        equal_nan=True,
    )
    assert pf["overlaps_bb"].dtype == bool


def test_classify_process_from_features_fixed_layer_activation():
    ds = _synthetic_full_ds()
    pf = build_process_features(
        ds,
        mode="fixed_layer",
        fixed_layer_top_m=300.0,
        fixed_layer_bottom_m=0.0,
        bb_bottom_m=10.0,
        bb_peak_m=50.0,
        bb_top_m=90.0,
    )
    classified = classify_process_from_features(pf)

    assert dict(classified.sizes) == {"time": 1}
    assert "layer" not in classified.dims
    assert "proc_label_base" in classified
    assert "proc_label" in classified
    assert "R" not in classified and "G" not in classified and "B" not in classified

    assert str(classified["proc_label_base"].values[0]) == "activation"
    assert str(classified["proc_label"].values[0]) == "activation"
    assert float(classified["strength"].values[0]) == 1.0


def test_classify_process_from_features_scan_activation():
    ds = _synthetic_full_ds()
    pf = build_process_features(
        ds,
        mode="scan",
        window_thickness_m=200.0,
        window_step_m=100.0,
        bb_bottom_m=10.0,
        bb_peak_m=50.0,
        bb_top_m=90.0,
    )
    classified = classify_process_from_features(pf)

    assert dict(classified.sizes) == {"time": 1, "layer": 2}
    labels = classified["proc_label"].values.astype(str).reshape((-1,))
    assert set(np.unique(labels)) == {"activation"}


def test_classify_rain_process_canonical_does_not_require_rgb_mapping():
    analysis = xr.Dataset(
        coords={"time": np.array(["2025-10-29T19:23:00"], dtype="datetime64[s]")}
    )
    for var in ("Dm", "Nw", "LWC"):
        analysis[f"trend_sign_{var}"] = xr.DataArray(
            np.array([+1], dtype=int), dims=("time",)
        )
        analysis[f"trend_strength_{var}"] = xr.DataArray(
            np.array([1.0], dtype=float), dims=("time",)
        )
        analysis[f"trend_p_{var}"] = xr.DataArray(
            np.array([0.01], dtype=float), dims=("time",)
        )

    classified = classify_rain_process(None, analysis=analysis)
    assert str(classified["proc_label"].values[0]) == "activation"


