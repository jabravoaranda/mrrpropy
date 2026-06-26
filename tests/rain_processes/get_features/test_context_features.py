import numpy as np
import xarray as xr

from mrrpropy.analysis.process_features import get_context


def test_get_context_fixed_layer_simple():
    ds = xr.Dataset(
        coords={
            "time": np.array(["2025-10-29T19:23:00"], dtype="datetime64[s]"),
            "range": np.array([0.0, 100.0, 200.0], dtype=float),
        }
    )
    ds["RR"] = xr.DataArray(np.array([[1.0, 2.0, 3.0]]), dims=("time", "range"))

    z_top = xr.DataArray(200.0)
    z_bottom = xr.DataArray(0.0)
    z_center = xr.DataArray(100.0)

    features = get_context(
        ds,
        mode="fixed_layer",
        z_top=z_top,
        z_bottom=z_bottom,
        z_center=z_center,
        bb_bottom_m=50.0,
        bb_peak_m=100.0,
        bb_top_m=150.0,
        RR_var="RR",
    )

    assert dict(features.sizes) == {"time": 1}
    assert float(features["RR_mean"].values[0]) == 2.0
    assert bool(features["overlaps_bb"].values[0]) is True
    assert float(features["dist_bb_peak"].values[0]) == 0.0


def test_get_context_scan_uses_z_center_as_layer_coord():
    ds = xr.Dataset(
        coords={
            "time": np.array(["2025-10-29T19:23:00"], dtype="datetime64[s]"),
            "range": np.array([0.0, 100.0, 200.0, 300.0], dtype=float),
        }
    )
    ds["RR"] = xr.DataArray(np.array([[1.0, 2.0, 3.0, 4.0]]), dims=("time", "range"))

    z_top = xr.DataArray(np.array([100.0, 300.0], dtype=float), dims=("layer",))
    z_bottom = xr.DataArray(np.array([0.0, 200.0], dtype=float), dims=("layer",))
    z_center = xr.DataArray(np.array([50.0, 250.0], dtype=float), dims=("layer",))

    features = get_context(
        ds,
        mode="scan",
        z_top=z_top,
        z_bottom=z_bottom,
        z_center=z_center,
        bb_bottom_m=50.0,
        bb_peak_m=100.0,
        bb_top_m=150.0,
        RR_var="RR",
    )

    assert dict(features.sizes) == {"time": 1, "layer": 2}
    assert np.allclose(features["layer"].values, np.array([50.0, 250.0], dtype=float))
    assert float(features["RR_mean"].values[0, 0]) == 1.5
    assert float(features["RR_mean"].values[0, 1]) == 3.5
    assert bool(features["overlaps_bb"].values[0, 0]) is True
    assert bool(features["overlaps_bb"].values[0, 1]) is False
    assert float(features["dist_bb_peak"].values[0, 0]) == -50.0
    assert float(features["dist_bb_peak"].values[0, 1]) == 150.0


