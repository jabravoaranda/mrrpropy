import numpy as np
import xarray as xr

from mrrpropy.analysis.process_features import get_spectral_features


def test_get_spectral_features_fixed_layer_simple_two_bin():
    ds = xr.Dataset(
        coords={
            "time": np.array(["2025-10-29T19:23:00"], dtype="datetime64[s]"),
            "range": np.array([0.0, 200.0], dtype=float),
            "velocity": np.array([-2.0, 0.0], dtype=float),
        }
    )
    spectrum = np.zeros((1, 2, 2), dtype=float)
    # top (200 m): all power at 0 m/s
    spectrum[0, 1, :] = np.array([0.0, 1.0])
    # bottom (0 m): all power at -2 m/s (downward motion)
    spectrum[0, 0, :] = np.array([1.0, 0.0])
    ds["spectrum"] = xr.DataArray(spectrum, dims=("time", "range", "velocity"))

    z_top = xr.DataArray(200.0)
    z_bottom = xr.DataArray(0.0)
    z_center = xr.DataArray(100.0)

    features = get_spectral_features(
        ds,
        mode="fixed_layer",
        z_top=z_top,
        z_bottom=z_bottom,
        z_center=z_center,
        spectrum_var="spectrum",
        velocity_coord="velocity",
    )

    assert dict(features.sizes) == {"time": 1}
    assert float(features["v_mean_top"].values[0]) == 0.0
    assert float(features["v_mean_bottom"].values[0]) == -2.0
    assert float(features["delta_v_mean"].values[0]) == -2.0

    assert float(features["v_std_top"].values[0]) == 0.0
    assert float(features["v_std_bottom"].values[0]) == 0.0
    assert float(features["delta_v_std"].values[0]) == 0.0

    assert float(features["v_p50_top"].values[0]) == 0.0
    assert float(features["v_p50_bottom"].values[0]) == -2.0


def test_get_spectral_features_scan_uses_z_center_as_layer_coord():
    ds = xr.Dataset(
        coords={
            "time": np.array(["2025-10-29T19:23:00"], dtype="datetime64[s]"),
            "range": np.array([0.0, 100.0, 200.0, 300.0], dtype=float),
            "velocity": np.array([-2.0, 0.0], dtype=float),
        }
    )
    spectrum = np.zeros((1, 4, 2), dtype=float)
    # layer 50 m: top (100 m) -> 0 m/s, bottom (0 m) -> -2 m/s
    spectrum[0, 1, :] = np.array([0.0, 1.0])
    spectrum[0, 0, :] = np.array([1.0, 0.0])
    # layer 250 m: top (300 m) -> 0 m/s, bottom (200 m) -> -2 m/s
    spectrum[0, 3, :] = np.array([0.0, 1.0])
    spectrum[0, 2, :] = np.array([1.0, 0.0])
    ds["spectrum"] = xr.DataArray(spectrum, dims=("time", "range", "velocity"))

    z_top = xr.DataArray(np.array([100.0, 300.0], dtype=float), dims=("layer",))
    z_bottom = xr.DataArray(np.array([0.0, 200.0], dtype=float), dims=("layer",))
    z_center = xr.DataArray(np.array([50.0, 250.0], dtype=float), dims=("layer",))

    features = get_spectral_features(
        ds,
        mode="scan",
        z_top=z_top,
        z_bottom=z_bottom,
        z_center=z_center,
        spectrum_var="spectrum",
        velocity_coord="velocity",
    )

    assert dict(features.sizes) == {"time": 1, "layer": 2}
    assert np.allclose(features["layer"].values, np.array([50.0, 250.0], dtype=float))
    assert float(features["delta_v_mean"].values[0, 0]) == -2.0
    assert float(features["delta_v_mean"].values[0, 1]) == -2.0
    assert float(features["v_p50_top"].values[0, 0]) == 0.0
    assert float(features["v_p50_bottom"].values[0, 0]) == -2.0


