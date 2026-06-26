import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mrrpropy.plotting import _spectra

matplotlib.use("Agg")

pytestmark = [pytest.mark.slow, pytest.mark.plot, pytest.mark.integration]


def test_transmittance_correction_increases_reflectivity_when_pia_applies(
    raprompro_subset_10min_loaded_mrr, artifact_dir
):
    pytest.importorskip("matplotlib")

    ds = raprompro_subset_10min_loaded_mrr.raprompro
    assert ds is not None

    ze = ds["Ze"].values.astype(float)
    zea = ds["Zea"].values.astype(float)
    dbpia = ds["DBPIA"].values.astype(float)
    hydrometeor_type = ds["Type"].values.astype(float)

    liquid_mask = np.isin(hydrometeor_type, [5.0, 10.0])
    corrected_mask = (
        liquid_mask
        & np.isfinite(ze)
        & np.isfinite(zea)
        & np.isfinite(dbpia)
        & (dbpia < 0.0)
    )

    assert (
        corrected_mask.any()
    ), "No liquid points with attenuation correction were found."

    delta = ze - zea
    assert np.nanmin(delta[corrected_mask]) >= -1e-6
    np.testing.assert_allclose(
        delta[corrected_mask],
        -dbpia[corrected_mask],
        atol=1e-6,
        rtol=1e-6,
    )

    x = (-dbpia[corrected_mask]).ravel()
    y = delta[corrected_mask].ravel()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=6, alpha=0.6)
    lo = float(np.nanmin(np.concatenate([x, y])))
    hi = float(np.nanmax(np.concatenate([x, y])))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("-DBPIA [dB]")
    ax.set_ylabel("Ze - Zea [dB]")
    ax.set_title("PIA correction consistency for liquid hydrometeors")
    fig.tight_layout()
    fig.savefig(artifact_dir / "transmittance_correction_consistency.png", dpi=150)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)

    correction = ds["Ze"] - ds["Zea"]
    correction_plot = correction.where(liquid_mask)

    fig, ax = plt.subplots(figsize=(12, 6))
    correction_plot.plot(
        ax=ax,
        x="time",
        y="range",
        cmap="viridis",
        vmin=0.0,
        robust=True,
        cbar_kwargs={"label": "Ze - Zea [dB]"},
    )
    ax.set_title("Transmittance correction quicklook (liquid hydrometeors only)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Range [m]")
    fig.tight_layout()
    fig.savefig(artifact_dir / "transmittance_correction_quicklook.png", dpi=150)
    plt.close(fig)


def test_plot_microphysical_properties_profiles_runs(
    raprompro_subset_10min_loaded_mrr, artifact_dir
):
    pytest.importorskip("matplotlib")

    (
        fig,
        axs,
        filepath,
    ) = raprompro_subset_10min_loaded_mrr.plot_microphysical_properties_profiles(
        target_datetime=datetime.datetime(2025, 10, 29, 19, 28, 0),
        savefig=True,
        output_dir=artifact_dir,
        dpi=120,
    )

    assert isinstance(fig, Figure)
    assert isinstance(axs, np.ndarray)
    assert all(isinstance(ax, Axes) for ax in axs.flatten())
    assert filepath is not None
    assert filepath.exists()
    plt.close(fig)


def test_plot_dealiased_spectrogram(raprompro_subset_10min_loaded_mrr, artifact_dir):
    pytest.importorskip("matplotlib")

    fig, _, filepath = raprompro_subset_10min_loaded_mrr.plot_spectrogram(
        target_datetime=datetime.datetime(2025, 10, 29, 19, 28, 0),
        spectrum_var="spe_3D",
        savefig=True,
        output_dir=artifact_dir,
        dpi=120,
    )

    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    plt.close(fig)


def test_dealiased_spectrogram_converts_legacy_positive_downward_speed():
    class Subject:
        path = "synthetic.nc"
        ds = xr.Dataset(
            coords={
                "time": np.array(["2025-10-29T19:28:00"], dtype="datetime64[s]"),
                "range": np.array([0.0, 100.0], dtype=float),
            }
        )

    legacy_speed = np.array([-1.0, 0.0, 1.0, 2.0], dtype=float)
    spe = np.array(
        [
            [
                [10.0, 20.0, 30.0, 40.0],
                [50.0, 60.0, 70.0, 80.0],
            ]
        ],
        dtype=float,
    )
    Subject.raprompro = xr.Dataset(
        {"spe_3D": (("time", "range", "speed"), spe)},
        coords={
            "time": Subject.ds["time"].values,
            "range": Subject.ds["range"].values,
            "speed": legacy_speed,
        },
    )

    _, _, vel, spec2d, _ = _spectra.get_spectrogram_2d(
        Subject,
        np.datetime64("2025-10-29T19:28:00"),
        spectrum_var="spe_3D",
    )

    assert np.allclose(vel, np.array([-2.0, -1.0, 0.0, 1.0]))
    assert np.allclose(spec2d[0], np.array([40.0, 30.0, 20.0, 10.0]))


def test_plot_dsdgram(raprompro_subset_10min_loaded_mrr, artifact_dir):
    pytest.importorskip("matplotlib")

    fig, _, filepath = raprompro_subset_10min_loaded_mrr.plot_DSDgram(
        target_datetime=datetime.datetime(2025, 10, 29, 19, 28, 0),
        savefig=True,
        output_dir=artifact_dir,
        dpi=120,
    )

    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    plt.close(fig)


def test_plot_dsd_by_range(raprompro_subset_10min_loaded_mrr, artifact_dir):
    fig, _, filepath = raprompro_subset_10min_loaded_mrr.plot_DSD_by_range(
        target_datetime=datetime.datetime(2025, 10, 29, 19, 28, 0),
        ranges=np.arange(500, 2500, 250),
        savefig=True,
        output_dir=artifact_dir,
        dpi=120,
    )

    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    plt.close(fig)


