import datetime

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")

pytestmark = [pytest.mark.slow, pytest.mark.plot]


def test_quickplot_reflectivity_runs(raw_subset_10min_mrr, artifact_dir):
    pytest.importorskip("matplotlib")

    variable = "Ze"
    fig, ax, _ = raw_subset_10min_mrr.quicklook(variable=variable, source="raw")
    fig.savefig(artifact_dir / f"test_quickplot_{variable}.png")

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_spectrum_saves_png(raw_subset_10min_mrr, artifact_dir):
    ds = raw_subset_10min_mrr.ds
    target_time = datetime.datetime(2025, 10, 29, 19, 28, 0)
    target_range = float(ds["range"].values[ds.sizes["range"] // 2])

    fig, _, filepath = raw_subset_10min_mrr.plot_spectrum(
        target_time,
        target_range,
        spectrum_var="spectrum_raw",
        savefig=True,
        output_dir=artifact_dir,
        dpi=120,
    )

    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    assert filepath.suffix.lower() == ".png"
    assert filepath.stat().st_size > 0
    plt.close(fig)


def test_plot_spectra_by_range_saves_png(raw_subset_10min_mrr, artifact_dir):
    ds = raw_subset_10min_mrr.ds
    target_time = datetime.datetime(2025, 10, 29, 19, 28, 0)
    ranges = ds["range"].values[[5, ds.sizes["range"] // 2, -5]].astype(float)

    fig, _, filepath = raw_subset_10min_mrr.plot_spectra_by_range(
        target_time,
        ranges,
        savefig=True,
        output_dir=artifact_dir,
        dpi=120,
    )

    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    assert filepath.suffix.lower() == ".png"
    assert filepath.stat().st_size > 0
    plt.close(fig)


def test_plot_spectrogram_saves_png(raw_subset_10min_mrr, artifact_dir):
    target_time = datetime.datetime(2025, 10, 29, 19, 28, 0)

    fig, _, filepath = raw_subset_10min_mrr.plot_spectrogram(
        target_time,
        spectrum_var="spectrum_raw",
        savefig=True,
        output_dir=artifact_dir,
        dpi=120,
    )

    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    assert filepath.suffix.lower() == ".png"
    assert filepath.stat().st_size > 0
    plt.close(fig)


