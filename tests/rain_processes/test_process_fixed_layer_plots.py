from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import xarray as xr

from mrrpropy.plotting import processes as process_plotting

matplotlib.use("Agg")

pytestmark = [pytest.mark.slow, pytest.mark.plot, pytest.mark.integration]


class _SyntheticPlotSubject:
    def __init__(self, ds: xr.Dataset):
        self.raprompro = ds

    def _is_processed(self):
        return True


@pytest.fixture(scope="session")
def analysis(raprompro_subset_10min_loaded_mrr):
    return raprompro_subset_10min_loaded_mrr.rain_process_analyze(
        period=(datetime(2025, 10, 29, 19, 23, 0), datetime(2025, 10, 29, 19, 33, 0)),
        k=11,
        selection_mode="fixed_layer",
        z_bottom_m=1000.0,
        z_top_m=2000.0,
        ze_th=-5.0,
        min_points_trend=10,
        eps_q=0.01,
        rgb_q=0.02,
        vars_trend=("Dm", "Nw", "LWC"),
    )


@pytest.fixture(scope="session")
def classified(raprompro_subset_10min_loaded_mrr, analysis):
    return raprompro_subset_10min_loaded_mrr.classify_rain_process(analysis=analysis)


def _fixed_layer_artifact_dir(artifact_dir: Path) -> Path:
    path = artifact_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_plot_rain_process_in_layer_2d(raprompro_subset_10min_loaded_mrr, artifact_dir):
    output_dir = _fixed_layer_artifact_dir(artifact_dir)
    fig, _, path = raprompro_subset_10min_loaded_mrr.plot_rain_process_in_layer_2D(
        target_datetime=(
            datetime(2025, 10, 29, 19, 23, 0),
            datetime(2025, 10, 29, 19, 33, 0),
        ),
        layer=(1000.0, 2000.0),
        x="Dm",
        y="Nw",
        z="LWC",
        savefig=True,
        marker_size=100,
        figsize=(12, 10),
        cmap="seismic",
        output_dir=output_dir,
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    assert path.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf")
    plt.close(fig)


def test_plot_rain_process_in_layer_hexagram(
    raprompro_subset_10min_loaded_mrr, analysis, artifact_dir
):
    output_dir = _fixed_layer_artifact_dir(artifact_dir)
    (
        fig,
        ax,
        filepath,
    ) = raprompro_subset_10min_loaded_mrr.plot_rain_process_in_layer_hexagram(
        analysis=analysis,
        savefig=True,
        output_dir=output_dir,
        dpi=200,
        alpha_hexagram=0.5,
        cmap="viridis",
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert filepath is not None
    assert filepath.exists()
    assert filepath.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf")

    axes = fig.get_axes()
    assert len(axes) >= 1
    ax0 = axes[0]
    assert isinstance(ax0, Axes)
    assert len(ax0.images) >= 1
    assert len(ax0.collections) >= 1
    plt.close(fig)


def test_plot_microphysics_summary_multipanel(
    raprompro_subset_10min_loaded_mrr,
    analysis,
    classified,
    artifact_dir,
):
    output_dir = _fixed_layer_artifact_dir(artifact_dir)
    fig, _, path = raprompro_subset_10min_loaded_mrr.plot_processes_evolution(
        classified=classified,
        analysis=analysis,
        savefig=True,
        output_dir=output_dir,
        figsize=(14, 10),
        cmap="viridis",
        alpha_hexagram=0.5,
        markersize=40.0,
        line_width=0.8,
        dpi=200,
    )
    if classified.attrs.get("max_tau_pvalue", "missing") is None:
        classified.attrs.pop("max_tau_pvalue", None)
    classified.attrs["rgb_mapping"] = str(classified.attrs["rgb_mapping"])
    classified.to_netcdf(
        artifact_dir
        / f"{raprompro_subset_10min_loaded_mrr.path.stem}_process_classification.nc"
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    plt.close(fig)


def test_plot_event_scatter(raprompro_subset_10min_loaded_mrr, artifact_dir):
    output_dir = _fixed_layer_artifact_dir(artifact_dir)
    fig, _, path = raprompro_subset_10min_loaded_mrr.plot_event_scatter(
        target_datetime=(
            datetime(2025, 10, 29, 19, 23, 0),
            datetime(2025, 10, 29, 19, 33, 0),
        ),
        layer=(1000.0, 2000.0),
        x="Dm",
        y="Nw",
        color="LWC",
        savefig=True,
        output_dir=output_dir,
        figsize=(12, 10),
        cmap="seismic",
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    plt.close(fig)


def test_plot_region_scatter(
    raprompro_subset_10min_loaded_mrr,
    classified,
    artifact_dir,
):
    output_dir = _fixed_layer_artifact_dir(artifact_dir)
    labels = sorted(
        {
            label
            for label in classified["proc_label"].values.astype(str)
            if label not in {"no_data", "unknown", "steady_or_weak"}
        }
    )
    processes = labels[:2] if labels else None

    fig, _, path = raprompro_subset_10min_loaded_mrr.plot_region_scatter(
        target_datetime=(
            datetime(2025, 10, 29, 19, 23, 0),
            datetime(2025, 10, 29, 19, 33, 0),
        ),
        z_bottom_m=1000.0,
        z_top_m=2000.0,
        x="Dm",
        y="Nw",
        color="LWC",
        processes=processes,
        classified=classified,
        savefig=True,
        output_dir=output_dir,
        figsize=(12, 10),
        cmap="seismic",
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    plt.close(fig)


def test_plot_process_scatter(
    raprompro_subset_10min_loaded_mrr,
    classified,
    artifact_dir,
):
    output_dir = _fixed_layer_artifact_dir(artifact_dir)
    labels = classified["proc_label"].values.astype(str)
    process = next(
        (
            label
            for label in labels
            if label not in {"no_data", "unknown", "steady_or_weak"}
        ),
        None,
    )
    if process is None:
        process = next((label for label in labels if label != "no_data"), None)
    if process is None:
        pytest.skip("No classified process is available in this fixture.")

    fig, _, path = raprompro_subset_10min_loaded_mrr.plot_process_scatter(
        classified=classified,
        process=process,
        target_datetime=(
            datetime(2025, 10, 29, 19, 23, 0),
            datetime(2025, 10, 29, 19, 33, 0),
        ),
        layer=(1000.0, 2000.0),
        x="Dm",
        y="Nw",
        color="LWC",
        savefig=True,
        output_dir=output_dir,
        figsize=(12, 10),
        cmap="seismic",
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()

    axes = fig.get_axes()
    assert len(axes) >= 1
    ax0 = axes[0]
    assert isinstance(ax0, Axes)
    assert len(ax0.collections) >= 1
    plt.close(fig)


def test_select_layer_event_data_uses_top_of_layer_as_percent_reference():
    ds = xr.Dataset(
        coords={
            "time": np.array(["2025-10-29T19:23:00"], dtype="datetime64[s]"),
            "range": np.array([1000.0, 1500.0, 2000.0], dtype=float),
        }
    )
    ds["Dm"] = xr.DataArray(np.array([[12.0, 11.0, 10.0]]), dims=("time", "range"))
    ds["Nw"] = xr.DataArray(np.array([[30.0, 20.0, 10.0]]), dims=("time", "range"))
    ds["LWC"] = xr.DataArray(np.array([[6.0, 4.0, 2.0]]), dims=("time", "range"))

    selected = process_plotting._select_layer_event_data(
        _SyntheticPlotSubject(ds),
        target_datetime=(
            datetime(2025, 10, 29, 19, 23, 0),
            datetime(2025, 10, 29, 19, 23, 1),
        ),
        layer=(1000.0, 2000.0),
        variables=("Dm", "Nw", "LWC"),
        use_relative_difference=True,
    )

    assert selected.attrs["profile_reference"] == "top_of_layer"
    assert selected["LWC"].sel(range=2000.0).values[0] == pytest.approx(0.0)
    assert selected["LWC"].sel(range=1000.0).values[0] > 0.0
    assert selected["Nw"].sel(range=1000.0).values[0] > 0.0


def test_plot_event_vertical_percent_profiles(
    raprompro_subset_10min_loaded_mrr,
    artifact_dir,
):
    output_dir = _fixed_layer_artifact_dir(artifact_dir)
    fig, _, path = raprompro_subset_10min_loaded_mrr.plot_event_vertical_percent_profiles(
        target_datetime=(
            datetime(2025, 10, 29, 19, 23, 0),
            datetime(2025, 10, 29, 19, 33, 0),
        ),
        layer=(1000.0, 2000.0),
        variables=("Dm", "Nw", "LWC"),
        savefig=True,
        output_dir=output_dir,
        figsize=(7, 6),
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    axes = fig.get_axes()
    assert len(axes) == 1
    assert len(axes[0].lines) >= 3
    plt.close(fig)


def test_plot_process_vertical_percent_profiles(
    raprompro_subset_10min_loaded_mrr,
    classified,
    artifact_dir,
):
    output_dir = _fixed_layer_artifact_dir(artifact_dir)
    labels = classified["proc_label"].values.astype(str)
    process = next(
        (
            label
            for label in labels
            if label not in {"no_data", "unknown", "steady_or_weak"}
        ),
        None,
    )
    if process is None:
        process = next((label for label in labels if label != "no_data"), None)
    if process is None:
        pytest.skip("No classified process is available in this fixture.")

    (
        fig,
        ax,
        path,
    ) = raprompro_subset_10min_loaded_mrr.plot_process_vertical_percent_profiles(
        classified=classified,
        process=process,
        target_datetime=(
            datetime(2025, 10, 29, 19, 23, 0),
            datetime(2025, 10, 29, 19, 33, 0),
        ),
        layer=(1000.0, 2000.0),
        variables=("Dm", "Nw", "LWC"),
        savefig=True,
        output_dir=output_dir,
        figsize=(7, 6),
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(path, Path)
    assert path.exists()
    axes = fig.get_axes()
    assert len(axes) == 1
    assert len(axes[0].lines) >= 3
    plt.close(fig)


