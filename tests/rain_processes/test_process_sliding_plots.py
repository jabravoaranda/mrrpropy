from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from mrrpropy.plotting import processes as process_plotting
from mrrpropy.rain_process_classification.rain_process_algorithm import (
    sliding_rain_classification_to_dataframe,
)

matplotlib.use("Agg")

pytestmark = [pytest.mark.slow, pytest.mark.plot, pytest.mark.integration]

WINDOW_THICKNESS_M = 500.0
WINDOW_STEP_M = (
    None  # Use the raw window step from the sliding, which is typically around 100 m
)
MIN_TAU_STRENGTH = 0.5


@pytest.fixture(scope="session")
def analysis(raprompro_subset_10min_loaded_mrr):
    return raprompro_subset_10min_loaded_mrr.sliding_rain_classification(
        period=(datetime(2025, 10, 29, 19, 23, 0), datetime(2025, 10, 29, 19, 33, 0)),
        k=11,
        window_thickness_m=WINDOW_THICKNESS_M,
        window_step_m=WINDOW_STEP_M,
        min_tau_strength=MIN_TAU_STRENGTH,
        ze_th=-5.0,
        min_points_trend=10,
        vars_trend=("Dm", "Nw", "LWC"),
    )


@pytest.fixture(scope="session")
def classified(raprompro_subset_10min_loaded_mrr, analysis):
    return analysis


def _sliding_artifact_dir(artifact_dir: Path) -> Path:
    path = artifact_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_plot_sliding_column_process_marker_mode_square():
    subject = SimpleNamespace(
        plot_cfg=SimpleNamespace(dpi=100, figsize_profiles=(6, 4)),
        raprompro=None,
    )
    sliding_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2025-10-29 19:23", "2025-10-29 19:24"]),
            "range": [1000.0, 1100.0],
            "proc_label": ["breakup", "coalescence"],
            "proc_strength": [1.0, 1.0],
        }
    )

    fig_process, _, _ = process_plotting.plot_sliding_column_process(
        subject,
        sliding_df=sliding_df,
        marker_mode="process",
        scale_by_strength=False,
    )
    fig_square, _, _ = process_plotting.plot_sliding_column_process(
        subject,
        sliding_df=sliding_df,
        marker_mode="square",
        render_mode="markers",
        scale_by_strength=False,
    )

    process_paths = [
        collection.get_paths()[0].vertices
        for collection in fig_process.axes[0].collections
    ]
    square_paths = [
        collection.get_paths()[0].vertices
        for collection in fig_square.axes[0].collections
    ]

    assert not np.array_equal(process_paths[0], process_paths[1])
    assert np.array_equal(square_paths[0], square_paths[1])

    plt.close(fig_process)
    plt.close(fig_square)


def test_plot_sliding_column_process_square_defaults_to_cells():
    subject = SimpleNamespace(
        plot_cfg=SimpleNamespace(dpi=100, figsize_profiles=(6, 4)),
        raprompro=None,
    )
    sliding_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2025-10-29 19:23", "2025-10-29 19:24"]),
            "range": [1000.0, 1100.0],
            "proc_label": ["breakup", "coalescence"],
            "proc_strength": [1.0, 1.0],
        }
    )

    fig, ax, _ = process_plotting.plot_sliding_column_process(
        subject,
        sliding_df=sliding_df,
        marker_mode="square",
        scale_by_strength=False,
    )

    assert len(ax.collections) == 2
    for collection in ax.collections:
        path = collection.get_paths()[0].vertices
        assert np.isclose(path[:, 0].max() - path[:, 0].min(), 0.000638888888)
        assert np.isclose(path[:, 1].max() - path[:, 1].min(), 0.092)

    plt.close(fig)


def test_plot_sliding_column_process(
    raprompro_subset_10min_loaded_mrr, analysis, artifact_dir
):
    output_dir = _sliding_artifact_dir(artifact_dir)
    sliding_df = analysis

    fig, _, path = raprompro_subset_10min_loaded_mrr.plot_sliding_column_process(
        sliding_df=sliding_df,
        savefig=True,
        output_dir=output_dir,
        figsize=(10, 6),
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    axes = fig.get_axes()
    assert len(axes) == 1
    assert len(axes[0].collections) >= 1
    plt.close(fig)


def test_plot_sliding_column_process_selected_processes(
    raprompro_subset_10min_loaded_mrr, analysis, artifact_dir
):
    output_dir = _sliding_artifact_dir(artifact_dir)
    sliding_df = analysis
    sliding_frame = sliding_rain_classification_to_dataframe(sliding_df)
    all_expected = {
        label
        for label in pd.unique(sliding_frame["proc_label"].astype(str))
        if label not in {"unknown", "no_data"}
    }
    selected_processes = sorted(
        {
            label
            for label in pd.unique(sliding_frame["proc_label"].astype(str))
            if label not in {"unknown", "no_data", "steady_or_weak"}
        }
    )
    if not selected_processes:
        pytest.skip("No identified processes are available in this fixture.")

    fig_all, _, _ = raprompro_subset_10min_loaded_mrr.plot_sliding_column_process(
        sliding_df=sliding_df,
        figsize=(10, 6),
    )
    (
        fig_selected,
        ax_selected,
        path_selected,
    ) = raprompro_subset_10min_loaded_mrr.plot_sliding_column_process(
        sliding_df=sliding_df,
        processes=selected_processes + ["steady_or_weakk"],
        savefig=True,
        output_dir=output_dir,
        figsize=(10, 6),
    )

    legend_all = fig_all.axes[0].get_legend()
    legend_selected = ax_selected.get_legend()
    assert legend_all is not None
    assert legend_selected is not None
    labels_all = {text.get_text() for text in legend_all.get_texts()}
    labels_selected = {text.get_text() for text in legend_selected.get_texts()}

    assert path_selected is not None
    assert path_selected.exists()
    assert labels_all == all_expected
    assert labels_selected == set(selected_processes)
    assert "steady_or_weak" not in labels_selected
    assert "steady_or_weakk" not in labels_selected

    plt.close(fig_all)
    plt.close(fig_selected)


def test_plot_sliding_column_process_hexagram_colors(
    raprompro_subset_10min_loaded_mrr, analysis, artifact_dir
):
    output_dir = _sliding_artifact_dir(artifact_dir)
    sliding_df = analysis

    fig, _, path = raprompro_subset_10min_loaded_mrr.plot_sliding_column_process(
        sliding_df=sliding_df,
        color_mode="hexagram",
        processes=[
            "breakup",
            "coalescence",
            "coalescence_loss",
            "coalescence_gain",
            "activation",
            "evaporation_strong",
            "growth",
        ],
        savefig=True,
        output_dir=output_dir,
        figsize=(10, 6),
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    axes = fig.get_axes()
    assert len(axes) == 1
    assert len(axes[0].collections) >= 1
    plt.close(fig)


def test_plot_sliding_process_scatter_compare(
    raprompro_subset_10min_loaded_mrr, analysis, artifact_dir
):
    output_dir = _sliding_artifact_dir(artifact_dir)
    sliding_df = analysis
    sliding_frame = sliding_rain_classification_to_dataframe(sliding_df)
    selected_processes = sorted(
        {
            label
            for label in pd.unique(sliding_frame["proc_label"].astype(str))
            if label not in {"unknown", "no_data", "steady_or_weak"}
        }
    )[:2]
    if len(selected_processes) < 1:
        pytest.skip("No identified sliding processes are available in this fixture.")

    (
        fig,
        _,
        path,
    ) = raprompro_subset_10min_loaded_mrr.plot_sliding_process_scatter_compare(
        sliding_df=sliding_df,
        processes=selected_processes,
        show_centroids=True,
        savefig=True,
        output_dir=output_dir,
        figsize=(10, 8),
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    axes = fig.get_axes()
    assert len(axes) >= 1
    assert len(axes[0].collections) >= 1
    plt.close(fig)
