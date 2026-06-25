from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
import pandas as pd

matplotlib.use("Agg")

pytestmark = [pytest.mark.slow]

WINDOW_THICKNESS_M = 500.0
WINDOW_STEP_M = (
    None  # Use the raw window step from the scan, which is typically around 100 m
)
MIN_TAU_STRENGTH = 0.5


@pytest.fixture(scope="session")
def analysis(raprompro_mrr):
    return raprompro_mrr.rain_process_analyze(
        period=(datetime(2025, 10, 29, 19, 23, 0), datetime(2025, 10, 29, 19, 33, 0)),
        k=11,
        selection_mode="sliding",
        window_thickness_m=WINDOW_THICKNESS_M,
        window_step_m=WINDOW_STEP_M,
        min_tau_strength=MIN_TAU_STRENGTH,
        ze_th=-5.0,
        min_points_trend=10,
        eps_q=0.01,
        rgb_q=0.02,
        vars_trend=("Dm", "Nw", "LWC"),
    )


@pytest.fixture(scope="session")
def classified(raprompro_mrr, analysis):
    return raprompro_mrr.classify_rain_process(analysis=analysis)


def test_plot_column_process_scan(raprompro_mrr, product_dir):
    output_dir = product_dir
    scan_df = raprompro_mrr.build_sliding_process_dataframe(
        period=(
            datetime(2025, 10, 29, 19, 23, 0),
            datetime(2025, 10, 29, 19, 33, 0),
        ),
        k=11,
        window_thickness_m=WINDOW_THICKNESS_M,
        window_step_m=WINDOW_STEP_M,
        min_tau_strength=MIN_TAU_STRENGTH,
    )

    fig, path = raprompro_mrr.plot.rain.column_scan(
        scan_df=scan_df,
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


def test_plot_column_process_scan_selected_processes(raprompro_mrr, product_dir):
    output_dir = product_dir
    scan_df = raprompro_mrr.build_sliding_process_dataframe(
        period=(
            datetime(2025, 10, 29, 19, 23, 0),
            datetime(2025, 10, 29, 19, 33, 0),
        ),
        k=11,
        window_thickness_m=WINDOW_THICKNESS_M,
        window_step_m=WINDOW_STEP_M,
        min_tau_strength=MIN_TAU_STRENGTH,
    )
    all_expected = {
        label
        for label in pd.unique(scan_df["proc_label"].astype(str))
        if label not in {"unknown", "no_data"}
    }
    selected_processes = sorted(
        {
            label
            for label in pd.unique(scan_df["proc_label"].astype(str))
            if label not in {"unknown", "no_data", "steady_or_weak"}
        }
    )
    if not selected_processes:
        pytest.skip("No identified processes are available in this fixture.")

    fig_all, _ = raprompro_mrr.plot.rain.column_scan(
        scan_df=scan_df,
        figsize=(10, 6),
    )
    (fig_selected, path_selected,) = raprompro_mrr.plot.rain.column_scan(
        scan_df=scan_df,
        processes=selected_processes + ["steady_or_weakk"],
        savefig=True,
        output_dir=output_dir,
        figsize=(10, 6),
    )

    legend_all = fig_all.axes[0].get_legend()
    legend_selected = fig_selected.axes[0].get_legend()
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


def test_plot_column_process_scan_hexagram_colors(raprompro_mrr, product_dir):
    output_dir = product_dir
    scan_df = raprompro_mrr.build_sliding_process_dataframe(
        period=(
            datetime(2025, 10, 29, 19, 23, 0),
            datetime(2025, 10, 29, 19, 33, 0),
        ),
        k=11,
        window_thickness_m=WINDOW_THICKNESS_M,
        window_step_m=WINDOW_STEP_M,
        min_tau_strength=MIN_TAU_STRENGTH,
    )

    fig, path = raprompro_mrr.plot.rain.column_scan(
        scan_df=scan_df,
        color_mode="hexagram",
        processes=[
            "breakup",
            "growth_depletion",
            "growth_depletion_loss",
            "growth_depletion_gain",
            "activation",
            "evaporation",
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


def test_plot_scan_process_scatter_compare(raprompro_mrr, product_dir):
    output_dir = product_dir
    scan_df = raprompro_mrr.build_sliding_process_dataframe(
        period=(
            datetime(2025, 10, 29, 19, 23, 0),
            datetime(2025, 10, 29, 19, 33, 0),
        ),
        k=11,
        window_thickness_m=WINDOW_THICKNESS_M,
        window_step_m=WINDOW_STEP_M,
        min_tau_strength=MIN_TAU_STRENGTH,
    )
    selected_processes = sorted(
        {
            label
            for label in pd.unique(scan_df["proc_label"].astype(str))
            if label not in {"unknown", "no_data", "steady_or_weak"}
        }
    )[:2]
    if len(selected_processes) < 1:
        pytest.skip("No identified scan processes are available in this fixture.")

    fig, path = raprompro_mrr.plot.rain.scan_scatter_compare(
        scan_df=scan_df,
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
