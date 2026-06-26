from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

pytestmark = [pytest.mark.slow, pytest.mark.plot, pytest.mark.integration]


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


def test_plot_classified_processes_on_hexagram(
    raprompro_subset_10min_loaded_mrr,
    analysis,
    classified,
    artifact_dir,
):
    fig, _, path = raprompro_subset_10min_loaded_mrr.plot_classified_processes_on_hexagram(
        classified=classified,
        analysis=analysis,
        savefig=True,
        show_background=True,
        output_dir=artifact_dir,
        figsize=(14, 10),
        cmap="viridis",
        alpha_hexagram=0.25,
        markersize=70.0,
        line_width=0.8,
        dpi=200,
        legend_fontsize=14,
    )

    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    plt.close(fig)


