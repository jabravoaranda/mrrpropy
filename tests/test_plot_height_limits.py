from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import pytest


pytestmark = [pytest.mark.slow, pytest.mark.plot, pytest.mark.integration]


def test_process_height_plots_use_dataset_range_by_default(
    raprompro_subset_10min_loaded_mrr,
    artifact_dir,
):
    ds = raprompro_subset_10min_loaded_mrr.raprompro
    assert ds is not None
    expected = (
        float(ds["range"].values.min() / 1000.0),
        float(ds["range"].values.max() / 1000.0),
    )

    fig, _, path = raprompro_subset_10min_loaded_mrr.plot_event_vertical_percent_profiles(
        target_datetime=(
            datetime(2025, 10, 29, 19, 23, 0),
            datetime(2025, 10, 29, 19, 33, 0),
        ),
        layer=(1000.0, 2000.0),
        variables=("Dm", "Nw", "LWC"),
        savefig=True,
        output_dir=artifact_dir,
    )

    assert path is not None
    assert tuple(round(value, 6) for value in fig.axes[0].get_ylim()) == tuple(
        round(value, 6) for value in expected
    )
    plt.close(fig)


def test_process_height_plots_allow_user_y_limits(
    raprompro_subset_10min_loaded_mrr,
    artifact_dir,
):
    expected = (0.8, 2.4)

    fig, _, path = raprompro_subset_10min_loaded_mrr.plot_column_process_scan(
        scan_df=raprompro_subset_10min_loaded_mrr.build_column_process_scan_dataframe(
            period=(
                datetime(2025, 10, 29, 19, 23, 0),
                datetime(2025, 10, 29, 19, 33, 0),
            ),
            k=11,
            window_thickness_m=1000.0,
            window_step_m=100.0,
            min_tau_strength=0.10,
        ),
        y_limits=expected,
        savefig=True,
        output_dir=artifact_dir,
    )

    assert path is not None
    assert tuple(round(value, 6) for value in fig.axes[0].get_ylim()) == expected
    plt.close(fig)


