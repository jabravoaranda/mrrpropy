from __future__ import annotations

from datetime import datetime
import pandas as pd
import pytest

from mrrpropy.analysis import sliding as sliding_analysis

from tests.rain_processes.test_process_scan_plots import (
    MIN_TAU_STRENGTH,
    WINDOW_STEP_M,
    WINDOW_THICKNESS_M,
)

pytestmark = [pytest.mark.slow]


@pytest.fixture(scope="session")
def scan_df(raprompro_mrr) -> pd.DataFrame:
    return raprompro_mrr.build_sliding_process_dataframe(
        period=(datetime(2025, 10, 29, 19, 23, 0), datetime(2025, 10, 29, 19, 33, 0)),
        k=11,
        window_thickness_m=WINDOW_THICKNESS_M,
        window_step_m=WINDOW_STEP_M,
        min_tau_strength=MIN_TAU_STRENGTH,
    )


def test_build_fused_column_process_dataframe(raprompro_mrr, scan_df):
    fused = sliding_analysis.build_fused_column_process_dataframe(
        raprompro_mrr,
        scan_df,
        min_consecutive=3,
        variable_threshold="Ze",
        threshold_value=-999.0,
        min_points_trend=3,
    )

    assert isinstance(fused, pd.DataFrame)
    assert not fused.empty

    for col in (
        "time",
        "run_process_label",
        "proc_label_fused",
        "z_top_fused",
        "z_bottom_fused",
        "thickness_fused",
        "n_windows_merged",
    ):
        assert col in fused.columns

    assert int(fused.loc[0, "n_windows_merged"]) >= 3
    assert float(fused.loc[0, "thickness_fused"]) > 0.0

