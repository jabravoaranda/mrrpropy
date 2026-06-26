from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import pytest

from mrrpropy.plotting.processes import plot_fused_process_quicklook

from tests.rain_processes.test_process_scan_plots import (
    MIN_TAU_STRENGTH,
    WINDOW_STEP_M,
    WINDOW_THICKNESS_M,
    _scan_artifact_dir,
)


matplotlib.use("Agg")

QUICKLOOK_PROCESSES = [
    "breakup",
    "growth_depletion",
    "growth_depletion_loss",
    "growth_depletion_gain",
    "activation",
    "evaporation",
    "growth",
]


@pytest.fixture(scope="session")
def scan_df(raprompro_subset_10min_loaded_mrr) -> pd.DataFrame:
    return raprompro_subset_10min_loaded_mrr.build_column_process_scan_dataframe(
        period=(datetime(2025, 10, 29, 19, 23, 0), datetime(2025, 10, 29, 19, 33, 0)),
        k=11,
        window_thickness_m=WINDOW_THICKNESS_M,
        window_step_m=WINDOW_STEP_M,
        min_tau_strength=MIN_TAU_STRENGTH,
    )


def _make_fused_df_from_scan_snapshot(scan_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a minimal fused_df-like dataframe from a single-time scan snapshot.

    This keeps plotting tests fast while still relying on the same scan_df data
    produced by the scan workflow tests.
    """
    if scan_df.empty:
        return pd.DataFrame()

    valid = scan_df.copy()
    valid["z_top_m"] = pd.to_numeric(valid["z_top_m"], errors="coerce")
    valid["z_bottom_m"] = pd.to_numeric(valid["z_bottom_m"], errors="coerce")
    valid = valid[valid["z_top_m"].gt(valid["z_bottom_m"])].copy()
    if valid.empty:
        return pd.DataFrame()

    plottable = valid[valid["proc_label"].astype(str).isin(QUICKLOOK_PROCESSES)].copy()
    source = plottable.iloc[0] if not plottable.empty else valid.iloc[0]
    time0 = pd.Timestamp(pd.to_datetime(source["time"]))
    label = str(source["proc_label"])
    if label not in QUICKLOOK_PROCESSES:
        label = QUICKLOOK_PROCESSES[-1]

    snap = valid[
        (valid["time"] == time0) & (valid["proc_label"].astype(str) == label)
    ].copy()
    if snap.empty:
        snap = pd.DataFrame([source])

    if "window_id" in snap.columns:
        snap = snap.sort_values("window_id", ascending=False)
    take = snap.head(3)
    if take.empty:
        return pd.DataFrame()

    z_top = float(pd.to_numeric(take["z_top_m"], errors="coerce").max())
    z_bottom = float(pd.to_numeric(take["z_bottom_m"], errors="coerce").min())

    return pd.DataFrame(
        {
            "time": [time0],
            "proc_label_fused": [label],
            "z_top_fused": [z_top],
            "z_bottom_fused": [z_bottom],
        }
    )


@pytest.fixture(scope="session")
def fused_df(scan_df) -> pd.DataFrame:
    return _make_fused_df_from_scan_snapshot(scan_df)


def test_plot_fused_process_quicklook_savefig(scan_df, fused_df, artifact_dir: Path):
    if fused_df.empty:
        pytest.skip("No fused events available for quicklook savefig test.")

    output_dir = _scan_artifact_dir(artifact_dir)

    fig, _, path = plot_fused_process_quicklook(
        scan_df,
        fused_df,
        processes=QUICKLOOK_PROCESSES,
        savefig=True,
        output_dir=output_dir,
        dpi=200,
    )
    assert isinstance(fig, Figure)
    assert isinstance(path, Path)
    assert path.exists()
    axes = fig.get_axes()
    assert len(axes) == 1
    assert len(axes[0].patches) >= 1
    plt.close(fig)


def test_plot_fused_process_quicklook_returns_none_path_by_default(scan_df, fused_df):
    fig, _, path = plot_fused_process_quicklook(scan_df, fused_df)
    assert isinstance(fig, Figure)
    assert path is None
    plt.close(fig)


def test_plot_fused_process_quicklook_processes_filter(scan_df, fused_df):
    if fused_df.empty:
        pytest.skip("No fused events available for filter test.")

    first_label = str(pd.unique(fused_df["proc_label_fused"].astype(str))[0])

    fig_keep, _, _ = plot_fused_process_quicklook(
        scan_df, fused_df, processes=[first_label]
    )
    assert len(fig_keep.axes[0].patches) >= 1
    plt.close(fig_keep)

    fig_drop, _, _ = plot_fused_process_quicklook(
        scan_df, fused_df, processes=["__no_such_process__"]
    )
    assert len(fig_drop.axes[0].patches) == 0
    plt.close(fig_drop)


