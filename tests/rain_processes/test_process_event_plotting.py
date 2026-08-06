from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, PolyCollection
import numpy as np
import pandas as pd
import xarray as xr

from mrrpropy.plotting.processes import (
    plot_rain_process_in_layer_hexagram,
    plot_sliding_column_process,
)

matplotlib.use("Agg")


class _Subject:
    path = ""
    plot_cfg = SimpleNamespace(
        figsize_profiles=(6, 4),
        figsize_multipanel=(6, 4),
        markersize=20.0,
        alpha_points=0.9,
        dpi=100,
    )
    raprompro = xr.Dataset(coords={"range": np.array([900.0, 2500.0])})

    def _is_processed(self) -> bool:
        return True


def test_event_mode_omits_below_threshold_zero_and_unavailable_intensity_rows():
    sliding_df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2024-10-28 22:00:00",
                    "2024-10-28 22:01:00",
                    "2024-10-28 22:02:00",
                    "2024-10-28 22:03:00",
                    "2024-10-28 22:04:00",
                ]
            ),
            "range": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            "proc_label": [
                "evaporation_strong",
                "evaporation_strong",
                "evaporation_strong",
                "breakup",
                "no_data",
            ],
            "proc_strength": [0.4, 0.2, 0.0, 0.8, np.nan],
            "trend_sign_Dm": [-1.0, -1.0, -1.0, 1.0, np.nan],
            "trend_sign_Nw": [-1.0, -1.0, -1.0, 1.0, np.nan],
            "trend_sign_LWC": [-1.0, -1.0, -1.0, 1.0, np.nan],
            "trend_strength_Dm": [0.4, 0.2, 0.0, 0.8, np.nan],
            "trend_strength_Nw": [0.5, 0.2, 0.0, 0.8, np.nan],
            "trend_strength_LWC": [0.6, 0.2, 0.0, 0.8, np.nan],
        }
    )
    sliding_df.attrs["min_tau_strength"] = 0.3

    fig, _, _ = plot_sliding_column_process(
        _Subject(),
        sliding_df=sliding_df,
        color_mode="event",
        event_process="evaporation_strong",
    )

    plotted_points = sum(
        len(collection.get_offsets()) for collection in fig.axes[0].collections
    )

    assert plotted_points == 1
    plt.close(fig)


def test_hexagram_mode_uses_pixelated_cells():
    sliding_df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2024-10-28 22:00:00",
                    "2024-10-28 22:00:00",
                    "2024-10-28 22:01:00",
                    "2024-10-28 22:01:00",
                ]
            ),
            "range": [1000.0, 1035.0, 1000.0, 1035.0],
            "proc_label": [
                "evaporation_strong",
                "evaporation_strong",
                "activation",
                "activation",
            ],
            "proc_strength": [0.8, 0.8, 0.7, 0.7],
            "R": [0.2, 0.3, 0.4, 0.5],
            "G": [0.4, 0.5, 0.6, 0.7],
            "B": [0.6, 0.7, 0.8, 0.9],
        }
    )

    fig, _, _ = plot_sliding_column_process(
        _Subject(),
        sliding_df=sliding_df,
        color_mode="hexagram",
    )

    patch_collections = [
        collection
        for collection in fig.axes[0].collections
        if isinstance(collection, PatchCollection)
    ]
    plotted_cells = sum(len(collection.get_paths()) for collection in patch_collections)

    assert plotted_cells == len(sliding_df)
    assert fig.axes[0].get_xlim()[0] > mdates.date2num(pd.Timestamp("2024-10-28"))
    plt.close(fig)


def test_contour_mode_draws_density_boundaries_with_date_limits():
    times = pd.date_range("2024-10-28 22:00:00", periods=6, freq="10s")
    sliding_df = pd.DataFrame(
        {
            "time": np.repeat(times, 4),
            "range": np.tile([1000.0, 1035.0, 1070.0, 1105.0], len(times)),
            "proc_label": "activation",
            "proc_strength": 0.8,
        }
    )

    fig, _, _ = plot_sliding_column_process(
        _Subject(),
        sliding_df=sliding_df,
        color_mode="contour",
        contour_background=False,
        contour_event_count=4,
        contour_top_n=2,
        contour_bins=(12, 8),
        contour_sigma=1.0,
    )

    contour_paths = sum(
        len(collection.get_paths()) for collection in fig.axes[0].collections
    )

    assert contour_paths > 0
    assert fig.axes[0].get_xlim()[0] > mdates.date2num(pd.Timestamp("2024-10-28"))
    plt.close(fig)


def test_layer_hexagram_can_render_samples_with_hexbin():
    analysis = xr.Dataset(
        {
            "hex_x": ("time", np.array([45.0, 46.0, 46.0, 47.0])),
            "hex_y": ("time", np.array([44.0, 45.0, 46.0, 47.0])),
            "minutes": ("time", np.array([0.0, 1.0, 2.0, 3.0])),
            "R": ("time", np.array([0.2, 0.3, 0.4, 0.5])),
            "G": ("time", np.array([0.3, 0.4, 0.5, 0.6])),
            "B": ("time", np.array([0.4, 0.5, 0.6, 0.7])),
        },
        coords={"time": pd.date_range("2024-10-28 22:00:00", periods=4, freq="10s")},
        attrs={
            "k": 11,
            "z_bottom_m": 1000.0,
            "z_top_m": 2000.0,
            "selection_mode": "fixed_layer",
            "period_start": "2024-10-28T22:00:00",
            "period_end": "2024-10-28T22:01:00",
        },
    )

    fig, _, _ = plot_rain_process_in_layer_hexagram(
        _Subject(),
        analysis=analysis,
        use_hexbin=True,
    )

    assert fig.axes[0].images
    assert any(
        isinstance(collection, PolyCollection) for collection in fig.axes[0].collections
    )
    plt.close(fig)
