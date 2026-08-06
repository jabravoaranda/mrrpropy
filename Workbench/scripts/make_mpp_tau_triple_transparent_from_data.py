from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mrrpropy.plotting.paper import plot_microphysical_tau_triple


matplotlib.use("Agg")


def main() -> None:
    csv_path = Path(
        "workbench/output/poster/"
        "poster_column_process_recomputed_minTau03_20251029_190000.csv"
    )
    output_dir = Path("Workbench/output/poster")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(csv_path, parse_dates=["time"])
    filename = "mpp_tau_triple_20251029_193000_w500_step35_layermean_y270_font24_transparent.png"
    fig, _, _ = plot_microphysical_tau_triple(
        frame,
        target_datetime="2025-10-29 19:30:00",
        variables=("Dm", "Nw", "LWC"),
        score_prefix="tau",
        figsize=(12.0, 6.0),
        y_limits=(0.9, 2.70),
        savefig=False,
        dpi=300,
        transparent=True,
    )
    for ax in fig.axes:
        ax.xaxis.label.set_size(24)
        ax.yaxis.label.set_size(24)
        ax.tick_params(axis="both", labelsize=24)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
