from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib
import pandas as pd
import xarray as xr

from mrrpropy import MRRProData

matplotlib.use("Agg")


DEFAULT_INPUT_FILE = Path(
    r"workbench/output/raprompro/2025/10/29/20251029_190000_raprompro.nc"
)
DEFAULT_OUTPUT_DIR = Path(r"workbench/output/rain_classification/2025/10/29/190000")
DEFAULT_LAYERS = [(2000.0, 2500.0), (1000.0, 2000.0)]


def _layer_dir_name(layer: tuple[float, float]) -> str:
    return f"layer_{int(layer[0])}_{int(layer[1])}m"


def _compact_name(
    *,
    start_time,
    layer: tuple[float, float],
    kind: str,
) -> str:
    t0 = pd.Timestamp(start_time).strftime("%Y%m%d_%H%M")
    return f"{t0}_{int(layer[0])}-{int(layer[1])}_{kind}.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the test_6-style rain classification figures for selected layers."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to the processed *_raprompro.nc file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory where layer-specific figure folders will be written.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=11,
        help="Hexagram resolution parameter.",
    )
    parser.add_argument(
        "--ze-th",
        type=float,
        default=-5.0,
        help="Reflectivity threshold passed to rain_process_analyze.",
    )
    parser.add_argument(
        "--min-points-ols",
        type=int,
        default=10,
        help="Minimum valid points for OLS trend estimation.",
    )
    parser.add_argument(
        "--eps-q",
        type=float,
        default=0.01,
        help="Quantile used for epsilon estimation.",
    )
    parser.add_argument(
        "--rgb-q",
        type=float,
        default=0.02,
        help="Quantile used for RGB scaling.",
    )
    args = parser.parse_args()

    input_file = args.input_file.resolve()
    output_dir = args.output_dir.resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Attach the processed NetCDF as the raprompro dataset used by the analysis
    # and plotting methods below.
    mrr = MRRProData(path=input_file, ds=xr.Dataset())
    mrr.load_raprompro(input_file)
    try:
        ds = mrr.raprompro
        assert ds is not None

        t0 = ds["time"].values[0]
        t1 = ds["time"].values[-1]
        period = (
            pd.Timestamp(t0).floor("s").to_pydatetime(),
            pd.Timestamp(t1).floor("s").to_pydatetime(),
        )

        print(f"Input file : {input_file}")
        print(f"Period     : {t0} -> {t1}")
        print(f"Output dir : {output_dir}")

        for layer in DEFAULT_LAYERS:
            layer_output_dir = output_dir / _layer_dir_name(layer)
            layer_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"[layer] {layer[0]:.0f}-{layer[1]:.0f} m")

            analysis = mrr.rain_process_analyze(
                period=period,
                layer=layer,
                k=args.k,
                ze_th=args.ze_th,
                min_points_ols=args.min_points_ols,
                eps_q=args.eps_q,
                rgb_q=args.rgb_q,
                vars_trend=("Dm", "Nw", "LWC"),
            )
            classified = mrr.classify_rain_process(analysis=analysis)

            fig_2d, path_2d = mrr.plot.rain.layer_2d(
                target_datetime=period,
                layer=layer,
                x="Dm",
                y="Nw",
                z="LWC",
                savefig=True,
                marker_size=100,
                figsize=(12, 10),
                cmap="seismic",
                output_dir=layer_output_dir,
            )
            fig_2d.clf()
            compact_2d = layer_output_dir / _compact_name(
                start_time=period[0],
                layer=layer,
                kind="2d",
            )
            if path_2d is not None and path_2d != compact_2d:
                shutil.move(str(path_2d), str(compact_2d))
                path_2d = compact_2d
            print(f"  [ok] 2D         -> {path_2d}")

            fig_hex, path_hex = mrr.plot.rain.layer_hexagram(
                analysis=analysis,
                savefig=True,
                output_dir=layer_output_dir,
                dpi=200,
                alpha_hexagram=0.5,
                cmap="viridis",
            )
            fig_hex.clf()
            compact_hex = layer_output_dir / _compact_name(
                start_time=period[0],
                layer=layer,
                kind="hex",
            )
            if path_hex is not None and path_hex != compact_hex:
                shutil.move(str(path_hex), str(compact_hex))
                path_hex = compact_hex
            print(f"  [ok] hexagram   -> {path_hex}")

            fig_sum, path_sum = mrr.plot.rain.evolution(
                classified=classified,
                analysis=analysis,
                savefig=True,
                output_dir=layer_output_dir,
                figsize=(14, 10),
                cmap="viridis",
                alpha_hexagram=0.5,
                markersize=40.0,
                line_width=0.8,
                dpi=200,
            )
            fig_sum.clf()
            compact_sum = layer_output_dir / _compact_name(
                start_time=period[0],
                layer=layer,
                kind="summary",
            )
            if path_sum is not None and path_sum != compact_sum:
                shutil.move(str(path_sum), str(compact_sum))
                path_sum = compact_sum
            print(f"  [ok] summary    -> {path_sum}")
    finally:
        mrr.close()


if __name__ == "__main__":
    main()
