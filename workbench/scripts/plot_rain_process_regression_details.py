from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from mrrpropy import MRRProData

matplotlib.use("Agg")


DEFAULT_INPUT_FILE = Path(
    r"workbench/output/raprompro/2025/10/29/20251029_190000_raprompro.nc"
)
DEFAULT_OUTPUT_DIR = Path(
    r"workbench/output/rain_regression_details/2025/10/29/190000"
)
DEFAULT_LAYER = (1000.0, 2000.0)
DEFAULT_LABEL = "activation"
DEFAULT_VARS = ("Dm", "Nw", "LWC")


def _parse_datetime_or_none(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def _slugify_timestamp(value: pd.Timestamp) -> str:
    return value.strftime("%Y%m%d_%H%M%S")


def _build_summary_rows(
    analysis: xr.Dataset,
    classified: xr.Dataset,
    *,
    selected_indices: np.ndarray,
    vars_trend: tuple[str, str, str],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    times = pd.to_datetime(classified["time"].values)

    for index in selected_indices:
        row: dict[str, float | int | str] = {
            "time": times[index].isoformat(),
            "proc_label": str(classified["proc_label"].isel(time=index).item()),
            "strength": float(classified["strength"].isel(time=index).item()),
            "R": float(classified["R"].isel(time=index).item()),
            "G": float(classified["G"].isel(time=index).item()),
            "B": float(classified["B"].isel(time=index).item()),
            "sign_R": int(classified["sign_R"].isel(time=index).item()),
            "sign_G": int(classified["sign_G"].isel(time=index).item()),
            "sign_B": int(classified["sign_B"].isel(time=index).item()),
            "hex_x": float(classified["hex_x"].isel(time=index).item()),
            "hex_y": float(classified["hex_y"].isel(time=index).item()),
        }

        for variable_name in vars_trend:
            row[f"b_{variable_name}"] = float(
                analysis[f"b_{variable_name}"].isel(time=index).item()
            )
            row[f"a_{variable_name}"] = float(
                analysis[f"a_{variable_name}"].isel(time=index).item()
            )
            row[f"r2_{variable_name}"] = float(
                analysis[f"r2_{variable_name}"].isel(time=index).item()
            )
            row[f"F_{variable_name}"] = float(
                analysis[f"F_{variable_name}"].isel(time=index).item()
            )
            row[f"eps_{variable_name}"] = float(
                analysis[f"eps_{variable_name}"].isel(time=index).item()
            )
            row[f"n_fit_{variable_name}"] = int(
                analysis[f"n_fit_{variable_name}"].isel(time=index).item()
            )

        rows.append(row)

    return rows


def _plot_regression_detail(
    mrr: MRRProData,
    analysis: xr.Dataset,
    classified: xr.Dataset,
    *,
    index: int,
    layer: tuple[float, float],
    vars_trend: tuple[str, str, str],
    output_dir: Path,
) -> Path:
    ds = mrr.raprompro
    if ds is None:
        raise RuntimeError("raprompro not loaded.")

    timestamp = pd.Timestamp(classified["time"].isel(time=index).item())
    profile = ds.sel(time=timestamp.to_datetime64(), method="nearest").sel(
        range=slice(*layer)
    )

    depth = analysis["depth"].values.astype(float)
    strength = float(classified["strength"].isel(time=index).item())
    label = str(classified["proc_label"].isel(time=index).item())

    fig, axes = plt.subplots(
        ncols=len(vars_trend),
        figsize=(5.6 * len(vars_trend), 5.5),
        constrained_layout=True,
        sharex=True,
        sharey=False,
    )
    if len(vars_trend) == 1:
        axes = [axes]

    for ax, variable_name in zip(axes, vars_trend, strict=True):
        values = profile[variable_name].values.astype(float)
        fit_mask = (
            analysis[f"mask_fit_{variable_name}"].isel(time=index).values.astype(bool)
        )
        valid_mask = np.isfinite(values) & (values > 0.0)
        eps_value = float(analysis[f"eps_{variable_name}"].isel(time=index).item())

        display_eps = eps_value if np.isfinite(eps_value) and eps_value > 0 else np.nan
        if not np.isfinite(display_eps):
            positive_values = values[valid_mask]
            display_eps = float(np.min(positive_values)) if positive_values.size else 1.0

        if np.any(valid_mask):
            ax.scatter(
                depth[valid_mask],
                np.log(np.maximum(values[valid_mask], display_eps)),
                s=40,
                facecolors="white",
                edgecolors="0.55",
                linewidths=0.8,
                alpha=0.95,
                label="valid",
                zorder=2,
            )

        if np.any(fit_mask):
            ax.scatter(
                depth[fit_mask],
                np.log(np.maximum(values[fit_mask], display_eps)),
                s=55,
                facecolors="#1f77b4",
                edgecolors="black",
                linewidths=0.5,
                alpha=0.95,
                label="used in fit",
                zorder=3,
            )

        b_value = float(analysis[f"b_{variable_name}"].isel(time=index).item())
        a_value = float(analysis[f"a_{variable_name}"].isel(time=index).item())
        r2_value = float(analysis[f"r2_{variable_name}"].isel(time=index).item())
        n_fit = int(analysis[f"n_fit_{variable_name}"].isel(time=index).item())

        if np.isfinite(b_value) and np.isfinite(a_value):
            x_line = np.linspace(depth.min(), depth.max(), 200)
            y_line = a_value + b_value * x_line
            ax.plot(
                x_line,
                y_line,
                color="#d62728",
                linewidth=2.0,
                label="OLS fit",
                zorder=4,
            )

        ax.set_title(variable_name)
        ax.set_xlabel("Depth from layer top (m)")
        ax.set_ylabel(f"log({variable_name})")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.axvline(0.0, color="0.3", linestyle=":", linewidth=0.8)
        ax.text(
            0.03,
            0.97,
            (
                f"b = {b_value:.4f}\n"
                f"a = {a_value:.3f}\n"
                f"R^2 = {r2_value:.3f}\n"
                f"n_fit = {n_fit}\n"
                f"eps = {display_eps:.3g}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.75"},
        )
        ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        (
            f"{label.upper()} | layer {int(layer[0])}-{int(layer[1])} m | "
            f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | strength={strength:.2f}"
        ),
        fontsize=16,
    )

    output_path = output_dir / (
        f"{_slugify_timestamp(timestamp)}_{int(layer[0])}-{int(layer[1])}_{label}_ols.png"
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot detailed OLS regression diagnostics for selected classified rain-process "
            "times within a processed *_raprompro.nc file."
        )
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
        help="Directory where CSV summaries and regression PNGs will be written.",
    )
    parser.add_argument(
        "--layer-top",
        type=float,
        default=DEFAULT_LAYER[0],
        help="Top of the analysed layer in metres.",
    )
    parser.add_argument(
        "--layer-base",
        type=float,
        default=DEFAULT_LAYER[1],
        help="Base of the analysed layer in metres.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=DEFAULT_LABEL,
        help="Process label to inspect, for example activation or evaporation.",
    )
    parser.add_argument(
        "--window-start",
        type=str,
        default=None,
        help="Optional start time filter, e.g. 2025-10-29T19:30:00.",
    )
    parser.add_argument(
        "--window-end",
        type=str,
        default=None,
        help="Optional end time filter, e.g. 2025-10-29T19:35:00.",
    )
    parser.add_argument(
        "--max-times",
        type=int,
        default=None,
        help="Optional maximum number of matching times to plot.",
    )
    parser.add_argument("--k", type=int, default=11, help="Hexagram resolution.")
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
        help="Minimum valid points for each OLS fit.",
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
    layer = (float(args.layer_top), float(args.layer_base))
    target_label = args.label.strip()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if layer[1] <= layer[0]:
        raise ValueError("--layer-base must be greater than --layer-top.")

    output_dir.mkdir(parents=True, exist_ok=True)

    mrr = MRRProData(path=input_file, ds=xr.Dataset())
    mrr.load_raprompro(input_file)
    try:
        ds = mrr.raprompro
        if ds is None:
            raise RuntimeError("raprompro not loaded.")

        full_period = (
            pd.Timestamp(ds["time"].values[0]).to_pydatetime(),
            pd.Timestamp(ds["time"].values[-1]).to_pydatetime(),
        )
        analysis = mrr.rain_process_analyze(
            period=full_period,
            layer=layer,
            k=args.k,
            ze_th=args.ze_th,
            min_points_ols=args.min_points_ols,
            eps_q=args.eps_q,
            rgb_q=args.rgb_q,
            vars_trend=DEFAULT_VARS,
        )
        classified = mrr.classify_rain_process(analysis=analysis)

        times = pd.to_datetime(classified["time"].values)
        labels = classified["proc_label"].values.astype(str)
        mask = labels == target_label

        window_start = _parse_datetime_or_none(args.window_start)
        window_end = _parse_datetime_or_none(args.window_end)
        if window_start is not None:
            mask &= times >= window_start
        if window_end is not None:
            mask &= times <= window_end

        selected_indices = np.where(mask)[0]
        if args.max_times is not None and args.max_times > 0:
            selected_indices = selected_indices[: args.max_times]

        if selected_indices.size == 0:
            raise RuntimeError(
                "No matching times found for the selected label/window. "
                "Try broadening the time window or changing --label."
            )

        summary_rows = _build_summary_rows(
            analysis,
            classified,
            selected_indices=selected_indices,
            vars_trend=DEFAULT_VARS,
        )
        summary_df = pd.DataFrame(summary_rows)

        label_slug = target_label.replace(" ", "_")
        csv_path = output_dir / (
            f"{label_slug}_{int(layer[0])}-{int(layer[1])}_summary.csv"
        )
        summary_df.to_csv(csv_path, index=False)

        print(f"Input file : {input_file}")
        print(f"Layer      : {layer[0]:.0f}-{layer[1]:.0f} m")
        print(f"Label      : {target_label}")
        print(f"Selected   : {selected_indices.size} times")
        print(f"Summary CSV: {csv_path}")

        for index in selected_indices:
            output_path = _plot_regression_detail(
                mrr,
                analysis,
                classified,
                index=int(index),
                layer=layer,
                vars_trend=DEFAULT_VARS,
                output_dir=output_dir,
            )
            print(f"[ok] {output_path}")
    finally:
        mrr.close()


if __name__ == "__main__":
    main()
