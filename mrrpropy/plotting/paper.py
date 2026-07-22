from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from mrrpropy.plotting import _spectra
from mrrpropy.rain_process_classification.rain_process_algorithm import (
    sliding_rain_classification_to_dataframe,
)
from mrrpropy.rain_process_classification.rain_process_info import (
    PROCESS_CODES,
    PROCESS_MARKERS,
)


class SupportsPaperPlotting(_spectra.SupportsSpectralAccess, Protocol):
    path: str | Path
    raprompro: xr.Dataset | None
    plot_cfg: Any

    def _is_processed(self) -> bool: ...


PROCESS_COLORS: dict[str, str] = {
    "breakup": "#12af54",
    "growth_depletion": "#1b9e77",
    "growth_depletion_gain": "#f808d0",
    "growth_depletion_loss": "#ff0000",
    "evaporation": "#000000",
    "growth": "#91209b",
    "activation": "#66a61e",
    "steady_or_weak": "#8f8f8f",
    "unknown": "#666666",
    "no_data": "#bdbdbd",
}

VARIABLE_LABELS: dict[str, str] = {
    "Dm": r"$D_m$ [mm]",
    "Dw": r"$D_w$ [mm]",
    "Nw": r"$\log_{10}(N_w)$ [mm$^{-1}$ m$^{-3}$]",
    "N": r"$N$ [m$^{-3}$ mm$^{-1}$]",
    "LWC": r"LWC [g m$^{-3}$]",
    "LWC_all": r"LWC [g m$^{-3}$]",
    "V": r"$V$ [m s$^{-1}$]",
    "W": r"$W$ [m s$^{-1}$]",
    "VEL": r"$V$ [m s$^{-1}$]",
    "delta_v_mean": r"$\Delta v$ [m s$^{-1}$]",
    "bb_distance_m": "BB distance [m]",
    "BB_distance_m": "BB distance [m]",
    "Ze": r"$Z_e$ [dBZ]",
}


def _paper_grid(ax: Axes) -> None:
    ax.set_facecolor("#fbfbfb")
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.55, alpha=0.8)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.35, alpha=0.7)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.8)


def _save(fig: Figure, path: Path | None, dpi: int) -> Path | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _to_dataframe(sliding_df: xr.Dataset | pd.DataFrame) -> pd.DataFrame:
    attrs = dict(getattr(sliding_df, "attrs", {}))
    if isinstance(sliding_df, xr.Dataset):
        df = sliding_rain_classification_to_dataframe(sliding_df)
    elif isinstance(sliding_df, pd.DataFrame):
        df = sliding_df.copy()
    else:
        raise TypeError("sliding_df must be an xr.Dataset or pandas DataFrame.")
    df.attrs.update(attrs)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    if "proc_label" in df.columns:
        df["proc_label"] = df["proc_label"].astype(str)
    return df


def _process_order(labels: pd.Series | np.ndarray) -> list[str]:
    preferred = [
        "activation",
        "growth",
        "growth_depletion_gain",
        "growth_depletion",
        "growth_depletion_loss",
        "breakup",
        "evaporation",
        "steady_or_weak",
        "unknown",
        "no_data",
    ]
    present = [str(label) for label in pd.unique(labels) if str(label)]
    return [label for label in preferred if label in present] + [
        label for label in present if label not in preferred
    ]


def _resolve_var(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise KeyError(f"None of these columns were found: {', '.join(candidates)}")


def _approximate_kde_domain(
    frame: pd.DataFrame,
    *,
    value_col: str,
    process_col: str,
    sigma: float | None,
    fallback_quantiles: tuple[float, float],
) -> tuple[float, float]:
    """Estimate a readable domain from per-process means and spread."""
    values = pd.to_numeric(frame[value_col], errors="coerce")
    finite = frame.loc[np.isfinite(values), [process_col]].copy()
    finite[value_col] = values.loc[finite.index]
    if finite.empty:
        return (0.0, 1.0)

    quantile_domain = finite[value_col].quantile(fallback_quantiles).to_numpy(dtype=float)
    if sigma is None:
        lower, upper = quantile_domain
    else:
        bounds: list[tuple[float, float]] = []
        for _, group in finite.groupby(process_col, sort=False)[value_col]:
            if len(group) < 2:
                bounds.append((float(group.iloc[0]), float(group.iloc[0])))
                continue
            mean = float(group.mean())
            spread = float(group.std(ddof=1))
            if np.isfinite(mean) and np.isfinite(spread):
                bounds.append((mean - sigma * spread, mean + sigma * spread))
        lower = min(bound[0] for bound in bounds) if bounds else quantile_domain[0]
        upper = max(bound[1] for bound in bounds) if bounds else quantile_domain[1]
        # Keep an extreme group from expanding the domain through a tiny
        # population of pathological values.
        if np.isfinite(quantile_domain[0]):
            lower = max(lower, float(quantile_domain[0]))
        if np.isfinite(quantile_domain[1]):
            upper = min(upper, float(quantile_domain[1]))

    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower = float(finite[value_col].min())
        upper = float(finite[value_col].max())
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1.0)
        lower -= padding
        upper += padding
    return lower, upper


def plot_quicklook_comparison(
    subject: SupportsPaperPlotting,
    *,
    variable: str = "Ze",
    raw_variable: str | None = None,
    processed_variable: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (7.2, 3.2),
    y_limits: tuple[float, float] | None = None,
    show_legend: bool = False,
    savefig: bool = False,
    output_dir: Path | None = None,
    filename: str | None = None,
    dpi: int | None = None,
) -> tuple[Figure, np.ndarray, Path | None]:
    """Paper-ready side-by-side raw vs processed quicklook without colorbars."""
    raw_name = raw_variable or variable
    processed_name = processed_variable or variable
    if raw_name not in subject.ds:
        raise KeyError(f"Variable '{raw_name}' not found in raw dataset.")
    if subject.raprompro is None or processed_name not in subject.raprompro:
        raise KeyError(f"Variable '{processed_name}' not found in raprompro dataset.")

    arrays = [subject.ds[raw_name], subject.raprompro[processed_name]]
    fig, axes = plt.subplots(ncols=2, figsize=figsize, sharey=True, constrained_layout=True)
    for ax, da, label in zip(axes, arrays, ("Raw", "Processed")):
        mesh = cast(Any, da.plot)
        mesh(
            ax=ax,
            x="time",
            y="range",
            add_colorbar=False,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title("")
        ax.text(
            0.02,
            0.96,
            label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        )
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.set_xlabel("Time")
        _paper_grid(ax)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[0].set_ylabel("Range [m]")
    axes[1].set_ylabel("")
    if show_legend:
        axes[1].legend(
            handles=[
                Line2D([], [], color="black", linewidth=2, label="Raw"),
                Line2D([], [], color="black", linewidth=2, label="Processed"),
            ],
            loc="upper right", fontsize=7, frameon=True,
        )

    out = None
    if savefig:
        outdir = Path.cwd() if output_dir is None else Path(output_dir)
        stem = filename or f"{Path(subject.path).stem}_{variable}_raw_processed_quicklook.png"
        out = _save(fig, outdir / stem, dpi or subject.plot_cfg.dpi)
    return fig, axes, out


def plot_microphysical_properties_triple(
    subject: SupportsPaperPlotting,
    *,
    target_datetime: datetime | np.datetime64 | str,
    variables: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
    y_limits: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (6.6, 3.8),
    savefig: bool = False,
    output_dir: Path | None = None,
    filename: str | None = None,
    dpi: int | None = None,
    **kwargs: Any,
) -> tuple[Figure, np.ndarray, Path | None]:
    """Three-panel MPP profile figure, excluding reflectivity panels."""
    if subject.raprompro is None:
        raise RuntimeError("raprompro not loaded. Use load_raprompro().")
    ds = subject.raprompro
    missing = [name for name in variables if name not in ds]
    if missing:
        raise KeyError(f"raprompro missing variables: {missing}")

    profile = ds.sel(time=np.datetime64(target_datetime), method="nearest")
    heights_km = profile["range"].values.astype(float) / 1000.0
    fig, axes = plt.subplots(ncols=3, figsize=figsize, sharey=True, constrained_layout=True)
    colors = kwargs.get("colors", ("#0072B2", "#009E73", "#D55E00"))
    for ax, variable, color in zip(axes, variables, colors):
        ax.plot(
            profile[variable].values.astype(float),
            heights_km,
            color=color,
            linewidth=float(kwargs.get("linewidth", 1.5)),
            marker=kwargs.get("marker", "o"),
            markersize=float(kwargs.get("markersize", 2.8)),
        )
        ax.set_xlabel(VARIABLE_LABELS.get(variable, variable))
        ax.set_title("")
        _paper_grid(ax)
        axes[0].set_ylabel("Height [km]")
    if y_limits is not None:
        for ax in axes:
            ax.set_ylim(*y_limits)

    out = None
    if savefig:
        outdir = Path.cwd() if output_dir is None else Path(output_dir)
        time_tag = str(np.datetime_as_string(profile["time"].values, unit="s")).replace(":", "")
        stem = filename or f"{Path(subject.path).stem}_mpp_triple_{time_tag}.png"
        out = _save(fig, outdir / stem, dpi or subject.plot_cfg.dpi)
    return fig, axes, out


def plot_single_column_events(
    sliding_df: xr.Dataset | pd.DataFrame,
    *,
    target_datetime: datetime | np.datetime64 | str,
    range_col: str = "range",
    figsize: tuple[float, float] = (2.4, 4.0),
    savefig: bool = False,
    output_dir: Path | None = None,
    filename: str = "single_column_events.png",
    dpi: int = 200,
) -> tuple[Figure, Axes, Path | None]:
    """Plot one classified process column at the time nearest target_datetime."""
    df = _to_dataframe(sliding_df)
    required = {"time", range_col, "proc_label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"sliding_df missing columns: {missing}")
    target = pd.Timestamp(target_datetime)
    nearest = df.loc[(df["time"] - target).abs().idxmin(), "time"]
    column = df[df["time"] == nearest].copy()
    column[range_col] = pd.to_numeric(column[range_col], errors="coerce")
    column = column[np.isfinite(column[range_col])].sort_values(range_col)
    if column.empty:
        raise ValueError("No finite column data available at the selected time.")

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    labels = _process_order(column["proc_label"])
    for label in labels:
        group = column[column["proc_label"] == label]
        ax.scatter(
            np.ones(len(group)),
            group[range_col].to_numpy(dtype=float) / 1000.0,
            s=52,
            marker=PROCESS_MARKERS.get(label, "o"),
            color=PROCESS_COLORS.get(label, "#333333"),
            edgecolors="none",
            label=label,
        )
    ax.set_xlim(0.7, 1.3)
    ax.set_xticks([])
    ax.set_ylabel("Height [km]")
    ax.set_title("")
    _paper_grid(ax)
    ax.legend(loc="best", fontsize=7, frameon=True)

    out = _save(fig, (Path.cwd() if output_dir is None else Path(output_dir)) / filename, dpi) if savefig else None
    return fig, ax, out


def plot_microphysical_tau_triple(
    sliding_df: xr.Dataset | pd.DataFrame,
    *,
    target_datetime: datetime | np.datetime64 | str,
    variables: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
    score_prefix: str = "tau",
    figsize: tuple[float, float] = (10.0, 6.0),
    y_limits: tuple[float, float] | None = None,
    savefig: bool = False,
    output_dir: Path | None = None,
    filename: str = "mpp_tau_triple.png",
    dpi: int = 300,
) -> tuple[Figure, np.ndarray, Path | None]:
    """Three MPP profiles colored point-by-point by Kendall/Thiel score."""
    df = _to_dataframe(sliding_df)
    if "time" not in df.columns:
        raise KeyError("sliding_df must contain 'time'.")
    target = pd.Timestamp(target_datetime)
    nearest = df.loc[(df["time"] - target).abs().idxmin(), "time"]
    column = df[df["time"] == nearest].copy()
    column["range"] = pd.to_numeric(column["range"], errors="coerce")
    column = column[np.isfinite(column["range"])].sort_values("range")
    if column.empty:
        raise ValueError("No finite column data available at the selected time.")

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, constrained_layout=True)
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(-1.0, 1.0)
    for ax, variable in zip(axes, variables):
        value_col = f"{variable}_top"
        score_col = f"{score_prefix}_{variable}"
        if value_col not in column or score_col not in column:
            raise KeyError(f"Missing '{value_col}' or '{score_col}' for tau MPP plot.")
        values = pd.to_numeric(column[value_col], errors="coerce").to_numpy(float)
        scores = pd.to_numeric(column[score_col], errors="coerce").to_numpy(float)
        height = column["range"].to_numpy(float) / 1000.0
        valid = np.isfinite(values) & np.isfinite(scores) & np.isfinite(height)
        ax.plot(values[valid], height[valid], color="#777777", linewidth=0.7, alpha=0.7)
        ax.scatter(
            values[valid], height[valid], c=np.clip(scores[valid], -1.0, 1.0),
            cmap=cmap, norm=norm, s=30, edgecolors="none",
        )
        ax.set_xlabel(VARIABLE_LABELS.get(variable, variable))
        _paper_grid(ax)
    axes[0].set_ylabel("Height [km]")
    if y_limits is not None:
        for ax in axes:
            ax.set_ylim(*y_limits)
    out = _save(fig, (Path.cwd() if output_dir is None else Path(output_dir)) / filename, dpi) if savefig else None
    return fig, axes, out


def plot_process_binary_points(
    data: xr.Dataset | pd.DataFrame,
    *,
    variables: tuple[str, str, str] = ("Dm", "Nw", "LWC"),
    figsize: tuple[float, float] = (6.5, 2.8),
    savefig: bool = False,
    output_dir: Path | None = None,
    filename: str = "process_binary_points.png",
    dpi: int = 200,
) -> tuple[Figure, Axes, Path | None]:
    """Plot 0/1 trend-sign indicators for each microphysical point."""
    if isinstance(data, xr.Dataset):
        df = data.to_dataframe().reset_index()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("data must be an xr.Dataset or pandas DataFrame.")
    if "time" not in df.columns:
        raise KeyError("data must contain a 'time' column or coordinate.")
    df["time"] = pd.to_datetime(df["time"])

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    offsets = np.arange(len(variables), dtype=float)
    variable_colors = {
        "Dm": "#0072B2",
        "Nw": "#009E73",
        "LWC": "#D55E00",
    }
    for offset, variable in zip(offsets, variables):
        sign_col = None
        for candidate in (
            f"trend_sign_{variable}",
            f"sign_{variable}",
            f"sign_{variable[0]}",
        ):
            if candidate in df.columns:
                sign_col = candidate
                break
        if sign_col is None:
            raise KeyError(f"No sign column found for {variable}.")
        values = pd.to_numeric(df[sign_col], errors="coerce").to_numpy(dtype=float)
        binary = np.where(values > 0, 1.0, 0.0)
        finite = np.isfinite(values)
        ax.scatter(
            df.loc[finite, "time"],
            binary[finite] + offset * 1.35,
            s=18,
            color=variable_colors.get(variable, "#0072B2"),
            edgecolors="none",
            label=variable,
        )
    ax.set_yticks(offsets * 1.35 + 0.5)
    ax.set_yticklabels(list(variables))
    ax.set_ylim(-0.25, offsets[-1] * 1.35 + 1.25)
    ax.set_xlabel("Time")
    ax.set_ylabel("Indicator")
    ax.set_title("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    _paper_grid(ax)

    out = _save(fig, (Path.cwd() if output_dir is None else Path(output_dir)) / filename, dpi) if savefig else None
    return fig, ax, out


def plot_column_process_scan_with_spectrogram(
    subject: SupportsPaperPlotting,
    *,
    sliding_df: xr.Dataset | pd.DataFrame,
    target_datetime: datetime | np.datetime64 | str,
    spectrum_var: str = "spe_3D",
    range_limits: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (10.0, 3.6),
    savefig: bool = False,
    output_dir: Path | None = None,
    filename: str = "column_process_scan_with_spectrogram.png",
    dpi: int | None = None,
) -> tuple[Figure, np.ndarray, Path | None]:
    """Original-process scan, hex-colored scan, and spectrogram side by side."""
    df = _to_dataframe(sliding_df)
    required = {"time", "range", "proc_label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"sliding_df missing columns: {missing}")

    fig, axes = plt.subplots(ncols=3, figsize=figsize, constrained_layout=True)
    labels = _process_order(df["proc_label"])
    for label in labels:
        group = df[df["proc_label"] == label]
        if group.empty:
            continue
        axes[0].scatter(
            group["time"],
            group["range"].to_numpy(dtype=float) / 1000.0,
            s=18,
            marker="s",
            color=PROCESS_COLORS.get(label, "#333333"),
            edgecolors="none",
            label=PROCESS_CODES.get(label, label),
        )
    axes[0].legend(loc="best", fontsize=7, frameon=True)

    if {"R", "G", "B"}.issubset(df.columns):
        rgb = df[["R", "G", "B"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite_rgb = np.isfinite(rgb).all(axis=1)
        axes[1].scatter(
            df.loc[finite_rgb, "time"],
            df.loc[finite_rgb, "range"].to_numpy(dtype=float) / 1000.0,
            s=18,
            marker="s",
            c=np.clip(rgb[finite_rgb], 0.0, 1.0),
            edgecolors="none",
        )
    else:
        for label in labels:
            group = df[df["proc_label"] == label]
            axes[1].scatter(
                group["time"],
                group["range"].to_numpy(dtype=float) / 1000.0,
                s=18,
                marker="s",
                color=PROCESS_COLORS.get(label, "#333333"),
                edgecolors="none",
            )

    for ax in axes[:2]:
        ax.set_xlabel("Time")
        ax.set_ylabel("Height [km]")
        ax.set_title("")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        _paper_grid(ax)

    t_sel, ranges, vel, spec2d, _units = _spectra.get_spectrogram_2d(
        subject,
        np.datetime64(target_datetime),
        spectrum_var=spectrum_var,
        range_limits=range_limits,
    )
    axes[2].imshow(
        spec2d,
        aspect="auto",
        extent=(float(vel[0]), float(vel[-1]), float(ranges[0]) / 1000.0, float(ranges[-1]) / 1000.0),
        origin="lower",
        cmap="viridis",
    )
    axes[2].axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel(r"Doppler velocity [m s$^{-1}$]")
    axes[2].set_ylabel("Height [km]")
    axes[2].set_title("")
    _paper_grid(axes[2])

    out = None
    if savefig:
        outdir = Path.cwd() if output_dir is None else Path(output_dir)
        out = _save(fig, outdir / filename, dpi or subject.plot_cfg.dpi)
    return fig, axes, out


def plot_column_process_scan(
    sliding_df: xr.Dataset | pd.DataFrame,
    *,
    color_mode: str = "rain_signature",
    figsize: tuple[float, float] = (10.0, 6.0),
    range_limits: tuple[float, float] | None = None,
    processes: list[str] | None = None,
    show_legend: bool = True,
    savefig: bool = False,
    output_dir: Path | None = None,
    filename: str = "column_process_scan.png",
    dpi: int = 300,
    marker_size: float = 52.0,
    label_fs: float = 13.0,
    tick_fs: float = 10.0,
) -> tuple[Figure, Axes, Path | None]:
    """Paper-ready process curtain, with optional RGB/hex coloring."""
    df = _to_dataframe(sliding_df)
    required = {"time", "range", "proc_label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"sliding_df missing columns: {missing}")
    df["range"] = pd.to_numeric(df["range"], errors="coerce")
    df = df[np.isfinite(df["range"])]
    if processes is not None:
        df = df[df["proc_label"].isin([str(value) for value in processes])]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    labels = _process_order(df["proc_label"])
    rgb_columns = ("R", "G", "B")
    if color_mode.lower() in {"hex", "rgb"} and not set(rgb_columns).issubset(df.columns):
        if {"tau_Dm", "tau_Nw", "tau_LWC"}.issubset(df.columns):
            df = df.copy()
            for channel, score_column in zip(rgb_columns, ("tau_Dm", "tau_Nw", "tau_LWC")):
                scores = pd.to_numeric(df[score_column], errors="coerce")
                df[channel] = 0.5 * (scores.clip(-1.0, 1.0) + 1.0)
    if color_mode.lower() in {"hex", "rgb"} and set(rgb_columns).issubset(df.columns):
        rgb = df[list(rgb_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = np.isfinite(rgb).all(axis=1)
        ax.scatter(
            df.loc[valid, "time"],
            df.loc[valid, "range"] / 1000.0,
            s=marker_size,
            marker="s",
            c=np.clip(rgb[valid], 0.0, 1.0),
            edgecolors="none",
        )
    else:
        for label in labels:
            group = df[df["proc_label"] == label]
            ax.scatter(
                group["time"],
                group["range"] / 1000.0,
                s=marker_size,
                marker=PROCESS_MARKERS.get(label, "o"),
                color=PROCESS_COLORS.get(label, "#333333"),
                edgecolors="none",
                label=PROCESS_CODES.get(label, label),
            )
    ax.set_xlabel("Time", fontsize=label_fs)
    ax.set_ylabel("Height [km]", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    if range_limits is not None:
        ax.set_ylim(range_limits[0] / 1000.0, range_limits[1] / 1000.0)
    if show_legend and color_mode.lower() not in {"hex", "rgb"} and labels:
        ax.legend(loc="best", fontsize=9, frameon=True)
    _paper_grid(ax)
    out = _save(fig, (Path.cwd() if output_dir is None else Path(output_dir)) / filename, dpi) if savefig else None
    return fig, ax, out


def plot_sliding_column_process_paper(
    subject: SupportsPaperPlotting,
    sliding_df: xr.Dataset | pd.DataFrame,
    *,
    processes: list[str] | None = None,
    figsize: tuple[float, float] = (14.0, 8.0),
    y_limits: tuple[float, float] | None = None,
    savefig: bool = False,
    output_dir: Path | None = None,
    filename: str = "sliding_column_process.png",
    dpi: int = 300,
) -> tuple[Figure, Axes, Path | None]:
    """Paper wrapper around the workflow's canonical CPS renderer."""
    fig, _ = subject.plot_sliding_column_process(
        sliding_df=sliding_df,
        processes=processes,
        color_mode="rain_signature",
        figsize=figsize,
        y_limits=y_limits,
        markersize=52.0,
        scale_by_strength=True,
        title_fs=14,
        label_fs=13,
        tick_fs=10,
        legend_fs=9,
        savefig=False,
    )
    ax = fig.axes[0]
    ax.set_title("")
    ax.set_ylabel("Height [km]")
    _paper_grid(ax)
    out = _save(fig, (Path.cwd() if output_dir is None else Path(output_dir)) / filename, dpi) if savefig else None
    return fig, ax, out


def plot_quicklook_cps_hex(
    subject: SupportsPaperPlotting,
    sliding_df: xr.Dataset | pd.DataFrame,
    *,
    variable: str = "Ze",
    processes: list[str] | None = None,
    vmin: float = -10.0,
    vmax: float = 40.0,
    range_limits: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (18.0, 6.0),
    savefig: bool = False,
    output_dir: Path | None = None,
    filename: str = "quicklook_cps_hex.png",
    dpi: int = 300,
) -> tuple[Figure, np.ndarray, Path | None]:
    """Three-panel quicklook, canonical CPS, and RGB/hex CPS view."""
    df = _to_dataframe(sliding_df)
    if processes is not None:
        df = df[df["proc_label"].isin([str(value) for value in processes])].copy()
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    if variable not in subject.raprompro:
        raise KeyError(f"Processed dataset missing '{variable}'.")
    quicklook = subject.raprompro[variable]
    if "range" in quicklook.coords:
        quicklook = quicklook.assign_coords(range=quicklook["range"] / 1000.0)
    cast(Any, quicklook.plot)(
        ax=axes[0], x="time", y="range", add_colorbar=False,
        cmap="viridis", vmin=vmin, vmax=vmax,
    )
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Height [km]")
    axes[0].set_ylim(*(value / 1000.0 for value in range_limits)) if range_limits else None
    axes[0].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    _paper_grid(axes[0])

    subject.plot_sliding_column_process(
        sliding_df=df,
        processes=processes,
        color_mode="rain_signature",
        ax=axes[1],
        markersize=52.0,
        scale_by_strength=True,
        title_fs=14,
        label_fs=13,
        tick_fs=10,
        legend_fs=9,
        show_legend=False,
    )
    axes[1].set_title("")
    axes[1].set_ylabel("")
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    _paper_grid(axes[1])

    rgb_columns = ("R", "G", "B")
    if not set(rgb_columns).issubset(df.columns) and {"tau_Dm", "tau_Nw", "tau_LWC"}.issubset(df.columns):
        df = df.copy()
        for channel, score_column in zip(rgb_columns, ("tau_Dm", "tau_Nw", "tau_LWC")):
            scores = pd.to_numeric(df[score_column], errors="coerce")
            df[channel] = 0.5 * (scores.clip(-1.0, 1.0) + 1.0)
    if set(rgb_columns).issubset(df.columns):
        rgb = df[list(rgb_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = np.isfinite(rgb).all(axis=1)
        axes[2].scatter(
            df.loc[valid, "time"], df.loc[valid, "range"] / 1000.0,
            s=20, marker="s", c=np.clip(rgb[valid], 0.0, 1.0), edgecolors="none",
        )
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("")
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    if range_limits:
        axes[1].set_ylim(*(value / 1000.0 for value in range_limits))
        axes[2].set_ylim(*(value / 1000.0 for value in range_limits))
    quicklook_times = pd.to_datetime(subject.raprompro["time"].values)
    scan_times = pd.to_datetime(df["time"])
    common_xlim = (min(quicklook_times.min(), scan_times.min()), max(quicklook_times.max(), scan_times.max()))
    for ax in axes:
        ax.set_xlim(*common_xlim)
        ax.tick_params(axis="both", labelsize=10)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Time", fontsize=13)
    axes[0].set_ylabel("Height [km]", fontsize=13)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    _paper_grid(axes[2])
    out = _save(fig, (Path.cwd() if output_dir is None else Path(output_dir)) / filename, dpi) if savefig else None
    return fig, axes, out


def plot_paper_spectrogram(
    subject: SupportsPaperPlotting,
    *,
    target_datetime: datetime | np.datetime64 | str,
    spectrum_var: str = "spe_3D",
    range_limits: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (10.0, 6.0),
    cmap: str = "viridis",
    savefig: bool = False,
    output_dir: Path | None = None,
    filename: str = "spectrogram.png",
    dpi: int = 300,
) -> tuple[Figure, Axes, Path | None]:
    """Paper-ready range-velocity spectrogram without a colorbar or title."""
    _t_sel, ranges, velocity, spectrum, _units = _spectra.get_spectrogram_2d(
        subject,
        np.datetime64(target_datetime),
        spectrum_var=spectrum_var,
        range_limits=range_limits,
    )
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.imshow(
        spectrum,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=(float(velocity[0]), float(velocity[-1]), float(ranges[0]) / 1000.0, float(ranges[-1]) / 1000.0),
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Doppler velocity [m s$^{-1}$]")
    ax.set_ylabel("Height [km]")
    _paper_grid(ax)
    out = _save(fig, (Path.cwd() if output_dir is None else Path(output_dir)) / filename, dpi) if savefig else None
    return fig, ax, out


def plot_process_kde_2x2(
    frame: pd.DataFrame,
    *,
    variables: tuple[str, str, str, str] = ("Dw", "V", "LWC", "N"),
    process_col: str = "proc_label",
    figsize: tuple[float, float] = (8.0, 6.0),
    clip_quantiles: tuple[float, float] = (0.005, 0.995),
    domain_sigma: float | None = 3.0,
    savefig: bool = False,
    output_dir: Path | None = None,
    filename: str = "process_kde_2x2.png",
    dpi: int = 200,
) -> tuple[Figure, np.ndarray, Path | None]:
    """2x2 KDE summary for large-folder process statistics."""
    if process_col not in frame.columns:
        raise KeyError(f"frame must contain '{process_col}'.")
    df = frame.copy()
    df[process_col] = df[process_col].astype(str)
    colors = {label: PROCESS_COLORS.get(label, "#333333") for label in _process_order(df[process_col])}

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, variable in zip(axes.flat, variables):
        aliases = {
            "Dw": ("Dw", "Dm", "Dm_top", "Dm_layer_mean", "Dm_mean"),
            "V": ("V", "W", "VEL", "v_mean_top", "v_mean_bottom", "V_layer_mean", "W_layer_mean", "VEL_layer_mean"),
            "LWC": ("LWC", "LWC_top", "LWC_bottom", "LWC_layer_mean", "LWC_mean"),
            "N": ("N", "Nw", "Nw_top", "Nw_bottom", "N_layer_mean", "Nw_layer_mean", "Nw_mean"),
        }
        column = _resolve_var(df, aliases.get(variable, (variable, f"{variable}_layer_mean", f"{variable}_mean")))
        plot_df = df[[process_col, column]].copy()
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
        plot_df = plot_df[np.isfinite(plot_df[column])]
        if plot_df.empty:
            continue
        lower, upper = _approximate_kde_domain(
            plot_df,
            value_col=column,
            process_col=process_col,
            sigma=domain_sigma,
            fallback_quantiles=clip_quantiles,
        )
        for process, color in colors.items():
            values = plot_df.loc[plot_df[process_col] == process, column]
            if values.nunique(dropna=True) < 2:
                continue
            sns.kdeplot(
                x=values,
                color=color,
                clip=(lower, upper),
                fill=False,
                linewidth=1.3,
                ax=ax,
                legend=False,
                warn_singular=False,
            )
        ax.set_xlim(lower, upper)
        ax.set_xlabel(VARIABLE_LABELS.get(variable, VARIABLE_LABELS.get(column, column)), fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_title("")
        _paper_grid(ax)

    fig.tight_layout()

    out = _save(fig, (Path.cwd() if output_dir is None else Path(output_dir)) / filename, dpi) if savefig else None
    return fig, axes, out


def plot_process_distance_velocity_kdes(
    frame: pd.DataFrame,
    *,
    bb_distance_col: str | None = None,
    delta_v_col: str | None = None,
    process_col: str = "proc_label",
    figsize: tuple[float, float] = (8.0, 3.4),
    clip_quantiles: tuple[float, float] = (0.005, 0.995),
    domain_sigma: float | None = 3.0,
    savefig: bool = False,
    output_dir: Path | None = None,
    filename: str = "process_bb_distance_delta_v_kdes.png",
    dpi: int = 200,
) -> tuple[Figure, np.ndarray, Path | None]:
    """KDE plots of bright-band distance and delta-v by process."""
    if process_col not in frame.columns:
        raise KeyError(f"frame must contain '{process_col}'.")
    df = frame.copy()
    df[process_col] = df[process_col].astype(str)
    bb_col = bb_distance_col or _resolve_var(
        df,
        ("bb_distance_m", "BB_distance_m", "dist_bb_peak", "dist_bb_bottom", "distance_to_bb_m", "z_center_minus_bb_m"),
    )
    dv_col = delta_v_col or _resolve_var(
        df,
        ("delta_v_mean", "delta_v_p50", "delta_v", "V_delta", "delta_V", "delta_velocity", "velocity_difference"),
    )
    colors = {label: PROCESS_COLORS.get(label, "#333333") for label in _process_order(df[process_col])}

    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    for ax, column, label in zip(
        axes,
        (bb_col, dv_col),
        (VARIABLE_LABELS.get(bb_col, bb_col), VARIABLE_LABELS.get(dv_col, dv_col)),
    ):
        plot_df = df[[process_col, column]].copy()
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
        plot_df = plot_df[np.isfinite(plot_df[column])]
        if plot_df.empty:
            continue
        lower, upper = _approximate_kde_domain(
            plot_df,
            value_col=column,
            process_col=process_col,
            sigma=domain_sigma,
            fallback_quantiles=clip_quantiles,
        )
        for process, color in colors.items():
            values = plot_df.loc[plot_df[process_col] == process, column]
            if values.nunique(dropna=True) < 2:
                continue
            sns.kdeplot(
                x=values,
                color=color,
                clip=(lower, upper),
                fill=False,
                linewidth=1.3,
                ax=ax,
                legend=False,
                warn_singular=False,
            )
        ax.set_xlim(lower, upper)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_title("")
        _paper_grid(ax)

    fig.tight_layout()

    out = _save(fig, (Path.cwd() if output_dir is None else Path(output_dir)) / filename, dpi) if savefig else None
    return fig, axes, out
