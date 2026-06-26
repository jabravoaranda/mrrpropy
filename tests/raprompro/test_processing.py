from __future__ import annotations

import csv
import os
from collections.abc import Iterator
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")

pytestmark = [pytest.mark.slow, pytest.mark.integration]

KEY_REGRESSION_PAIRS = [
    ("Ze", "Ze", "dBZ"),
    ("RR", "RR", "mm/hr"),
    ("LWC", "LWC", "g/m3"),
    ("VEL", "W", "m/s"),
]


def _align_2d(a: xr.DataArray, b: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    if "range" in a.dims and "height" in b.dims:
        b = b.rename({"height": "range"})
    if "height" in a.dims and "range" in b.dims:
        a = a.rename({"height": "range"})
    try:
        a2, b2 = xr.align(a, b, join="inner")
    except Exception:
        a2 = a
        b2 = b
    return a2, b2


def _stats(x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return None
    xx = x[mask].astype(float)
    yy = y[mask].astype(float)
    bias = float(np.mean(yy - xx))
    rmse = float(np.sqrt(np.mean((yy - xx) ** 2)))
    corr = float(np.corrcoef(xx, yy)[0, 1]) if xx.size > 1 else np.nan
    slope, intercept = np.polyfit(xx, yy, 1)
    return {
        "n": int(mask.sum()),
        "bias": bias,
        "rmse": rmse,
        "corr": corr,
        "slope": float(slope),
        "intercept": float(intercept),
    }


@pytest.fixture(scope="session")
def raw_dataset(raw_dataset_path: Path) -> Iterator[xr.Dataset]:
    ds = xr.open_dataset(raw_dataset_path)
    yield ds
    ds.close()


@pytest.fixture(scope="session")
def generated_raprompro_dataset(generated_raprompro_path: Path) -> Iterator[xr.Dataset]:
    ds = xr.open_dataset(generated_raprompro_path)
    yield ds
    ds.close()


def test_processing_raprompro_smoke(generated_raprompro_path: Path) -> None:
    assert generated_raprompro_path.exists()
    assert generated_raprompro_path.suffix == ".nc"
    assert generated_raprompro_path.stat().st_size > 0


def test_processing_raprompro_from_raw(generated_raprompro_path: Path) -> None:
    if os.getenv("MRRPRO_FORCE_REPROCESS", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        pytest.skip(
            "Set MRRPRO_FORCE_REPROCESS=1 to force regeneration from RAW in this test."
        )

    assert generated_raprompro_path.exists()
    assert generated_raprompro_path.suffix == ".nc"
    assert generated_raprompro_path.stat().st_size > 0


def test_process_raprompro_regression(
    raw_dataset: xr.Dataset,
    generated_raprompro_dataset: xr.Dataset,
    artifact_dir: Path,
) -> None:
    rows: list[dict[str, float | int | str]] = []

    for v0, v1, units in KEY_REGRESSION_PAIRS:
        if v0 not in raw_dataset or v1 not in generated_raprompro_dataset:
            pytest.fail(
                f"Missing regression variable pair {v0}->{v1} in generated output"
            )

        raw_var, generated_var = _align_2d(
            raw_dataset[v0], generated_raprompro_dataset[v1]
        )

        x = raw_var.values.ravel()
        y = generated_var.values.ravel()

        if v1 == "W":
            x = -x
        if v1 == "DBPIA":
            y = np.abs(y)

        stats = _stats(x, y)
        if stats is None:
            pytest.fail(f"Not enough valid points for regression check {v0}->{v1}")

        stats.update({"var_before": v0, "var_after": v1, "units": units})
        rows.append(stats)

        assert stats["corr"] > 0.6, f"Low correlation for {v0}: {stats['corr']}"

    with open(artifact_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "var_before",
                "var_after",
                "units",
                "n",
                "bias",
                "rmse",
                "corr",
                "slope",
                "intercept",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})


def test_process_raprompro_generated_metadata(
    raw_subset_10min_mrr,
) -> None:
    ds = raw_subset_10min_mrr.process_raprompro(
        save=False,
        save_spe_3d=True,
        save_dsd_3d=True,
    )

    missing_long_name = [
        variable_name
        for variable_name in ds.data_vars
        if not ds[variable_name].attrs.get("long_name")
    ]
    missing_units = [
        variable_name
        for variable_name in ds.data_vars
        if not ds[variable_name].attrs.get("units")
    ]

    assert missing_long_name == []
    assert missing_units == []
    assert ds.attrs["velocity_convention"].startswith(
        "Public RaProMPro velocity outputs use negative-downward"
    )
    assert ds["W"].attrs["positive"] == "up"
    assert "negative" in ds["W"].attrs["convention"]
    assert np.nanmedian(ds["W"].values) < 0.0
    assert ds["speed"].attrs["positive"] == "up"
    assert np.all(np.diff(ds["speed"].values) > 0.0)


@pytest.mark.plot
def test_process_raprompro_generated_visual_comparison(
    raw_dataset: xr.Dataset,
    generated_raprompro_dataset: xr.Dataset,
    artifact_dir: Path,
) -> None:
    for v0, v1, units in KEY_REGRESSION_PAIRS:
        if v0 not in raw_dataset or v1 not in generated_raprompro_dataset:
            continue

        raw_var, generated_var = _align_2d(
            raw_dataset[v0], generated_raprompro_dataset[v1]
        )
        x = raw_var.values.ravel()
        y = generated_var.values.ravel()
        if v1 == "W":
            x = -x

        idx = np.where(np.isfinite(x) & np.isfinite(y))[0]
        if idx.size < 10:
            continue
        if idx.size > 200_000:
            idx = np.random.default_rng(0).choice(idx, size=200_000, replace=False)

        xx = x[idx].astype(float)
        yy = y[idx].astype(float)
        stats = _stats(xx, yy)
        if stats is None:
            continue

        fig = plt.figure()
        plt.scatter(xx, yy, s=1)
        lo = np.nanpercentile(np.concatenate([xx, yy]), 1)
        hi = np.nanpercentile(np.concatenate([xx, yy]), 99)
        plt.plot([lo, hi], [lo, hi])
        plt.xlabel(f"{v0} (raw) [{units}]")
        plt.ylabel(f"{v1} (generated) [{units}]")
        plt.title(
            f"generated {v0} vs {v1} | n={stats['n']} bias={stats['bias']:.3g} rmse={stats['rmse']:.3g} r={stats['corr']:.3f}",
            fontsize=12,
        )
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        fig.savefig(
            artifact_dir / f"generated_correlation_check_{v0}_vs_{v1}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


