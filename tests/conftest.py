from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import xarray as xr

from mrrpropy import MRRProData


RAW_FIXTURE_PATH = Path("./tests/data/RAW/mrrpro81/2025/10/29/20251029_192300_10min.nc")
RAW_ROOT = Path("./tests/data/RAW")
PRODUCTS_ROOT = Path("./tests/data/PRODUCTS")


def _is_valid_raprompro_product(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 1024:
        return False
    try:
        ds = xr.open_dataset(path)
        ok = (
            "time" in ds.coords
            and ds.sizes.get("time", 0) > 0
            and len(ds.data_vars) > 0
            and str(ds.attrs.get("velocity_convention", "")).startswith(
                "Public RaProMPro velocity outputs use negative-downward"
            )
        )
        ds.close()
        return ok
    except Exception:
        return False


@pytest.fixture()
def product_dir(request: pytest.FixtureRequest) -> Path:
    tests_root = Path(__file__).resolve().parent
    test_path = Path(str(request.node.fspath)).resolve()
    try:
        relative = test_path.relative_to(tests_root).with_suffix("")
    except ValueError:
        relative = Path(test_path.stem)

    clean_parts = [
        re.sub(r"[^A-Za-z0-9_.-]+", "_", part).strip("._") or "products"
        for part in relative.parts
    ]
    path = PRODUCTS_ROOT.resolve() / Path(*clean_parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def raw_mrr() -> Iterator[MRRProData]:
    if not RAW_FIXTURE_PATH.exists():
        pytest.skip(f"Missing raw fixture file: {RAW_FIXTURE_PATH}")
    mrr = MRRProData.from_file(RAW_FIXTURE_PATH)
    yield mrr
    mrr.close()


@pytest.fixture(scope="session")
def generated_raprompro_path() -> Path:
    if not RAW_FIXTURE_PATH.exists():
        pytest.skip(f"Missing raw fixture file: {RAW_FIXTURE_PATH}")

    raw_root = RAW_ROOT.resolve()
    raw_resolved = RAW_FIXTURE_PATH.resolve()
    try:
        relative = raw_resolved.relative_to(raw_root)
    except ValueError:
        relative = Path(raw_resolved.name)

    product_relative = relative.with_suffix("")
    output_path = (
        PRODUCTS_ROOT.resolve()
        / product_relative.parent
        / f"{product_relative.name}_raprompro.nc"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_valid_raprompro_product(output_path):
        return output_path

    mrr = MRRProData.from_file(RAW_FIXTURE_PATH)
    try:
        out = mrr.process_raprompro(
            save_dsd_3d=True,
            save_spe_3d=True,
            save=True,
            output_dir=output_path.parent,
        )
        out.close()
    finally:
        mrr.close()

    if not output_path.exists():
        raise FileNotFoundError(
            f"Expected generated file was not created: {output_path}"
        )
    return output_path


@pytest.fixture(scope="session")
def raprompro_mrr(
    generated_raprompro_path: Path,
) -> Iterator[MRRProData]:
    if not RAW_FIXTURE_PATH.exists():
        pytest.skip(f"Missing raw fixture file: {RAW_FIXTURE_PATH}")
    mrr = MRRProData.from_file(RAW_FIXTURE_PATH)
    mrr.load_raprompro(generated_raprompro_path)
    yield mrr
    mrr.close()
