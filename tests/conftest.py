from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import xarray as xr

from mrrpropy.raw_class import MRRProData


RAW_FIXTURE_PATH = Path(
    r"./tests/data/RAW/mrrpro81/2025/10/29/20251029_192300_10min.nc"
)
PRODUCTS_ROOT = Path(r"./tests/data/PRODUCTS")


def _sanitize_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "artifacts"


def _request_relative_artifact_name(request: pytest.FixtureRequest) -> str:
    test_path = Path(str(request.node.fspath)).resolve()
    tests_root = Path(__file__).resolve().parent
    try:
        relative = test_path.relative_to(tests_root)
    except ValueError:
        relative = Path(test_path.name)

    relative_no_suffix = relative.with_suffix("")
    parts = [_sanitize_path_part(part) for part in relative_no_suffix.parts]
    return str(Path(*parts))


def _raw_to_generated_product_path(raw_path: Path, products_root: Path) -> Path:
    raw_resolved = raw_path.resolve()
    raw_root = Path("./tests/data/RAW").resolve()
    try:
        relative = raw_resolved.relative_to(raw_root)
    except ValueError:
        relative = Path(raw_resolved.name)

    product_relative = relative.with_suffix("")
    return (
        products_root
        / product_relative.parent
        / f"{product_relative.name}_raprompro.nc"
    )


def _env_truthy(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _is_valid_generated_product(path: Path) -> bool:
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


@pytest.fixture(scope="session")
def raw_dataset_path() -> Path:
    raw_path = Path(os.getenv("MRRPRO_RAW_DATA_PATH", str(RAW_FIXTURE_PATH)))
    if not raw_path.exists():
        pytest.skip(f"Missing raw fixture file: {raw_path}")
    return raw_path


@pytest.fixture(scope="session")
def raw_subset_10min_path(raw_dataset_path: Path) -> Path:
    return raw_dataset_path


@pytest.fixture(scope="session")
def figure_root() -> Path:
    root = Path(os.getenv("MRRPRO_TEST_FIGURES", "./tests/figures")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def artifact_dir(figure_root: Path, request: pytest.FixtureRequest) -> Path:
    path = figure_root / _request_relative_artifact_name(request)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def generated_root() -> Path:
    root = Path(os.getenv("MRRPRO_TEST_GENERATED", "./tests/generated")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def generated_dir(generated_root: Path, request: pytest.FixtureRequest) -> Path:
    path = generated_root / _request_relative_artifact_name(request)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def raw_mrr(raw_dataset_path: Path) -> Iterator[MRRProData]:
    mrr = MRRProData.from_file(raw_dataset_path)
    yield mrr
    mrr.close()


@pytest.fixture(scope="session")
def raprompro_loaded_mrr(
    raw_dataset_path: Path,
    generated_raprompro_path: Path,
) -> Iterator[MRRProData]:
    mrr = MRRProData.from_file(raw_dataset_path)
    mrr.load_raprompro(generated_raprompro_path)
    yield mrr
    mrr.close()


@pytest.fixture(scope="session")
def raw_subset_10min_mrr(raw_mrr: MRRProData) -> MRRProData:
    return raw_mrr


@pytest.fixture(scope="session")
def raprompro_subset_10min_loaded_mrr(
    raprompro_loaded_mrr: MRRProData,
) -> MRRProData:
    return raprompro_loaded_mrr


@pytest.fixture(scope="session")
def generated_raprompro_path(
    raw_dataset_path: Path,
) -> Path:
    products_root = Path(
        os.getenv("MRRPRO_GENERATED_PRODUCT_ROOT", str(PRODUCTS_ROOT))
    ).resolve()
    output_path = _raw_to_generated_product_path(raw_dataset_path, products_root)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    force_reprocess = _env_truthy("MRRPRO_FORCE_REPROCESS")

    if not force_reprocess and _is_valid_generated_product(output_path):
        return output_path

    mrr = MRRProData.from_file(raw_dataset_path)
    try:
        out = mrr.process_raprompro(
            save_dsd_3d=True,
            save_spe_3d=True,
            save=True,
            output_dir=output_dir,
        )
        out.close()
    finally:
        mrr.close()

    if not output_path.exists():
        raise FileNotFoundError(
            f"Expected generated file was not created: {output_path}"
        )
    return output_path
