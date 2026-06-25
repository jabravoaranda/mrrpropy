# mrrpropy

MRR PRO code for processing and analysis.

## Scientific references

The retained RaProMPro processing implementation used for MRR-PRO data is associated
with:

- Garcia-Benadi A, Bech J, Gonzalez S, Udina M, Codina B. A New Methodology to
  Characterise the Radar Bright Band Using Doppler Spectral Moments from Vertically
  Pointing Radar Observations. Remote Sensing. 2021;13(21):4323.
  https://doi.org/10.3390/rs13214323

The corresponding original code repository is:

- https://github.com/AlbertGBena/RaProM-Pro

For MRR-2 data, a related implementation is distributed separately as `RaProM.py`:

- Garcia-Benadi A, Bech J, Gonzalez S, Udina M, Codina B, Georgis JF.
  Precipitation Type Classification of Micro Rain Radar Data Using an Improved
  Doppler Spectral Processing Methodology. Remote Sensing. 2020;12(24):4113.
  https://doi.org/10.3390/rs12244113

## Repository workflow

This repository keeps the scientific processing code intact while standardizing the
developer workflow around it.

- Install the project in editable mode with `uv sync --group dev` or `pip install -e .`.
- Import the package as `mrrpropy`.
- Use the optional CLI entry point as `mrrpropy version`.
- Keep scientific algorithm changes confined to the processing modules and treat
  workflow, packaging, tests, and CI as separate concerns.

## Rain-process trends

The microphysical rain-process workflow now defaults to a non-parametric vertical
trend characterization:

- Kendall's tau describes the direction and consistency of monotonic change in a layer.
- Theil-Sen slope describes the robust magnitude of that change.
- Downstream RGB and classification consume canonical `trend_*` variables instead
  of method-specific names, so the trend method can be swapped without changing
  the rest of the pipeline.
- OLS trend fitting remains available only as a legacy or diagnostic comparison path.

## Velocity sign convention

Public `mrrpropy` outputs use Doppler/fall velocity with negative values indicating
downward hydrometeor motion. In RaProMPro products this applies to `W` and to the
`speed` coordinate of `spe_3D`; spectral plotting and rain-process spectral
features use the same negative-downward convention. The retained RaProMPro
algorithm keeps its original positive-downward convention internally, and the sign
is converted only at the public output and plotting/feature boundary.

## Development

Typical local commands:

```bash
uv sync --group dev
uv run python -c "import mrrpropy"
uv run pytest -m "not slow"
uv run pytest -m slow
uv run mypy
uv run black --check mrrpropy/cli tests
uv run pre-commit install
uv run python scripts/benchmark_raprompro.py --quick --repeats 1
```

## Production

Before using the package in production, follow the release checklist in
`PRODUCTION.md`.

## Releasing

To build and publish the package, follow `RELEASING.md`. The repository includes
GitHub Actions workflows for release validation and Trusted Publishing to PyPI.

## Tests

The test suite is organized into:

- fast checks for import and basic data access,
- integration checks for end-to-end workflow behavior,
- slow plotting regressions that write figures under `tests/data/PRODUCTS/`,
- generated NetCDF and other non-figure test outputs under `tests/generated/`.

Bundled NetCDF files under `tests/data/` should stay minimal. The repository keeps
only the small RAW fixture required for exercising the workflow, while generated
RaProMPro products should go to ignored output paths under `tests/data/PRODUCTS/`
or other configured generated directories.

## Benchmarking

For quick performance checks of the canonical processing path, use the bundled
10-minute RAW subset:

```bash
uv run python scripts/benchmark_raprompro.py --quick --repeats 1
```

The benchmark is intentionally aimed at the small bundled RAW fixture so the repo
does not need to carry large reproducible products.

## Documentation

The repository includes a static documentation site for GitHub Pages.

- Build locally with `python scripts/build_docs.py` after installing `.[docs]`.
- Preview locally with `python -m http.server 8000 --directory site` after the build.
- The landing pages live under `docs/`.
- The API reference is generated with `pdoc` from package docstrings and signatures.
