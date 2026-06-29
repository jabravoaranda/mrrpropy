# Changelog

All notable changes to this project should be documented in this file.

The format is based on Keep a Changelog and the project follows semantic-style
versioning for published package releases.

## [Unreleased]

### Added

### Changed

### Fixed

## [0.3.0]

### Added

- Plot examples gallery in the documentation, including regenerated example
  figures for processed spectra and microphysical profiles.
- Public velocity-convention metadata on RaProMPro outputs and spectral
  rain-process features.

### Changed

- Public Doppler/fall velocity outputs now use the negative-downward convention:
  `W`, `spe_3D.speed`, spectral plots, and spectral process features expose
  falling hydrometeors as negative velocity.
- `MRRProData.plot_spectrogram(spectrum_var="spe_3D")` converts legacy
  positive-downward processed spectra at the plotting boundary when loaded
  products do not declare the new convention.
- The microphysical profile example now distinguishes `LWC_all` and `LWC` with
  clearer colors, widths, and markers.
- The plot examples gallery uses a fixed two-column desktop layout with mobile
  fallback and wraps long method names.

### Fixed

- Loading an older RaProMPro product now warns when its velocity convention is
  ambiguous, preventing silent interpretation of legacy positive-downward
  `W`/`speed` as current public output.

## [0.2.0]

### Added

- Phase A rain-process feature extraction and Phase B classification bridge for
  scan-based rain-process workflows.
- Single-hour processing script for GAIA campaign analysis, including explicit
  window-parameter controls.
- Poster/report documentation for the rain-process analysis workflow.
- Local pre-commit Black hook matching the CI formatting lane.

### Changed

- Improved typing coverage in rain-process analysis to keep the typed lane green.
- Black-formatted rain-process tests to match the CI formatter.

### Fixed

- Fused quicklook plotting tests now build a deterministic plottable fused event,
  avoiding CI failures when the first scan snapshot is labelled `no_data`.

## [0.1.1]

### Added

- Non-parametric rain-process trend diagnostics based on Kendall's tau and
  Theil-Sen slope, with canonical `trend_*` outputs for downstream RGB and
  classification.
- Regression coverage for the new monotonic-trend utilities and for RaProMPro
  metadata completeness.

### Changed

- Rain-process classification now uses canonical trend diagnostics instead of
  OLS slopes by default, while keeping OLS as an explicit legacy/diagnostic path.
- Test modules are now grouped by domain under `tests/raw_mrr`,
  `tests/raprompro`, `tests/rain_processes`, and `tests/hexagram`.
- The documentation site, production guide, and release-validation workflow now
  reflect the non-parametric trend workflow and the reorganized test suite.

### Fixed

- RaProMPro outputs now populate `long_name` and `units` consistently across
  generated data variables, including the saved reference product.
- Release validation now uses `MRRPRO_GENERATED_PRODUCT_ROOT` for forced product
  regeneration instead of the old generated-output path convention.

## [0.1.0]

### Added

- Production readiness checklist in `PRODUCTION.md`.
- Release validation workflow for forced RAW reprocessing checks.
- Packaging and release preparation docs for PyPI publication.

### Changed

- `MRRProData` now delegates plotting, rain-process analysis, and RaProMPro
  preprocessing/load responsibilities into dedicated modules.
- The enforced `mypy` lane now covers the extracted typed subset of the package.

## [0.0.1]

### Added

- Initial packaged release of `mrrpropy`.
