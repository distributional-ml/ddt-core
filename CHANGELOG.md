# Changelog

## [1.1.0] - 2026-10-07
### Added
- `split_weighting` parameter (`"none"` default, `"sqrt"`, `"crps"`) for the split criterion.
- `atom_threshold`, EVT tail parameters and `evt_tail_fraction_lower` are now constructor parameters.
- `tail_fraction="auto"`; float values of `tail_fraction` are now honoured when selecting the tail size.
- `ConformalResolutionWarning`, `calibration_strict` and `hybrid_min_calib` for conformal calibration.
- `max_depth=None` is accepted for unlimited depth.

### Changed
- Body/EVT splice: quantiles near the tail entry point are remapped so that quantiles no longer cross. Tail shapes are unchanged. The tail entry point uses the raw leaf mass when `smooth_leaves=True`.
- Lower tail is disabled on leaves where the lower and upper tails would overlap.
- Conformal calibration uses an order-statistic shift with monotone repair. Calibrated outputs are slightly more conservative; the requested quantile is used for strict calibration.
- Hybrid calibration groups use a `"rest"` group; hybrid mode is active at realistic calibration sizes. Use `calibration_mode="marginal"` to opt out.
- The GPD tail honours a bounded endpoint (negative shape).
- The documented split criterion is a binned, scale-normalised Wasserstein-1 difference between child target distributions.

### Fixed
- `ddt/_evt.py` was not importable in 1.0.0 because of an export indentation error.
- NaN/Inf inputs raise `ValueError` on all predict routes; native calls validate array bounds.
- `fit` clears stale calibration and EVT state; EVT activation is read from the fitted state.
- Package is ruff-clean (`ruff check` and `ruff format --check`).
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-04
### Added
- First public release of `ddt-core`.
- High-performance C++ inference engine.
- EVT tails for lower and upper bounds.
- Dirichlet smoothing for leaf histograms.
- `predict_quantiles` for optimized, batched quantile inference.
