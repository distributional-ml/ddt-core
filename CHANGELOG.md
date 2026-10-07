# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [v1.1.0]
### Changed
- **Body/EVT Splice Remap**: Quantiles near `F_u`/`F_l` change; no more crossing. Tails are unchanged.
- **Smoothed `F_u` Recomputation Removed**: Tail entry point uses raw leaf mass when `smooth_leaves=True`.
- **`EVT_STATUS_TAIL_OVERLAP`**: Lower tail disabled on degenerate leaves.
- **Conformal Order Statistic + Monotone Shift**: Calibrated outputs shift slightly conservative; no crossing.
- **Conformal Resolution Limits**: Raises `ConformalResolutionWarning` when requested `q` exceeds what `n_calib` can resolve. If `calibration_strict=True`, returns `±inf`.
- **Hybrid Groups**: `hybrid_min_calib=50` and rest group added. Hybrid mode becomes active at realistic sizes. Opt-out via `calibration_mode="marginal"`.
- **Split Weighting**: Threaded `split_weighting` parameter (defaults to `"none"`).
- **Tree Depth limits**: `max_depth=None` is now explicitly accepted for unlimited depth.

### Fixed
- **Input Validation**: NaN/Inf now raise `ValueError` across all predict routes.
- **Stale State**: `fit` now properly clears stale calibration and EVT states.

## [v1.0.0] - Core Stabilisation Release
### Added
- **Conformal Calibration Parity**: Exact split-conformal calibration integrated directly into the `predict_quantiles` API, providing empirical coverage across bounds simultaneously.
- **EVT Intermittent Demand Support**: Added robust atom-detection for zero-inflated targets, stabilizing shape parameter $\xi$ via `evt_borrow_factor` across extreme quantiles.
- **Unified Inference Pipeline**: A single linear interpolation kernel (`invert_cdf`) across C++ and Python paths yields `1e-12` parity.

### Changed
- **GIL released in native bindings**: `build_tree*`, `predict_quantiles_*`, `predict_leaves`, `quantize_features`, and `smooth_tree` now release the GIL during C++ compute, so Python threads can run them concurrently.
- **Build flags**: default builds are portable. `-march=native` (or `/arch:AVX2` on MSVC) is now opt-in via `DDT_NATIVE=1` (setup.py) / `-DDDT_NATIVE=ON` (CMake).

### Removed
- Dead `#pragma omp` directives (OpenMP was never enabled in any build).

## [v0.3.0] - The C++ Consolidation Sprint
### Added
- Pipeline execution order guards to prevent invalid operation sequences.

### Fixed
- **Unweighted Fallback Mean Bias**: Fixed a structural bias in empty-leaf inference.

## [v0.2.0] - Stability and Scaling Sprint
### Added
- **Dirichlet Smoothing** (`smooth_leaves=True`): Hierarchical Empirical Bayes smoothing for terminal leaf PMFs. Prevents overconfident quantile estimates from small-sample histograms.

## [v0.1.0] - Tackling the Tails Sprint
### Added
- **Extreme Value Theory (EVT)**: Integrated Generalized Pareto Distribution (GPD) fitting on terminal leaves to capture unbounded long-tail distributions and vastly reduce Realized CVaR.
- **Log-Width & Hybrid Binning**: Shifted from pure equal-width bins to dynamic bin resolutions to accurately represent dense regions while retaining tail span.

### Fixed
- **EVT Precision Loss**: Fixed catastrophic cancellation (`1.0 - F_u`) in GPD upper-tail quantile calculations.
- **Body-Tail Splice Discontinuity**: Ensured `F_u` and `S_u` are properly recomputed from the smoothed PMF.

## [v0.0.0] - The Foundation (Pre-Sprints)
### Added
- Initial project conception.
- Core C++ micro-kernel for single-tree fitting.
- Cache-friendly 1D Wasserstein kernel over equal-width bins, written as flat loops for compiler auto-vectorization (no SIMD intrinsics are used).
- Base Pybind11 wrapper.
