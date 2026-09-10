# Distributional Decision Trees (DDT) — Core

A non-parametric machine learning framework that preserves full Empirical Cumulative Distribution Functions (ECDF) in terminal leaves by maximizing the 1D Wasserstein distance during tree construction.

## Why DDT?

Standard regression trees predict a single point estimate per leaf, discarding distributional information. DDT retains the full empirical distribution in every leaf, enabling native quantile queries (P10, P50, P90, P99) without distributional assumptions — useful for risk quantification, uncertainty-aware forecasting, and any domain where tail behavior matters.

## Features

- **Zero-Heap C++ Engine:** The core split-finding loop operates natively on continuous arrays in L1 cache, avoiding all heap allocations during inference.
- **Pure Integer Math:** Uses target binning to evaluate the Wasserstein distance purely via integer sum-of-absolute-differences.
- **Multiple Bin Strategies:** Choose from `equal_width`, `log_width`, `hybrid`, or `manual` target binning to match your target distribution.
- **Scikit-Learn Compatible:** Exposes a seamless Python API (`DDTRegressor`) with `fit()`, `predict_distribution()`, and `predict_quantiles()`.

## Installation

Requires a modern C++17 compiler (GCC, Clang, or MSVC).

```bash
git clone https://github.com/distributional-ml/ddt-core.git
cd ddt-core
pip install -e .[dev]
```

## Quickstart

```python
from ddt import DDTRegressor
import numpy as np

# Generate synthetic data
X = np.random.rand(1000, 10)
y = np.sum(X[:, :3], axis=1) + np.random.randn(1000) * 0.1

# Fit model with default equal-width bins
model = DDTRegressor(max_depth=5, n_target_bins=32)
model.fit(X, y)

# Extract P10, P50 (Median), and P90 quantiles
quantiles = model.predict_quantiles(X[:5], [0.10, 0.50, 0.90])
print(quantiles)
```

## Target Bin Strategies

The `bin_strategy` parameter controls how the continuous target `y` is discretized before tree construction. Choosing the right strategy can significantly improve distributional fidelity for skewed or heavy-tailed targets.

### `"equal_width"` (default)

Uniform bins across `[y_min, y_max]`. Recommended for general-purpose use and required for FPGA / bare-metal inference where arithmetic bin lookup (no lookup table) is needed.

```python
model = DDTRegressor(n_target_bins=30, bin_strategy="equal_width")
```

### `"log_width"`

Logarithmically spaced bin edges. Produces narrow bins near the origin (high resolution for the common case) and exponentially wider bins toward the tail. Best suited for strictly positive right-skewed targets such as prices, lead times, and durations.

> **Requires `y_min > 0` after winsorization.** Cannot be combined with `target_transform='log1p'`.

```python
model = DDTRegressor(
    n_target_bins=30,
    bin_strategy="log_width",
    winsorize_tails=0.001,  # ensures positive y_min
)
```

### `"hybrid"`

Combines a linear core over the IQR (Q25–Q75) with log-spaced bins in both tails. This provides fine resolution where data is densest while preserving tail sensitivity. The `core_fraction` parameter controls how many bins are allocated to the core vs. the tails.

```python
model = DDTRegressor(
    n_target_bins=40,
    bin_strategy="hybrid",
    core_fraction=0.6,  # 60% of bins cover the IQR, 40% cover the tails
)
```

### `"manual"`

User-supplied bin edges for regulatory, actuarial, or domain-expert boundaries. Provide a strictly monotonically increasing array of length `n_target_bins + 1`.

> Cannot be combined with `target_transform`.

```python
import numpy as np
edges = np.array([0, 100, 500, 1000, 5000, 10000, 50000], dtype=float)

model = DDTRegressor(
    n_target_bins=6,
    bin_strategy="manual",
    manual_bin_edges=edges,
)
```

## Target Transform

For targets that span multiple orders of magnitude (e.g., insurance claims, revenue), applying a log-transform before binning can improve resolution significantly. Use `target_transform='log1p'` in combination with `bin_strategy='equal_width'`:

```python
model = DDTRegressor(
    n_target_bins=30,
    bin_strategy="equal_width",
    target_transform="log1p",  # applies log1p(y) before binning
)
```

> `target_transform` cannot be combined with `bin_strategy` other than `"equal_width"`. For log-spaced bins on positive targets, use `bin_strategy="log_width"` instead.

## API Reference

### `DDTRegressor`

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `max_depth` | `int` | `10` | Maximum tree depth (0 = unlimited) |
| `min_samples_leaf` | `int` | `20` | Minimum samples per leaf |
| `n_target_bins` | `int` | `30` | Number of target bins B ∈ [2, 255] |
| `bin_strategy` | `str` | `"equal_width"` | Bin edge strategy: `equal_width`, `log_width`, `hybrid`, `manual` |
| `target_transform` | `str\|None` | `None` | Pre-binning transform: `'log1p'` or `None` |
| `core_fraction` | `float` | `0.5` | IQR bin fraction for `hybrid` strategy |
| `manual_bin_edges` | `array\|None` | `None` | Custom edges for `manual` strategy |
| `winsorize_tails` | `float\|None` | `0.001` | Tail clip fraction; `None` to disable |
| `consolidate_bins` | `bool` | `False` | Merge always-empty bins after fit |
| `n_feature_bins` | `int` | `256` | Quantile bins for feature discretization |
| `calibration_fraction` | `float` | `0.0` | Fraction of data held out for conformal calibration |
| `calibration_mode` | `str` | `"marginal"` | Conformal mode: `marginal` or `hybrid` |
| `calibration_order_by` | `int\|str\|None` | `None` | Feature column (index or name) used to sort conformal calibration samples |
| `calibration_order_asc` | `bool` | `True` | Sort order for conformal calibration feature |

### Key Methods

```python
model.fit(X, y)                          # Train the tree
model.predict(X)                         # Predict leaf ECDF mean
model.predict_distribution(X)           # Predict raw bin count histograms
model.predict_quantiles(X, [0.1, 0.9])  # Predict quantiles (dict output)
model.predict_quantile(X, 0.95)         # Predict a single quantile
model.get_tree_info()                    # Tree metadata (nodes, leaves, depth)
```

## License

Licensed under Apache 2.0. See [LICENSE](LICENSE) for details.
