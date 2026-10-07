# ddt-core: Distributional Decision Trees

**Predict an outcome distribution, and learn the conditions that change its shape.**

Two populations can have the same average demand and need different stock levels. Two operating regimes can have the same average error and very different upper quantiles.

DDT builds trees from differences between child target distributions. Each leaf stores a compact histogram approximation, so one fitted model can answer multiple quantile queries. The body does not require a Normal or lognormal family.

## Same mean, different uncertainty

This small example gives both populations an exact training mean of 100. Their spreads differ. A depth-one tree separates them and returns different uncertainty intervals.

```python
import numpy as np
from ddt import DDTRegressor

rng = np.random.default_rng(42)
z = rng.normal(size=1000)
noise = np.concatenate([z, -z])
X = np.repeat([[0.0], [1.0]], len(noise), axis=0)
y = np.concatenate([100 + noise, 100 + 6 * noise])

model = DDTRegressor(
    max_depth=1,
    min_samples_leaf=100,
    n_target_bins=64,
    winsorize_tails=None,
)
model.fit(X, y)

queries = np.array([[0.0], [1.0]])
quantiles = model.predict_quantiles(queries, [0.1, 0.5, 0.9])
for regime in range(2):
    print(regime, [round(float(quantiles[q][regime]), 2) for q in [0.1, 0.5, 0.9]])
```

Illustrative output from the development engine on 6 October 2026:

```text
0 [98.7, 100.0, 101.3]
1 [92.36, 100.0, 107.64]
```

This demonstrates a split mechanism on controlled samples. It is not evidence of superiority on independent data or a timing benchmark. Results should be checked against the installed build.

## Why use a distributional tree?

| Need | Representation |
|---|---|
| Distinguish stable and variable outcomes with similar means | Splits sensitive to represented CDF differences |
| Query several service levels or risk limits | Multiple quantiles from one fitted distribution |
| Represent skewness, modes, and repeated outcomes | A shared histogram grid with atom-aware quantile inversion |
| Stabilize sparse leaf estimates | Optional hierarchical Dirichlet shrinkage |
| Extend queries into extreme tails | Optional upper/lower Generalized Pareto tail models |
| Keep distributional output within a resource budget | Quantized features and count histograms evaluated by a C++ engine |

Equal-width, log-width, hybrid, and manual target grids trade resolution between the body and tails. The default split score is based on binned, scale-normalized Wasserstein-1 differences. Bin widths and target transformations affect the comparison.

The integration is the motivation: partitions identify differing outcome profiles, the leaf representation supports several questions, and histogram computation limits the cost. Accuracy and runtime advantages depend on the dataset, configuration, and competing methods.

## Installation

From a source checkout:

```bash
pip install .
```

Requires Python >= 3.9, a C++17 compiler, and pybind11 to build the extension.

## Outputs and assumptions

- `predict(X)` returns a mean reconstructed from the body histogram.
- `predict_quantiles(X, quantiles)` answers several quantile queries, with configured tail routing.
- `predict_distribution(X)` returns body counts, or normalized leaf PMFs when smoothing is enabled. It does not expose a complete body-plus-tail CDF.

Finite bins approximate within-bin detail. Atom detection depends on the grid and data. Shrinkage can reduce sparse sampling irregularities but may also weaken local modes. GPD extrapolation is parametric and depends on threshold choice and sufficient tail observations.

Use held-out distribution scores, interval/event scores, decision costs, and end-to-end runtime measurements to evaluate the model. Include appropriate conditional distributional competitors. A few quantile losses answer only part of the question; the architecture does not replace empirical validation.

## License

Apache 2.0.
