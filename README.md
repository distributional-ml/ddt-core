# ddt-core: Distributional Decision Trees

`ddt-core` is a fast, rigorous tree-based model for full distributional forecasting. 

Unlike standard decision trees that predict a single scalar mean, DDTs predict the full empirical probability distribution at each leaf. This allows for rigorous risk management, tail estimation, and arbitrary quantile queries at zero extra training cost.

## Features
- **Distributional Prediction:** Full probability mass functions (PMFs) at every leaf.
- **Extreme Value Theory (EVT) Tails:** Automatically fits Generalized Pareto Distributions (GPD) to the upper and lower tails for robust extrapolation beyond the training data.
- **Dirichlet Smoothing:** Regularizes leaf distributions using a hierarchical prior, ensuring smooth density estimates even in sparse regions.
- **Wasserstein Splits:** Uses the Wasserstein-1 metric for split criteria, optimizing directly for distributional fidelity.
- **Fast C++ Engine:** Highly optimized C++ core for both training and inference.

## Quickstart

```python
from ddt import DDTRegressor
import numpy as np

# Generate dummy data
X = np.random.randn(1000, 10)
y = X[:, 0] + np.random.randn(1000)

# Train the model
model = DDTRegressor(max_depth=5, evt_tails=True, smooth_leaves=True)
model.fit(X, y)

# Predict scalar means
means = model.predict(X[:5])

# Predict multiple quantiles efficiently
quantiles = model.predict_quantiles(X[:5], [0.1, 0.5, 0.9])
print("P90s:", quantiles[0.9])
```

