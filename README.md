# Distributional Decision Trees (DDT)

**Learn which conditions change the distribution of an outcome, and use that distribution to make decisions.**

Two customer groups can have the same average demand and need different stock levels. Two operating regimes can have the same average error and very different failure probabilities. A model of the conditional mean leaves those differences unanswered.

DDT builds a tree using differences between child target distributions. Each leaf stores a compact histogram approximation to the conditional distribution. One fitted model can answer several quantile queries and expose the outcome profile of each discovered subgroup.

The value proposition is the combination: **distribution-sensitive partitions, flexible leaf distributions, inspectable rules, and a C++ histogram engine designed to keep the representation affordable.**

## What this makes useful

| Need | What DDT provides | Evidence to look for |
|---|---|---|
| Separate behavior hidden by an average | Target-guided subgroups sensitive to represented CDF differences | Held-out distribution scores and recovered subgroup profiles |
| Choose a buffer, limit, or risk threshold | Multiple quantiles from one fitted distribution | Decision cost, interval score, coverage, and threshold-probability accuracy |
| Represent intermittent or multimodal outcomes | Histogram bodies with no prescribed Normal or lognormal family; atom-aware quantile grids | Zero-mass accuracy, mode preservation, and bin-resolution sensitivity |
| Work with sparse leaves and extreme outcomes | Optional hierarchical shrinkage and GPD tail models | Separate smoothing and tail ablations, with enough observations |
| Inspect a prediction or changing regime | Tree paths, leaf distributions, and labeled out-of-sample leaf comparisons | Stable rules, local sample counts, and diagnostic false alarms |
| Fit distributional modeling into a resource budget | Quantized features, count histograms, and C++ training/inference | End-to-end latency, peak memory, and accuracy under the same budget |

## Why the architecture fits the problem

- **Distribution-sensitive splits:** the default criterion compares child CDFs through a binned, scale-normalized Wasserstein-1 score. Bin widths matter; target transforms change the scale on which differences are measured. The default is unweighted by child size, with alternative `split_weighting` settings available.
- **A shared target grid:** each leaf stores counts on a common grid. Equal-width, log-width, hybrid, and manual grids trade body resolution against tail resolution. Detail within a bin is approximated.
- **Atoms:** the target grid can represent detected point masses using zero-width bins. Detection depends on the data and grid; repeated values and nearby continuous values need separate validation.
- **Sparse-leaf regularization:** optional Dirichlet shrinkage borrows an ancestor distribution. It can reduce sampling irregularities, including comb-like histograms; excessive shrinkage can weaken real minority modes.
- **Tail extension:** optional upper/lower GPD models extend quantile queries beyond the empirical body. Their usefulness depends on threshold choice, sample size, and tail assumptions.
- **Compute:** histogram sweeps reuse counts over quantized features during split search. Quantization has an upfront cost, and probability normalization, variable widths, smoothing, and EVT use floating-point arithmetic.

The empirical body does not prescribe a distribution family. The optional EVT extension does.

## Installation

```bash
pip install .
```

Requires Python >= 3.9, a C++17 compiler, and pybind11 to build the extension.

## Working with the output

```python
# Mean reconstructed from the body histogram.
means = model.predict(queries)

# Several quantiles from the same model.
quantiles = model.predict_quantiles(queries, [0.1, 0.5, 0.9, 0.99])

# Raw histogram counts, or normalized PMFs when smoothing is enabled.
body = model.predict_distribution(queries)

# Inspect the conditions defining the two outcome profiles.
print(model.export_text(feature_names=["regime"]))
```

`predict_distribution()` exposes the body histogram; it is not a complete body-plus-EVT CDF. Use the quantile methods for configured tail routing. Calibration settings may modify quantile outputs without modifying the returned histogram.

## License

Apache 2.0.
