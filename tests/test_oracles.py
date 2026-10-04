"""
DDT Statistical Oracle Tests — Wall 2 Validation
=================================================
Verifies statistical oracle properties:
    1. Monotonicity Oracle (FR-VAL-02): A feature perfectly correlated with target y
       forces strong/maximal split gain and clean separation.
    2. White Noise Adversarial Oracle (FR-VAL-03): Pure white noise features force
       early stopping at the root node (or no spurious deep splits).
"""

import numpy as np
import pytest

from ddt import DDTRegressor


class TestMonotonicityOracle:
    """FR-VAL-02: Monotonicity Oracle tests."""

    def test_perfect_step_function_split(self):
        """A deterministic step feature X -> y must produce a perfect split at threshold."""
        rng = np.random.RandomState(42)
        n_samples = 1000
        X = rng.uniform(-10.0, 10.0, size=(n_samples, 2))

        # Step function at X[:, 0] == 0:
        # Left cluster centered at -50, Right cluster centered at +50
        y = np.where(X[:, 0] < 0.0, -50.0 + rng.randn(n_samples) * 0.1, 50.0 + rng.randn(n_samples) * 0.1)

        model = DDTRegressor(max_depth=1, min_samples_leaf=50, n_target_bins=50, min_divergence_decrease=0.01)
        model.fit(X, y)

        tree = model.tree_data_
        assert len(tree["is_leaf"]) == 3, "Root node should have split into exactly 2 leaves"
        assert not tree["is_leaf"][0]
        assert tree["split_feature_idx"][0] == 0, "Split must occur on the predictive feature (index 0)"
        assert tree["wasserstein_gain"][0] > 10.0, f"Expected high Wasserstein gain, got {tree['wasserstein_gain'][0]}"

    def test_strictly_monotonic_signal(self):
        """Strictly monotonic relationship creates informative splits with descending leaf medians."""
        rng = np.random.RandomState(123)
        n_samples = 1500
        x0 = np.linspace(-5.0, 5.0, n_samples)
        noise_feature = rng.randn(n_samples)
        X = np.column_stack([x0, noise_feature])
        y = 3.0 * x0 + rng.randn(n_samples) * 0.2

        model = DDTRegressor(max_depth=3, min_samples_leaf=50, n_target_bins=40)
        model.fit(X, y)

        # Predictions along X[:, 0] must be non-decreasing
        test_x0 = np.linspace(-4.5, 4.5, 50)
        test_X = np.column_stack([test_x0, np.zeros_like(test_x0)])
        preds = model.predict(test_X)

        diffs = np.diff(preds)
        assert np.all(diffs >= -1e-6), f"Predicted means are not monotonic: {preds}"


class TestWhiteNoiseOracle:
    """FR-VAL-03: White Noise Adversarial Oracle tests."""

    def test_pure_white_noise_early_stopping(self):
        """Pure white noise features with no signal should stop early with min_divergence_decrease."""
        rng = np.random.RandomState(999)
        n_samples = 800
        n_features = 5
        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)  # Completely independent Gaussian noise

        # With adequate min_divergence_decrease (above random sampling fluctuation ~1.5-2.0),
        # pure noise does not split
        model = DDTRegressor(
            max_depth=5,
            min_samples_leaf=50,
            n_target_bins=30,
            min_divergence_decrease=3.0,
        )
        model.fit(X, y)

        tree_info = model.get_tree_info()
        tree_info = model.get_tree_info()
        assert tree_info["n_nodes"] == 1, (
            f"Pure white noise should not split with threshold=3.0, got {tree_info['n_nodes']} nodes"
        )
        assert tree_info["n_leaves"] == 1


class TestDegenerateOracles:
    """FR-VAL-02: Degenerate edge cases."""

    def test_constant_target(self):
        # A-6: Constant target degeneracy
        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y = np.full(100, 42.0)

        model = DDTRegressor(max_depth=3)
        with pytest.raises(ValueError, match="Target y has zero range"):
            model.fit(X, y)
