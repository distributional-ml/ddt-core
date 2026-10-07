"""
DDT Smoke Tests — Phase 1 Gate Validation.
===========================================
Minimal tests to confirm the core engine works end-to-end:
    - fit/predict runs without errors
    - predict_distribution returns valid histograms
    - predict_quantile returns monotonic quantiles
    - NaN/Inf inputs are rejected (FR-CORE-05)
    - Tree structure is valid
"""

import numpy as np
import pytest


def make_synthetic_data(n_samples=1000, n_features=5, seed=42):
    """
    Generate synthetic regression data with a clear signal.

    y = 2*X[:,0] + noise, creating a distributional shift
    that DDT should capture.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = 2.0 * X[:, 0] + rng.randn(n_samples) * 0.5
    return X, y


class TestFitPredict:
    """Test that fit/predict runs end-to-end."""

    def test_fit_predict_runs(self):
        """Fit on synthetic data and predict — output shape must match."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=1000)
        model = DDTRegressor(max_depth=5, min_samples_leaf=20, n_target_bins=30)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == (1000,), f"Expected (1000,), got {predictions.shape}"
        assert np.all(np.isfinite(predictions)), "Predictions contain NaN or Inf"

    def test_predict_single_sample(self):
        """Predict on a single sample should work."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(max_depth=3, min_samples_leaf=30)
        model.fit(X, y)

        pred = model.predict(X[:1])
        assert pred.shape == (1,)
        assert np.isfinite(pred[0])

    def test_fit_returns_self(self):
        """fit() should return self for method chaining."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=200)
        model = DDTRegressor(max_depth=3)
        result = model.fit(X, y)
        assert result is model


class TestDistribution:
    """Test distributional output."""

    def test_distribution_output(self):
        """predict_distribution returns valid histograms."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        n_bins = 25
        model = DDTRegressor(max_depth=4, min_samples_leaf=20, n_target_bins=n_bins)
        model.fit(X, y)

        distributions = model.predict_distribution(X)
        assert len(distributions) == 500

        for dist in distributions:
            assert len(dist) == n_bins, f"Expected {n_bins} bins, got {len(dist)}"
            assert np.all(dist >= 0), "Histogram contains negative counts"
            assert np.sum(dist) > 0, "Histogram is empty"

    def test_quantile_monotonicity(self):
        """Predicted quantiles should be monotonically non-decreasing."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(max_depth=4, min_samples_leaf=20, n_target_bins=40)
        model.fit(X, y)

        q10 = model.predict_quantile(X[:10], 0.10)
        q50 = model.predict_quantile(X[:10], 0.50)
        q90 = model.predict_quantile(X[:10], 0.90)

        # For each sample, P10 <= P50 <= P90.
        for i in range(10):
            assert q10[i] <= q50[i] + 1e-10, f"Sample {i}: P10={q10[i]} > P50={q50[i]}"
            assert q50[i] <= q90[i] + 1e-10, f"Sample {i}: P50={q50[i]} > P90={q90[i]}"

    def test_bimodal_distribution_preservation(self):
        # A-5: Bimodal distribution preservation
        import scipy.signal

        from ddt import DDTRegressor

        rng = np.random.RandomState(42)
        X = np.ones((1000, 1))
        # mixture of N(-5, 1) and N(+5, 1)
        y = np.concatenate([rng.randn(500) - 5, rng.randn(500) + 5])

        model = DDTRegressor(max_depth=1, n_target_bins=50, quantize_engine="python")
        model.fit(X, y)

        dist = model.predict_distribution(X[:1])[0]
        # find peaks
        peaks, _ = scipy.signal.find_peaks(dist, prominence=max(dist) * 0.2)
        assert len(peaks) >= 2, "Bimodal distribution should have at least 2 peaks"

    def test_conformal_coverage_iid(self):
        # A-7: Conformal coverage on IID data
        from ddt import DDTRegressor

        rng = np.random.RandomState(42)
        X = rng.randn(2000, 2)
        y = 2.0 * X[:, 0] + rng.randn(2000) * 1.0

        model = DDTRegressor(max_depth=4, min_samples_leaf=20, calibration_fraction=0.2, calibration_mode="hybrid")
        model.fit(X, y)

        X_test = rng.randn(1000, 2)
        y_test = 2.0 * X_test[:, 0] + rng.randn(1000) * 1.0

        preds = model.predict_quantiles(X_test, [0.05, 0.95])
        p5, p95 = preds[0.05], preds[0.95]

        coverage = np.mean((y_test >= p5) & (y_test <= p95))
        # Expected coverage 90%, allow ±5%
        assert 0.85 <= coverage <= 0.95


class TestInputValidation:
    """Test NaN/Inf rejection (FR-CORE-05)."""

    def test_nan_rejection_X(self):
        """fit() should raise ValueError on NaN in X."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=100)
        X[5, 2] = np.nan

        model = DDTRegressor()
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_inf_rejection_X(self):
        """fit() should raise ValueError on Inf in X."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=100)
        X[3, 0] = np.inf

        model = DDTRegressor()
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_nan_rejection_y(self):
        """fit() should raise ValueError on NaN in y."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=100)
        y[10] = np.nan

        model = DDTRegressor()
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_nan_rejection_predict(self):
        """predict() should raise ValueError on NaN in X."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=100)
        model = DDTRegressor(max_depth=3).fit(X, y)

        X_bad = X[:5].copy()
        X_bad[2, 0] = np.nan
        with pytest.raises(ValueError):
            model.predict(X_bad)

    def test_quantile_out_of_range(self):
        """predict_quantile should reject q outside [0, 1]."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=100)
        model = DDTRegressor(max_depth=3).fit(X, y)

        with pytest.raises(ValueError, match="q must be in"):
            model.predict_quantile(X[:5], 1.5)


class TestTreeStructure:
    """Test tree metadata and structure."""

    def test_tree_info(self):
        """get_tree_info() should return valid metadata."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(max_depth=4, min_samples_leaf=20)
        model.fit(X, y)

        info = model.get_tree_info()
        assert info["n_nodes"] > 0
        assert info["n_leaves"] > 0
        assert info["n_internal"] >= 0
        assert info["n_leaves"] + info["n_internal"] == info["n_nodes"]
        assert info["max_depth"] <= 4
        assert info["divergence"] == "wasserstein"

    def test_max_depth_respected(self):
        """Tree should not exceed max_depth."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=2000)
        model = DDTRegressor(max_depth=2, min_samples_leaf=5)
        model.fit(X, y)

        info = model.get_tree_info()
        assert info["max_depth"] <= 2

    def test_divergence_param(self):
        """Divergence parameter should be stored and reported."""
        from ddt import DDTRegressor

        model = DDTRegressor(divergence="wasserstein")
        X, y = make_synthetic_data(n_samples=200)
        model.fit(X, y)
        assert model.get_tree_info()["divergence"] == "wasserstein"

    def test_invalid_divergence(self):
        """Unknown divergence should raise an error."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=200)
        model = DDTRegressor(divergence="unknown_metric")
        with pytest.raises(Exception):
            model.fit(X, y)


class TestSmoothParamsAPI:
    def test_smooth_params_round_trip(self):
        from sklearn.base import clone

        from ddt import DDTRegressor

        model = DDTRegressor(smooth_leaves=True, smooth_prior_weight=50.0, smooth_min_samples=100)
        params = model.get_params()
        assert params["smooth_leaves"] is True
        assert params["smooth_prior_weight"] == 50.0
        assert params["smooth_min_samples"] == 100
        cloned = clone(model)
        assert cloned.get_params() == params


class TestPickleSmoothedState:
    def test_pickle_preserves_smoothed_pmf(self):
        import pickle

        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(max_depth=3, min_samples_leaf=10, smooth_leaves=True).fit(X, y)
        dumped = pickle.dumps(model)
        loaded = pickle.loads(dumped)
        assert "smoothed_pmf" in loaded.tree_data_

    def test_pickle_size_reduction(self):
        # B-4: Pickle size reduction
        import pickle

        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(max_depth=5, min_samples_leaf=5).fit(X, y)

        # Original tree_data size
        raw_size = len(pickle.dumps(model.tree_data_))

        dumped = pickle.dumps(model)
        loaded = pickle.loads(dumped)
        compact_size = len(pickle.dumps(loaded.tree_data_))

        # Model's tree_data pickle should be smaller because of stripped arrays
        assert compact_size < raw_size * 0.95


class TestSmoothParamValidation:
    def test_negative_weight_raises(self):
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=100)
        with pytest.raises(ValueError, match="smooth_prior_weight must be > 0"):
            DDTRegressor(smooth_leaves=True, smooth_prior_weight=-5.0).fit(X, y)

    def test_zero_weight_raises(self):
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=100)
        with pytest.raises(ValueError, match="smooth_prior_weight must be > 0"):
            DDTRegressor(smooth_leaves=True, smooth_prior_weight=0.0).fit(X, y)

    def test_negative_min_samples_raises(self):
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=100)
        with pytest.raises(ValueError, match="smooth_min_samples must be >= 1"):
            DDTRegressor(smooth_leaves=True, smooth_min_samples=0).fit(X, y)


class TestDistributionSummary:
    def test_distribution_summary(self):
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(max_depth=3, min_samples_leaf=10).fit(X, y)
        summary = model.distribution_summary()
        assert "n_leaves" in summary
        assert summary["smoothing_applied"] is False

    def test_distribution_summary_smoothed(self):
        # B-7: distribution_summary on smoothed model
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(max_depth=3, min_samples_leaf=10, smooth_leaves=True).fit(X, y)
        summary = model.distribution_summary()
        assert summary["smoothing_applied"] is True


class TestEdgeCases:
    def test_single_feature(self):
        # B-5: 1-feature, 2-bin edge case
        from ddt import DDTRegressor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 1)
        y = rng.randn(100)

        model = DDTRegressor(max_depth=3, n_target_bins=2)
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert len(preds) == 5

    def test_default_model_uses_fast_cpp_path(self):
        # C-2: Default model uses fast C++ path
        import warnings

        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=100)
        model = DDTRegressor(max_depth=2, quantize_engine="cpp").fit(X, y)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.predict_quantiles(X[:5], [0.5])

            # No warnings about Python fallback should be raised
            fallback_warnings = [
                warn for warn in w if "disables the fast C++ quantile inference path" in str(warn.message)
            ]
            assert len(fallback_warnings) == 0

    @pytest.mark.slow
    def test_large_n_smoke_test(self):
        # B-6: Large-N smoke test
        import time

        from ddt import DDTRegressor

        rng = np.random.RandomState(42)
        X = rng.randn(50000, 20)
        y = X[:, 0] + X[:, 1] + rng.randn(50000)

        start_t = time.time()
        model = DDTRegressor(max_depth=8, min_samples_leaf=10).fit(X, y)
        fit_time = time.time() - start_t

        assert fit_time < 30.0, f"Fit took too long: {fit_time:.2f}s"
        preds = model.predict(X[:10])
        assert np.all(np.isfinite(preds))

    def test_get_params_all(self):
        # B-8: get_params round-trip for ALL params
        from sklearn.base import clone

        from ddt import DDTRegressor

        model = DDTRegressor(
            max_depth=5,
            min_samples_leaf=20,
            n_target_bins=30,
            min_divergence_decrease=0.01,
            target_transform="log1p",
            winsorize_tails=0.001,
            n_feature_bins=256,
            divergence="wasserstein",
            quantize_engine="cpp",
            smooth_leaves=True,
            smooth_prior_weight=5.0,
            smooth_min_samples=100,
            evt_tails=True,
            evt_min_samples=50,
            evt_tail_fraction=0.1,
            bin_strategy="hybrid",
            core_fraction=0.6,
            consolidate_bins=True,
            manual_bin_edges=None,
            calibration_fraction=0.2,
            calibration_mode="hybrid",
        )

        params = model.get_params()
        cloned = clone(model)
        assert cloned.get_params() == params
