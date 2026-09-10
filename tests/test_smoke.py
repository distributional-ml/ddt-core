"""
DDT Smoke Tests
===========================================
Minimal tests to confirm the core engine works end-to-end:
    - fit/predict runs without errors
    - predict_distribution returns valid histograms
    - predict_quantile returns monotonic quantiles
    - NaN/Inf inputs are rejected
    - Tree structure is valid
"""

import pickle

import numpy as np
import pytest


def make_synthetic_data(n_samples=1000, n_features=5, seed=42):
    """Generate synthetic regression data with a clear signal.

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
            assert q10[i] <= q50[i] + 1e-10, (
                f"Sample {i}: P10={q10[i]} > P50={q50[i]}"
            )
            assert q50[i] <= q90[i] + 1e-10, (
                f"Sample {i}: P50={q50[i]} > P90={q90[i]}"
            )


class TestInputValidation:
    """Test NaN/Inf rejection."""

    def test_nan_rejection_X(self):
        """fit() should raise ValueError on NaN in X."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=100)
        X[5, 2] = np.nan

        model = DDTRegressor()
        with pytest.raises(ValueError, match="NaN"):
            model.fit(X, y)

    def test_inf_rejection_X(self):
        """fit() should raise ValueError on Inf in X."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=100)
        X[3, 0] = np.inf

        model = DDTRegressor()
        with pytest.raises(ValueError, match="Inf"):
            model.fit(X, y)

    def test_nan_rejection_y(self):
        """fit() should raise ValueError on NaN in y."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=100)
        y[10] = np.nan

        model = DDTRegressor()
        with pytest.raises(ValueError, match="NaN"):
            model.fit(X, y)

    def test_nan_rejection_predict(self):
        """predict() should raise ValueError on NaN in X."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=100)
        model = DDTRegressor(max_depth=3).fit(X, y)

        X_bad = X[:5].copy()
        X_bad[2, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
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
        assert info["n_target_bins"] == 30  # default
        assert info["n_bins_active"] > 0

    def test_max_depth_respected(self):
        """Tree should not exceed max_depth."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=2000)
        model = DDTRegressor(max_depth=2, min_samples_leaf=5)
        model.fit(X, y)

        info = model.get_tree_info()
        assert info["max_depth"] <= 2

    def test_tree_max_depth_cached(self):
        """tree_max_depth_ should be set during fit so it survives pickling."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(max_depth=4, min_samples_leaf=20)
        model.fit(X, y)

        assert hasattr(model, "tree_max_depth_")
        assert model.tree_max_depth_ <= 4


class TestSerialization:
    """Test pickle serialization roundtrip (custom __getstate__/__setstate__)."""

    def test_pickle_roundtrip_predictions_match(self):
        """Unpickling a model should produce identical predictions."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(max_depth=5, min_samples_leaf=20, n_target_bins=30)
        model.fit(X, y)

        preds_before = model.predict(X)

        blob = pickle.dumps(model)
        model2 = pickle.loads(blob)

        preds_after = model2.predict(X)
        np.testing.assert_array_almost_equal(
            preds_before, preds_after, decimal=10,
            err_msg="Predictions differ after pickle roundtrip"
        )

    def test_pickle_strips_diagnostic_arrays(self):
        """Pickled state should not contain wasserstein_gain, depth, total_samples."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(max_depth=4, min_samples_leaf=20)
        model.fit(X, y)

        state = model.__getstate__()
        tree_data = state["tree_data_"]
        assert "wasserstein_gain" not in tree_data, "wasserstein_gain was not stripped"
        assert "depth" not in tree_data, "depth was not stripped"
        assert "total_samples" not in tree_data, "total_samples was not stripped"

    def test_pickle_get_tree_info_works_after_load(self):
        """get_tree_info() should work correctly on an unpickled model."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(max_depth=3, min_samples_leaf=20)
        model.fit(X, y)

        info_before = model.get_tree_info()

        model2 = pickle.loads(pickle.dumps(model))
        info_after = model2.get_tree_info()

        assert info_before["max_depth"] == info_after["max_depth"]
        assert info_before["n_nodes"] == info_after["n_nodes"]
        assert info_before["n_leaves"] == info_after["n_leaves"]

    def test_pickle_quantiles_match(self):
        """Quantile predictions should be identical before and after pickling."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(max_depth=5, min_samples_leaf=20)
        model.fit(X, y)

        q_before = model.predict_quantiles(X[:50], [0.1, 0.5, 0.9])

        model2 = pickle.loads(pickle.dumps(model))
        q_after = model2.predict_quantiles(X[:50], [0.1, 0.5, 0.9])

        for q in [0.1, 0.5, 0.9]:
            np.testing.assert_array_almost_equal(
                q_before[q], q_after[q], decimal=10,
                err_msg=f"Q{q} differs after pickle roundtrip"
            )

    def test_feature_quantizer_pickle_roundtrip(self):
        """FeatureQuantizer should reconstruct bin_edges_ on load."""
        from ddt import FeatureQuantizer

        X, _ = make_synthetic_data(n_samples=500)
        fq = FeatureQuantizer(n_bins=64)
        fq.fit(X)

        X_q_before = fq.transform(X)

        fq2 = pickle.loads(pickle.dumps(fq))
        # bin_edges_ should have been lazily reconstructed
        assert fq2.bin_edges_ is not None
        assert len(fq2.bin_edges_) == X.shape[1]

        X_q_after = fq2.transform(X)
        np.testing.assert_array_equal(X_q_before, X_q_after)


class TestCalibration:
    """Smoke tests for conformal calibration."""

    def test_marginal_calibration_runs(self):
        """calibration_fraction > 0 with mode='marginal' should not crash."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(
            max_depth=4,
            min_samples_leaf=20,
            calibration_fraction=0.2,
            calibration_mode="marginal",
        )
        model.fit(X, y)

        q50 = model.predict_quantile(X, 0.5)
        assert q50.shape == (len(X),)
        assert np.all(np.isfinite(q50))

    def test_hybrid_calibration_runs(self):
        """calibration_fraction > 0 with mode='hybrid' should not crash."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(
            max_depth=4,
            min_samples_leaf=20,
            calibration_fraction=0.2,
            calibration_mode="hybrid",
        )
        model.fit(X, y)

        q50 = model.predict_quantile(X, 0.5)
        assert q50.shape == (len(X),)
        assert np.all(np.isfinite(q50))

    def test_invalid_calibration_mode(self):
        """Invalid calibration_mode should raise ValueError at construction."""
        from ddt import DDTRegressor

        with pytest.raises(ValueError, match="calibration_mode"):
            DDTRegressor(calibration_mode="bogus")

    def test_calibration_pickle_roundtrip(self):
        """Calibrated model should produce identical results after pickling."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(
            max_depth=4,
            min_samples_leaf=20,
            calibration_fraction=0.2,
            calibration_mode="marginal",
        )
        model.fit(X, y)

        q_before = model.predict_quantile(X[:20], 0.9)
        model2 = pickle.loads(pickle.dumps(model))
        q_after = model2.predict_quantile(X[:20], 0.9)
        np.testing.assert_array_almost_equal(q_before, q_after, decimal=10)


class TestConsolidation:
    """Test bin consolidation and reconstruction."""

    def test_consolidation_runs(self):
        """consolidate_bins=True should produce valid, smaller active bin count."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(
            max_depth=4, min_samples_leaf=20, n_target_bins=30, consolidate_bins=True
        )
        model.fit(X, y)

        info = model.get_tree_info()
        assert info["n_bins_active"] <= 30
        assert info["n_bins_active"] >= 1

    def test_reconstruct_original_grid(self):
        """reconstruct_original_grid expands active distribution losslessly."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(
            max_depth=3, min_samples_leaf=20, n_target_bins=30, consolidate_bins=True
        )
        model.fit(X, y)

        binner = model.target_binner_
        if binner.active_bins_mask_ is None:
            pytest.skip("No empty bins to consolidate in this dataset")

        dists = model.predict_distribution(X[:10])  # shape (10, n_bins_active_)
        full = binner.reconstruct_original_grid(dists)

        assert full.shape == (10, 30)
        # Mass must be conserved
        np.testing.assert_array_equal(dists.sum(axis=1), full.sum(axis=1))
        # Empty bin positions must be zero
        empty_positions = ~binner.active_bins_mask_
        assert np.all(full[:, empty_positions] == 0)

    def test_no_consolidation_reconstruct_raises(self):
        """reconstruct_original_grid without consolidation should raise RuntimeError."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=300)
        model = DDTRegressor(max_depth=3, min_samples_leaf=20, consolidate_bins=False)
        model.fit(X, y)

        dists = model.predict_distribution(X[:5])
        with pytest.raises(RuntimeError, match="consolidate_empty_bins"):
            model.target_binner_.reconstruct_original_grid(dists)
