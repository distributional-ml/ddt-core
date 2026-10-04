import numpy as np

from ddt import DDTRegressor
from ddt._smoothing import smooth_leaf_pmf


def make_sparse_data(n_samples=500, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, 5))
    y = X[:, 0] * 10 + rng.exponential(2, n_samples)
    return X, y


class TestSmoothingEngine:
    def test_smooth_leaf_pmf(self):
        leaf_counts = np.array([0, 10, 0, 0], dtype=np.int32)
        parent_pmf = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

        # prior_weight = 10, N = 10 -> w = 0.5
        smoothed = smooth_leaf_pmf(leaf_counts, parent_pmf, prior_weight=10.0)

        expected_leaf_pmf = np.array([0.0, 1.0, 0.0, 0.0])
        expected_smoothed = 0.5 * expected_leaf_pmf + 0.5 * parent_pmf

        assert np.allclose(smoothed, expected_smoothed)
        assert np.isclose(smoothed.sum(), 1.0)

    def test_smooth_leaf_pmf_empty(self):
        leaf_counts = np.array([0, 0], dtype=np.int32)
        parent_pmf = np.array([0.2, 0.8], dtype=np.float64)
        smoothed = smooth_leaf_pmf(leaf_counts, parent_pmf, prior_weight=10.0)
        assert np.allclose(smoothed, parent_pmf)


class TestIntegration:
    def test_estimator_integration(self):
        X, y = make_sparse_data()
        model = DDTRegressor(max_depth=4, min_samples_leaf=20, smooth_leaves=True).fit(X, y)
        assert "smoothed_pmf" in model.tree_data_

        # Check invariants
        pmf = model.tree_data_["smoothed_pmf"]
        assert pmf.dtype == np.float64
        assert np.all(pmf >= 0)
        assert np.allclose(pmf.sum(axis=1), 1.0)

    def test_quantile_monotonicity(self):
        X, y = make_sparse_data()
        model = DDTRegressor(max_depth=4, min_samples_leaf=20, smooth_leaves=True).fit(X, y)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q = model.predict_quantiles(X[:10], [0.1, 0.5, 0.9])
        assert np.all(q[0.1] <= q[0.5] + 1e-9)
        assert np.all(q[0.5] <= q[0.9] + 1e-9)


class TestLeafFusionEVTPooling:
    pass


class TestSmoothingBehavior:
    def test_shrinkage_strength_scales(self):
        # A-2: Shrinkage strength scales with lambda
        X, y = make_sparse_data(n_samples=200, seed=42)
        model_low = DDTRegressor(max_depth=3, smooth_leaves=True, smooth_prior_weight=2.0).fit(X, y)
        model_high = DDTRegressor(max_depth=3, smooth_leaves=True, smooth_prior_weight=200.0).fit(X, y)

        # High lambda should pull leaf distributions much closer to the root distribution (or uniform-ish if root is uniform)
        # We can just verify the PMFs differ between the two models
        pmf_low = model_low.tree_data_["smoothed_pmf"]
        pmf_high = model_high.tree_data_["smoothed_pmf"]

        assert not np.allclose(pmf_low, pmf_high), "High lambda should produce different PMFs than low lambda"
        # High lambda should have lower variance across leaves (more homogeneous)
        assert np.var(pmf_high) < np.var(pmf_low), "High lambda should heavily regularize leaves toward the mean"

    def test_ancestor_traversal(self):
        # A-3: Ancestor traversal fallback
        X, y = make_sparse_data(n_samples=500, seed=42)
        # min_samples_leaf=5 allows very small leaves
        # smooth_min_samples=200 forces it to walk far up the tree to find a prior
        model = DDTRegressor(max_depth=6, min_samples_leaf=5, smooth_leaves=True, smooth_min_samples=200).fit(X, y)

        assert "smoothed_pmf" in model.tree_data_
        assert np.all(np.isfinite(model.tree_data_["smoothed_pmf"]))

    def test_sparse_leaf_smoothing_effectiveness(self):
        # A-1: Sparse-leaf smoothing effectiveness
        X, y = make_sparse_data(n_samples=500, seed=42)
        model = DDTRegressor(max_depth=4, min_samples_leaf=10, smooth_leaves=True, smooth_prior_weight=20.0).fit(X, y)

        counts = model.tree_data_["distribution_counts"]
        pmfs = model.tree_data_["smoothed_pmf"]
        is_leaf = model.tree_data_["is_leaf"]

        # Find a sparse leaf
        for i in range(len(is_leaf)):
            if is_leaf[i] == 1 and counts[i].sum() < 20:
                # Calculate raw PMF
                raw_pmf = counts[i] / counts[i].sum()
                smoothed_pmf = pmfs[i]

                # Check that it was pulled away from raw empirical
                assert not np.allclose(raw_pmf, smoothed_pmf, atol=1e-3)
                break

    def test_default_model_has_no_smoothed_pmf(self):
        # C-1: Default model has no smoothed_pmf
        X, y = make_sparse_data(n_samples=100)
        model = DDTRegressor(max_depth=3).fit(X, y)
        assert "smoothed_pmf" not in model.tree_data_

        dists = model.predict_distribution(X[:5])
        assert np.all(dists >= 0)
        assert np.all(dists.sum(axis=1) > 0)
