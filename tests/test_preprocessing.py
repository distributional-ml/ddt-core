import numpy as np
import pytest

from ddt import DDTRegressor
from ddt._preprocessor import TargetBinner


def test_target_binner_winsorize():
    # Create target with an extreme outlier
    y = np.linspace(0, 10, 99)
    y = np.append(y, [1000])  # extreme outlier

    binner = TargetBinner(n_bins=10, winsorize_tails=0.02)  # clip top and bottom 2%
    binner.fit(y)

    # 1000 should be clipped to the 98th percentile which is around 9.8
    assert binner.y_max_ < 20
    assert binner.winsorize_upper_ < 20
    assert binner.y_max_ == binner.winsorize_upper_

    # Transform should still work, mapping 1000 to the highest bin (9)
    y_q = binner.transform([1000])
    assert y_q[0] == 9


def test_target_binner_log1p():
    y = np.array([0, 10, 100, 1000])

    binner = TargetBinner(n_bins=5, target_transform="log1p", winsorize_tails=None)
    binner.fit(y)

    assert np.isclose(binner.y_max_, np.log1p(1000))

    y_q = binner.transform(y)

    # Centers should be reconstructed using expm1
    centers = binner.inverse_transform_bin_centers()
    # The max center should be roughly expm1 of the top bin center
    assert centers[-1] > 100


def test_target_binner_log1p_negative():
    y = np.array([-2, 0, 10])
    binner = TargetBinner(n_bins=5, target_transform="log1p", winsorize_tails=None)
    with pytest.raises(ValueError, match="y contains values <= -1"):
        binner.fit(y)


def test_ddt_regressor_integration():
    X = np.random.rand(100, 2)
    y = np.random.rand(99)
    y = np.append(y, 100)  # outlier

    model = DDTRegressor(winsorize_tails=0.01, target_transform="log1p")
    model.fit(X, y)

    assert model.target_binner_.target_transform == "log1p"
    assert model.target_binner_.winsorize_upper_ < 10

    # predict should return correctly scaled values (expm1 applied by inverse transform)
    preds = model.predict(X)
    assert np.max(preds) < 10  # Since 100 was clipped and we reconstruct via expm1



from ddt import DDTRandomForestRegressor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n=800, n_features=4, seed=0):
    """Strictly positive target (required for log_width tests)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = np.abs(X[:, 0]) * 5.0 + rng.uniform(0.5, 2.0, size=n)  # y > 0 always
    return X, y


def _make_sparse_target(n=600, n_bins=20, seed=1):
    """Target concentrated in a small range so many bins will be empty."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    # All values in [1, 5]; with n_bins=20 over [0.5, 200] many bins are empty
    y = rng.uniform(1.0, 5.0, size=n)
    return X, y


# ===========================================================================
# 1. TargetBinner — bin strategy unit tests
# ===========================================================================


class TestTargetBinnerEqualWidth:
    def test_bin_edges_are_linspace(self):
        y = np.linspace(1.0, 10.0, 500)
        binner = TargetBinner(n_bins=20, winsorize_tails=None)
        binner.fit(y)
        expected = np.linspace(binner.y_min_, binner.y_max_, 21)
        np.testing.assert_allclose(binner.bin_edges_, expected)

    def test_delta_x_norm_is_none_without_consolidation(self):
        """Fast equal-width path must not allocate delta_x_norm_."""
        y = np.linspace(1.0, 10.0, 500)
        binner = TargetBinner(n_bins=20, winsorize_tails=None)
        binner.fit(y)
        assert binner.delta_x_norm_ is None

    def test_n_bins_active_equals_n_bins_without_consolidation(self):
        y = np.linspace(1.0, 10.0, 500)
        binner = TargetBinner(n_bins=20, winsorize_tails=None)
        binner.fit_transform(y)
        assert binner.n_bins_active_ == 20


class TestTargetBinnerLogWidth:
    def test_bin_edges_are_geomspace(self):
        y = np.geomspace(1.0, 100.0, 400)
        binner = TargetBinner(n_bins=15, bin_strategy="log_width", winsorize_tails=None)
        binner.fit(y)
        expected = np.geomspace(binner.y_min_, binner.y_max_, 16)
        np.testing.assert_allclose(binner.bin_edges_, expected, rtol=1e-10)

    def test_delta_x_norm_non_none(self):
        y = np.geomspace(1.0, 100.0, 400)
        binner = TargetBinner(n_bins=15, bin_strategy="log_width", winsorize_tails=None)
        binner.fit(y)
        assert binner.delta_x_norm_ is not None
        assert binner.delta_x_norm_.shape == (15,)

    def test_delta_x_norm_mean_is_one(self):
        """Normalised bin widths must have mean == 1.0 by construction."""
        y = np.geomspace(1.0, 1000.0, 500)
        binner = TargetBinner(n_bins=20, bin_strategy="log_width", winsorize_tails=None)
        binner.fit(y)
        np.testing.assert_allclose(np.mean(binner.delta_x_norm_), 1.0, rtol=1e-12)

    def test_transform_output_in_range(self):
        y = np.geomspace(1.0, 100.0, 400)
        binner = TargetBinner(n_bins=15, bin_strategy="log_width", winsorize_tails=None)
        y_q = binner.fit_transform(y)
        assert y_q.min() >= 0
        assert y_q.max() <= 14


class TestTargetBinnerManual:
    def test_manual_edges_stored_correctly(self):
        edges = np.array([0.0, 1.0, 3.0, 7.0, 14.0, 30.0], dtype=float)
        binner = TargetBinner(n_bins=5, bin_strategy="manual", manual_bin_edges=edges, winsorize_tails=None)
        y = np.linspace(0.1, 29.0, 300)
        binner.fit(y)
        np.testing.assert_array_equal(binner.bin_edges_, edges)

    def test_delta_x_norm_reflects_manual_widths(self):
        edges = np.array([0.0, 1.0, 3.0, 7.0, 14.0, 30.0], dtype=float)
        binner = TargetBinner(n_bins=5, bin_strategy="manual", manual_bin_edges=edges, winsorize_tails=None)
        y = np.linspace(0.1, 29.0, 300)
        binner.fit(y)
        widths = np.diff(edges)
        expected_norm = widths / widths.mean()
        expected_norm = widths / widths.mean()
        np.testing.assert_allclose(binner.delta_x_norm_, expected_norm, rtol=1e-12)


class TestTargetBinnerHybrid:
    def test_bin_edges_hybrid_construction(self):
        y = np.linspace(-10.0, 50.0, 1000)  # y_min < 0, IQR = [5, 35]
        # With n_bins=10 and core_fraction=0.4:
        # n_core = 4 (Q25 to Q75), n_tail = 6 -> n_lo = 3, n_hi = 3
        binner = TargetBinner(n_bins=10, bin_strategy="hybrid", core_fraction=0.4, winsorize_tails=None)
        binner.fit(y)

        edges = binner.bin_edges_
        assert len(edges) == 11

        # Check core section: Q25=5.0, Q75=35.0, n_core=4
        # edges 3, 4, 5, 6, 7 should be [5, 12.5, 20, 27.5, 35]
        core_expected = np.linspace(5.0, 35.0, 5)
        np.testing.assert_allclose(edges[3:8], core_expected, rtol=1e-7)

        # Ensure delta_x_norm is set properly
        assert binner.delta_x_norm_ is not None
        np.testing.assert_allclose(np.mean(binner.delta_x_norm_), 1.0, rtol=1e-12)

    def test_hybrid_positive_ymin(self):
        y = np.linspace(10.0, 100.0, 1000)  # y_min > 0
        binner = TargetBinner(n_bins=12, bin_strategy="hybrid", core_fraction=0.5, winsorize_tails=None)
        binner.fit(y)
        assert len(binner.bin_edges_) == 13
        assert binner.bin_edges_[0] == 10.0
        assert binner.bin_edges_[-1] == 100.0


# ===========================================================================
# 2. TargetBinner — validation errors
# ===========================================================================


class TestTargetBinnerValidation:
    def test_log_width_with_log1p_raises(self):
        with pytest.raises(ValueError, match="Cannot combine"):
            TargetBinner(n_bins=10, bin_strategy="log_width", target_transform="log1p")

    def test_manual_without_edges_raises(self):
        with pytest.raises(ValueError, match="manual_bin_edges is required"):
            TargetBinner(n_bins=5, bin_strategy="manual")

    def test_manual_with_target_transform_raises(self):
        with pytest.raises(ValueError, match="Cannot combine"):
            TargetBinner(
                n_bins=5, bin_strategy="manual", manual_bin_edges=np.linspace(0, 10, 6), target_transform="log1p"
            )

    def test_log_width_with_non_positive_ymin_raises(self):
        binner = TargetBinner(n_bins=10, bin_strategy="log_width", winsorize_tails=None)
        y = np.linspace(-1.0, 10.0, 200)  # y_min < 0
        with pytest.raises(ValueError, match="y_min > 0"):
            binner.fit(y)

    def test_manual_wrong_edge_count_raises(self):
        binner = TargetBinner(n_bins=5, bin_strategy="manual", manual_bin_edges=np.linspace(0, 10, 4))  # needs 6
        y = np.linspace(0.5, 9.5, 200)
        with pytest.raises(ValueError, match="length n_bins \\+ 1"):
            binner.fit(y)

    def test_manual_non_monotone_edges_raises(self):
        edges = np.array([0.0, 5.0, 3.0, 10.0, 15.0, 20.0])  # not monotone
        binner = TargetBinner(n_bins=5, bin_strategy="manual", manual_bin_edges=edges)
        y = np.linspace(0.5, 19.0, 200)
        with pytest.raises(ValueError, match="monotonically increasing"):
            binner.fit(y)

    def test_hybrid_invalid_core_fraction_raises(self):
        with pytest.raises(ValueError, match="core_fraction must be strictly in"):
            TargetBinner(n_bins=10, bin_strategy="hybrid", core_fraction=1.5).fit(np.linspace(1, 10, 100))

    def test_hybrid_too_few_core_bins_raises(self):
        with pytest.raises(ValueError, match="yields too few core bins"):
            # 10 * 0.1 = 1 core bin < 2
            TargetBinner(n_bins=10, bin_strategy="hybrid", core_fraction=0.1).fit(np.linspace(1, 10, 100))

    def test_hybrid_zero_iqr_raises(self):
        y = np.concatenate([np.ones(100), np.array([2.0])])  # Q25 = Q75 = 1.0
        with pytest.raises(ValueError, match="Target y has identical 25th and 75th percentiles"):
            TargetBinner(n_bins=10, bin_strategy="hybrid", winsorize_tails=None).fit(y)


# ===========================================================================
# 3. Bin consolidation unit tests
# ===========================================================================


class TestBinConsolidation:
    def _make_sparse_binner(self, n_bins=30):
        """Return a fitted binner where several bins will be empty."""
        X, y = _make_sparse_target(n=600, n_bins=n_bins)
        # Wide range ensures many bins are empty.
        edges = np.concatenate([np.linspace(0.5, 5.5, n_bins), [200.0]])
        binner = TargetBinner(
            n_bins=n_bins,
            bin_strategy="manual",
            manual_bin_edges=edges,
            consolidate_bins=True,
            winsorize_tails=None,
        )
        y_q = binner.fit_transform(y)
        return binner, y_q

    def test_n_bins_active_leq_n_bins(self):
        binner, _ = self._make_sparse_binner(n_bins=30)
        assert binner.n_bins_active_ <= binner.n_bins
        assert binner.n_bins_active_ >= 1

    def test_active_bins_mask_shape_and_count(self):
        binner, _ = self._make_sparse_binner(n_bins=30)
        assert binner.active_bins_mask_.shape == (30,)
        assert int(binner.active_bins_mask_.sum()) == binner.n_bins_active_

    def test_bin_remap_shape_and_range(self):
        binner, _ = self._make_sparse_binner(n_bins=30)
        assert binner.bin_remap_.shape == (30,)
        assert binner.bin_remap_.min() >= 0
        assert binner.bin_remap_.max() == binner.n_bins_active_ - 1

    def test_y_q_output_in_active_range(self):
        binner, y_q = self._make_sparse_binner(n_bins=30)
        assert y_q.min() >= 0
        assert y_q.max() <= binner.n_bins_active_ - 1

    def test_delta_x_norm_set_after_consolidation(self):
        """Consolidation on equal_width must trigger the weighted path."""
        y = np.linspace(1.0, 3.0, 400)  # narrow range → many empty bins at edges
        binner = TargetBinner(n_bins=30, consolidate_bins=True, winsorize_tails=None)
        binner.fit_transform(y)
        # If any bin was empty, delta_x_norm_ must be set.
        if binner.n_bins_active_ < binner.n_bins:
            assert binner.delta_x_norm_ is not None
            assert binner.delta_x_norm_.shape == (binner.n_bins_active_,)

    def test_delta_x_norm_mean_is_one_after_consolidation(self):
        binner, _ = self._make_sparse_binner(n_bins=30)
        if binner.delta_x_norm_ is not None:
            np.testing.assert_allclose(np.mean(binner.delta_x_norm_), 1.0, rtol=1e-12)


# ===========================================================================
# 4. reconstruct_original_grid
# ===========================================================================


class TestReconstructOriginalGrid:
    def _consolidated_binner_and_dist(self):
        X, y = _make_sparse_target(n=600)
        edges = np.concatenate([np.linspace(0.5, 5.5, 20), [200.0]])
        binner = TargetBinner(
            n_bins=20,
            bin_strategy="manual",
            manual_bin_edges=edges,
            consolidate_bins=True,
            winsorize_tails=None,
        )
        binner.fit_transform(y)
        # Fake a distribution over active bins.
        rng = np.random.default_rng(7)
        active_dist = rng.dirichlet(np.ones(binner.n_bins_active_), size=5).astype(np.float32)
        return binner, active_dist

    def test_output_shape(self):
        binner, active_dist = self._consolidated_binner_and_dist()
        result = binner.reconstruct_original_grid(active_dist)
        assert result.shape == (5, 20)

    def test_mass_conservation(self):
        """Sum over original grid must equal sum over active grid per sample."""
        binner, active_dist = self._consolidated_binner_and_dist()
        result = binner.reconstruct_original_grid(active_dist)
        np.testing.assert_allclose(result.sum(axis=1), active_dist.sum(axis=1), rtol=1e-6)

    def test_empty_bins_are_zero(self):
        binner, active_dist = self._consolidated_binner_and_dist()
        result = binner.reconstruct_original_grid(active_dist)
        empty_positions = ~binner.active_bins_mask_
        assert np.all(result[:, empty_positions] == 0.0)

    def test_raises_without_consolidation(self):
        binner = TargetBinner(n_bins=10, winsorize_tails=None)
        binner.fit(np.linspace(1, 10, 200))
        fake_dist = np.ones((3, 10), dtype=np.float32)
        with pytest.raises(RuntimeError, match="consolidate_empty_bins"):
            binner.reconstruct_original_grid(fake_dist)

    def test_raises_on_wrong_last_dim(self):
        binner, active_dist = self._consolidated_binner_and_dist()
        bad_dist = np.ones((5, binner.n_bins_active_ + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="n_bins_active_"):
            binner.reconstruct_original_grid(bad_dist)


# ===========================================================================
# 5. DDTRegressor shape guards
# ===========================================================================


class TestEstimatorShapeGuards:
    """Ensure predict_distribution returns (n_samples, n_bins_active_) always."""

    def test_default_shape_equals_n_target_bins(self):
        """Default path: n_bins_active_ == n_target_bins, shape unchanged."""
        X, y = _make_data()
        model = DDTRegressor(n_target_bins=20, max_depth=4, min_samples_leaf=20)
        model.fit(X, y)
        dist = model.predict_distribution(X[:10])
        assert dist.shape == (10, 20)
        assert dist.shape[1] == model.target_binner_.n_bins_active_

    def test_log_width_shape_equals_n_target_bins(self):
        """log_width with no consolidation: n_bins_active_ == n_target_bins."""
        X, y = _make_data()
        model = DDTRegressor(
            n_target_bins=20,
            bin_strategy="log_width",
            max_depth=4,
            min_samples_leaf=20,
            winsorize_tails=None,
        )
        model.fit(X, y)
        dist = model.predict_distribution(X[:10])
        assert dist.shape == (10, 20)
        assert dist.shape[1] == model.target_binner_.n_bins_active_

    def test_consolidation_shape_uses_n_bins_active(self):
        """With consolidation the last dim must be n_bins_active_, not n_target_bins."""
        X, y = _make_sparse_target(n=600)
        edges = np.concatenate([np.linspace(0.5, 5.5, 25), [200.0]])
        model = DDTRegressor(
            n_target_bins=25,
            bin_strategy="manual",
            manual_bin_edges=edges,
            consolidate_bins=True,
            max_depth=4,
            min_samples_leaf=15,
            winsorize_tails=None,
        )
        model.fit(X, y)
        n_active = model.target_binner_.n_bins_active_
        dist = model.predict_distribution(X[:10])

        # The shape must match n_bins_active_, not the original n_target_bins=25.
        assert dist.shape == (10, n_active), (
            f"Expected (10, {n_active}), got {dist.shape}. n_target_bins=25 but n_bins_active_={n_active}."
        )
        # Sanity: if any bins were consolidated the active count is strictly less.
        if n_active < 25:
            assert dist.shape[1] < 25

    def test_predict_returns_finite_values_log_width(self):
        X, y = _make_data()
        model = DDTRegressor(
            n_target_bins=15,
            bin_strategy="log_width",
            max_depth=4,
            min_samples_leaf=20,
            winsorize_tails=None,
        )
        model.fit(X, y)
        preds = model.predict(X[:20])
        assert preds.shape == (20,)
        assert np.all(np.isfinite(preds))

    def test_quantile_monotonicity_log_width(self):
        X, y = _make_data()
        model = DDTRegressor(
            n_target_bins=15,
            bin_strategy="log_width",
            max_depth=4,
            min_samples_leaf=20,
            winsorize_tails=None,
        )
        model.fit(X, y)
        q10 = model.predict_quantile(X[:10], 0.10)
        q50 = model.predict_quantile(X[:10], 0.50)
        q90 = model.predict_quantile(X[:10], 0.90)
        assert np.all(q10 <= q50 + 1e-9)
        assert np.all(q50 <= q90 + 1e-9)

    def test_get_tree_info_exposes_n_bins_active(self):
        X, y = _make_data()
        model = DDTRegressor(
            n_target_bins=20,
            bin_strategy="log_width",
            max_depth=3,
            min_samples_leaf=20,
            winsorize_tails=None,
        )
        model.fit(X, y)
        info = model.get_tree_info()
        assert "n_bins_active" in info
        assert info["n_bins_active"] == model.target_binner_.n_bins_active_

    def test_manual_bins_end_to_end(self):
        X, y = _make_data()
        edges = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=float)
        model = DDTRegressor(
            n_target_bins=7,
            bin_strategy="manual",
            manual_bin_edges=edges,
            max_depth=3,
            min_samples_leaf=20,
            winsorize_tails=None,
        )
        model.fit(X, y)
        dist = model.predict_distribution(X[:5])
        assert dist.shape == (5, 7)


# ===========================================================================
# 6. DDTRandomForestRegressor shape guards
# ===========================================================================


class TestForestShapeGuards:
    def test_default_shape_equals_n_target_bins(self):
        X, y = _make_data()
        model = DDTRandomForestRegressor(
            n_estimators=5,
            n_target_bins=20,
            max_depth=3,
            min_samples_leaf=20,
            random_state=0,
        )
        model.fit(X, y)
        dist = model.predict_distribution(X[:10])
        assert dist.shape == (10, 20)
        assert dist.shape[1] == model.target_binner_.n_bins_active_

    def test_log_width_shape_correct(self):
        X, y = _make_data()
        model = DDTRandomForestRegressor(
            n_estimators=5,
            n_target_bins=15,
            bin_strategy="log_width",
            max_depth=3,
            min_samples_leaf=20,
            random_state=0,
            winsorize_tails=None,
        )
        model.fit(X, y)
        n_active = model.target_binner_.n_bins_active_
        dist = model.predict_distribution(X[:10])
        assert dist.shape == (10, n_active)

    def test_consolidation_shape_uses_n_bins_active(self):
        X, y = _make_sparse_target(n=600)
        edges = np.concatenate([np.linspace(0.5, 5.5, 25), [200.0]])
        model = DDTRandomForestRegressor(
            n_estimators=5,
            n_target_bins=25,
            bin_strategy="manual",
            manual_bin_edges=edges,
            consolidate_bins=True,
            max_depth=3,
            min_samples_leaf=15,
            random_state=0,
            winsorize_tails=None,
        )
        model.fit(X, y)
        n_active = model.target_binner_.n_bins_active_
        dist = model.predict_distribution(X[:10])
        assert dist.shape == (10, n_active), f"Expected (10, {n_active}), got {dist.shape}."

    def test_forest_log_width_predict_finite(self):
        X, y = _make_data()
        model = DDTRandomForestRegressor(
            n_estimators=5,
            n_target_bins=15,
            bin_strategy="log_width",
            max_depth=3,
            min_samples_leaf=20,
            random_state=0,
            winsorize_tails=None,
        )
        model.fit(X, y)
        preds = model.predict(X[:20])
        assert np.all(np.isfinite(preds))

    def test_all_trees_share_same_n_bins_active(self):
        """Every tree in the forest must be built with the same bin count."""
        X, y = _make_data()
        model = DDTRandomForestRegressor(
            n_estimators=8,
            n_target_bins=15,
            bin_strategy="log_width",
            max_depth=3,
            min_samples_leaf=20,
            random_state=0,
            winsorize_tails=None,
        )
        model.fit(X, y)
        n_active = model.target_binner_.n_bins_active_
        for tree in model.estimators_:
            # Each tree's internal n_bins (from C++) must equal n_bins_active_.
            assert tree.tree_data_["n_bins"] == n_active, (
                f"Tree has n_bins={tree.tree_data_['n_bins']}, expected {n_active}."
            )


# ===========================================================================
# 7. Regression guard: equal_width default is unchanged
# ===========================================================================


class TestEqualWidthRegressionGuard:
    def test_tree_data_n_bins_equals_n_target_bins(self):
        X, y = _make_data()
        model = DDTRegressor(n_target_bins=30, max_depth=5, min_samples_leaf=20)
        model.fit(X, y)
        assert model.tree_data_["n_bins"] == 30

    def test_distribution_shape_unchanged(self):
        X, y = _make_data()
        model = DDTRegressor(n_target_bins=30, max_depth=5, min_samples_leaf=20)
        model.fit(X, y)
        dist = model.predict_distribution(X)
        assert dist.shape == (len(X), 30)

    def test_delta_x_norm_none_on_default(self):
        X, y = _make_data()
        model = DDTRegressor(n_target_bins=30, max_depth=5, min_samples_leaf=20)
        model.fit(X, y)
        assert model.target_binner_.delta_x_norm_ is None

    def test_predict_is_finite_on_default(self):
        X, y = _make_data()
        model = DDTRegressor(n_target_bins=30, max_depth=5, min_samples_leaf=20)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))


# ===========================================================================
# 8. Weighted C++ path dispatch sentinel
# ===========================================================================


class TestWeightedPathDispatch:
    def test_equal_width_no_consolidation_uses_fast_path(self):
        X, y = _make_data()
        model = DDTRegressor(n_target_bins=20, max_depth=3, min_samples_leaf=20)
        model.fit(X, y)
        assert model.target_binner_.delta_x_norm_ is None

    def test_log_width_uses_weighted_path(self):
        X, y = _make_data()
        model = DDTRegressor(
            n_target_bins=20,
            bin_strategy="log_width",
            max_depth=3,
            min_samples_leaf=20,
            winsorize_tails=None,
        )
        model.fit(X, y)
        assert model.target_binner_.delta_x_norm_ is not None

    def test_manual_uses_weighted_path(self):
        X, y = _make_data()
        edges = np.array([0.0, 0.5, 1.0, 2.5, 6.0, 15.0, 35.0], dtype=float)
        model = DDTRegressor(
            n_target_bins=6,
            bin_strategy="manual",
            manual_bin_edges=edges,
            max_depth=3,
            min_samples_leaf=20,
            winsorize_tails=None,
        )
        model.fit(X, y)
        assert model.target_binner_.delta_x_norm_ is not None

    def test_equal_width_with_consolidation_triggers_weighted_path_when_bins_merged(self):
        """If consolidation merges any bin, delta_x_norm_ must be set."""
        X, y = _make_sparse_target(n=600)
        edges = np.concatenate([np.linspace(0.5, 5.5, 25), [200.0]])
        model = DDTRegressor(
            n_target_bins=25,
            bin_strategy="manual",
            manual_bin_edges=edges,
            consolidate_bins=True,
            max_depth=3,
            min_samples_leaf=15,
            winsorize_tails=None,
        )
        model.fit(X, y)
        if model.target_binner_.n_bins_active_ < 25:
            assert model.target_binner_.delta_x_norm_ is not None


# ===========================================================================
# 9. KS criterion monotonicity
# ===========================================================================


class TestKSCriterionMonotonicity:
    """Larger epsilon must prune at least as many nodes as smaller epsilon."""

    def _fitted_model(self):
        X, y = _make_data(n=1000)
        model = DDTRegressor(
            n_target_bins=30,
            max_depth=6,
            min_samples_leaf=10,
            min_divergence_decrease=0.0,
        )
        model.fit(X, y)
        return model


# ===========================================================================
# 10. Hybrid Strategy End-to-End
# ===========================================================================


class TestHybridEndToEnd:
    def test_hybrid_fits_and_predicts(self):
        X, y = _make_data()
        y = y - 5.0  # Shift so y_min < 0
        model = DDTRegressor(
            n_target_bins=15,
            bin_strategy="hybrid",
            core_fraction=0.5,
            max_depth=3,
            min_samples_leaf=20,
            winsorize_tails=None,
        )
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert np.all(np.isfinite(preds))

    def test_quantile_monotonicity_hybrid(self):
        X, y = _make_data()
        model = DDTRegressor(
            n_target_bins=15,
            bin_strategy="hybrid",
            core_fraction=0.6,
            max_depth=4,
            min_samples_leaf=20,
            winsorize_tails=None,
        )
        model.fit(X, y)
        q10 = model.predict_quantile(X[:10], 0.10)
        q50 = model.predict_quantile(X[:10], 0.50)
        q90 = model.predict_quantile(X[:10], 0.90)
        assert np.all(q10 <= q50 + 1e-9)
        assert np.all(q50 <= q90 + 1e-9)

    def test_forest_hybrid_aligned_bins(self):
        X, y = _make_data()
        model = DDTRandomForestRegressor(
            n_estimators=3,
            n_target_bins=15,
            bin_strategy="hybrid",
            core_fraction=0.5,
            max_depth=3,
            min_samples_leaf=20,
            random_state=0,
            winsorize_tails=None,
        )
        model.fit(X, y)
        n_active = model.target_binner_.n_bins_active_
        for tree in model.estimators_:
            assert tree.tree_data_["n_bins"] == n_active
