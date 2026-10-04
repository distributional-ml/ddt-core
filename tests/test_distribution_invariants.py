"""
DDT Conservation Law Tests — Wall 1 Validation
================================================
Verifies strict physical conservation laws that must hold across
all node splits (FR-VAL-01):

    1. Mass Conservation: N_parent == N_left + N_right
    2. Bin Conservation:  counts_parent[b] == counts_left[b] + counts_right[b] for all b
    3. Wasserstein Symmetry: W_1(P, Q) == W_1(Q, P)
    4. Non-negativity: W_1(P, Q) >= 0
    5. Identity: W_1(P, P) == 0
"""

import numpy as np


def make_synthetic_data(n_samples=500, n_features=3, seed=42):
    """Generate synthetic data for conservation tests."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = 2.0 * X[:, 0] + rng.randn(n_samples) * 0.5
    return X, y


class TestMassConservation:
    """Verify N_parent == N_left + N_right for all internal nodes."""

    def test_mass_conservation(self):
        """Total sample count is conserved across every split."""
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        model = DDTRegressor(max_depth=5, min_samples_leaf=10, n_target_bins=20)
        model.fit(X, y)

        tree = model.tree_data_
        for i in range(len(tree["is_leaf"])):
            if not tree["is_leaf"][i]:
                parent_total = tree["total_samples"][i]
                left_total = tree["total_samples"][tree["left_child_id"][i]]
                right_total = tree["total_samples"][tree["right_child_id"][i]]

                assert parent_total == left_total + right_total, (
                    f"Node {i}: mass not conserved. Parent={parent_total}, Left={left_total}, Right={right_total}"
                )


class TestBinConservation:
    """Verify per-bin count conservation across splits."""

    def test_bin_conservation(self):
        """counts_parent[b] == counts_left[b] + counts_right[b] for all bins."""
        from ddt import DDTRegressor, _ddt_core

        X, y = make_synthetic_data(n_samples=500)
        n_bins = 20
        model = DDTRegressor(max_depth=5, min_samples_leaf=10, n_target_bins=n_bins)
        model.fit(X, y)

        tree = model.tree_data_

        for i in range(len(tree["is_leaf"])):
            if not tree["is_leaf"][i]:
                parent_counts = np.asarray(_ddt_core.get_node_distribution(tree, i))
                left_counts = np.asarray(_ddt_core.get_node_distribution(tree, tree["left_child_id"][i]))
                right_counts = np.asarray(_ddt_core.get_node_distribution(tree, tree["right_child_id"][i]))

                np.testing.assert_array_equal(
                    parent_counts,
                    left_counts + right_counts,
                    err_msg=f"Node {i}: per-bin counts not conserved",
                )


class TestWassersteinProperties:
    """Verify mathematical properties of the Wasserstein distance."""

    def test_symmetry(self):
        """W_1(P, Q) == W_1(Q, P)."""
        from ddt import _ddt_core

        left = np.array([10, 5, 3, 2, 0], dtype=np.int32)
        right = np.array([0, 2, 3, 5, 10], dtype=np.int32)

        w_lr = _ddt_core.calculate_wasserstein(left, right)
        w_rl = _ddt_core.calculate_wasserstein(right, left)

        assert abs(w_lr - w_rl) < 1e-12, f"Wasserstein not symmetric: W(L,R)={w_lr}, W(R,L)={w_rl}"

    def test_non_negativity(self):
        """W_1(P, Q) >= 0 for all P, Q."""
        from ddt import _ddt_core

        rng = np.random.RandomState(123)
        for _ in range(50):
            left = rng.randint(0, 20, size=10).astype(np.int32)
            right = rng.randint(0, 20, size=10).astype(np.int32)
            # Ensure non-empty histograms.
            if left.sum() == 0:
                left[0] = 1
            if right.sum() == 0:
                right[0] = 1
            w = _ddt_core.calculate_wasserstein(left, right)
            assert w >= -1e-15, f"Wasserstein negative: {w}"

    def test_identity(self):
        """W_1(P, P) == 0."""
        from ddt import _ddt_core

        counts = np.array([5, 10, 15, 10, 5], dtype=np.int32)
        w = _ddt_core.calculate_wasserstein(counts, counts)
        assert abs(w) < 1e-12, f"W_1(P,P) should be 0, got {w}"

    def test_maximum_separation(self):
        """Maximally separated distributions should have highest W1."""
        from ddt import _ddt_core

        # All mass in first bin vs. all mass in last bin.
        left = np.array([20, 0, 0, 0, 0], dtype=np.int32)
        right = np.array([0, 0, 0, 0, 20], dtype=np.int32)
        w_max = _ddt_core.calculate_wasserstein(left, right)

        # Partially overlapping distributions.
        left2 = np.array([10, 10, 0, 0, 0], dtype=np.int32)
        right2 = np.array([0, 0, 0, 10, 10], dtype=np.int32)
        w_partial = _ddt_core.calculate_wasserstein(left2, right2)

        assert w_max > w_partial, f"Max separation W1={w_max} should exceed partial W1={w_partial}"

    def test_wasserstein_gain_monotonicity(self):
        # C-4: Wasserstein gain monotonicity
        from ddt import DDTRegressor

        X, y = make_synthetic_data(n_samples=500)
        min_gain = 0.05
        model = DDTRegressor(max_depth=4, min_samples_leaf=10, min_divergence_decrease=min_gain).fit(X, y)

        tree = model.tree_data_
        is_leaf = tree["is_leaf"]
        gain = tree["wasserstein_gain"]

        for i in range(len(is_leaf)):
            if not is_leaf[i]:
                # At internal nodes, gain must be >= min_divergence_decrease
                assert gain[i] >= min_gain - 1e-9, f"Node {i} split with gain {gain[i]} < {min_gain}"


class TestRootNode:
    """Verify root node properties."""

    def test_root_contains_all_samples(self):
        """Root node should contain exactly N samples."""
        from ddt import DDTRegressor

        n_samples = 300
        X, y = make_synthetic_data(n_samples=n_samples)
        model = DDTRegressor(max_depth=4, min_samples_leaf=10)
        model.fit(X, y)

        root_samples = model.tree_data_["total_samples"][0]
        assert root_samples == n_samples, f"Root has {root_samples} samples, expected {n_samples}"

    def test_leaf_samples_sum_to_total(self):
        """Sum of all leaf sample counts should equal total samples."""
        from ddt import DDTRegressor

        n_samples = 400
        X, y = make_synthetic_data(n_samples=n_samples)
        model = DDTRegressor(max_depth=5, min_samples_leaf=10)
        model.fit(X, y)

        tree = model.tree_data_
        leaf_total = sum(tree["total_samples"][i] for i in range(len(tree["is_leaf"])) if tree["is_leaf"][i])
        assert leaf_total == n_samples, f"Leaf sum={leaf_total}, expected {n_samples}"
