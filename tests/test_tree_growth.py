import numpy as np
import pytest

from ddt import DDTRegressor


@pytest.fixture
def make_data():
    def _make(n_samples=500, n_features=5, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, size=(n_samples, n_features))
        # Make y depend heavily on feature 0 and 1 so they are chosen to split
        y = np.sin(3 * X[:, 0]) + X[:, 1] + rng.normal(0, 0.1, size=n_samples)
        return X, y

    return _make


def test_g1_feature_limited_to_1_split(make_data):
    X, y = make_data()
    # Feature 0 should naturally split many times. We limit it to 1.
    model = DDTRegressor(max_depth=5, min_samples_leaf=5, max_splits_per_feature={0: 1})
    model.fit(X, y)

    td = model.tree_data_
    is_leaf = td["is_leaf"]
    split_feat = td["split_feature_idx"][is_leaf == 0]

    assert np.count_nonzero(split_feat == 0) <= 1


def test_g2_feature_limited_to_0_splits(make_data):
    X, y = make_data()
    model = DDTRegressor(max_depth=5, min_samples_leaf=5, max_splits_per_feature={0: 0})
    model.fit(X, y)

    td = model.tree_data_
    is_leaf = td["is_leaf"]
    split_feat = td["split_feature_idx"][is_leaf == 0]

    assert np.count_nonzero(split_feat == 0) == 0


def test_g3_unconstrained_features_unaffected(make_data):
    X, y = make_data()

    model_unconstrained = DDTRegressor(max_depth=5, min_samples_leaf=5)
    model_unconstrained.fit(X, y)

    model_constrained = DDTRegressor(max_depth=5, min_samples_leaf=5, max_splits_per_feature={0: 0})
    model_constrained.fit(X, y)

    # Feature 1 should still split freely
    td_c = model_constrained.tree_data_
    is_leaf_c = td_c["is_leaf"]
    split_feat_c = td_c["split_feature_idx"][is_leaf_c == 0]

    assert np.count_nonzero(split_feat_c == 1) > 0
    # Depth should not be trivially 0
    assert np.max(td_c["depth"]) > 0


def test_g4_default_matches_existing(make_data):
    X, y = make_data()

    model_unconstrained = DDTRegressor(max_depth=5, min_samples_leaf=5, max_splits_per_feature=None)
    model_unconstrained.fit(X, y)

    model_empty_dict = DDTRegressor(max_depth=5, min_samples_leaf=5, max_splits_per_feature={})
    model_empty_dict.fit(X, y)

    # Should be identical
    np.testing.assert_array_equal(
        model_unconstrained.tree_data_["split_feature_idx"], model_empty_dict.tree_data_["split_feature_idx"]
    )
    np.testing.assert_array_equal(
        model_unconstrained.tree_data_["split_threshold"], model_empty_dict.tree_data_["split_threshold"]
    )


def test_g6_weighted_tree_path(make_data):
    X, y = make_data()
    # Using consolidate_bins=True triggers the weighted path (delta_x_norm is passed)
    model = DDTRegressor(max_depth=5, min_samples_leaf=5, consolidate_bins=True, max_splits_per_feature={0: 1})
    model.fit(X, y)

    assert model.target_binner_.delta_x_norm_ is not None

    td = model.tree_data_
    is_leaf = td["is_leaf"]
    split_feat = td["split_feature_idx"][is_leaf == 0]

    assert np.count_nonzero(split_feat == 0) <= 1


def test_g7_string_keys_with_pandas(make_data):
    X, y = make_data()
    import pandas as pd

    X_df = pd.DataFrame(X, columns=["feat_A", "feat_B", "feat_C", "feat_D", "feat_E"])

    model = DDTRegressor(max_depth=5, min_samples_leaf=5, max_splits_per_feature={"feat_A": 1})
    model.fit(X_df, y)

    td = model.tree_data_
    is_leaf = td["is_leaf"]
    split_feat = td["split_feature_idx"][is_leaf == 0]

    assert np.count_nonzero(split_feat == 0) <= 1


def test_g8_string_keys_without_pandas_raises(make_data):
    X, y = make_data()

    model = DDTRegressor(max_depth=5, min_samples_leaf=5, max_splits_per_feature={"feat_A": 1})
    with pytest.raises(ValueError, match="not a pandas DataFrame"):
        model.fit(X, y)
