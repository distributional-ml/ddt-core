import pickle

import numpy as np
import pytest

from ddt import DDTRegressor


def generate_calibration_data(n_samples=2000):
    np.random.seed(42)
    X = np.random.rand(n_samples, 2)
    # Heteroskedastic noise to make calibration meaningful
    y = X[:, 0] * 2 + np.random.randn(n_samples) * (1.0 + X[:, 1])
    return X, y


@pytest.mark.parametrize("q", [0.05, 0.5, 0.95, 0.99])
def test_calibration_coverage_integrity(q):
    X, y = generate_calibration_data()
    n_train = 1000
    X_train, y_train = X[:n_train], y[:n_train]

    calib_frac = 0.2
    n_calib = int(n_train * calib_frac)

    model = DDTRegressor(
        max_depth=5,
        min_samples_leaf=20,
        n_target_bins=32,
        calibration_fraction=calib_frac,
        evt_tails=True,
        smooth_leaves=True,
        quantile_interpolation="linear",
    )
    model.fit(X_train, y_train)

    assert model.is_calibrated_

    # Get calibration set predictions
    X_calib = model.X_calib_
    y_calib = model.y_calib_

    # Predict on the calibration set and check coverage
    preds = model.predict_quantiles(X_calib, quantiles=[q])[q]

    # Coverage logic: for a calibrated lower bound q, we expect P(y <= preds) ~ q
    coverage = np.mean(y_calib <= preds)

    # Conformal target coverage should be accurate to within 1/n_calib
    tolerance = 1.0 / n_calib
    assert abs(coverage - q) <= tolerance, f"Coverage {coverage} too far from target {q}"


def test_calibrated_pickle_round_trip():
    X, y = generate_calibration_data(n_samples=1000)

    model = DDTRegressor(calibration_fraction=0.2, evt_tails=True, smooth_leaves=True)
    model.fit(X, y)

    grid_before = model.target_binner_.quantile_grid()
    preds_before = model.predict_quantiles(X[:10], quantiles=[0.1, 0.9])

    serialized = pickle.dumps(model)
    model_loaded = pickle.loads(serialized)

    grid_after = model_loaded.target_binner_.quantile_grid()
    preds_after = model_loaded.predict_quantiles(X[:10], quantiles=[0.1, 0.9])

    np.testing.assert_array_equal(grid_before[0], grid_after[0])
    np.testing.assert_array_equal(grid_before[1], grid_after[1])
    np.testing.assert_array_equal(grid_before[2], grid_after[2])

    np.testing.assert_array_equal(preds_before[0.1], preds_after[0.1])
    np.testing.assert_array_equal(preds_before[0.9], preds_after[0.9])
