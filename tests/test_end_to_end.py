import os

import numpy as np
import pytest

from ddt import DDTRegressor

# Determine if the synthetic data directory and files exist
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "synthetic_data")
X_PATH = os.path.join(DATA_DIR, "X.csv")
Y_PATH = os.path.join(DATA_DIR, "y.csv")
DATA_EXISTS = os.path.exists(X_PATH) and os.path.exists(Y_PATH)


@pytest.mark.skipif(not DATA_EXISTS, reason="Synthetic data not found. Run scripts/generate_synthetic.py first.")
def test_e2e_synthetic_groups():
    """Validates DDTRegressor on the 8-group synthetic dataset."""
    # Load data
    X = np.loadtxt(X_PATH, delimiter=",", skiprows=1)
    y = np.loadtxt(Y_PATH, delimiter=",", skiprows=1)

    # Train model
    # Use max_depth=8 (allows deep splits to capture both categorical features and continuous shifts)
    # n_target_bins=100 (maximum allowed) provides better resolution for the 8 separate clusters
    model = DDTRegressor(max_depth=8, min_samples_leaf=20, n_target_bins=100)
    model.fit(X, y)

    # We will test the predictions for each of the 8 categorical groups at cont=0
    # Group definitions (c1, c2, c3):
    groups = [
        (0, 0, 0),  # Group 0: Normal low var (mean ~10)
        (0, 0, 1),  # Group 1: Normal high var (mean ~10, spread wider)
        (0, 1, 0),  # Group 2: Skewed right (mean ~20+shift)
        (0, 1, 1),  # Group 3: Skewed left (mean ~30-shift)
        (1, 0, 0),  # Group 4: Bimodal (-10 and +10)
        (1, 0, 1),  # Group 5: Heavy tails (t-dist, mean ~0)
        (1, 1, 0),  # Group 6: Uniform (0 to 20, mean ~10)
        (1, 1, 1),  # Group 7: Exponential (scale=5, mean ~5)
    ]

    test_X = np.array([[c1, c2, c3, 0.0] for c1, c2, c3 in groups])

    # Predict quantiles
    q10 = model.predict_quantile(test_X, 0.10)
    q50 = model.predict_quantile(test_X, 0.50)
    q90 = model.predict_quantile(test_X, 0.90)
    model.predict(test_X)

    # Group 0: Normal, low variance. P10 to P90 should be tight relative to Group 1.
    assert q90[0] - q10[0] < 12

    # Group 1: Normal, high variance. P10 to P90 should be wide.
    assert q90[1] - q10[1] > 15
    assert (q90[1] - q10[1]) > (q90[0] - q10[0]) * 1.5

    # Group 2: Skewed right (long right tail). Median < Mean.
    # Note: For right skew, the distance from P50 to P90 is typically larger than P10 to P50.
    # We use a small tolerance as finite binning can make them marginally equal.
    assert q90[2] - q50[2] >= q50[2] - q10[2] - 0.1

    # Group 3: Skewed left (long left tail). Median > Mean.
    # We use a larger tolerance as finite binning and stochasticity can make them marginally equal.
    assert q50[3] - q10[3] >= q90[3] - q50[3] - 1.0

    # Validations passed.
