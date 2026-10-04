import numpy as np

from ddt import DDTRegressor


def generate_intermittent_demand(n_samples=5000):
    np.random.seed(42)
    X = np.random.rand(n_samples, 3)
    is_zero = np.random.rand(n_samples) < 0.8
    spikes = np.random.lognormal(mean=2.0, sigma=1.0, size=n_samples)
    y = np.where(is_zero, 0.0, spikes)
    n_train = int(0.8 * n_samples)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def test_atoms_intermittent_demand():
    X_train, y_train, X_test, y_test = generate_intermittent_demand()

    # Model with atom_threshold enabled
    model = DDTRegressor(
        max_depth=5,
        n_target_bins=32,
        bin_strategy="equal_width",
        quantile_interpolation="linear",
        winsorize_tails=None,
    )
    model.fit(X_train, y_train)

    # Check atoms in binner
    binner = model.target_binner_
    bin_lo, bin_width, bin_rep = binner.quantile_grid()

    atom_mask = bin_width == 0.0
    assert np.any(atom_mask), "Should detect at least one atom bin"

    # Evaluate P50 prediction
    preds_p50 = model.predict_quantiles(X_test, quantiles=[0.5])[0.5]

    # Check that predictions return EXACTLY 0.0 when P(Y=0) > 0.5
    exact_zeros = np.sum(preds_p50 == 0.0)
    assert exact_zeros > 0, "Should predict exactly 0.0 for some samples"


def test_no_atoms_for_continuous_target():
    np.random.seed(42)
    X = np.random.rand(1000, 3)
    # Target with no ties
    y = X[:, 0] * 2 + np.random.randn(1000)

    model = DDTRegressor(n_target_bins=16, winsorize_tails=None)
    model.fit(X, y)

    bin_lo, bin_width, bin_rep = model.target_binner_.quantile_grid()
    assert np.all(bin_width > 0.0), "Should not detect any atoms in a continuous target"


def test_no_atoms_at_winsorization_bounds():
    np.random.seed(42)
    X = np.random.rand(5000, 3)
    # Heavy tail target
    y = np.random.lognormal(mean=2.0, sigma=2.0, size=5000)

    model = DDTRegressor(n_target_bins=32, winsorize_tails=0.1)  # 10% winsorization
    model.fit(X, y)

    bin_lo, bin_width, bin_rep = model.target_binner_.quantile_grid()
    assert np.all(bin_width > 0.0), "Should not detect atoms at winsorization bounds"
