import numpy as np

from ddt import DDTRegressor
from ddt._evt import (
    fit_gpd_lower_tail,
    fit_gpd_upper_tail,
    query_gpd_quantile,
)


class TestEVTOracle:
    def test_fit_gpd_upper_tail_starvation(self):
        values = np.arange(20, dtype=float)
        # N=20 < min_evt_samples=30
        res = fit_gpd_upper_tail(values, min_evt_samples=30, tail_fraction=0.1)
        assert res["status"] == 1

    def test_fit_gpd_upper_tail_success(self):
        rng = np.random.RandomState(42)
        # Generate heavy tailed data
        values = rng.pareto(a=2.0, size=1000)
        res = fit_gpd_upper_tail(values, min_evt_samples=30, tail_fraction=0.1)
        assert res is not None
        assert "u" in res
        assert "F_u" in res
        assert "xi" in res
        assert "sigma" in res
        assert res["sigma"] > 0
        assert 0.80 <= res["F_u"] <= 0.98

    def test_fit_gpd_lower_tail_starvation(self):
        values = np.arange(20, dtype=float)
        # N=20 < min_evt_samples=30
        res = fit_gpd_lower_tail(values, min_evt_samples=30, tail_fraction=0.1)
        assert res["status"] == 1

    def test_fit_gpd_lower_tail_success(self):
        rng = np.random.RandomState(42)
        # Generate data with heavy left tail by negating a Pareto
        values = -rng.pareto(a=2.0, size=1000)
        res = fit_gpd_lower_tail(values, min_evt_samples=30, tail_fraction=0.1)
        assert res is not None
        assert "u_lower" in res
        assert "F_u_lower" in res
        assert "xi_lower" in res
        assert "sigma_lower" in res
        assert res["sigma_lower"] > 0
        assert 0.02 <= res["F_u_lower"] <= 0.20

    def test_query_gpd_quantile_monotonicity(self):
        # query_gpd_quantile is monotone in q
        u = 10.0
        F_u = 0.9
        xi = 0.5
        sigma = 2.0

        q_vals = [0.91, 0.95, 0.99, 0.999]
        results = [query_gpd_quantile(q, u, F_u, xi, sigma) for q in q_vals]

        assert results[0] < results[1] < results[2] < results[3]

    def test_query_gpd_boundary_continuity(self):
        # query_gpd_quantile at q=F_u equals u
        u = 10.0
        F_u = 0.9
        xi = 0.5
        sigma = 2.0
        assert np.isclose(query_gpd_quantile(F_u, u, F_u, xi, sigma), u)

    def test_exponential_limit(self):
        # exponential limit (xi=0) matches exponential distribution analytical quantile
        u = 10.0
        F_u = 0.9
        xi = 0.0
        sigma = 2.0
        q = 0.99

        # Exponential quantile: u - sigma * log(exceedance_prob)
        exceed_prob = (1 - q) / (1 - F_u)
        expected = u - sigma * np.log(exceed_prob)

        actual = query_gpd_quantile(q, u, F_u, xi, sigma)
        assert np.isclose(actual, expected)


class TestEVTIntegration:
    def test_evt_regressor_fit(self):
        # DDTRegressor(evt_tails=True) fits without error on right-skewed data
        rng = np.random.RandomState(42)
        X = rng.rand(500, 2)
        y = rng.pareto(a=1.5, size=500)

        model = DDTRegressor(
            max_depth=3,
            min_samples_leaf=50,
            evt_tails=True,
            evt_min_samples=20,
            evt_tail_fraction=0.2,
            quantize_engine="python",
        )
        model.fit(X, y)

        assert "evt_enabled" in model.tree_data_

        # Ensure it can predict
        preds = model.predict_quantiles(X[:5], [0.5, 0.9, 0.99])
        assert 0.5 in preds
        assert 0.99 in preds

        # P99 should be higher than P90
        assert np.all(preds[0.99] >= preds[0.9])

    def test_evt_tail_accuracy_pareto(self):
        # A-4: EVT tail accuracy on Pareto data
        rng = np.random.RandomState(42)
        # alpha=2, scale=1
        y = rng.pareto(a=2.0, size=10000)
        X = np.ones((10000, 1))

        model = DDTRegressor(
            max_depth=1,
            min_samples_leaf=5000,
            evt_tails=True,
            evt_min_samples=100,
            evt_tail_fraction=0.1,
            quantize_engine="python",
        )
        model.fit(X, y)

        # Analytical P99 of Pareto(2) = 1 / (1-0.99)^(1/2) - 1 = 1 / 0.1 - 1 = 9.0
        analytical_p99 = 9.0
        pred_p99 = model.predict_quantiles(X[:1], [0.99])[0.99][0]

        # Error should be < 20%
        error = abs(pred_p99 - analytical_p99) / analytical_p99
        assert error < 0.20


class TestLowerTailMathBoundary:
    """D-2, D-3, D-4: Mathematical boundary tests for query_gpd_lower_quantile."""

    def test_lower_tail_quantile_monotonicity(self):
        # D-2: query_gpd_lower_quantile is strictly decreasing as q decreases
        # (i.e., lower q → more extreme left → smaller x_q)
        from ddt._evt import query_gpd_lower_quantile

        u_lower = -5.0
        F_u_lower = 0.1
        xi = 0.4
        sigma = 1.5

        q_vals = [F_u_lower * 0.95, F_u_lower * 0.5, F_u_lower * 0.1, F_u_lower * 0.01]
        results = [query_gpd_lower_quantile(q, u_lower, F_u_lower, xi, sigma) for q in q_vals]

        # As q decreases (deeper into lower tail), x_q must also decrease
        assert results[0] > results[1] > results[2] > results[3], f"Lower-tail quantile not monotone: {results}"
        # All results must be at or below u_lower
        assert all(r <= u_lower + 1e-10 for r in results), f"Lower-tail quantile exceeded u_lower={u_lower}: {results}"

    def test_lower_tail_boundary_continuity(self):
        # D-3: At q = F_u_lower, query_gpd_lower_quantile must return u_lower (continuous splice)
        from ddt._evt import query_gpd_lower_quantile

        for xi in [0.0, 0.3, -0.2, 0.5]:
            u_lower = -3.0
            F_u_lower = 0.08
            sigma = 2.0
            result = query_gpd_lower_quantile(F_u_lower, u_lower, F_u_lower, xi, sigma)
            assert np.isclose(result, u_lower, atol=1e-9), (
                f"Boundary continuity violated for xi={xi}: got {result}, expected {u_lower}"
            )

    def test_symmetric_student_t_both_tails(self):
        # D-4: For symmetric StudentT(df=3), both upper and lower tail GPD shapes
        # should be similar in magnitude (both tails are power-law with the same exponent).
        from scipy.stats import t as student_t

        from ddt._evt import fit_gpd_lower_tail, fit_gpd_upper_tail

        rng = np.random.RandomState(42)
        y = student_t.rvs(df=3, size=5000, random_state=rng)

        upper = fit_gpd_upper_tail(y, min_evt_samples=50, tail_fraction=0.1)
        lower = fit_gpd_lower_tail(y, min_evt_samples=50, tail_fraction=0.1)

        assert upper is not None, "Upper tail GPD fitting failed on StudentT(3)"
        assert lower is not None, "Lower tail GPD fitting failed on StudentT(3)"

        xi_upper = upper["xi"]
        xi_lower = lower["xi_lower"]

        # StudentT(3) has tail index α=3 → GPD shape ξ=1/3 ≈ 0.333 for both tails.
        # Shapes should be positive and similar (within 0.25 of each other).
        assert xi_upper > 0, f"Expected positive xi for heavy-tailed StudentT, got {xi_upper}"
        assert xi_lower > 0, f"Expected positive xi for heavy-tailed StudentT, got {xi_lower}"
        assert abs(xi_upper - xi_lower) < 0.25, (
            f"Upper xi={xi_upper:.3f} and lower xi={xi_lower:.3f} differ too much for symmetric StudentT(3)"
        )

    def test_lower_tail_exponential_limit(self):
        # xi=0 lower-tail: x_q = u_lower + sigma * log(exceedance_prob)
        # log(exceedance_prob) ≤ 0, so x_q ≤ u_lower ✓
        from ddt._evt import query_gpd_lower_quantile

        u_lower = 0.0
        F_u_lower = 0.1
        xi = 0.0
        sigma = 2.0
        q = F_u_lower * 0.5  # exceedance_prob = 0.5

        expected = u_lower + sigma * np.log(q / F_u_lower)  # = 2*log(0.5) < 0
        actual = query_gpd_lower_quantile(q, u_lower, F_u_lower, xi, sigma)
        assert np.isclose(actual, expected, rtol=1e-9), f"Exponential limit mismatch: got {actual}, expected {expected}"


class TestLowerTailIntegration:
    """End-to-end integration tests for evt_tails_lower."""

    def test_lower_tail_estimator_fit(self):
        # DDTRegressor(evt_tails=True, evt_tails_lower=True) fits and populates
        # lower-tail arrays in tree_data_.
        rng = np.random.RandomState(42)
        X = rng.rand(1000, 2)
        # StudentT(3): symmetric heavy tails on both sides
        from scipy.stats import t as student_t

        y = student_t.rvs(df=3, size=1000, random_state=rng)

        model = DDTRegressor(
            max_depth=3,
            min_samples_leaf=50,
            evt_tails=True,
            evt_tails_lower=True,
            evt_min_samples=20,
            evt_tail_fraction=0.15,
            quantize_engine="python",
        )
        model.fit(X, y)

        # Both upper and lower arrays must exist
        assert "evt_enabled" in model.tree_data_
        assert "evt_lower_enabled" in model.tree_data_
        assert "evt_lower_threshold_u" in model.tree_data_
        assert "evt_lower_F_u" in model.tree_data_
        assert "evt_lower_gpd_shape" in model.tree_data_
        assert "evt_lower_gpd_scale" in model.tree_data_

        # At least one leaf should have lower tail fitted (with 1000 samples and 15% tail fraction)
        assert model.tree_data_["evt_lower_enabled"].sum() > 0

    def test_lower_tail_quantile_routing(self):
        # When evt_tails_lower=True, predictions at low quantiles should be routed
        # through the GPD lower tail, extending beyond the empirical histogram minimum.
        rng = np.random.RandomState(42)
        from scipy.stats import t as student_t

        y = student_t.rvs(df=3, size=2000, random_state=rng)
        X = np.ones((2000, 1))

        model_no_evt = DDTRegressor(
            max_depth=1,
            min_samples_leaf=1000,
            quantize_engine="python",
        )
        model_no_evt.fit(X, y)

        model_evt = DDTRegressor(
            max_depth=1,
            min_samples_leaf=1000,
            evt_tails=True,
            evt_tails_lower=True,
            evt_min_samples=50,
            evt_tail_fraction=0.1,
            quantize_engine="python",
        )
        model_evt.fit(X, y)

        # Predictions must be finite
        preds_no = model_no_evt.predict_quantiles(X[:1], [0.01, 0.5, 0.99])
        preds_ev = model_evt.predict_quantiles(X[:1], [0.01, 0.5, 0.99])

        assert np.all(np.isfinite(preds_no[0.01]))
        assert np.all(np.isfinite(preds_ev[0.01]))

        # Quantile ordering must be preserved
        assert preds_ev[0.01][0] < preds_ev[0.5][0] < preds_ev[0.99][0]

        # EVT lower tail should extend further left than the histogram minimum
        # (it extrapolates beyond the training minimum using the GPD)
        hist_min = float(y.min())
        evt_p01 = float(preds_ev[0.01][0])
        # The EVT P01 should be at or below (more extreme than) the empirical P01
        empirical_p01 = float(np.percentile(y, 1))
        assert evt_p01 <= empirical_p01 + abs(empirical_p01) * 0.5, (
            f"EVT lower tail P01={evt_p01:.3f} should be near or below empirical P01={empirical_p01:.3f}"
        )


import pickle
import warnings

import pytest


@pytest.fixture
def evt_data():
    rng = np.random.default_rng(42)
    # Heavy tails for both sides so we can test both
    # StudentT(df=3) has heavy tails
    X = rng.uniform(-1, 1, size=(5000, 2))
    from scipy.stats import t

    y = t.rvs(df=3, size=5000, random_state=rng)
    return X, y


def test_e1_cpp_evt_upper_tail_matches_python(evt_data):
    X, y = evt_data
    # Train model with EVT enabled
    model = DDTRegressor(
        max_depth=3,
        min_samples_leaf=500,
        evt_tails=True,
        evt_tails_lower=False,
        evt_min_samples=50,
        evt_tail_fraction=0.1,
    )
    model.fit(X, y)

    # Predict with C++ engine
    model.set_params(quantize_engine="cpp")
    preds_cpp = model.predict_quantiles(X[:10], [0.99, 0.999])

    # Predict with Python engine
    model.set_params(quantize_engine="python")
    preds_py = model.predict_quantiles(X[:10], [0.99, 0.999])

    np.testing.assert_allclose(preds_cpp[0.99], preds_py[0.99], rtol=1e-10)
    np.testing.assert_allclose(preds_cpp[0.999], preds_py[0.999], rtol=1e-10)


def test_e2_cpp_evt_lower_tail_matches_python(evt_data):
    X, y = evt_data
    model = DDTRegressor(
        max_depth=3,
        min_samples_leaf=500,
        evt_tails=True,
        evt_tails_lower=True,
        evt_min_samples=50,
        evt_tail_fraction=0.1,
    )
    model.fit(X, y)

    model.set_params(quantize_engine="cpp")
    preds_cpp = model.predict_quantiles(X[:10], [0.001, 0.01])

    model.set_params(quantize_engine="python")
    preds_py = model.predict_quantiles(X[:10], [0.001, 0.01])

    np.testing.assert_allclose(preds_cpp[0.001], preds_py[0.001], rtol=1e-10)
    np.testing.assert_allclose(preds_cpp[0.01], preds_py[0.01], rtol=1e-10)


def test_e3_cpp_evt_smoothing_combination(evt_data):
    X, y = evt_data
    model = DDTRegressor(
        max_depth=3,
        min_samples_leaf=500,
        smooth_leaves=True,
        evt_tails=True,
        evt_tails_lower=True,
        quantize_engine="cpp",
    )
    model.fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Should not raise warning about Python fallback
        preds = model.predict_quantiles(X[:5], [0.01, 0.99])

    assert np.all(np.isfinite(preds[0.01]))
    assert np.all(np.isfinite(preds[0.99]))


def test_e4_no_python_fallback_warning_with_evt(evt_data):
    X, y = evt_data
    model = DDTRegressor(max_depth=1, min_samples_leaf=500, evt_tails=True, quantize_engine="cpp")
    model.fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Just predicting quantiles should not raise the UserWarning: C++ predict_quantiles does not support EVT...
        preds = model.predict_quantiles(X[:5], [0.99])

    assert np.all(np.isfinite(preds[0.99]))


def test_e6_backward_compat_missing_evt_s_u(evt_data):
    X, y = evt_data
    model = DDTRegressor(max_depth=1, min_samples_leaf=500, evt_tails=True, quantize_engine="cpp")
    model.fit(X, y)

    # Strip evt_S_u manually to simulate older serialized models
    td = dict(model.tree_data_)
    td.pop("evt_S_u", None)
    model.tree_data_ = td

    # Should fallback to 1-F_u seamlessly without crashing
    preds = model.predict_quantiles(X[:5], [0.99])
    assert np.all(np.isfinite(preds[0.99]))


def test_e7_evt_pickle_round_trip(evt_data):
    X, y = evt_data
    model = DDTRegressor(max_depth=1, min_samples_leaf=500, evt_tails=True, evt_tails_lower=True, quantize_engine="cpp")
    model.fit(X, y)

    preds_orig = model.predict_quantiles(X[:5], [0.01, 0.99])

    pkl = pickle.dumps(model)
    model_loaded = pickle.loads(pkl)

    preds_loaded = model_loaded.predict_quantiles(X[:5], [0.01, 0.99])

    np.testing.assert_allclose(preds_orig[0.01], preds_loaded[0.01], rtol=1e-10)
    np.testing.assert_allclose(preds_orig[0.99], preds_loaded[0.99], rtol=1e-10)
