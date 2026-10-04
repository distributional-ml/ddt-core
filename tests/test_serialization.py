import pickle

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing

from ddt import DDTRandomForestRegressor, DDTRegressor


@pytest.fixture(scope="module")
def data():
    X, y = fetch_california_housing(return_X_y=True)
    return X[:500], y[:500]


@pytest.mark.parametrize("model_cls", [DDTRegressor, DDTRandomForestRegressor])
@pytest.mark.parametrize(
    "config",
    [
        {"fast_inference": True},
        {"smooth_leaves": True},
        {"evt_tails": True},
        {"evt_tails": True, "evt_tails_lower": True},
        {"compact_inference": True},
        {"calibration_fraction": 0.2},
    ],
)
def test_serialization_roundtrip(data, model_cls, config):
    X, y = data
    # Forest doesn't support fast_inference directly yet, skip or use compact_inference
    if model_cls == DDTRandomForestRegressor and config.get("fast_inference"):
        pytest.skip("Forest doesn't support fast_inference yet")

    kwargs = dict(max_depth=4, n_target_bins=16, **config)

    if model_cls == DDTRandomForestRegressor:
        kwargs.pop("evt_tails_lower", None)
        kwargs.pop("calibration_fraction", None)
        kwargs.pop("fast_inference", None)
        kwargs["n_estimators"] = 5

    # random_state is standard on Forest but not on DDTRegressor natively
    if "random_state" in model_cls.__init__.__code__.co_varnames:
        kwargs["random_state"] = 42

    model = model_cls(**kwargs)
    model.fit(X, y)

    # Check predictions before
    X_test = X[:10]
    preds_before = model.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])

    # Pickle round trip
    dump = pickle.dumps(model)
    model_loaded = pickle.loads(dump)

    # Check predictions after
    preds_after = model_loaded.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])

    for q in [0.1, 0.5, 0.9]:
        np.testing.assert_allclose(preds_before[q], preds_after[q], err_msg=f"Mismatch for quantile {q}")

    # Check that methods don't raise KeyError
    if hasattr(model_loaded, "distribution_summary"):
        model_loaded.distribution_summary()
