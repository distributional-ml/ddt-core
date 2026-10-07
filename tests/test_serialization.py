import pickle

import numpy as np
import pytest
from sklearn.datasets import make_regression

from ddt import DDTRegressor

_MODEL_CLASSES = [DDTRegressor]


@pytest.fixture(scope="module")
def data():
    return make_regression(n_samples=500, n_features=8, noise=10.0, random_state=42)


@pytest.mark.parametrize("model_cls", _MODEL_CLASSES)
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

    kwargs = dict(max_depth=4, n_target_bins=16, **config)

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
