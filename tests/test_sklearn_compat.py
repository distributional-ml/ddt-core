"""Test scikit-learn estimator compatibility."""

from sklearn.utils.estimator_checks import check_estimator

from ddt import DDTRegressor


def test_sklearn_check_estimator():
    check_estimator(DDTRegressor())
