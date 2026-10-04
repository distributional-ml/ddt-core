"""
Distributional Decision Trees (DDT)
====================================
A non-parametric machine learning framework that preserves full Empirical
Cumulative Distribution Functions (ECDF) in terminal leaves by maximizing
1D Wasserstein distance during tree construction.

Main API:
    DDTRegressor - scikit-learn compatible distributional regression tree

Architecture:
    C++ Compute Engine: C++17 / Pybind11 core engine (O(N + K*B) split evaluation)
    Python API & Validation: Validation harnesses & statistical oracles
"""

from ._estimator import DDTRegressor
from ._preprocessor import FeatureQuantizer, TargetBinner

__all__ = [
    "DDTRegressor",
    "FeatureQuantizer",
    "TargetBinner",
]
__version__ = "1.0.0"
