"""
Distributional Decision Trees (DDT).
====================================
A histogram-based machine learning framework that models conditional target distributions
with compact histograms in terminal leaves by maximizing
scale-normalised 1-Wasserstein distance between child target distributions (gain = (B/R)·W₁, unweighted by child size by default; see `split_weighting`). Note that with `target_transform="log1p"` it is measured on the transformed scale.

Main API:
    DDTRegressor - distributional regression tree (scikit-learn compatible in tested versions)

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
__version__ = "1.1.0"
