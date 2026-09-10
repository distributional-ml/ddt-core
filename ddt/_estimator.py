"""
DDT Estimator — Scikit-Learn Compatible Wrapper
=================================================
DDTRegressor implements sklearn's BaseEstimator and RegressorMixin APIs,
providing fit(X, y) / predict(X) / predict_distribution(X) methods.

The estimator wraps the C++17 core engine via pybind11, handling:
    - Input validation (NaN/Inf rejection)
    - Double discretization (feature quantization + target binning)
    - Tree building via C++ core engine (pure integer Wasserstein path)
    - Prediction exclusively via the C++ predict_quantiles_fast path

Inherits from sklearn.base.BaseEstimator and RegressorMixin
for full scikit-learn pipeline compatibility (cross-validation, GridSearchCV, etc.).
"""

import logging
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._preprocessor import FeatureQuantizer, TargetBinner

logger = logging.getLogger("ddt")


class DDTRegressor(BaseEstimator, RegressorMixin):
    """Distributional Decision Tree Regressor.

    A non-parametric regression tree that preserves full empirical
    cumulative distribution functions (ECDF) in terminal leaves by
    maximizing 1D Wasserstein distance (Earth Mover's Distance) between
    child distributions at each split.

    Instead of predicting a single point estimate, DDTRegressor captures
    the complete distributional shape — fat tails, multimodality, and
    skewness — enabling risk-aware quantile queries (P90, P95, P99).

    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of the tree. Set to 0 for unlimited depth.

    min_samples_leaf : int, default=20
        Minimum number of samples required in each leaf node.
        The tree will not split a node if either child would have
        fewer than min_samples_leaf samples.

    n_target_bins : int, default=30
        Number of bins for target discretization (B). Higher values
        capture finer distributional detail but increase computation.
        Must be in [2, 255].

    winsorize_tails : float, default=0.001
        Fraction of data to clip from both tails of the target distribution
        prior to binning (e.g. 0.001 clips the 0.1th and 99.9th percentiles).
        This protects bin boundaries from being stretched by extreme outliers.
        Set to None to disable.

    bin_strategy : str, default="equal_width"
        Bin edge strategy for target discretization. One of:

        - ``"equal_width"`` — uniform bins across ``[y_min, y_max]``.
          Required for FPGA / bare-metal inference.
        - ``"log_width"`` — logarithmically spaced bins. Best for strictly
          positive right-skewed targets (prices, durations). Requires
          ``y_min > 0`` after winsorization.
        - ``"hybrid"`` — linear bins over the IQR core (Q25–Q75) with
          log-spaced bins in both tails. Controlled by ``core_fraction``.
        - ``"manual"`` — user-supplied edges via ``manual_bin_edges``.

    target_transform : str or None, default=None
        Transform applied to ``y`` before binning. Only ``'log1p'`` is
        supported. Cannot be combined with ``bin_strategy != 'equal_width'``.

    core_fraction : float, default=0.5
        Fraction of bins allocated to the linear IQR core when using
        ``bin_strategy='hybrid'``. Must be strictly in (0, 1).

    manual_bin_edges : array-like or None, default=None
        Required when ``bin_strategy='manual'``. Strictly monotonically
        increasing array of length ``n_target_bins + 1``.

    n_feature_bins : int, default=256
        Number of quantile bins for feature discretization (K).
        Maximum 256 (uint8 range). Reducing may speed up computation
        at the cost of split resolution.
    Attributes
    ----------
    tree_data_ : list[dict]
        Serialized tree structure from the C++ core engine.

    feature_quantizer_ : FeatureQuantizer
        Fitted feature quantizer for transforming continuous features.

    target_binner_ : TargetBinner
        Fitted target binner for transforming continuous targets.

    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from ddt import DDTRegressor
    >>> import numpy as np
    >>> X = np.random.randn(1000, 5)
    >>> y = X[:, 0] * 2 + np.random.randn(1000) * 0.5
    >>> model = DDTRegressor(max_depth=5, min_samples_leaf=30)
    >>> model.fit(X, y)
    DDTRegressor(max_depth=5, min_samples_leaf=30)
    >>> predictions = model.predict(X)
    >>> distributions = model.predict_distribution(X)
    """

    def __init__(
        self,
        max_depth=10,
        min_samples_leaf=20,
        n_target_bins=30,
        winsorize_tails=0.001,
        bin_strategy="equal_width",
        target_transform=None,
        core_fraction=0.5,
        manual_bin_edges=None,
        consolidate_bins=False,
        n_feature_bins=256,
        calibration_fraction=0.0,
        calibration_order_by=None,
        calibration_order_asc=True,
        calibration_mode="marginal",
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_target_bins = n_target_bins
        self.winsorize_tails = winsorize_tails
        self.bin_strategy = bin_strategy
        self.target_transform = target_transform
        self.core_fraction = core_fraction
        self.manual_bin_edges = manual_bin_edges
        self.consolidate_bins = consolidate_bins
        self.n_feature_bins = n_feature_bins
        self.calibration_fraction = calibration_fraction
        self.calibration_order_by = calibration_order_by
        self.calibration_order_asc = calibration_order_asc
        if calibration_mode not in ["marginal", "hybrid"]:
            raise ValueError("calibration_mode must be 'marginal' or 'hybrid'")
        self.calibration_mode = calibration_mode

    def fit(self, X, y):
        """Build a distributional decision tree from training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix. Must not contain NaN or Inf.

        y : array-like, shape (n_samples,)
            Training target vector. Must not contain NaN or Inf.

        Returns
        -------
        self
            Fitted estimator.

        Raises
        ------
        ValueError
            If X or y contain NaN or Inf values.
        ValueError
            If X and y have inconsistent number of samples.
        """
        # Import C++ extension (deferred to avoid import errors during install).
        from . import _ddt_core

        X_orig = X
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Validate inputs.
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")
        if len(y) != X.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"X has {X.shape[0]}, y has {len(y)}."
            )
        self._validate_no_nan_inf(X, "X")
        self._validate_no_nan_inf(y, "y")

        self.n_features_in_ = X.shape[1]

        # Extract calibration set if requested
        if self.calibration_fraction > 0:
            if not (0 < self.calibration_fraction < 1):
                raise ValueError("calibration_fraction must be between 0 and 1.")

            n_calib = int(len(X) * self.calibration_fraction)
            if n_calib == 0:
                raise ValueError(
                    "calibration_fraction is too small to yield any samples."
                )

            if self.calibration_order_by is not None:
                if isinstance(self.calibration_order_by, int):
                    col_idx = self.calibration_order_by
                elif (
                    hasattr(X_orig, "columns")
                    and self.calibration_order_by in X_orig.columns
                ):
                    col_idx = X_orig.columns.get_loc(self.calibration_order_by)
                else:
                    raise ValueError(
                        f"Could not find column {self.calibration_order_by} in X"
                    )

                sort_idx = np.argsort(X[:, col_idx])
                if not self.calibration_order_asc:
                    sort_idx = sort_idx[::-1]

                X_sorted = X[sort_idx]
                y_sorted = y[sort_idx]

                # Take the end of the array as calibration
                self.X_calib_ = X_sorted[-n_calib:]
                self.y_calib_ = y_sorted[-n_calib:]
                X_train = X_sorted[:-n_calib]
                y_train = y_sorted[:-n_calib]
            else:
                # Random split
                rng = np.random.default_rng(42)
                indices = rng.permutation(len(X))
                calib_idx = indices[:n_calib]
                train_idx = indices[n_calib:]

                self.X_calib_ = X[calib_idx]
                self.y_calib_ = y[calib_idx]
                X_train = X[train_idx]
                y_train = y[train_idx]
        else:
            self.X_calib_ = None
            self.y_calib_ = None
            X_train = X
            y_train = y

        # Step 1: Fit and apply double discretization.
        self.feature_quantizer_ = FeatureQuantizer(
            n_bins=self.n_feature_bins
        )
        X_q = self.feature_quantizer_.fit_transform(X_train)

        self.target_binner_ = TargetBinner(
            n_bins=self.n_target_bins,
            winsorize_tails=self.winsorize_tails,
            bin_strategy=self.bin_strategy,
            target_transform=self.target_transform,
            core_fraction=self.core_fraction,
            manual_bin_edges=self.manual_bin_edges,
            consolidate_bins=self.consolidate_bins,
        )
        y_q = self.target_binner_.fit_transform(y_train)

        # Ensure contiguous C-order arrays for zero-copy C++ access.
        X_q = np.ascontiguousarray(X_q, dtype=np.uint8)
        y_q = np.ascontiguousarray(y_q, dtype=np.int32)

        # Step 2: Build tree via C++ core engine (pure integer Wasserstein path).
        n_bins_active = self.target_binner_.n_bins_active_
        self.tree_data_ = _ddt_core.build_tree(
            X=X_q,
            y=y_q,
            n_bins=n_bins_active,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )

        # Cache structural metadata so we can safely drop diagnostic arrays during pickling
        self.tree_max_depth_ = int(np.max(self.tree_data_.get("depth", [0])))

        # Step 3: Cache calibration distributions for fast O(1) inference
        self._refresh_calibration()

        return self

    def _refresh_calibration(self):
        """Cache calibration distributions for fast O(1) inference."""
        from . import _ddt_core

        if (
            getattr(self, "calibration_fraction", 0.0) > 0
            and getattr(self, "X_calib_", None) is not None
        ):
            X_q_calib = self._preprocess_X(self.X_calib_)
            self._calib_leaves_ = _ddt_core.predict_leaves(self.tree_data_, X_q_calib)

            unique_leaves = np.unique(self._calib_leaves_)
            max_node_id = int(np.max(unique_leaves)) if len(unique_leaves) > 0 else 0

            # n_bins_active_ matches the bin count the tree was actually built with.
            n_bins_active = self.target_binner_.n_bins_active_
            dist_lookup = np.zeros((max_node_id + 1, n_bins_active), dtype=np.int32)
            for leaf_idx in unique_leaves:
                dist_lookup[leaf_idx] = _ddt_core.get_node_distribution(
                    self.tree_data_, int(leaf_idx)
                )

            self._calib_distributions_ = dist_lookup[self._calib_leaves_]

    def predict(self, X):
        """Predict target mean for each sample (sklearn compatibility).

        Returns the mean of the leaf ECDF for each sample. For
        risk-aware predictions, use predict_distribution() to access
        the full distribution and query specific quantiles.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for prediction.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted mean values.
        """
        check_is_fitted(self)

        distributions = self.predict_distribution(X)
        bin_centers = self.target_binner_.inverse_transform_bin_centers()

        totals = distributions.sum(axis=1)
        safe_totals = np.where(totals > 0, totals, 1.0)

        probs = distributions / safe_totals[:, np.newaxis]
        predictions = np.dot(probs, bin_centers)

        valid = totals > 0
        predictions[~valid] = np.mean(bin_centers)

        return predictions

    def predict_distribution(self, X):
        """Predict full histogram distributions for each sample.

        Returns the raw bin counts from the leaf ECDF for each sample,
        enabling downstream quantile queries, ECDF reconstruction,
        and risk analysis.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for prediction.

        Returns
        -------
        np.ndarray, shape (n_samples, n_bins_active_)
            2D array of int32 bin counts. When ``consolidate_bins=False`` (the
            default) this equals ``(n_samples, n_target_bins)``. When
            consolidation is active the last dimension is
            ``target_binner_.n_bins_active_``, which may be smaller.
            Call ``target_binner_.reconstruct_original_grid(dist)`` to expand
            back to ``n_target_bins`` if needed.
        """
        check_is_fitted(self)
        from . import _ddt_core

        X_q = self._preprocess_X(X)
        leaf_indices = _ddt_core.predict_leaves(self.tree_data_, X_q)

        # Only query C++ engine for unique leaves to avoid O(N) Python overhead.
        # The tree was built with n_bins_active_ bins, so get_node_distribution
        # returns arrays of exactly that length — use it here to avoid a shape
        # mismatch when consolidation is active.
        unique_leaves = np.unique(leaf_indices)
        max_node_id = int(np.max(unique_leaves)) if len(unique_leaves) > 0 else 0

        n_bins_active = self.target_binner_.n_bins_active_
        dist_lookup = np.zeros((max_node_id + 1, n_bins_active), dtype=np.int32)
        for leaf_idx in unique_leaves:
            dist_lookup[leaf_idx] = _ddt_core.get_node_distribution(
                self.tree_data_, int(leaf_idx)
            )

        return dist_lookup[leaf_indices]

    def _predict_raw_quantiles(self, X, quantiles, distributions=None):
        """Internal method to compute raw uncalibrated quantiles for a batch of quantiles.

        Uses linear interpolation between bin centers to recover continuous
        quantile resolution from the discrete PMF.
        """
        if distributions is None:
            distributions = self.predict_distribution(X)
        bin_centers = self.target_binner_.inverse_transform_bin_centers()

        totals = distributions.sum(axis=1, keepdims=True)
        safe_totals = np.where(totals > 0, totals, 1.0)

        ecdfs = np.cumsum(distributions, axis=1) / safe_totals
        valid = totals.ravel() > 0
        n_samples = len(distributions)

        results = {}
        mean_center = np.mean(bin_centers)
        sample_range = np.arange(n_samples)

        for q in quantiles:
            # Find first bin where ECDF >= q for all samples
            idx = np.argmax(ecdfs >= q, axis=1)

            # Gather CDF values at the crossing bin and just before
            cdf_hi = ecdfs[sample_range, idx]
            prev_idx = np.maximum(idx - 1, 0)
            cdf_lo = ecdfs[sample_range, prev_idx]
            cdf_lo = np.where(idx > 0, cdf_lo, 0.0)

            # Linear interpolation fraction
            denom = cdf_hi - cdf_lo
            safe_denom = np.where(denom > 1e-12, denom, 1.0)
            frac = np.where(denom > 1e-12, (q - cdf_lo) / safe_denom, 0.0)

            # Interpolated quantile values
            center_hi = bin_centers[idx]
            center_lo = bin_centers[prev_idx]
            interpolated = center_lo + frac * (center_hi - center_lo)

            # Assemble: use interpolation when idx > 0, snap to bin center at idx==0
            q_vals = np.where(idx > 0, interpolated, center_hi)
            # Override with mean for invalid (empty) distributions
            q_vals = np.where(valid, q_vals, mean_center)

            results[q] = q_vals

        return results

    def predict_quantiles(self, X, quantiles):
        """Predict multiple quantiles via the C++ fast-path exclusively.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for prediction.
        quantiles : list of float
            List of quantiles to predict, in [0, 1]. E.g., [0.1, 0.5, 0.9].

        Returns
        -------
        dict
            Dictionary mapping each requested quantile to its np.ndarray of
            predicted values in the original target space.

        Notes
        -----
        All bin strategies (``'equal_width'``, ``'log_width'``, ``'manual'``)
        are handled uniformly: bin center values are fetched from
        ``target_binner_.inverse_transform_bin_centers()``, which accounts for
        variable bin spacing, so returned quantile values are correct in the
        original target space regardless of strategy.
        """
        for q in quantiles:
            if not 0.0 <= q <= 1.0:
                raise ValueError(f"q must be in [0, 1], got {q}")

        check_is_fitted(self)
        from . import _ddt_core

        X_arr = np.asarray(X, dtype=np.float64)
        self._validate_no_nan_inf(X_arr, "X")

        bin_centers = self.target_binner_.inverse_transform_bin_centers().astype(
            np.float64
        )
        q_array = np.array(quantiles, dtype=np.float64)

        # C++ monolithic inference: single pass over the tree for all quantiles.
        fast_results = _ddt_core.predict_quantiles_fast(
            self.tree_data_,
            X_arr,
            self.feature_quantizer_._flat_bin_edges_,
            self.feature_quantizer_._bin_offsets_,
            self.feature_quantizer_.n_bins,
            q_array,
            bin_centers,
        )
        raw_preds = {q: fast_results[i] for i, q in enumerate(quantiles)}

        # Conformal quantile calibration (marginal or leaf-level hybrid).
        if self.calibration_fraction > 0 and self.X_calib_ is not None:
            calib_preds = self._predict_raw_quantiles(
                self.X_calib_, quantiles, distributions=self._calib_distributions_
            )

            if self.calibration_mode == "hybrid":
                calib_leaves = self._calib_leaves_
                X_q_test = self._preprocess_X(X)
                test_leaves = _ddt_core.predict_leaves(self.tree_data_, X_q_test)
                unique_calib_leaves, calib_counts = np.unique(
                    calib_leaves, return_counts=True
                )
                threshold = self.min_samples_leaf * 2

            for q in quantiles:
                residuals = self.y_calib_ - calib_preds[q]
                s_global = np.quantile(residuals, q)

                if self.calibration_mode == "hybrid":
                    shifts = np.full(len(X), s_global, dtype=np.float64)
                    for leaf, count in zip(unique_calib_leaves, calib_counts):
                        if count >= threshold:
                            leaf_mask = calib_leaves == leaf
                            s_leaf = np.quantile(residuals[leaf_mask], q)
                            shifts[test_leaves == leaf] = s_leaf
                    raw_preds[q] += shifts
                else:
                    raw_preds[q] += s_global

        return raw_preds

    def predict_quantile(self, X, q):
        """Predict a specific quantile from the leaf ECDF with optional conformal calibration.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for prediction.
        q : float
            Quantile to predict, in [0, 1]. E.g., 0.95 for P95.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted quantile values.
        """
        return self.predict_quantiles(X, [q])[q]

    def get_tree_info(self):
        """Return summary information about the fitted tree.

        Returns
        -------
        dict
            Tree metadata including node count, leaf count, max depth,
            and active bins.
        """
        check_is_fitted(self)

        n_nodes = len(self.tree_data_["is_leaf"])
        n_leaves = int(np.sum(self.tree_data_["is_leaf"]))
        
        # Use cached depth if available (loaded models drop the 'depth' array to save memory)
        max_depth = getattr(self, "tree_max_depth_", int(np.max(self.tree_data_.get("depth", [0]))))

        return {
            "n_nodes": n_nodes,
            "n_leaves": n_leaves,
            "n_internal": n_nodes - n_leaves,
            "max_depth": max_depth,
            "n_target_bins": self.n_target_bins,
            "n_bins_active": self.target_binner_.n_bins_active_,
        }

    def __getstate__(self):
        """Strip diagnostic arrays from tree_data_ to reduce pickle size.

        Note: X_calib_ and y_calib_ are intentionally retained so that the
        conformal calibration state can be fully restored after unpickling
        (e.g. for re-calibration or auditing). If pickle size is critical for
        calibrated models, callers can zero these out manually after saving.
        """
        state = self.__dict__.copy()
        if "tree_data_" in state:
            # Copy the dict so we don't mutate the live object's tree_data_
            tree_data = state["tree_data_"].copy()
            # Remove arrays not required for inference
            tree_data.pop("wasserstein_gain", None)
            tree_data.pop("depth", None)
            tree_data.pop("total_samples", None)
            state["tree_data_"] = tree_data
        return state

    def __setstate__(self, state):
        """Restore state."""
        self.__dict__.update(state)

    def _preprocess_X(self, X):
        """Validate and quantize feature matrix for prediction."""
        X = np.asarray(X, dtype=np.float64)
        self._validate_no_nan_inf(X, "X")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but DDTRegressor was fitted "
                f"with {self.n_features_in_} features."
            )

        X_q = self.feature_quantizer_.transform(X)
        return np.ascontiguousarray(X_q, dtype=np.uint8)

    @staticmethod
    def _validate_no_nan_inf(arr, name):
        """Reject NaN/Inf inputs."""
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name} contains NaN values. DDT requires finite inputs.")
        if np.any(np.isinf(arr)):
            raise ValueError(f"{name} contains Inf values. DDT requires finite inputs.")

