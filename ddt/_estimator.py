"""
DDT Estimator — Scikit-Learn Compatible Wrapper.
=================================================
DDTRegressor implements sklearn's BaseEstimator and RegressorMixin APIs,
providing fit(X, y) / predict(X) / predict_distribution(X) methods.

The estimator wraps the C++17 core engine via pybind11, handling:
    - Input validation (NaN/Inf rejection per FR-CORE-05)
    - Double discretization (feature quantization + target binning per FR-CORE-01)
    - Tree building via C++ core engine
    - Prediction via leaf traversal and ECDF reconstruction

NFR-COMP-01: Inherits from sklearn.base.BaseEstimator and RegressorMixin
for full scikit-learn pipeline compatibility (cross-validation, GridSearchCV, etc.).
"""

import logging
import platform
import subprocess
import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class ConformalResolutionWarning(UserWarning):
    pass


from ._preprocessor import FeatureQuantizer, TargetBinner
from ._tree_state import TreeState

logger = logging.getLogger("ddt.simplifier")


class DDTRegressor(
    RegressorMixin,
    BaseEstimator,
):
    """
    Distributional Decision Tree Regressor.

    A histogram-based regression tree that models conditional target distributions
    with compact histograms in terminal leaves by
    maximizing 1D Wasserstein distance (Earth Mover's Distance) between
    child distributions at each split.

    Instead of predicting a single point estimate, DDTRegressor captures
    the complete distributional shape — fat tails, multimodality, and
    skewness — enabling risk-aware quantile queries (P90, P95, P99).

    Parameters
    ----------
    max_depth : int or None, default=10
        Maximum depth of the tree. Set to 0 or None for unlimited depth.

    min_samples_leaf : int, default=20
        Minimum number of samples required in each leaf node.
        The tree will not split a node if either child would have
        fewer than min_samples_leaf samples.

    n_target_bins : int, default=30
        Number of equal-width bins for target discretization (B).
        Higher values capture finer distributional detail but increase
        computation. Must be in [2, 100].

    min_divergence_decrease : float, default=0.01
        Minimum divergence gain required to allow a split. Splits with
        gain below this threshold are rejected (early stopping).

    target_transform : str, default=None
        Transform to apply to the target before binning. Options: 'log1p' or None.
        Using 'log1p' causes bins to grow exponentially in the original space,
        which natively models heavy-tailed distributions.

    winsorize_tails : float, default=1e-4
        Fraction of data to clip from both tails of the target distribution
        prior to binning (e.g. 0.001 clips the 0.1th and 99.9th percentiles).
        This affects only the body bin grid; EVT always observes the raw `y`
        values (i.e. EVT threshold `u` can exceed `winsorize_upper_`).
        Set to None to disable.

    bin_strategy : str, default="equal_width"
        Bin edge strategy for the target. Options:
        ``"equal_width"`` (default, FPGA/edge-safe), ``"log_width"`` (log-spaced
        edges for strictly positive targets), ``"manual"`` (user-supplied edges).

        **Note on bin strategy and split behavior:**
        When using ``bin_strategy="log_width"`` or ``"hybrid"``, the Wasserstein split
        criterion weights bin differences by their physical width. This means the tree will
        preferentially split on features that differentiate the *tails* of the distribution,
        not the *modes*. If your application requires high body resolution (e.g., detecting
        bimodality in the core IQR), consider using ``bin_strategy="equal_width"`` or
        increasing the ``core_fraction`` parameter.

    n_feature_bins : int, default=256
        Number of quantile bins for feature discretization (K).
        Maximum 256 (uint8 range). Reducing may speed up computation
        at the cost of split resolution.

    divergence : str, default="wasserstein"
        Divergence metric for split evaluation.
        Currently supported: "wasserstein" (1D Earth Mover's Distance).
        Future: "sinkhorn" (entropic regularized OT for gradient boosting).

    quantize_engine : str, default="cpp"
        Engine used for feature quantization during inference.
        "cpp" uses a highly optimized C++ implementation.
        "python" falls back to scikit-learn's KBinsDiscretizer.

    smooth_leaves : bool, default=False
        If True, apply Dirichlet (Empirical Bayes) smoothing to all terminal
        leaf PMFs after tree construction. Sparse leaves are shrunk toward
        their parent's (or ancestor's) more stable distribution, preventing
        overconfident quantile estimates from small-sample histograms.

        When enabled, ``predict_distribution()`` returns float64 probability
        arrays (smoothed PMFs) instead of int32 raw counts. The raw counts
        are always preserved in ``tree_data_["distribution_counts"]``.

    smooth_prior_weight : float or None, default=None
        Dirichlet concentration parameter (lambda) controlling shrinkage
        strength. If None, defaults to ``min_samples_leaf``. Higher values
        produce stronger shrinkage toward the parent. A leaf with N samples
        retains data weight w = N / (N + lambda), so a leaf at exactly
        ``min_samples_leaf`` samples receives 50% shrinkage by default.

    smooth_min_samples : int or None, default=None
        Minimum total_samples an ancestor node must have to serve as the
        smoothing prior source. If None, defaults to ``4 * min_samples_leaf``.
        The smoother walks up the tree from each leaf until finding an
        ancestor meeting this threshold. If none qualifies (including root),
        the root's marginal PMF is used.

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
        min_divergence_decrease=0.01,
        target_transform=None,
        winsorize_tails=1e-4,
        bin_strategy="equal_width",
        manual_bin_edges=None,
        core_fraction=0.5,
        consolidate_bins=False,
        n_feature_bins=256,
        divergence="wasserstein",
        quantize_engine="cpp",
        calibration_fraction=0.0,
        calibration_order_by=None,
        calibration_order_asc=True,
        calibration_mode="marginal",
        calibration_strict=False,
        hybrid_min_calib=50,
        evt_tails=False,
        evt_tails_lower=False,
        evt_min_samples=30,
        evt_tail_fraction="auto",
        evt_tail_fraction_lower=None,
        smooth_leaves=False,
        smooth_prior_weight=None,
        smooth_min_samples=None,
        compact_inference=False,
        fast_inference=False,
        max_splits_per_feature=None,
        split_weighting="crps",
        quantile_interpolation="linear",
        atom_threshold=0.9,
        evt_borrow_factor=1.0,
        evt_gof_alpha=0.05,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_target_bins = n_target_bins
        self.min_divergence_decrease = min_divergence_decrease
        self.target_transform = target_transform
        self.winsorize_tails = winsorize_tails
        self.bin_strategy = bin_strategy
        self.manual_bin_edges = manual_bin_edges
        self.core_fraction = core_fraction
        self.consolidate_bins = consolidate_bins
        self.n_feature_bins = n_feature_bins
        self.divergence = divergence
        self.quantize_engine = quantize_engine
        self.atom_threshold = atom_threshold
        self.calibration_fraction = calibration_fraction
        self.calibration_order_by = calibration_order_by
        self.calibration_order_asc = calibration_order_asc
        self.calibration_mode = calibration_mode
        self.calibration_strict = calibration_strict
        self.hybrid_min_calib = hybrid_min_calib
        self.evt_tails = evt_tails
        self.evt_tails_lower = evt_tails_lower
        self.evt_min_samples = evt_min_samples
        self.evt_tail_fraction = evt_tail_fraction
        self.evt_tail_fraction_lower = evt_tail_fraction_lower
        self.evt_borrow_factor = evt_borrow_factor
        self.evt_gof_alpha = evt_gof_alpha
        self.smooth_leaves = smooth_leaves
        self.smooth_prior_weight = smooth_prior_weight
        self.smooth_min_samples = smooth_min_samples
        self.compact_inference = compact_inference
        self.fast_inference = fast_inference
        self.max_splits_per_feature = max_splits_per_feature
        self.split_weighting = split_weighting
        self.quantile_interpolation = quantile_interpolation

    def fit(self, X, y):
        if self.evt_tail_fraction != "auto" and not (0 < float(self.evt_tail_fraction) < 0.5):
            raise ValueError(f"evt_tail_fraction must be 'auto' or in (0, 0.5), got {self.evt_tail_fraction}")
        if self.evt_tail_fraction_lower is not None:
            if self.evt_tail_fraction_lower != "auto" and not (0 < float(self.evt_tail_fraction_lower) < 0.5):
                raise ValueError(
                    f"evt_tail_fraction_lower must be 'auto' or in (0, 0.5), got {self.evt_tail_fraction_lower}"
                )

        """
        Build a distributional decision tree from training data.

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
            If X or y contain NaN or Inf values (FR-CORE-05).
        ValueError
            If X and y have inconsistent number of samples.

        """
        if self.calibration_mode not in ["marginal", "hybrid"]:
            raise ValueError("calibration_mode must be 'marginal' or 'hybrid'")
        if self.quantile_interpolation not in ["linear", "snap"]:
            raise ValueError("quantile_interpolation must be 'linear' or 'snap'")

        # Import C++ extension (deferred to avoid import errors during install).
        from sklearn.utils.validation import check_X_y

        from . import _ddt_core

        for attr in [
            "X_calib_",
            "y_calib_",
            "_calib_leaves_",
            "calibration_shifts_",
            "_Q_cal_",
            "evt_active_",
            "evt_lower_active_",
            "pipeline_stage_",
            "split_threshold_float_",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)

        X_orig = X
        if hasattr(self, "_validate_data"):
            X, y = self._validate_data(X, y, dtype=np.float64, y_numeric=True)
        else:
            X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        self._validate_no_nan_inf(X, "X")
        self._validate_no_nan_inf(y, "y")

        if X.shape[0] < 2:
            raise ValueError("n_samples=1 is not supported by DDTRegressor")

        if self.smooth_leaves:
            if self.smooth_prior_weight is not None and self.smooth_prior_weight <= 0:
                raise ValueError(f"smooth_prior_weight must be > 0, got {self.smooth_prior_weight}")
            if self.smooth_min_samples is not None and self.smooth_min_samples < 1:
                raise ValueError(f"smooth_min_samples must be >= 1, got {self.smooth_min_samples}")

        self.n_features_in_ = X.shape[1]

        # Extract calibration set if requested
        if self.calibration_fraction > 0:
            if not (0 < self.calibration_fraction < 1):
                raise ValueError("calibration_fraction must be between 0 and 1.")

            n_calib = int(len(X) * self.calibration_fraction)
            if n_calib == 0:
                raise ValueError("calibration_fraction is too small to yield any samples.")

            if self.calibration_order_by is not None:
                if isinstance(self.calibration_order_by, int):
                    col_idx = self.calibration_order_by
                elif hasattr(X_orig, "columns") and self.calibration_order_by in X_orig.columns:
                    col_idx = X_orig.columns.get_loc(self.calibration_order_by)
                else:
                    raise ValueError(f"Could not find column {self.calibration_order_by} in X")

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

        # Step 1: Fit and apply double discretization (FR-CORE-01).
        self.feature_quantizer_ = FeatureQuantizer(n_bins=self.n_feature_bins, engine=self.quantize_engine)
        X_q = self.feature_quantizer_.fit_transform(X_train)

        self.target_binner_ = TargetBinner(
            n_bins=self.n_target_bins,
            target_transform=self.target_transform,
            winsorize_tails=self.winsorize_tails,
            bin_strategy=self.bin_strategy,
            manual_bin_edges=(
                np.asarray(self.manual_bin_edges, dtype=np.float64) if self.manual_bin_edges is not None else None
            ),
            core_fraction=self.core_fraction,
            consolidate_bins=self.consolidate_bins,
            atom_threshold=self.atom_threshold,
        )
        y_q = self.target_binner_.fit_transform(y_train)

        # Ensure contiguous C-order arrays for zero-copy C++ access.
        X_q = np.ascontiguousarray(X_q, dtype=np.uint8)
        y_q = np.ascontiguousarray(y_q, dtype=np.int32)

        # Step 2: Build tree via C++ core engine.
        # Use the weighted path whenever delta_x_norm_ is set — which happens for
        # any non-equal_width strategy, or equal_width + consolidation (merged bins
        # are no longer uniform width).  The unweighted build_tree path is used only
        # for the default equal_width + no-consolidation case, preserving zero
        # behavioural change for existing users.
        n_bins_active = self.target_binner_.n_bins_active_
        use_weighted = self.target_binner_.delta_x_norm_ is not None

        # Phase 5c: convert Optional[Dict[int, int]] → dense int32 array.
        # Unconstrained features get np.iinfo(np.int32).max (no effective limit).
        max_splits_arr = None
        if self.max_splits_per_feature is not None:
            n_feat = X_q.shape[1]
            _INT32_MAX = np.iinfo(np.int32).max
            max_splits_arr = np.full(n_feat, _INT32_MAX, dtype=np.int32)
            for feat_key, limit in self.max_splits_per_feature.items():
                if isinstance(feat_key, str):
                    if not hasattr(X_orig, "columns"):
                        raise ValueError(
                            f"max_splits_per_feature contains string key '{feat_key}', "
                            "but X does not have columns (not a pandas DataFrame)."
                        )
                    if feat_key not in X_orig.columns:
                        raise ValueError(f"max_splits_per_feature key '{feat_key}' not found in X columns.")
                    feat_idx = X_orig.columns.get_loc(feat_key)
                else:
                    feat_idx = int(feat_key)

                if feat_idx < 0 or feat_idx >= n_feat:
                    raise ValueError(f"max_splits_per_feature key {feat_idx} is out of range for {n_feat} features.")
                max_splits_arr[feat_idx] = int(limit)

        _c_max_depth = self.max_depth if self.max_depth is not None else 0
        if use_weighted:
            tree_dict = _ddt_core.build_tree_weighted(
                X=X_q,
                y=y_q,
                delta_x_norm=np.ascontiguousarray(self.target_binner_.delta_x_norm_, dtype=np.float64),
                n_bins=n_bins_active,
                max_depth=_c_max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_divergence_decrease=self.min_divergence_decrease,
                divergence=self.divergence,
                max_splits_per_feature=max_splits_arr,
                split_weighting=self.split_weighting,
            )
        else:
            tree_dict = _ddt_core.build_tree(
                X=X_q,
                y=y_q,
                n_bins=n_bins_active,
                max_depth=_c_max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_divergence_decrease=self.min_divergence_decrease,
                divergence=self.divergence,
                max_splits_per_feature=max_splits_arr,
                split_weighting=self.split_weighting,
            )

        # Ensure numpy-owned copies (break capsule dependency)
        for key in list(tree_dict.keys()):
            if isinstance(tree_dict[key], np.ndarray):
                tree_dict[key] = np.array(tree_dict[key])

        self.tree_data_ = tree_dict
        self.tree_state_ = TreeState.from_dict(self.tree_data_)

        # Order: build_tree → smooth → EVT
        if self.smooth_leaves:
            self.smooth_leaves_()

        if self.evt_tails:
            from ._evt import fit_all_leaf_gpds

            evt_params = fit_all_leaf_gpds(
                tree_data=self.tree_data_,
                y_train=y_train,
                target_binner=self.target_binner_,
                feature_quantizer=self.feature_quantizer_,
                X_q=X_q,
                min_samples=self.evt_min_samples,
                tail_fraction=self.evt_tail_fraction,
                tail_fraction_lower=self.evt_tail_fraction_lower
                if self.evt_tail_fraction_lower is not None
                else self.evt_tail_fraction,
                fit_lower_tail=bool(getattr(self, "evt_tails_lower", False)),
            )
            self.tree_data_.update(evt_params)
            self.evt_active_ = True
            if getattr(self, "evt_tails_lower", False):
                self.evt_lower_active_ = True

        # Cache structural metadata so we can safely drop extraneous arrays during pickling
        self.tree_max_depth_ = int(np.max(self.tree_data_.get("depth", [0])))

        # Step 3: Pre-compute float split thresholds when fast_inference=True
        if getattr(self, "fast_inference", False):
            self.compute_float_thresholds()
        # Step 3: Cache calibration distributions for fast O(1) inference
        self._refresh_calibration()

        # Step 4: Post-training memory reduction if requested
        self._compact()

        return self

    @property
    def is_calibrated_(self) -> bool:
        """True if the model has been calibrated via conformal calibration."""
        return hasattr(self, "y_calib_") and hasattr(self, "_calib_leaves_")

    def smooth_leaves_(self):
        """
        Apply Dirichlet smoothing to all terminal leaf PMFs.

        This method can be called manually after fit() if the model was
        instantiated with smooth_leaves=False, allowing separate timing
        of the tree build and smoothing phases.
        """
        check_is_fitted(self)

        from . import _ddt_core

        _prior_weight = (
            self.smooth_prior_weight if self.smooth_prior_weight is not None else float(self.min_samples_leaf)
        )
        _min_ancestor = self.smooth_min_samples if self.smooth_min_samples is not None else 4 * self.min_samples_leaf
        # Phase 6: C++ Dirichlet smoothing
        self.tree_data_["smoothed_pmf"] = _ddt_core.smooth_tree(
            tree_data=self.tree_data_,
            prior_weight=_prior_weight,
            min_ancestor_samples=int(_min_ancestor),
        )
        if hasattr(self, "tree_state_"):
            self.tree_state_.smoothed_pmf = self.tree_data_["smoothed_pmf"]
        return self

    def _compact(self):
        """
        Phase 2: Post-training memory reduction.

        Converts the potentially massive float64 or int32 distribution arrays
        to float16 PMFs for inference, stripping the original arrays.
        """
        if not getattr(self, "compact_inference", False):
            return

        n_bins = self.target_binner_.n_bins_active_
        dtype = np.float16 if n_bins <= 64 else np.float32

        tree_data = self.tree_data_

        if "smoothed_pmf" in tree_data:
            tree_data["inference_pmf"] = tree_data["smoothed_pmf"].astype(dtype)
            del tree_data["smoothed_pmf"]
            if hasattr(self, "tree_state_"):
                self.tree_state_.inference_pmf = tree_data["inference_pmf"]
                self.tree_state_.smoothed_pmf = None
        elif "distribution_counts" in tree_data:
            counts = tree_data["distribution_counts"]
            row_sums = counts.sum(axis=1, keepdims=True)
            # Avoid divide by zero for empty nodes
            safe_sums = np.where(row_sums == 0, 1, row_sums)
            tree_data["inference_pmf"] = (counts / safe_sums).astype(dtype)
            del tree_data["distribution_counts"]
            if hasattr(self, "tree_state_"):
                self.tree_state_.inference_pmf = tree_data["inference_pmf"]
                self.tree_state_.distribution_counts = None

    def compute_float_thresholds(self):
        """
        Phase 5: Pre-compute float32 split boundaries for O(1) inference.

        Maps internal tree splits from uint8 bin indices back to their
        original float64 feature boundaries, eliminating the need for
        bin_edges and bin_offsets lookup arrays during prediction.
        """
        if (
            not hasattr(self, "feature_quantizer_")
            or getattr(self.feature_quantizer_, "_flat_bin_edges_", None) is None
        ):
            return

        flat_edges = self.feature_quantizer_._flat_bin_edges_
        bin_offsets = self.feature_quantizer_._bin_offsets_

        split_feat = self.tree_data_["split_feature_idx"]
        split_thresh = self.tree_data_["split_threshold"]
        n_nodes = len(split_feat)

        float_thresholds = np.zeros(n_nodes, dtype=np.float64)

        # Vectorized mapping: for each node, threshold float is flat_edges[bin_offsets[f] + thresh + 1]
        # Only valid for internal nodes (split_feat >= 0)
        internal_mask = split_feat >= 0
        if np.any(internal_mask):
            f_idx = split_feat[internal_mask]
            t_idx = split_thresh[internal_mask]

            # Look up the actual threshold value (right edge of the bin)
            offsets = bin_offsets[f_idx]
            float_thresholds[internal_mask] = flat_edges[offsets + t_idx + 1]

        self.tree_data_["split_threshold_float"] = float_thresholds

    def _refresh_calibration(self):
        """Cache calibration distributions for fast O(1) inference."""
        from . import _ddt_core

        if getattr(self, "calibration_fraction", 0.0) > 0 and getattr(self, "X_calib_", None) is not None:
            X_q_calib = self._preprocess_X(self.X_calib_)
            self._calib_leaves_ = _ddt_core.predict_leaves(self.tree_data_, X_q_calib)

            # Pre-compute Q_cal grid and shifts per group (Phase 3)
            Q_cal = [0.001, 0.0025, 0.005] + np.arange(0.01, 0.991, 0.01).tolist() + [0.995, 0.9975, 0.999]
            Q_cal = np.array(Q_cal)
            self._Q_cal_ = Q_cal
            self.calibration_shifts_ = {}
            self.calibration_counts_ = {}

            route = self._inference_route()
            calib_preds = route.from_leaves(self._calib_leaves_, Q_cal)

            def compute_shifts_for_group(y_group, mask, group_name):
                n = len(y_group)
                self.calibration_counts_[group_name] = n
                s_raw = np.zeros(len(Q_cal))

                for i, q in enumerate(Q_cal):
                    r = np.sort(y_group - calib_preds[q][mask])
                    k = int(np.ceil((n + 1) * q)) if q >= 0.5 else int(np.floor((n + 1) * q))
                    if 1 <= k <= n:
                        s_raw[i] = r[k - 1]
                    else:
                        s_raw[i] = r[-1] if q >= 0.5 else r[0]

                s = np.zeros(len(Q_cal))
                median_idx = np.argmin(np.abs(Q_cal - 0.5))
                s[median_idx] = s_raw[median_idx]
                for i in range(median_idx + 1, len(Q_cal)):
                    s[i] = max(s_raw[i], s[i - 1])
                for i in range(median_idx - 1, -1, -1):
                    s[i] = min(s_raw[i], s[i + 1])
                return s

            if self.calibration_mode == "hybrid":
                unique_calib_leaves, calib_counts = np.unique(self._calib_leaves_, return_counts=True)
                min_calib = getattr(self, "hybrid_min_calib", 50)

                large_leaves = unique_calib_leaves[calib_counts >= min_calib]
                for leaf in large_leaves:
                    mask = self._calib_leaves_ == leaf
                    self.calibration_shifts_[leaf] = compute_shifts_for_group(self.y_calib_[mask], mask, leaf)

                rest_mask = np.isin(self._calib_leaves_, large_leaves, invert=True)
                if np.any(rest_mask):
                    self.calibration_shifts_["rest"] = compute_shifts_for_group(
                        self.y_calib_[rest_mask], rest_mask, "rest"
                    )
            else:
                mask = np.ones(len(self.y_calib_), dtype=bool)
                self.calibration_shifts_["global"] = compute_shifts_for_group(self.y_calib_, mask, "global")

    def predict(self, X):
        """
        Predict target mean for each sample (sklearn compatibility).

        Returns the representative-point mean of the leaf ECDF for each sample. This
        is not the exact mean of the predictive body distribution `G` or the spliced
        distribution `H`. For risk-aware predictions, use predict_distribution() to
        access the full distribution and query specific quantiles.
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
        bin_rep = self.target_binner_.quantile_grid()[2]

        totals = distributions.sum(axis=1)
        safe_totals = np.where(totals > 0, totals, 1.0)

        probs = distributions / safe_totals[:, np.newaxis]
        predictions = np.dot(probs, bin_rep)

        valid = totals > 0
        predictions[~valid] = np.mean(bin_rep)

        return predictions

    def predict_distribution(self, X):
        """
        Predict full histogram distributions for each sample.

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
            When ``smooth_leaves=False`` (default): 2D int32 array of raw bin
            counts.  When ``smooth_leaves=True``: 2D float64 array of smoothed
            PMF probabilities (each row sums to 1.0).  When
            ``consolidate_bins=False`` the last dimension equals
            ``n_target_bins``; when consolidation is active it equals
            ``target_binner_.n_bins_active_``.

        """
        check_is_fitted(self)
        from . import _ddt_core

        X_q = self._preprocess_X(X)
        leaf_indices = _ddt_core.predict_leaves(self.tree_data_, X_q)

        # Smoothed path: return float64 PMFs directly from smoothed_pmf array.
        # This is a pure NumPy fancy-index — O(n_samples), faster than the
        # per-leaf C++ loop below for large inference batches.
        if hasattr(self, "tree_state_") and self.tree_state_.has_smooth_pmf:
            if "smoothed_pmf" in self.tree_data_:
                return self.tree_data_["smoothed_pmf"][leaf_indices]
            elif "inference_pmf" in self.tree_data_:
                return self.tree_data_["inference_pmf"][leaf_indices].astype(np.float64)

        # Default path: return int32 raw counts via C++ get_node_distribution.
        # Only query C++ engine for unique leaves to avoid O(N) Python overhead.
        # The tree was built with n_bins_active_ bins, so get_node_distribution
        # returns arrays of exactly that length — use it here to avoid a shape
        # mismatch when consolidation is active.
        unique_leaves = np.unique(leaf_indices)
        max_node_id = int(np.max(unique_leaves)) if len(unique_leaves) > 0 else 0

        n_bins_active = self.target_binner_.n_bins_active_
        dist_lookup = np.zeros((max_node_id + 1, n_bins_active), dtype=np.int32)
        for leaf_idx in unique_leaves:
            dist_lookup[leaf_idx] = _ddt_core.get_node_distribution(self.tree_data_, int(leaf_idx))

        return dist_lookup[leaf_indices]

    def _inference_route(self):
        return _Route(self)

    def _predict_raw_quantiles(self, X, quantiles, distributions=None):
        """Vectorised NumPy implementation of CDF inversion (§3.2)."""
        if distributions is None:
            distributions = self.predict_distribution(X)

        N, B = distributions.shape
        C = np.cumsum(distributions, axis=1).astype(np.float64)
        T = C[:, -1]

        bin_lo, bin_width, bin_rep = self.target_binner_.quantile_grid()
        interpolation = getattr(self, "quantile_interpolation", "linear")

        results = {}
        for q in quantiles:
            thr = q * T

            mask_leq_0 = thr <= 0
            mask_gt_0 = ~mask_leq_0

            b = np.zeros(N, dtype=np.int32)
            if np.any(mask_leq_0):
                b[mask_leq_0] = np.argmax(C[mask_leq_0] > 0, axis=1)

            if np.any(mask_gt_0):
                thr_gt = np.minimum(thr[mask_gt_0], T[mask_gt_0])
                b[mask_gt_0] = np.argmax(C[mask_gt_0] >= thr_gt[:, None], axis=1)

            b = np.clip(b, 0, B - 1)

            if interpolation == "snap":
                q_vals = bin_rep[b].copy()
            else:
                prev = np.zeros(N, dtype=np.float64)
                mask_b_gt_0 = b > 0
                prev[mask_b_gt_0] = C[np.arange(N)[mask_b_gt_0], b[mask_b_gt_0] - 1]

                denom = C[np.arange(N), b] - prev
                frac = np.zeros(N, dtype=np.float64)

                mask_valid = (denom > 0) & mask_gt_0
                frac[mask_valid] = (thr[mask_valid] - prev[mask_valid]) / denom[mask_valid]
                frac = np.clip(frac, 0.0, 1.0)

                q_vals = bin_lo[b] + frac * bin_width[b]

            results[q] = q_vals

        return results

    def predict_quantiles(self, X, quantiles):
        """
        Predict multiple quantiles efficiently without re-preprocessing.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for prediction.
        quantiles : list of float
            List of quantiles to predict, in [0, 1]. E.g., [0.1, 0.5, 0.9].

        Returns
        -------
        dict
            Dictionary mapping each requested quantile to its np.ndarray of predictions.

        """
        for q in quantiles:
            if not 0.0 <= q <= 1.0:
                raise ValueError(f"q must be in [0, 1], got {q}")

        check_is_fitted(self)

        route = self._inference_route()
        raw_preds = route.from_X(X, quantiles)

        # 2. Conformal quantile calibration
        if self.calibration_fraction > 0 and getattr(self, "y_calib_", None) is not None:
            if self.calibration_mode == "hybrid":
                from . import _ddt_core

                X_q_test = self._preprocess_X(X)
                test_leaves = _ddt_core.predict_leaves(self.tree_data_, X_q_test)

            Q_cal = getattr(self, "_Q_cal_", None)

            warned = False
            for q in quantiles:
                if self.calibration_mode == "hybrid":
                    shifts = np.zeros(len(X), dtype=np.float64)

                    for leaf, s_grid in self.calibration_shifts_.items():
                        if leaf == "rest":
                            continue
                        n = self.calibration_counts_[leaf]
                        k = int(np.ceil((n + 1) * q)) if q >= 0.5 else int(np.floor((n + 1) * q))
                        unres = q <= 0.0 or q >= 1.0 or (q >= 0.5 and k > n) or (q < 0.5 and k < 1)
                        s_interp = np.interp(q, Q_cal, s_grid)
                        if unres:
                            if getattr(self, "calibration_strict", False):
                                s_interp = np.inf if q >= 0.5 else -np.inf
                            elif not warned:
                                warnings.warn(
                                    "Cannot resolve conformal quantile. Returning extreme residual.",
                                    ConformalResolutionWarning,
                                    stacklevel=2,
                                )
                                warned = True
                        shifts[test_leaves == leaf] = s_interp

                    if "rest" in self.calibration_shifts_:
                        known_leaves = [k for k in self.calibration_shifts_.keys() if isinstance(k, (int, np.integer))]
                        rest_mask = ~np.isin(test_leaves, known_leaves)
                        if np.any(rest_mask):
                            n = self.calibration_counts_["rest"]
                            k = int(np.ceil((n + 1) * q)) if q >= 0.5 else int(np.floor((n + 1) * q))
                            unres = q <= 0.0 or q >= 1.0 or (q >= 0.5 and k > n) or (q < 0.5 and k < 1)
                            s_interp = np.interp(q, Q_cal, self.calibration_shifts_["rest"])
                            if unres:
                                if getattr(self, "calibration_strict", False):
                                    s_interp = np.inf if q >= 0.5 else -np.inf
                                elif not warned:
                                    warnings.warn(
                                        "Cannot resolve conformal quantile. Returning extreme residual.",
                                        ConformalResolutionWarning,
                                        stacklevel=2,
                                    )
                                    warned = True
                            shifts[rest_mask] = s_interp

                    raw_preds[q] += shifts
                else:
                    s_grid = self.calibration_shifts_["global"]
                    n = self.calibration_counts_["global"]
                    k = int(np.ceil((n + 1) * q)) if q >= 0.5 else int(np.floor((n + 1) * q))
                    unres = q <= 0.0 or q >= 1.0 or (q >= 0.5 and k > n) or (q < 0.5 and k < 1)
                    s_interp = np.interp(q, Q_cal, s_grid)
                    if unres:
                        if getattr(self, "calibration_strict", False):
                            s_interp = np.inf if q >= 0.5 else -np.inf
                        elif not warned:
                            warnings.warn(
                                "Cannot resolve conformal quantile. Returning extreme residual.",
                                ConformalResolutionWarning,
                                stacklevel=2,
                            )
                            warned = True
                    raw_preds[q] += s_interp

        return raw_preds

    def predict_quantile(self, X, q):
        """
        Predict a specific quantile from the leaf ECDF with optional conformal calibration.

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
        """
        Return summary information about the fitted tree.

        Returns
        -------
        dict
            Tree metadata including node count, leaf count, max depth,
            and divergence metric used.

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
            "divergence": self.divergence,
        }

    def __getstate__(self):
        """Strip internal arrays and large calibration artifacts to reduce pickle size."""
        state = self.__dict__.copy()

        # Strip tree_state_ because it duplicates tree_data_ and prevents garbage collection
        if "tree_state_" in state:
            del state["tree_state_"]

        if "tree_data_" in state:
            # Copy the dict so we don't mutate the live object's tree_data_
            tree_data = state["tree_data_"].copy()
            # Remove arrays not required for inference (keep total_samples)
            tree_data.pop("wasserstein_gain", None)
            tree_data.pop("depth", None)
            state["tree_data_"] = tree_data

        # Strip massive calibration artifacts while keeping what's needed for inference
        state.pop("X_calib_", None)

        state.pop("_compiled_grid_cache", None)  # C++ object, rebuilt lazily

        return state

    def __setstate__(self, state):
        """Restore state."""
        self.__dict__.update(state)
        if hasattr(self, "tree_data_"):
            self.tree_state_ = TreeState.from_dict(self.tree_data_)

    def distribution_summary(self) -> dict:
        """
        Return summary statistics about leaf population and smoothing state.

        Returns
        -------
        dict with keys:
            n_leaves : int
            n_internal : int
            leaf_samples_min : int
            leaf_samples_max : int
            leaf_samples_median : float
            leaf_samples_mean : float
            n_sparse_leaves : int  (leaves with N < 2 * min_samples_leaf)
            sparse_fraction : float
            smoothing_applied : bool
            evt_enabled_count : int  (leaves with EVT fitted)
            evt_eligible_count : int  (leaves with N >= min_evt_samples)

        """
        check_is_fitted(self)
        is_leaf = self.tree_data_["is_leaf"]
        total_samples = self.tree_data_["total_samples"]

        leaf_mask = is_leaf.astype(bool)
        leaf_samples = total_samples[leaf_mask]

        sparse_threshold = 2 * self.min_samples_leaf
        n_sparse = int(np.sum(leaf_samples < sparse_threshold))

        result = {
            "n_leaves": int(leaf_mask.sum()),
            "n_internal": int((~leaf_mask).sum()),
            "leaf_samples_min": int(leaf_samples.min()) if len(leaf_samples) > 0 else 0,
            "leaf_samples_max": int(leaf_samples.max()) if len(leaf_samples) > 0 else 0,
            "leaf_samples_median": float(np.median(leaf_samples)) if len(leaf_samples) > 0 else 0.0,
            "leaf_samples_mean": float(leaf_samples.mean()) if len(leaf_samples) > 0 else 0.0,
            "n_sparse_leaves": n_sparse,
            "sparse_fraction": float(n_sparse / len(leaf_samples)) if len(leaf_samples) > 0 else 0.0,
            "smoothing_applied": "smoothed_pmf" in self.tree_data_,
            "evt_enabled_count": int(self.tree_data_.get("evt_enabled", np.zeros(0)).sum()),
            "evt_eligible_count": int(np.sum(leaf_samples >= getattr(self, "evt_min_samples", 30))),
        }
        return result

    # (Removed duplicated __getstate__)

    def _preprocess_X(self, X):
        """Validate and quantize feature matrix for prediction."""
        X_raw = np.asarray(X, dtype=np.float64)
        self._validate_no_nan_inf(X_raw, "X")

        if hasattr(self, "_validate_data"):
            X = self._validate_data(X, reset=False, dtype=np.float64)
        else:
            from sklearn.utils.validation import check_array

            X = check_array(X, dtype=np.float64)
            if X.shape[1] != getattr(self, "n_features_in_", X.shape[1]):
                raise ValueError(
                    f"X has {X.shape[1]} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input."
                )

        X_q = self.feature_quantizer_.transform(X)
        return np.ascontiguousarray(X_q, dtype=np.uint8)

    @staticmethod
    def _validate_no_nan_inf(arr, name):
        """Reject NaN/Inf inputs (FR-CORE-05)."""
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name} contains NaN values. DDT requires finite inputs.")
        if np.any(np.isinf(arr)):
            raise ValueError(f"{name} contains inf values. DDT requires finite inputs.")

    @staticmethod
    def _get_l1_cache_size_bytes():
        """Attempt to detect L1 data cache size in bytes."""
        system = platform.system()
        try:
            if system == "Linux":
                with open("/sys/devices/system/cpu/cpu0/cache/index0/size") as f:
                    raw = f.read().strip()
                    mult = 1024 if raw.endswith("K") else (1024**2 if raw.endswith("M") else 1)
                    return int(raw.rstrip("KM")) * mult
            elif system == "Darwin":
                out = subprocess.check_output(["sysctl", "-n", "hw.l1dcachesize"]).decode().strip()
                return int(out)
            elif system == "Windows":
                out = subprocess.check_output(
                    "wmic cpu get L1CacheSize /value",
                    shell=True,
                    stderr=subprocess.DEVNULL,
                ).decode()
                for line in out.splitlines():
                    if "L1CacheSize" in line and "=" in line:
                        val = line.split("=")[1].strip()
                        if val:
                            return int(val) * 1024
        except Exception:
            pass
        return 32 * 1024  # conservative 32 KB fallback

    def memory_report(self):
        """
        Generate a report on whether the tree fits in L1 cache.

        Returns
        -------
        dict
            A dictionary containing L1 cache size, tree memory footprint,
            and whether it fits.

        """
        import warnings

        warnings.warn(
            "memory_report() is deprecated and will be removed in a future version. Use model_footprint() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        l1_bytes = self._get_l1_cache_size_bytes()
        td = getattr(self, "tree_data_", None)
        if td is None:
            return {"fitted": False}

        footprint = self.model_footprint()
        total_bytes = footprint["total_bytes"]
        fits = total_bytes <= l1_bytes

        return {
            "fitted": True,
            "l1_size_bytes": l1_bytes,
            "tree_size_bytes": total_bytes,
            "node_bytes": footprint["tree_navigation_bytes"],
            "leaf_bytes": footprint["distribution_bytes"],
            "fits_in_l1": fits,
        }


class _Route:
    def __init__(self, estimator):
        self.est = estimator
        self.use_cpp = estimator.quantize_engine == "cpp"
        self.has_smooth_pmf = hasattr(estimator, "tree_state_") and getattr(
            estimator.tree_state_, "has_smooth_pmf", False
        )
        self.has_evt = getattr(estimator, "evt_active_", False) or getattr(estimator, "evt_lower_active_", False)
        # Backwards compatibility for models pickled before v1.1.0
        if not hasattr(estimator, "evt_active_"):
            self.has_evt = (
                self.has_evt
                or (estimator.evt_tails and "evt_enabled" in estimator.tree_data_)
                or (getattr(estimator, "evt_tails_lower", False) and "evt_lower_enabled" in estimator.tree_data_)
            )
        self.fast_float = (
            getattr(estimator, "fast_inference", False) and "split_threshold_float" in estimator.tree_data_
        )

        self.bin_lo, self.bin_width, self.bin_rep = estimator.target_binner_.quantile_grid()
        self.bin_centers = estimator.target_binner_.inverse_transform_bin_centers().astype(np.float64)

        self.interpolation = getattr(estimator, "quantile_interpolation", "linear")
        if self.interpolation == "snap":
            self.bin_lo_arg = self.bin_rep
            self.bin_width_arg = np.zeros_like(self.bin_width)
            self.bin_rep_arg = self.bin_rep
        else:
            self.bin_lo_arg = self.bin_lo
            self.bin_width_arg = self.bin_width
            self.bin_rep_arg = self.bin_rep

        # Grid validated once and cached on the estimator:
        # the C++ side then skips per-call array conversion/validation.  The cache is
        # keyed on the identity of the binner's arrays (which are replaced on refit) and
        # holds references to them so their ids cannot be recycled.
        src = (self.bin_lo, self.bin_width, self.bin_rep)
        key = (tuple(id(a) for a in src), self.interpolation)
        cached = getattr(estimator, "_compiled_grid_cache", None)
        if cached is not None and cached[0] == key:
            self.grid = cached[2]
        else:
            from . import _ddt_core

            self.grid = _ddt_core.CompiledGrid(self.bin_lo_arg, self.bin_width_arg, self.bin_rep_arg)
            estimator._compiled_grid_cache = (key, src, self.grid)

    def from_X(self, X, quantiles):
        from . import _ddt_core

        q_array = np.array(quantiles, dtype=np.float64)
        X_arr = np.asarray(X, dtype=np.float64)
        self.est._validate_no_nan_inf(X_arr, "X")

        if self.use_cpp:
            fq = self.est.feature_quantizer_
            if self.has_smooth_pmf and self.has_evt:
                res = _ddt_core.predict_quantiles_fast_smooth_evt(
                    self.est.tree_data_, X_arr, fq._flat_bin_edges_, fq._bin_offsets_, fq.n_bins, q_array, self.grid
                )
            elif self.has_evt:
                res = _ddt_core.predict_quantiles_fast_evt(
                    self.est.tree_data_, X_arr, fq._flat_bin_edges_, fq._bin_offsets_, fq.n_bins, q_array, self.grid
                )
            elif self.has_smooth_pmf:
                res = _ddt_core.predict_quantiles_fast_smooth(
                    self.est.tree_data_, X_arr, fq._flat_bin_edges_, fq._bin_offsets_, fq.n_bins, q_array, self.grid
                )
            elif self.fast_float:
                res = _ddt_core.predict_quantiles_fast_float(self.est.tree_data_, X_arr, q_array, self.grid)
            else:
                res = _ddt_core.predict_quantiles_fast(
                    self.est.tree_data_, X_arr, fq._flat_bin_edges_, fq._bin_offsets_, fq.n_bins, q_array, self.grid
                )
            return {q: res[i] for i, q in enumerate(quantiles)}
        else:
            X_q = self.est._preprocess_X(X)
            leaf_ids = _ddt_core.predict_leaves(self.est.tree_data_, X_q)
            return self.from_leaves(leaf_ids, quantiles)

    def from_leaves(self, leaf_ids, quantiles):
        from . import _ddt_core

        q_array = np.array(quantiles, dtype=np.float64)

        if self.est.compact_inference or not self.use_cpp:
            return self._from_leaves_python(leaf_ids, quantiles)

        try:
            leaf_ids_arr = np.asarray(leaf_ids, dtype=np.int32)
            res = _ddt_core.predict_quantiles_from_leaves(
                self.est.tree_data_,
                leaf_ids_arr,
                q_array,
                self.bin_lo_arg,
                self.bin_width_arg,
                self.bin_rep_arg,
                self.has_smooth_pmf,
                self.has_evt,
            )
            return {q: res[:, i] for i, q in enumerate(quantiles)}
        except Exception:
            return self._from_leaves_python(leaf_ids, quantiles)

    def _from_leaves_python(self, leaf_ids, quantiles):
        from . import _ddt_core

        unique_leaves, inverse = np.unique(leaf_ids, return_inverse=True)
        max_node_id = int(np.max(unique_leaves)) if len(unique_leaves) > 0 else 0

        if self.has_smooth_pmf:
            if "smoothed_pmf" in self.est.tree_data_:
                dist = self.est.tree_data_["smoothed_pmf"][leaf_ids]
            elif "inference_pmf" in self.est.tree_data_:
                dist = self.est.tree_data_["inference_pmf"][leaf_ids].astype(np.float64)
        else:
            n_bins_active = self.est.target_binner_.n_bins_active_
            dist_lookup = np.zeros((max_node_id + 1, n_bins_active), dtype=np.int32)
            for leaf_idx in unique_leaves:
                dist_lookup[leaf_idx] = _ddt_core.get_node_distribution(self.est.tree_data_, int(leaf_idx))
            dist = dist_lookup[leaf_ids]

        has_upper_evt = self.est.evt_tails and "evt_enabled" in self.est.tree_data_
        has_lower_evt = getattr(self.est, "evt_tails_lower", False) and "evt_lower_enabled" in self.est.tree_data_

        if not (has_upper_evt or has_lower_evt):
            return self.est._predict_raw_quantiles(None, quantiles, distributions=dist)

        # exact body/EVT splice (NumPy mirror of
        # ddt_core.hpp::quantiles_for_leaf; single implementation in _evt.py).
        from ._evt import spliced_quantiles_from_dist

        res = spliced_quantiles_from_dist(
            dist,
            np.asarray(leaf_ids),
            self.est.tree_data_,
            quantiles,
            self.bin_lo,
            self.bin_width,
            self.bin_rep,
            self.interpolation == "snap",
            bool(has_upper_evt),
            bool(has_lower_evt),
        )
        return {q: res[float(q)] for q in quantiles}
