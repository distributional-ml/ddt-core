"""
DDT Preprocessor — Feature Quantization & Target Binning
=========================================================
Handles FR-CORE-01 (double discretization) and FR-CORE-02 (boundary clamping).

Feature Quantization:
    Continuous features X ∈ R^(N×F) → uint8 X_q ∈ {0, ..., 255}^(N×F)
    via quantile-based binning (K=256 bins).

Target Binning:
    Continuous target y ∈ R^N → int32 y_q ∈ {0, ..., B-1}^N
    via equal-width binning with explicit boundary clamping.
"""

import warnings

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class FeatureQuantizer:
    """
    Quantize continuous features into K=256 quantile bins (uint8).

    Uses sklearn's KBinsDiscretizer with quantile strategy to produce
    approximately equal-frequency bins. Output dtype is uint8 for
    direct consumption by the C++ core engine.

    Parameters
    ----------
    n_bins : int, default=256
        Number of quantile bins per feature. Maximum 256 (uint8 range).

    """

    def __init__(self, n_bins: int = 256, add_jitter: bool = True, jitter_scale: float = 1e-8, engine: str = "cpp"):
        if n_bins < 2 or n_bins > 256:
            raise ValueError(f"n_bins must be between 2 and 256, got {n_bins}")
        self.n_bins = n_bins
        self.add_jitter = add_jitter
        self.jitter_scale = jitter_scale
        if engine not in ["cpp", "python"]:
            raise ValueError("engine must be 'cpp' or 'python'")
        self.engine = engine
        self._discretizer = None
        self.bin_edges_ = None
        self._bin_offsets_ = None
        self._flat_bin_edges_ = None

    def fit(self, X: np.ndarray) -> "FeatureQuantizer":
        """
        Fit the quantizer on training features.

        Parameters
        ----------
        X : np.ndarray, shape (N, F)
            Continuous feature matrix.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X contains NaN or Inf values (FR-CORE-05).

        """
        X = np.asarray(X, dtype=np.float64)
        self._validate_no_nan_inf(X, "X")

        if self.add_jitter:
            # Add small random noise to break ties for highly concentrated features
            # (e.g., lots of zeros), ensuring we can still compute unique quantiles.
            rng = np.random.default_rng(42)
            noise = rng.uniform(-self.jitter_scale, self.jitter_scale, size=X.shape)
            X_fit = X + noise
        else:
            X_fit = X

        if self.engine == "cpp":
            # Fast vectorized quantiles bypassing KBinsDiscretizer overhead
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            all_edges = np.quantile(X_fit, quantiles, axis=0)

            bin_edges_list = []
            for f in range(X_fit.shape[1]):
                edges = np.unique(all_edges[:, f])
                bin_edges_list.append(edges)

            self._bin_offsets_ = np.zeros(len(bin_edges_list) + 1, dtype=np.int32)
            for i, edges in enumerate(bin_edges_list):
                self._bin_offsets_[i + 1] = self._bin_offsets_[i] + len(edges)

            self._flat_bin_edges_ = np.concatenate(bin_edges_list).astype(np.float64)
            self.bin_edges_ = np.array(bin_edges_list, dtype=object)
        else:
            self._discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode="ordinal",
                strategy="quantile",
                subsample=None,  # Use all samples for bin edges
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Bins whose width are too small",
                    category=UserWarning,
                )
                self._discretizer.fit(X_fit)
            self.bin_edges_ = self._discretizer.bin_edges_

        return self

    def __getstate__(self):
        """Drop redundant python object arrays for smaller pickle size."""
        state = self.__dict__.copy()
        if "bin_edges_" in state:
            state["bin_edges_"] = None
        # The sklearn discretizer is not needed after fit if using cpp engine
        if self.engine == "cpp" and "_discretizer" in state:
            state["_discretizer"] = None
        return state

    def __setstate__(self, state):
        """Reconstruct python object arrays on load."""
        self.__dict__.update(state)
        if (
            self.bin_edges_ is None
            and getattr(self, "_bin_offsets_", None) is not None
            and getattr(self, "_flat_bin_edges_", None) is not None
        ):
            bin_edges_list = []
            for i in range(len(self._bin_offsets_) - 1):
                start = self._bin_offsets_[i]
                end = self._bin_offsets_[i + 1]
                bin_edges_list.append(self._flat_bin_edges_[start:end])
            self.bin_edges_ = np.array(bin_edges_list, dtype=object)

    def get_feature_threshold(self, feat_idx: int, binned_thresh: int) -> float:
        """
        Get the real-valued threshold corresponding to a quantized bin index.

        Parameters
        ----------
        feat_idx : int
            Feature index.
        binned_thresh : int
            Quantized threshold index (in uint8 range [0, 255]).

        Returns
        -------
        float
            Real-valued continuous threshold.

        """
        if self.bin_edges_ is None:
            raise RuntimeError("FeatureQuantizer has not been fitted yet.")
        edges = self.bin_edges_[feat_idx]
        if len(edges) == 0:
            return 0.0
        idx = binned_thresh + 1
        if idx < len(edges):
            return float(edges[idx])
        return float(edges[-1])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features to quantized uint8 representation.

        Parameters
        ----------
        X : np.ndarray, shape (N, F)
            Continuous feature matrix.

        Returns
        -------
        np.ndarray, shape (N, F), dtype=uint8
            Quantized feature matrix with values in [0, n_bins-1].

        Raises
        ------
        ValueError
            If X contains NaN or Inf values (FR-CORE-05).

        """
        X = np.asarray(X, dtype=np.float64)
        self._validate_no_nan_inf(X, "X")

        if self.engine == "cpp":
            from . import _ddt_core

            X_q = _ddt_core.quantize_features(X, self._flat_bin_edges_, self._bin_offsets_, self.n_bins)
        else:
            X_q = self._discretizer.transform(X)
            # Explicit clamp for safety (FR-CORE-02).
            X_q = np.clip(X_q, 0, self.n_bins - 1).astype(np.uint8)

        return X_q

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    @staticmethod
    def _validate_no_nan_inf(arr: np.ndarray, name: str) -> None:
        """Reject NaN/Inf inputs (FR-CORE-05)."""
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name} contains NaN values. DDT requires finite inputs.")
        if np.any(np.isinf(arr)):
            raise ValueError(f"{name} contains Inf values. DDT requires finite inputs.")


class TargetBinner:
    """
    Discretize continuous target into B bins (int32).

    Supports three bin edge strategies:

    ``"equal_width"`` (default)
        Uniform bin widths across ``[y_min, y_max]``. Recommended for general use
        and required for FPGA / bare-metal inference where arithmetic bin lookup
        (no lookup table) is needed. The C++ equal-width fast path is used.

    ``"log_width"``
        Logarithmically spaced bin edges. Narrow bins near the origin (high
        resolution for the common case) and exponentially wider toward the tail.
        Suitable for strictly positive right-skewed targets (prices, lead times,
        durations). Requires ``y_min > 0`` after winsorization. Uses the C++
        variable-width Wasserstein path. Cannot be combined with
        ``target_transform='log1p'``.

    ``"manual"``
        User-supplied bin edges array of length ``n_bins + 1``. Useful for
        regulatory / actuarial brackets or domain-expert boundaries. Cannot be
        combined with ``target_transform``.

    Parameters
    ----------
    n_bins : int, default=30
        Number of target bins B. Must be in [2, 100].
    target_transform : str or None, default=None
        Transform applied to y before binning. Options: ``'log1p'`` or ``None``.
        Cannot be combined with ``bin_strategy='log_width'`` or ``'manual'``.
    winsorize_tails : float or None, default=0.001
        Fraction of data clipped from both tails before binning (e.g. 0.001 clips
        P0.1 and P99.9). Must be in (0, 0.5). Pass ``None`` to disable.
        Note: ``winsorize_tails=0.0`` is invalid — use ``None`` to disable.
    bin_strategy : str, default="equal_width"
        Bin edge strategy. One of ``"equal_width"``, ``"log_width"``, ``"manual"``.

        **Note on bin strategy and split behavior:**
        When using ``bin_strategy="log_width"`` or ``"hybrid"``, the Wasserstein split
        criterion weights bin differences by their physical width. This means the tree will
        preferentially split on features that differentiate the *tails* of the distribution,
        not the *modes*. If your application requires high body resolution (e.g., detecting
        bimodality in the core IQR), consider using ``bin_strategy="equal_width"`` or
        increasing the ``core_fraction`` parameter.
    manual_bin_edges : array-like or None, default=None
        Required when ``bin_strategy="manual"``. Strictly monotonically increasing
        array of length ``n_bins + 1`` specifying bin boundaries in target space.
    consolidate_bins : bool, default=False
        If True, merge always-empty bins (zero training samples) into their
        left neighbour after fitting. Reduces ``n_bins_active_`` and the C++
        loop overhead. Consolidation makes ``delta_x_norm_`` non-None even for
        ``"equal_width"`` models — the C++ weighted path will be used.

    Attributes (fitted)
    -------------------
    bin_edges_ : ndarray, shape (n_bins + 1,)
        Original configured bin edges — never mutated after fit, even if
        consolidation is applied.
    n_bins_active_ : int
        Number of active bins after consolidation. Equals ``n_bins`` when
        ``consolidate_bins=False``.
    active_bins_mask_ : ndarray, bool, shape (n_bins,) or None
        True for bins that survived consolidation. None if not consolidated.
    bin_remap_ : ndarray, int32, shape (n_bins,) or None
        Maps original bin index -> active bin index. None if not consolidated.
    consolidated_edges_ : ndarray, shape (n_bins_active_ + 1,) or None
        Bin edges for active bins only. None if not consolidated.
    bin_widths_ : ndarray, shape (n_bins_active_,) or None
        Physical width of each active bin in target space. None for
        ``"equal_width"`` without consolidation.
    delta_x_norm_ : ndarray, shape (n_bins_active_,) or None
        Normalised bin widths: ``bin_widths_ / mean(bin_widths_)``. Non-None
        whenever the C++ weighted path must be used (variable strategy or
        consolidation applied). Exactly ``[1.0, ..., 1.0]`` conceptually for
        equal-width with no consolidation, but stored as None to avoid the overhead.
    bin_lo_ : ndarray, shape (n_bins_active_,)
        Lower edge of each active bin in original target space (or atom value).
    bin_width_ : ndarray, shape (n_bins_active_,)
        Width of each active bin in original target space (0.0 for atom bins).
    bin_rep_ : ndarray, shape (n_bins_active_,)
        Point representative (centroid, midpoint, or atom) for each active bin.

    """

    _VALID_STRATEGIES = ("equal_width", "log_width", "manual", "hybrid")

    def __init__(
        self,
        n_bins: int = 30,
        target_transform: str = None,
        winsorize_tails: float = 0.001,
        bin_strategy: str = "equal_width",
        manual_bin_edges: np.ndarray = None,
        core_fraction: float = 0.5,
        consolidate_bins: bool = False,
        atom_threshold: float = 0.9,
    ):
        if n_bins < 2 or n_bins > 100:
            raise ValueError(f"n_bins must be in [2, 100], got {n_bins}")
        if bin_strategy not in self._VALID_STRATEGIES:
            raise ValueError(f"bin_strategy must be one of {self._VALID_STRATEGIES}, got {bin_strategy!r}")
        if bin_strategy == "manual" and manual_bin_edges is None:
            raise ValueError("manual_bin_edges is required when bin_strategy='manual'.")
        if bin_strategy != "equal_width" and target_transform is not None:
            raise ValueError(
                f"Cannot combine bin_strategy={bin_strategy!r} with "
                f"target_transform={target_transform!r}. Apply the transform "
                f"externally before fitting, or use bin_strategy='equal_width'."
            )

        self.n_bins = n_bins
        self.target_transform = target_transform
        self.winsorize_tails = winsorize_tails
        self.bin_strategy = bin_strategy
        self.manual_bin_edges = manual_bin_edges
        self.core_fraction = core_fraction
        self.consolidate_bins = consolidate_bins
        self.atom_threshold = atom_threshold

        # Fitted attributes — set in fit()
        self.bin_edges_ = None  # Original edges, never mutated
        self.y_min_ = None
        self.y_max_ = None
        self.winsorize_lower_ = None
        self.winsorize_upper_ = None

        # Consolidation attributes — set in consolidate_empty_bins()
        self.n_bins_active_ = n_bins  # Updated after consolidation
        self.active_bins_mask_ = None  # bool (n_bins,)
        self.bin_remap_ = None  # int32 (n_bins,)
        self.consolidated_edges_ = None  # float64 (n_bins_active_+1,)
        self.bin_widths_ = None  # float64 (n_bins_active_,)
        self.delta_x_norm_ = None  # float64 (n_bins_active_,)

        # Grid attributes — set in _compute_quantile_grid()
        self.bin_lo_ = None
        self.bin_width_ = None
        self.bin_rep_ = None

    @property
    def has_variable_bins(self) -> bool:
        """
        True when the C++ weighted Wasserstein path must be used.

        This is the case when bin_strategy != 'equal_width' OR when
        consolidation has produced non-uniform bin widths.
        """
        return self.delta_x_norm_ is not None

    def fit(self, y: np.ndarray) -> "TargetBinner":
        """
        Fit the binner on training targets.

        Computes bin edges according to ``bin_strategy``. For ``"equal_width"``,
        produces B+1 linearly spaced edges. For ``"log_width"``, produces
        logarithmically spaced edges. For ``"manual"``, validates and stores the
        user-supplied edges.

        Parameters
        ----------
        y : np.ndarray, shape (N,)
            Continuous target vector.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If y contains NaN or Inf values (FR-CORE-05).
        ValueError
            If y has zero range (all identical values).
        ValueError
            If ``bin_strategy='log_width'`` and y_min <= 0 after winsorization.

        """
        y = np.asarray(y, dtype=np.float64).ravel()
        self._validate_no_nan_inf(y, "y")

        if self.winsorize_tails is not None:
            if not (0 < self.winsorize_tails < 0.5):
                raise ValueError("winsorize_tails must be in (0, 0.5)")
            self.winsorize_lower_ = np.percentile(y, self.winsorize_tails * 100)
            self.winsorize_upper_ = np.percentile(y, (1 - self.winsorize_tails) * 100)
            y = np.clip(y, self.winsorize_lower_, self.winsorize_upper_)

        if self.target_transform == "log1p":
            if np.any(y <= -1.0):
                raise ValueError("y contains values <= -1.0, cannot apply log1p.")
            y = np.log1p(y)
        elif self.target_transform is not None:
            raise ValueError(f"Unknown target_transform: {self.target_transform}")

        self.y_min_ = float(np.min(y))
        self.y_max_ = float(np.max(y))

        if self.y_min_ == self.y_max_:
            raise ValueError("Target y has zero range (all values identical). Cannot create meaningful bins.")

        # Compute bin edges per strategy
        if self.bin_strategy == "equal_width":
            self.bin_edges_ = np.linspace(self.y_min_, self.y_max_, self.n_bins + 1)

        elif self.bin_strategy == "log_width":
            if self.y_min_ <= 0.0:
                raise ValueError(
                    f"bin_strategy='log_width' requires y_min > 0 after winsorization, "
                    f"got y_min={self.y_min_:.6g}. "
                    f"Consider using target_transform='log1p' instead (works with y >= 0), "
                    f"or shift y before fitting."
                )
            self.bin_edges_ = np.geomspace(self.y_min_, self.y_max_, self.n_bins + 1)

        elif self.bin_strategy == "hybrid":
            if not (0.0 < self.core_fraction < 1.0):
                raise ValueError(
                    f"core_fraction must be strictly in (0, 1) for 'hybrid' strategy, got {self.core_fraction}"
                )
            n_core = int(round(self.n_bins * self.core_fraction))
            if n_core < 2:
                raise ValueError(
                    f"core_fraction={self.core_fraction} with n_bins={self.n_bins} "
                    f"yields too few core bins ({n_core}). Minimum is 2."
                )
            n_tail = self.n_bins - n_core
            n_lo = n_tail // 2
            n_hi = n_tail - n_lo
            if n_lo < 1 or n_hi < 1:
                raise ValueError(
                    f"core_fraction={self.core_fraction} with n_bins={self.n_bins} "
                    f"leaves too few tail bins. Decrease core_fraction or increase n_bins."
                )

            # Compute quantiles Q25 and Q75 on the (possibly winsorized) data
            Q25, Q75 = float(np.percentile(y, 25)), float(np.percentile(y, 75))
            if Q25 == Q75:
                raise ValueError(
                    f"Target y has identical 25th and 75th percentiles ({Q25}). "
                    f"'hybrid' strategy requires a measurable IQR. Use 'equal_width' instead."
                )

            # Core edges (linear)
            core_edges = np.linspace(Q25, Q75, n_core + 1)

            # Lower tail (log-spaced from y_min to Q25)
            # Use geomspace on shifted values so the finest resolution is near Q25
            if self.y_min_ > 0:
                lo_edges = np.geomspace(self.y_min_, Q25, n_lo + 1)
            else:
                shift = abs(self.y_min_) + 1.0
                lo_edges = np.geomspace(self.y_min_ + shift, Q25 + shift, n_lo + 1) - shift

            # Upper tail (log-spaced from Q75 to y_max)
            # Use geomspace on shifted values so finest resolution is near Q75
            shift_hi = abs(Q75) + 1.0 if Q75 <= 0 else 0.0
            hi_edges = np.geomspace(Q75 + shift_hi, self.y_max_ + shift_hi, n_hi + 1) - shift_hi

            # Concatenate, dropping duplicate boundary points
            self.bin_edges_ = np.concatenate([lo_edges, core_edges[1:], hi_edges[1:]])

        else:  # "manual"
            edges = np.asarray(self.manual_bin_edges, dtype=np.float64).ravel()
            if len(edges) != self.n_bins + 1:
                raise ValueError(f"manual_bin_edges must have length n_bins + 1 = {self.n_bins + 1}, got {len(edges)}.")
            if not np.all(np.diff(edges) > 0):
                raise ValueError("manual_bin_edges must be strictly monotonically increasing.")
            self.bin_edges_ = edges.copy()

        # Compute bin widths and normalised weights for variable-bin strategies.
        # For equal_width, these stay None until consolidation (if any) is applied.
        if self.bin_strategy != "equal_width":
            self._compute_bin_widths(self.bin_edges_)

        return self

    def _compute_bin_widths(self, edges: np.ndarray) -> None:
        """Compute bin_widths_ and delta_x_norm_ from a set of bin edges."""
        widths = np.diff(edges)
        mean_w = float(np.mean(widths))
        self.bin_widths_ = widths
        self.delta_x_norm_ = widths / mean_w  # normalised: mean = 1.0

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform target to quantized int32 bin indices.

        Applies explicit boundary clamping (FR-CORE-02):
            idx = clip(digitize(y) - 1, 0, B - 1)

        When ``consolidate_empty_bins()`` has been called, remaps indices using
        ``bin_remap_`` so output is in ``[0, n_bins_active_ - 1]``.

        Parameters
        ----------
        y : np.ndarray, shape (N,)
            Continuous target vector.

        Returns
        -------
        np.ndarray, shape (N,), dtype=int32
            Quantized target with values in ``[0, n_bins_active_ - 1]``.

        Raises
        ------
        ValueError
            If y contains NaN or Inf values (FR-CORE-05).

        Notes
        -----
        **Winsorization is NOT re-applied during transform.** The clipping
        bounds computed in ``fit`` are stored as ``winsorize_lower_`` and
        ``winsorize_upper_`` for inspection, but inference targets outside
        the training range are handled by bin clamping (FR-CORE-02) rather
        than value clipping. This preserves the original inference values in
        any downstream pipeline while still preventing C++ out-of-bounds access.
        If you want consistent clipping at inference time, apply it manually:
        ``y_clipped = np.clip(y, binner.winsorize_lower_, binner.winsorize_upper_)``
        before calling ``transform``.

        """
        y = np.asarray(y, dtype=np.float64).ravel()
        self._validate_no_nan_inf(y, "y")

        # Apply transform (inference: clamp instead of reject for log1p)
        if self.target_transform == "log1p":
            n_neg = int(np.sum(y <= -1.0))
            if n_neg > 0:
                warnings.warn(
                    f"log1p transform: {n_neg} sample(s) have y <= -1.0 and will be "
                    f"silently clamped to y = -1.0 + 1e-15 before applying log1p. "
                    f"These will map to target bin 0. If this is unexpected, consider "
                    f"winsorizing the target before calling transform.",
                    UserWarning,
                    stacklevel=2,
                )
            y_safe = np.maximum(y, -1.0 + 1e-15)
            y = np.log1p(y_safe)

        # np.digitize returns indices in [1, B] for values within range.
        # Subtract 1 to get [0, B-1]. Clamp for out-of-range safety.
        indices = np.digitize(y, self.bin_edges_[1:-1])
        indices = np.clip(indices, 0, self.n_bins - 1).astype(np.int32)

        # Apply consolidation remap if active
        if self.bin_remap_ is not None:
            indices = self.bin_remap_[indices]

        return indices

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        If ``consolidate_bins=True``, calls ``consolidate_empty_bins()``
        automatically after fitting. Also computes the quantile grid.
        """
        y_orig = np.asarray(y, dtype=np.float64).ravel()
        self.fit(y_orig)
        y_q = self.transform(y_orig)
        if self.consolidate_bins:
            y_q = self.consolidate_empty_bins(y_q)

        if self.winsorize_tails is not None:
            y_orig_win = np.clip(y_orig, self.winsorize_lower_, self.winsorize_upper_)
        else:
            y_orig_win = y_orig

        self._compute_quantile_grid(y_orig_win, y_q)
        return y_q

    def _compute_quantile_grid(self, y_orig_win: np.ndarray, y_q: np.ndarray) -> None:
        B = self.n_bins_active_
        N = len(y_orig_win)
        edges = self.consolidated_edges_ if self.consolidated_edges_ is not None else self.bin_edges_

        if self.target_transform == "log1p":
            edges_orig = np.expm1(edges)
        else:
            edges_orig = edges

        bin_lo = edges_orig[:-1].copy()
        bin_hi = edges_orig[1:].copy()
        bin_width = bin_hi - bin_lo
        bin_rep = (bin_lo + bin_hi) / 2.0

        for b in range(B):
            mask = y_q == b
            count = np.sum(mask)
            if count > 0:
                y_b = y_orig_win[mask]
                bin_rep[b] = np.mean(y_b)

                if self.atom_threshold is not None:
                    unique_vals, unique_counts = np.unique(y_b, return_counts=True)
                    max_idx = np.argmax(unique_counts)
                    atom_val = unique_vals[max_idx]
                    atom_count = unique_counts[max_idx]

                    cond1 = atom_count >= self.atom_threshold * count
                    cond2 = atom_count >= 0.01 * N
                    cond3 = True
                    if self.winsorize_tails is not None:
                        if atom_val == self.winsorize_lower_ or atom_val == self.winsorize_upper_:
                            cond3 = False

                    if cond1 and cond2 and cond3:
                        bin_lo[b] = atom_val
                        bin_rep[b] = atom_val
                        bin_width[b] = 0.0

        self.bin_lo_ = bin_lo
        self.bin_width_ = bin_width
        self.bin_rep_ = bin_rep

    def quantile_grid(self):
        """Return the (bin_lo, bin_width, bin_rep) arrays for quantile inversion."""
        if self.bin_lo_ is None:
            raise RuntimeError("TargetBinner has not been fitted, or was fitted with an older version.")
        return self.bin_lo_, self.bin_width_, self.bin_rep_

    def consolidate_empty_bins(self, y_q: np.ndarray) -> np.ndarray:
        """
        Merge always-empty bins into their left neighbours.

        Detects which bins have zero samples in ``y_q`` (the training set
        quantised targets) and merges each empty bin into its left neighbour
        (or right neighbour for bin 0). Updates ``bin_edges_``,
        ``consolidated_edges_``, ``bin_widths_``, ``delta_x_norm_``,
        ``n_bins_active_``, ``active_bins_mask_``, and ``bin_remap_``.

        Because consolidation only merges EMPTY bins, reconstruction via
        ``reconstruct_original_grid()`` is exact and lossless.

        Parameters
        ----------
        y_q : np.ndarray, shape (N,), dtype int32
            Quantised training targets from ``transform()``. Must use the
            original ``n_bins`` indexing (not yet remapped).

        Returns
        -------
        np.ndarray, shape (N,), dtype int32
            Remapped quantised targets in ``[0, n_bins_active_ - 1]``.

        """
        bin_counts = np.bincount(y_q, minlength=self.n_bins)
        empty_mask = bin_counts == 0

        if not np.any(empty_mask):
            # No empty bins — set attributes to reflect no-op consolidation
            self.active_bins_mask_ = np.ones(self.n_bins, dtype=bool)
            self.bin_remap_ = np.arange(self.n_bins, dtype=np.int32)
            self.n_bins_active_ = self.n_bins
            self.consolidated_edges_ = self.bin_edges_.copy()
            # Compute widths for equal_width if not already set
            if self.bin_widths_ is None:
                self._compute_bin_widths(self.consolidated_edges_)
            return y_q

        # Build remap: empty bin b is absorbed into its left neighbour
        remap = np.arange(self.n_bins, dtype=np.int32)
        for b in range(self.n_bins):
            if empty_mask[b]:
                if b > 0:
                    remap[b] = remap[b - 1]  # absorb into left neighbour
                else:
                    # bin 0 is empty: find first non-empty to the right
                    for b2 in range(1, self.n_bins):
                        if not empty_mask[b2]:
                            remap[b] = np.int32(b2)
                            break

        # Compact remap to contiguous [0, n_active-1]
        unique_old = np.unique(remap)  # sorted ascending old-bin indices
        compact = np.searchsorted(unique_old, remap).astype(np.int32)

        self.active_bins_mask_ = np.zeros(self.n_bins, dtype=bool)
        self.active_bins_mask_[unique_old] = True
        self.bin_remap_ = compact
        self.n_bins_active_ = len(unique_old)

        # Build consolidated edges:
        # active bin `a` spans from original left edge of unique_old[a]
        # to the original left edge of unique_old[a+1] (or the final right edge).
        consolidated = np.empty(self.n_bins_active_ + 1, dtype=np.float64)
        for a, old_b in enumerate(unique_old):
            consolidated[a] = self.bin_edges_[old_b]
        consolidated[-1] = self.bin_edges_[-1]
        self.consolidated_edges_ = consolidated

        # Recompute bin widths on consolidated edges
        self._compute_bin_widths(consolidated)

        return compact[y_q]

    def reconstruct_original_grid(self, active_distribution: np.ndarray) -> np.ndarray:
        """
        Expand a consolidated distribution back to the original n_bins grid.

        Because consolidation only ever merges EMPTY bins (zero training samples),
        reconstruction is exact and lossless:
        - Empty bin positions receive probability mass 0.0.
        - Active bin masses are placed at their original indices unchanged.
        - Mass is conserved: ``sum(output, axis=-1) == sum(input, axis=-1)``.

        Parameters
        ----------
        active_distribution : np.ndarray, shape (..., n_bins_active_)
            Probability distribution(s) over active bins. Can be any batch shape.

        Returns
        -------
        np.ndarray, shape (..., n_bins)
            Distribution(s) on the original ``n_bins`` grid.

        Raises
        ------
        RuntimeError
            If ``consolidate_empty_bins()`` was not called.
        ValueError
            If ``active_distribution.shape[-1] != n_bins_active_``.

        """
        if self.active_bins_mask_ is None:
            raise RuntimeError(
                "consolidate_empty_bins() was not called. "
                "Fit with consolidate_bins=True or call consolidate_empty_bins() manually."
            )
        if active_distribution.shape[-1] != self.n_bins_active_:
            raise ValueError(
                f"active_distribution.shape[-1] = {active_distribution.shape[-1]} "
                f"does not match n_bins_active_ = {self.n_bins_active_}."
            )
        original_indices = np.where(self.active_bins_mask_)[0]  # (n_bins_active_,)
        out_shape = active_distribution.shape[:-1] + (self.n_bins,)
        result = np.zeros(out_shape, dtype=active_distribution.dtype)
        result[..., original_indices] = active_distribution
        return result

    def inverse_transform_bin_centers(self) -> np.ndarray:
        """
        Return the centre value of each active bin.

        Note: This is kept for backward compatibility. For inference,
        prefer ``quantile_grid()`` which returns edges and representatives.

        Uses ``consolidated_edges_`` when consolidation has been applied,
        otherwise uses ``bin_edges_``. Applies ``expm1`` inverse if
        ``target_transform='log1p'`` was used.

        Returns
        -------
        np.ndarray, shape (n_bins_active_,), dtype=float64
            Centre value for each active bin in original target space.

        """
        if self.bin_edges_ is None:
            raise RuntimeError("TargetBinner has not been fitted yet.")

        edges = self.consolidated_edges_ if self.consolidated_edges_ is not None else self.bin_edges_
        centers = (edges[:-1] + edges[1:]) / 2.0

        if self.target_transform == "log1p":
            centers = np.expm1(centers)

        return centers

    @staticmethod
    def _validate_no_nan_inf(arr: np.ndarray, name: str) -> None:
        """Reject NaN/Inf inputs (FR-CORE-05)."""
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name} contains NaN values. DDT requires finite inputs.")
        if np.any(np.isinf(arr)):
            raise ValueError(f"{name} contains Inf values. DDT requires finite inputs.")
