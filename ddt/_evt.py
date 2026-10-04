"""
DDT EVT Oracle — Extreme Value Theory Tail Modeling
====================================================
Peak-Over-Threshold (POT) Generalized Pareto Distribution (GPD) fitting
for terminal leaf tails. Applied post-build; has zero impact on the
split-finding path or empirical histogram body.

Design:
  - scipy is a soft dependency: imported inside functions, not at module level.
    Importing ddt._evt when scipy is absent does not crash; only calling
    fit_gpd_upper_tail() / fit_gpd_lower_tail() / fit_all_leaf_gpds() with
    evt_tails=True does.
  - _ddt_core is imported inside fit_all_leaf_gpds() to mirror the pattern
    used throughout the codebase and avoid circular-import risk at module load.
  - Upper tail: exceedances above threshold u.
  - Lower tail: exceedances below threshold u_lower,
    re-expressed as positive distances z = u_lower - y for y < u_lower.
    The inversion carries a minus sign: x_q = u_lower - GPD_quantile(p).
    Lower tail is independently gated by evt_tails_lower=True.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# EVT Status Codes
EVT_STATUS_OK = 0
EVT_STATUS_STARVED = 1
EVT_STATUS_FIT_FAILED = 2
EVT_STATUS_GOF_REJECTED = 3
EVT_STATUS_SUPPORT_VIOLATION = 4


def _select_k(N: int, tail_fraction: float, min_k: int = 5) -> int:
    import math

    k = max(min_k, min(round(1.5 * math.sqrt(N)), N // 3))
    return min(k, N // 3)


def fit_gpd_upper_tail(
    leaf_values: np.ndarray,
    min_evt_samples: int = 30,
    tail_fraction: float = 0.05,
    evt_borrow_factor: float = 1.0,
    evt_gof_alpha: float = 0.05,
):
    try:
        from scipy.stats import genpareto, kstest
    except ImportError as exc:
        raise ImportError("EVT tail modeling requires scipy.") from exc

    N = len(leaf_values)
    if N == 0:
        return {"status": EVT_STATUS_STARVED}

        k_s = _select_k(N, tail_fraction, min_k=5)
    sorted_vals = np.sort(leaf_values)
    idx_s = max(0, N - k_s - 1)
    u_s = float(sorted_vals[idx_s])
    
        z = leaf_values[leaf_values > u_s] - u_s
    if len(z) < min_evt_samples:
        return {"status": EVT_STATUS_STARVED}
    
    try:
        xi, _loc, s_s = genpareto.fit(z, floc=0)
    except Exception:
        return {"status": EVT_STATUS_FIT_FAILED}
    
    if not np.isfinite(xi) or not np.isfinite(s_s):
        return {"status": EVT_STATUS_FIT_FAILED}
    
    if s_s <= 0:
        return {"status": EVT_STATUS_SUPPORT_VIOLATION}
    
    F_u = float(np.mean(leaf_values <= u_s))
    if F_u >= 1.0:
        return {"status": EVT_STATUS_STARVED}
    S_u = float(np.mean(leaf_values > u_s))
    
    return {"status": EVT_STATUS_OK, "u": float(u_s), "F_u": F_u, "S_u": S_u, "xi": float(xi), "sigma": float(s_s)}
    

def fit_gpd_lower_tail(
    leaf_values: np.ndarray,
    min_evt_samples: int = 30,
    tail_fraction: float = 0.05,
    evt_borrow_factor: float = 1.0,
    evt_gof_alpha: float = 0.05,
):
    try:
        from scipy.stats import genpareto, kstest
    except ImportError as exc:
        raise ImportError("EVT tail modeling requires scipy.") from exc

    N = len(leaf_values)
    if N == 0:
        return {"status": EVT_STATUS_STARVED}

        k_s = _select_k(N, tail_fraction, min_k=5)
    sorted_vals = np.sort(leaf_values)
    idx_s = min(N - 1, k_s)
    u_s = float(sorted_vals[idx_s])
    
        z = u_s - leaf_values[leaf_values < u_s]
    if len(z) < min_evt_samples:
        return {"status": EVT_STATUS_STARVED}
    
    try:
        xi, _loc, s_s = genpareto.fit(z, floc=0)
    except Exception:
        return {"status": EVT_STATUS_FIT_FAILED}
    
    if not np.isfinite(xi) or not np.isfinite(s_s):
        return {"status": EVT_STATUS_FIT_FAILED}
    
    if s_s <= 0:
        return {"status": EVT_STATUS_SUPPORT_VIOLATION}
    
    F_u = float(np.mean(leaf_values <= u_s))
    if F_u <= 0.0:
        return {"status": EVT_STATUS_STARVED}
    
    return {
        "status": EVT_STATUS_OK,
        "u_lower": float(u_s),
        "F_u_lower": F_u,
        "xi_lower": float(xi),
        "sigma_lower": float(s_s),
    }
    

def query_gpd_lower_quantile(q: float, u_lower: float, F_u_lower: float, xi: float, sigma: float) -> float:
    """
    Analytical inverse CDF for the GPD lower tail.

    Caller guarantees q <= F_u_lower.  Returns the physical target value at
    quantile q, which lies *below* u_lower by construction.

    The exceedance probability in the lower tail is defined as::

        exceedance_prob = q / F_u_lower

    so that exceedance_prob → 1 when q → F_u_lower (boundary: x_q → u_lower)
    and exceedance_prob → 0 when q → 0 (deep tail).

    Inversion formulae (mirror of query_gpd_quantile with a minus sign)::

        xi != 0:  x_q = u_lower - (sigma / xi) * (exceedance_prob^(-xi) - 1)
        xi == 0:  x_q = u_lower + sigma * log(exceedance_prob)

    Both converge to u_lower when exceedance_prob → 1, giving continuity at
    the body–tail splice point.

    Parameters
    ----------
    q : float
        Target quantile in (0, F_u_lower].  Caller must ensure q <= F_u_lower.
    u_lower : float
        Lower threshold (the physical value, not an excess).
    F_u_lower : float
        Empirical CDF mass at u_lower; P(Y <= u_lower).  Must be > 0.
    xi : float
        GPD shape parameter (lower-tail fit).
    sigma : float
        GPD scale parameter (lower-tail fit).  Must be > 0.

    Returns
    -------
    float
        Predicted value at quantile q.  Returns -inf when q <= 0 (caller
        must clamp).

    """
    if q <= 0.0:
        return -np.inf  # caller must clamp
    exceedance_prob = q / F_u_lower
    if abs(xi) < 1e-6:
        # Exponential tail limit (Gumbel domain)
        # log(exceedance_prob) <= 0 for exceedance_prob in (0, 1]
        # so x_q <= u_lower as expected
        return u_lower + sigma * np.log(exceedance_prob)
    # General Pareto inversion (leftward)
    return u_lower - (sigma / xi) * (exceedance_prob ** (-xi) - 1.0)


def query_gpd_quantile(
    q: float,
    u: float,
    F_u: float,
    xi: float,
    sigma: float,
    S_u: float = 0.0,
) -> float:
    """
    Analytical inverse CDF for the GPD upper tail.

    Caller guarantees q >= F_u. Returns the physical target value at quantile q.

    For |xi| > 1e-6 (Pareto tail):
        x_q = u + (sigma / xi) * (exceedance_prob^(-xi) - 1)

    For |xi| <= 1e-6 (exponential / Gumbel limit):
        x_q = u - sigma * log(exceedance_prob)

    where exceedance_prob = (1 - q) / S_u, and S_u = 1 - F_u.
    """
    if q >= 1.0:
        return np.inf  # caller must clamp
    if S_u > 0.0:
        exceedance_prob = (1.0 - q) / S_u
    else:
        denom = 1.0 - F_u
        if denom <= 0.0:
            return np.inf
        exceedance_prob = (1.0 - q) / denom
    if abs(xi) < 1e-6:
        # Exponential tail limit (Gumbel domain)
        return u - sigma * np.log(exceedance_prob)
    return u + (sigma / xi) * (exceedance_prob ** (-xi) - 1.0)


def fit_all_leaf_gpds(
    tree_data: dict,
    y_train: np.ndarray,
    target_binner,
    feature_quantizer,
    X_q: np.ndarray,
        min_samples: int = 30,
    tail_fraction: float = 0.05,
    fit_lower_tail: bool = False,
    ) -> dict:
        """
    Fit GPD upper-tail parameters for every terminal leaf in tree_data.

    Routes each training sample to its leaf via _ddt_core.predict_leaves, groups
    the raw y_train values by leaf, and calls fit_gpd_upper_tail for each leaf.
    Returns a dict of five numpy arrays keyed by EVT field name, with sentinel
    values for leaves where fitting was skipped (starvation gate or failure).

    Parameters
    ----------
    tree_data : dict
        Pybind11-serialized tree dict returned by build_tree[_weighted].
    y_train : np.ndarray, shape (N,)
        Raw (untransformed) training target values for the samples that were
        used to build this tree.  For forests this must be the bootstrap-sample
        slice, not the full training set.
    target_binner : TargetBinner
        Unused in the current implementation (reserved for future extensions).
    feature_quantizer : FeatureQuantizer
        Unused in the current implementation (reserved for future extensions).
    X_q : np.ndarray, shape (N, F), dtype uint8
        Quantized feature matrix for the same N samples as y_train.  Used to
        route samples to leaves via predict_leaves.
    min_samples : int, default=30
        Forwarded to fit_gpd_upper_tail and fit_gpd_lower_tail as
        min_evt_samples.
    tail_fraction : float, default=0.05
        Forwarded to fit_gpd_upper_tail and fit_gpd_lower_tail as tail_fraction.
    fit_lower_tail : bool, default=False
        When True, also fit the lower-tail GPD for each leaf and return the
        five additional lower-tail arrays in the result dict.  Controlled by
        ``evt_tails_lower`` on the estimator.

    Returns
    -------
    dict
        Always contains five upper-tail arrays of shape (n_nodes,):
        ``evt_enabled`` (uint8), ``evt_threshold_u`` (float64),
        ``evt_F_u`` (float64), ``evt_gpd_shape`` (float64),
        ``evt_gpd_scale`` (float64).
        Sentinel values for non-leaf and starvation-gated nodes:
        enabled=0, u=0, F_u=1, xi=0, sigma=1.

        When fit_lower_tail=True, also contains five lower-tail arrays:
        ``evt_lower_enabled`` (uint8), ``evt_lower_threshold_u`` (float64),
        ``evt_lower_F_u`` (float64), ``evt_lower_gpd_shape`` (float64),
        ``evt_lower_gpd_scale`` (float64).
        Sentinel values for lower tail: enabled=0, u=0, F_u=0, xi=0, sigma=1.

    """
    # Deferred import: mirrors codebase convention and avoids circular import
    from . import _ddt_core

    n_nodes = int(tree_data["left_child_id"].shape[0])

    # --- Upper-tail sentinel initialisation ---
    evt_enabled = np.zeros(n_nodes, dtype=np.uint8)
    evt_status_upper = np.full(n_nodes, EVT_STATUS_STARVED, dtype=np.uint8)
    evt_threshold_u = np.zeros(n_nodes, dtype=np.float64)
    evt_F_u = np.ones(n_nodes, dtype=np.float64)  # sentinel 1.0 → no upper tail
    evt_S_u = np.zeros(n_nodes, dtype=np.float64)  # sentinel 0.0 → fall back to 1-F_u
    evt_gpd_shape = np.zeros(n_nodes, dtype=np.float64)
    evt_gpd_scale = np.ones(n_nodes, dtype=np.float64)  # sentinel 1.0 (non-zero)

    # --- Lower-tail sentinel initialisation ---
    evt_lower_enabled = np.zeros(n_nodes, dtype=np.uint8)
    evt_status_lower = np.full(n_nodes, EVT_STATUS_STARVED, dtype=np.uint8)
    evt_lower_threshold_u = np.zeros(n_nodes, dtype=np.float64)
    evt_lower_F_u = np.zeros(n_nodes, dtype=np.float64)  # sentinel 0.0 → no lower tail
    evt_lower_gpd_shape = np.zeros(n_nodes, dtype=np.float64)
    evt_lower_gpd_scale = np.ones(n_nodes, dtype=np.float64)  # sentinel 1.0 (non-zero)

    # Route all training samples to terminal leaves
    leaf_indices = _ddt_core.predict_leaves(tree_data, X_q)

    # Fit GPD independently for each terminal leaf
    for leaf_idx in np.unique(leaf_indices):
        mask = leaf_indices == leaf_idx
        leaf_values = y_train[mask]

        # --- Upper tail ---
                upper_params = fit_gpd_upper_tail(
            leaf_values,
            min_evt_samples=min_samples,
            tail_fraction=tail_fraction,
        )
                evt_status_upper[leaf_idx] = upper_params["status"]
        if upper_params["status"] == EVT_STATUS_OK:
            evt_enabled[leaf_idx] = 1
            u = upper_params["u"]
            F_u = upper_params["F_u"]
            S_u = upper_params["S_u"]

            if "smoothed_pmf" in tree_data:
                bin_centers = target_binner.inverse_transform_bin_centers()
                pmf = tree_data["smoothed_pmf"][leaf_idx]
                F_u = np.sum(pmf[bin_centers <= u])
                S_u = 1.0 - F_u

            evt_threshold_u[leaf_idx] = u
            evt_F_u[leaf_idx] = F_u
            evt_S_u[leaf_idx] = S_u
            evt_gpd_shape[leaf_idx] = upper_params["xi"]
            evt_gpd_scale[leaf_idx] = upper_params["sigma"]

        # --- Lower tail (only when explicitly requested) ---
        if fit_lower_tail:
                        lower_params = fit_gpd_lower_tail(
                leaf_values,
                min_evt_samples=min_samples,
                tail_fraction=tail_fraction,
            )
                        evt_status_lower[leaf_idx] = lower_params["status"]
            if lower_params["status"] == EVT_STATUS_OK:
                u_lower = lower_params["u_lower"]
                F_u_lower = lower_params["F_u_lower"]

                if "smoothed_pmf" in tree_data:
                    bin_centers = target_binner.inverse_transform_bin_centers()
                    pmf = tree_data["smoothed_pmf"][leaf_idx]
                    F_u_lower = np.sum(pmf[bin_centers <= u_lower])

                evt_lower_enabled[leaf_idx] = 1
                evt_lower_threshold_u[leaf_idx] = u_lower
                evt_lower_F_u[leaf_idx] = F_u_lower
                evt_lower_gpd_shape[leaf_idx] = lower_params["xi_lower"]
                evt_lower_gpd_scale[leaf_idx] = lower_params["sigma_lower"]

    result = {
        "evt_enabled": evt_enabled,
        "evt_status_upper": evt_status_upper,
        "evt_threshold_u": evt_threshold_u,
        "evt_F_u": evt_F_u,
        "evt_S_u": evt_S_u,
        "evt_gpd_shape": evt_gpd_shape,
        "evt_gpd_scale": evt_gpd_scale,
    }

    if fit_lower_tail:
        result.update(
            {
                "evt_lower_enabled": evt_lower_enabled,
                "evt_status_lower": evt_status_lower,
                "evt_lower_threshold_u": evt_lower_threshold_u,
                "evt_lower_F_u": evt_lower_F_u,
                "evt_lower_gpd_shape": evt_lower_gpd_shape,
                "evt_lower_gpd_scale": evt_lower_gpd_scale,
            }
        )

    return result


def validate_splice_continuity(tree_data: dict, bin_centers: np.ndarray, tol: float = 0.02):
    """
    Check body-tail splice continuity for all EVT-enabled leaves.

    For each EVT-enabled leaf, computes:
        - F_body(u): the smoothed CDF value at the EVT threshold u
        - F_u: the raw empirical CDF value used to define the EVT splice

    If |F_body(u) - F_u| > tol, the splice is discontinuous and the
    predicted quantile function has a jump at the body-tail boundary.

    Parameters
    ----------
    tree_data : dict
        Fitted tree dict (must have smoothed_pmf and evt_enabled).
    bin_centers : ndarray, shape (B,)
        Target bin center positions.
    tol : float, default=0.02
        Maximum acceptable CDF discontinuity at splice point.

    Returns
    -------
    dict
        {"n_checked": int, "n_violated": int, "max_gap": float,
         "violations": list of (node_id, gap)}

    """
    if "smoothed_pmf" not in tree_data or "evt_enabled" not in tree_data:
        return {"n_checked": 0, "n_violated": 0, "max_gap": 0.0, "violations": []}

    is_leaf = tree_data["is_leaf"]
    evt_enabled = tree_data["evt_enabled"]
    evt_threshold_u = tree_data["evt_threshold_u"]
    evt_F_u = tree_data["evt_F_u"]
    smoothed_pmf = tree_data["smoothed_pmf"]
    n_bins = smoothed_pmf.shape[1]

    violations = []
    n_checked = 0

    for nid in range(len(is_leaf)):
        if not is_leaf[nid] or not evt_enabled[nid]:
            continue
        n_checked += 1

        u = evt_threshold_u[nid]
        F_u = evt_F_u[nid]
        pmf = smoothed_pmf[nid]

        # Compute F_smooth(u): CDF of smoothed PMF up to bin containing u
        F_body = 0.0
        for b in range(n_bins):
            if bin_centers[b] <= u:
                F_body += pmf[b]
            else:
                break

        gap = abs(F_body - F_u)
        if gap > tol:
            violations.append((nid, gap))

    max_gap = max((v[1] for v in violations), default=0.0)
    return {
        "n_checked": n_checked,
        "n_violated": len(violations),
        "max_gap": max_gap,
        "violations": violations,
    }
