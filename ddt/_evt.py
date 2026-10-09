"""
DDT EVT Oracle — Extreme Value Theory Tail Modeling.
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

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# EVT Status Codes
EVT_STATUS_OK = 0
EVT_STATUS_STARVED = 1
EVT_STATUS_FIT_FAILED = 2
EVT_STATUS_GOF_REJECTED = 3
EVT_STATUS_SUPPORT_VIOLATION = 4
EVT_STATUS_TAIL_OVERLAP = 5  # lower tail disabled (u_l >= u or F_l >= F_u)


def _select_k(N: int, tail_fraction: float | str, min_k: int = 5) -> int:
    import math

    if tail_fraction == "auto":
        k = round(1.5 * math.sqrt(N))
    else:
        k = math.ceil(float(tail_fraction) * N)
    return min(max(k, min_k), max(1, N // 3))


def fit_gpd_upper_tail(
    leaf_values: np.ndarray,
    min_evt_samples: int = 30,
    tail_fraction: float | str = "auto",
    evt_borrow_factor: float = 1.0,
    evt_gof_alpha: float = 0.05,
):
    try:
        from scipy.stats import genpareto
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
    tail_fraction: float | str = "auto",
    evt_borrow_factor: float = 1.0,
    evt_gof_alpha: float = 0.05,
):
    try:
        from scipy.stats import genpareto
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
        if xi < -1e-6:
            return u_lower + sigma / xi
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
        if xi < -1e-6:
            return u - sigma / xi
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
    tail_fraction: float | str = "auto",
    tail_fraction_lower: float | str = "auto",
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

            # F_u / S_u stay the RAW leaf empirical values.
            # The body/EVT splice is made continuous at inference time by
            # remapping the body CDF (ddt_core.hpp::quantiles_for_leaf), so no
            # smoothed-PMF recomputation (a third, bin-centre CDF convention)
            # is needed or wanted here.
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
                tail_fraction=tail_fraction_lower,
            )
            evt_status_lower[leaf_idx] = lower_params["status"]
            if lower_params["status"] == EVT_STATUS_OK:
                u_lower = lower_params["u_lower"]
                F_u_lower = lower_params["F_u_lower"]

                evt_lower_enabled[leaf_idx] = 1
                evt_lower_threshold_u[leaf_idx] = u_lower
                evt_lower_F_u[leaf_idx] = F_u_lower
                evt_lower_gpd_shape[leaf_idx] = lower_params["xi_lower"]
                evt_lower_gpd_scale[leaf_idx] = lower_params["sigma_lower"]

                # validity gate: the two tails must not overlap, or the
                # body between them has no support.  Disable the lower tail.
                if evt_enabled[leaf_idx] and (u_lower >= evt_threshold_u[leaf_idx] or F_u_lower >= evt_F_u[leaf_idx]):
                    evt_lower_enabled[leaf_idx] = 0
                    evt_lower_threshold_u[leaf_idx] = 0.0
                    evt_lower_F_u[leaf_idx] = 0.0
                    evt_lower_gpd_shape[leaf_idx] = 0.0
                    evt_lower_gpd_scale[leaf_idx] = 1.0
                    evt_status_lower[leaf_idx] = EVT_STATUS_TAIL_OVERLAP

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


def invert_cdf_rows(C, T, thr, bin_lo, bin_width, bin_rep=None, snap=False):
    """
    Vectorised row-wise CDF inversion (NumPy mirror of ``invert_cdf``).

    ``C`` is the (N, B) cumulative mass, ``T = C[:, -1]`` and ``thr`` the
    per-row target mass (``q' * T``).  Uniform-within-bin unless ``snap``.
    """
    N, B = C.shape
    mask_leq_0 = thr <= 0
    mask_gt_0 = ~mask_leq_0

    b = np.zeros(N, dtype=np.int64)
    if np.any(mask_leq_0):
        b[mask_leq_0] = np.argmax(C[mask_leq_0] > 0, axis=1)
    if np.any(mask_gt_0):
        thr_gt = np.minimum(thr[mask_gt_0], T[mask_gt_0])
        b[mask_gt_0] = np.argmax(C[mask_gt_0] >= thr_gt[:, None], axis=1)
    b = np.clip(b, 0, B - 1)

    if snap:
        return bin_rep[b].copy()

    idx = np.arange(N)
    prev = np.where(b > 0, C[idx, np.maximum(b - 1, 0)], 0.0)
    denom = C[idx, b] - prev
    frac = np.zeros(N, dtype=np.float64)
    mask_valid = (denom > 0) & mask_gt_0
    frac[mask_valid] = (thr[mask_valid] - prev[mask_valid]) / denom[mask_valid]
    frac = np.clip(frac, 0.0, 1.0)
    return bin_lo[b] + frac * bin_width[b]


def spliced_quantiles_from_dist(dist, leaf_ids, td, quantiles, bin_lo, bin_width, bin_rep, snap, has_upper, has_lower):
    """
    Exact body/EVT spliced quantiles for per-sample leaf distributions.

    NumPy mirror of ``quantiles_for_leaf`` in ``ddt_core.hpp``,
    section 3.1).  When a tail is active the body CDF ``G`` is affinely
    re-indexed so the body quantile function ends exactly at the GPD
    thresholds ``u_l`` / ``u``::

        q' = g_l + (q - F_l) * (g_u - g_l) / (F_u - F_l),
        g_l = G(u_l),  g_u = G(u)

    ``F_l`` / ``F_u`` stay the raw leaf empirical tail probabilities.  The
    result is non-decreasing in ``q`` and continuous at both junctions.  A
    degenerate body (``g_u <= g_l`` or ``F_u <= F_l``) falls back to a linear
    bridge between the two thresholds.

    Parameters
    ----------
    dist : ndarray (N, B)
        Per-sample leaf mass (counts or smoothed PMF).
    leaf_ids : ndarray (N,)
        Leaf node id of each sample (indexes the EVT arrays of ``td``).
    td : dict
        Tree dict holding the EVT arrays.
    quantiles : sequence of float
    bin_lo, bin_width, bin_rep : ndarray (B,)
        Quantile grid.  With ``snap=True`` the body snaps to ``bin_rep``.
    has_upper, has_lower : bool
        Whether the upper / lower tail is in use.

    Returns
    -------
    dict
        ``{q: ndarray (N,)}``.

    """
    leaf_ids = np.asarray(leaf_ids)
    N, B = dist.shape
    C = np.cumsum(dist, axis=1, dtype=np.float64)
    T = C[:, -1]
    valid = T > 0.0

    if snap:
        lo, w = bin_rep, np.zeros_like(bin_width)
    else:
        lo, w = bin_lo, bin_width

    idx = np.arange(N)

    def eval_cdf_rows(x):
        b = np.clip(np.searchsorted(lo, x, side="right") - 1, 0, B - 1)
        prev = np.where(b > 0, C[idx, np.maximum(b - 1, 0)], 0.0)
        wb, lob = w[b], lo[b]
        frac = np.where(
            wb > 0.0,
            np.clip((x - lob) / np.where(wb > 0.0, wb, 1.0), 0.0, 1.0),
            (x >= lob).astype(np.float64),
        )
        g = (prev + frac * (C[idx, b] - prev)) / np.where(valid, T, 1.0)
        return np.clip(g, 0.0, 1.0)

    up = np.zeros(N, dtype=bool)
    lo_act = np.zeros(N, dtype=bool)
    F_u = np.ones(N, dtype=np.float64)
    F_l = np.zeros(N, dtype=np.float64)
    u_u = np.zeros(N, dtype=np.float64)
    u_l = np.zeros(N, dtype=np.float64)
    if has_upper:
        up = np.asarray(td["evt_enabled"]).astype(bool)[leaf_ids] & valid
        F_u = np.where(up, np.asarray(td["evt_F_u"], dtype=np.float64)[leaf_ids], 1.0)
        u_u = np.where(up, np.asarray(td["evt_threshold_u"], dtype=np.float64)[leaf_ids], 0.0)
    if has_lower:
        lo_act = np.asarray(td["evt_lower_enabled"]).astype(bool)[leaf_ids] & valid
        F_l = np.where(lo_act, np.asarray(td["evt_lower_F_u"], dtype=np.float64)[leaf_ids], 0.0)
        u_l = np.where(lo_act, np.asarray(td["evt_lower_threshold_u"], dtype=np.float64)[leaf_ids], 0.0)

    g_u = np.where(up, eval_cdf_rows(u_u), 1.0)
    g_l = np.where(lo_act, eval_cdf_rows(u_l), 0.0)

    active = up | lo_act
    bridge = active & (~(g_u > g_l) | ~(F_u > F_l))
    remap = active & ~bridge
    den = np.where(remap, F_u - F_l, 1.0)

    x_lo_end = np.zeros(N, dtype=np.float64)
    x_hi_end = np.zeros(N, dtype=np.float64)
    if bridge.any():
        x_lo_end = np.where(lo_act, u_l, invert_cdf_rows(C, T, np.zeros(N), bin_lo, bin_width, bin_rep, snap))
        x_hi_end = np.where(up, u_u, invert_cdf_rows(C, T, T.copy(), bin_lo, bin_width, bin_rep, snap))
        x_hi_end = np.maximum(x_hi_end, x_lo_end)

    # GPD parameters (only gathered where the tail is active)
    if has_upper:
        S_arr = np.asarray(td["evt_S_u"], dtype=np.float64) if "evt_S_u" in td else None
        xi_u_arr = np.asarray(td["evt_gpd_shape"], dtype=np.float64)
        sg_u_arr = np.asarray(td["evt_gpd_scale"], dtype=np.float64)
    if has_lower:
        xi_l_arr = np.asarray(td["evt_lower_gpd_shape"], dtype=np.float64)
        sg_l_arr = np.asarray(td["evt_lower_gpd_scale"], dtype=np.float64)

    results = {}
    for q in quantiles:
        q = float(q)
        q_eff = np.where(remap, g_l + (q - F_l) * (g_u - g_l) / den, q)
        raw = invert_cdf_rows(C, T, q_eff * T, bin_lo, bin_width, bin_rep, snap)

        if bridge.any():
            t = np.where(den > 0.0, (q - F_l) / np.where(F_u - F_l > 0.0, F_u - F_l, 1.0), 0.0)
            t = np.where(F_u - F_l > 0.0, t, 0.0)
            raw = np.where(bridge, x_lo_end + t * (x_hi_end - x_lo_end), raw)

        mask_u = up & (q >= F_u)
        if mask_u.any():
            lid = leaf_ids[mask_u]
            if q >= 1.0:
                xi, sg = xi_u_arr[lid], sg_u_arr[lid]
                finite = xi < -1e-6
                raw[mask_u] = np.where(finite, u_u[mask_u] - sg / np.where(finite, xi, 1.0), np.inf)
            else:
                Fu_m = F_u[mask_u]
                if S_arr is not None:
                    S_m = S_arr[lid]
                    safe = np.where(1.0 - Fu_m > 0.0, 1.0 - Fu_m, 1.0)
                    expo = np.where(S_m > 0.0, (1.0 - q) / np.where(S_m > 0.0, S_m, 1.0), (1.0 - q) / safe)
                else:
                    safe = np.where(1.0 - Fu_m > 0.0, 1.0 - Fu_m, 1.0)
                    expo = (1.0 - q) / safe
                xi, sg = xi_u_arr[lid], sg_u_arr[lid]
                small = np.abs(xi) < 1e-6
                xi_safe = np.where(small, 1.0, xi)
                raw[mask_u] = np.where(
                    small,
                    u_u[mask_u] - sg * np.log(expo),
                    u_u[mask_u] + (sg / xi_safe) * (np.power(expo, -xi_safe) - 1.0),
                )

        # C++ routing checks the upper tail first.
        mask_l = lo_act & (q <= F_l) & ~mask_u
        if mask_l.any():
            lid = leaf_ids[mask_l]
            if q <= 0.0:
                xi, sg = xi_l_arr[lid], sg_l_arr[lid]
                finite = xi < -1e-6
                raw[mask_l] = np.where(finite, u_l[mask_l] + sg / np.where(finite, xi, 1.0), -np.inf)
            else:
                expo = q / F_l[mask_l]
                xi, sg = xi_l_arr[lid], sg_l_arr[lid]
                small = np.abs(xi) < 1e-6
                xi_safe = np.where(small, 1.0, xi)
                raw[mask_l] = np.where(
                    small,
                    u_l[mask_l] + sg * np.log(expo),
                    u_l[mask_l] - (sg / xi_safe) * (np.power(expo, -xi_safe) - 1.0),
                )

        results[q] = raw
    return results


def _gpd_tail_ratio(z, xi, sg):
    """GPD exceedance ratio ``(1 + xi z / sigma)^(-1/xi)`` in [0, 1] (vectorised)."""
    z = np.maximum(z, 0.0)
    sg = np.where(sg > 0.0, sg, 1.0)
    small = np.abs(xi) < 1e-6
    xi_s = np.where(small, 1.0, xi)
    base = 1.0 + xi_s * z / sg
    pos = base > 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gen = np.where(pos, np.power(np.where(pos, base, 1.0), -1.0 / xi_s), 0.0)
        expo = np.exp(-z / sg)
    return np.where(small, expo, gen)


def spliced_cdf_rows(dist, leaf_ids, td, bin_lo, bin_width, bin_rep, snap, has_upper, has_lower, pop_mean=0.0):
    """
    Per-sample spliced CDF ``H(x)`` of one tree.

    The exact inverse of :func:`spliced_quantiles_from_dist` and a NumPy
    mirror of ``SplicedLeaf::cdf`` in ``ddt_core.cpp``:

    * ``x >= u``:      ``1 - S_u (1 + xi (x - u) / sigma)^(-1/xi)``
    * ``x <= u_l``:    ``F_l (1 + xi (u_l - x) / sigma)^(-1/xi)``
    * body in between: ``F_l + (G(x) - g_l) (F_u - F_l) / (g_u - g_l)``
      (linear bridge when degenerate).

    Empty leaves are a step at ``pop_mean``.  Returns ``H`` mapping an
    ``(N,)`` array of abscissae to the ``(N,)`` CDF values.
    """
    leaf_ids = np.asarray(leaf_ids)
    N, B = dist.shape
    C = np.cumsum(dist, axis=1, dtype=np.float64)
    T = C[:, -1]
    valid = T > 0.0
    if snap:
        lo, w = bin_rep, np.zeros_like(bin_width)
    else:
        lo, w = bin_lo, bin_width
    idx = np.arange(N)

    def eval_cdf_rows(x):
        b = np.clip(np.searchsorted(lo, x, side="right") - 1, 0, B - 1)
        prev = np.where(b > 0, C[idx, np.maximum(b - 1, 0)], 0.0)
        wb, lob = w[b], lo[b]
        frac = np.where(
            wb > 0.0,
            np.clip((x - lob) / np.where(wb > 0.0, wb, 1.0), 0.0, 1.0),
            (x >= lob).astype(np.float64),
        )
        g = (prev + frac * (C[idx, b] - prev)) / np.where(valid, T, 1.0)
        return np.clip(g, 0.0, 1.0)

    def gather(name, default):
        return np.asarray(td[name], dtype=np.float64)[leaf_ids] if name in td else np.full(N, default)

    up = np.zeros(N, dtype=bool)
    lo_act = np.zeros(N, dtype=bool)
    F_u = np.ones(N)
    F_l = np.zeros(N)
    u_u = np.zeros(N)
    u_l = np.zeros(N)
    if has_upper:
        up = np.asarray(td["evt_enabled"]).astype(bool)[leaf_ids] & valid
        F_u = np.where(up, gather("evt_F_u", 1.0), 1.0)
        u_u = np.where(up, gather("evt_threshold_u", 0.0), 0.0)
    if has_lower:
        lo_act = np.asarray(td["evt_lower_enabled"]).astype(bool)[leaf_ids] & valid
        F_l = np.where(lo_act, gather("evt_lower_F_u", 0.0), 0.0)
        u_l = np.where(lo_act, gather("evt_lower_threshold_u", 0.0), 0.0)

    g_u = np.where(up, eval_cdf_rows(u_u), 1.0)
    g_l = np.where(lo_act, eval_cdf_rows(u_l), 0.0)
    active = up | lo_act
    bridge = active & (~(g_u > g_l) | ~(F_u > F_l))
    remap = active & ~bridge
    den = np.where(remap, g_u - g_l, 1.0)

    x_lo_end = np.zeros(N)
    x_hi_end = np.zeros(N)
    if bridge.any():
        x_lo_end = np.where(lo_act, u_l, invert_cdf_rows(C, T, np.zeros(N), bin_lo, bin_width, bin_rep, snap))
        x_hi_end = np.where(up, u_u, invert_cdf_rows(C, T, T.copy(), bin_lo, bin_width, bin_rep, snap))
        x_hi_end = np.maximum(x_hi_end, x_lo_end)

    S_u = np.zeros(N)
    xi_u = sg_u = xi_l = sg_l = np.zeros(N)
    if has_upper:
        S_raw = gather("evt_S_u", 0.0)
        S_u = np.where(S_raw > 0.0, S_raw, 1.0 - F_u)
        xi_u, sg_u = gather("evt_gpd_shape", 0.0), gather("evt_gpd_scale", 1.0)
    if has_lower:
        xi_l, sg_l = gather("evt_lower_gpd_shape", 0.0), gather("evt_lower_gpd_scale", 1.0)

    def H(x):
        x = np.asarray(x, dtype=np.float64)
        g = eval_cdf_rows(x)
        h = np.where(remap, np.clip(F_l + (g - g_l) * (F_u - F_l) / den, F_l, F_u), g)
        if bridge.any():
            wd = x_hi_end - x_lo_end
            t = np.where(wd > 0.0, (x - x_lo_end) / np.where(wd > 0.0, wd, 1.0), (x >= x_lo_end).astype(np.float64))
            h = np.where(bridge, F_l + np.clip(t, 0.0, 1.0) * (F_u - F_l), h)
        m_up = up & (x >= u_u)
        if has_lower:
            m_lo = lo_act & (x <= u_l) & ~m_up
            h = np.where(m_lo, F_l * _gpd_tail_ratio(u_l - x, xi_l, sg_l), h)
        if has_upper:
            h = np.where(m_up, 1.0 - S_u * _gpd_tail_ratio(x - u_u, xi_u, sg_u), h)
        return np.where(valid, h, (x >= pop_mean).astype(np.float64))

    return H


def bisect_mixture_quantile(cdfs, q, lo, hi, max_iter=60, tol=1e-10):
    """
    Smallest ``x`` in ``[lo, hi]`` with ``mean_m H_m(x) >= q`` (vectorised bisection).

    ``cdfs`` is a list of callables from :func:`spliced_cdf_rows`; ``lo`` / ``hi``
    are the per-sample bracket ``[min_m x_m(q), max_m x_m(q)]``.  Mirrors
    ``invert_spliced_mixture`` in ``ddt_core.cpp``: each sample stops at
    ``hi - lo <= tol (1 + |hi|)`` or after ``max_iter`` iterations.
    """
    lo = np.array(lo, dtype=np.float64)
    hi = np.array(hi, dtype=np.float64)
    inv_m = 1.0 / len(cdfs)
    finite = np.isfinite(lo) & np.isfinite(hi)
    for _ in range(max_iter):
        running = finite & (hi - lo > tol * (1.0 + np.abs(hi)))
        if not running.any():
            break
        mid = 0.5 * (lo + hi)
        h = np.zeros_like(mid)
        for H in cdfs:
            h += H(mid)
        h *= inv_m
        ge = h >= q
        hi = np.where(running & ge, mid, hi)
        lo = np.where(running & ~ge, mid, lo)
    return hi


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
