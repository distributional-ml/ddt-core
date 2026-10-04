"""
DDT Smoothing — Hierarchical Empirical Bayes (Dirichlet) Leaf Regularisation
=============================================================================
Post-build shrinkage of sparse leaf PMFs toward their parent's (or ancestor's)
more stable distribution. Implements the Dirichlet-Multinomial conjugate model
with concentration parameters derived from the hierarchical tree structure.

Mathematical basis
------------------
Each leaf's histogram ``counts[0..B-1]`` is modelled as a draw from a
Multinomial distribution. The conjugate prior for the Multinomial is the
Dirichlet distribution with concentration parameters ``alpha[b] = lambda *
p_parent[b]``, where ``lambda`` is the user-controlled prior weight.

The Bayes-optimal point estimate under squared loss (Dirichlet posterior mean)
for the smoothed PMF is:

    theta_b_smooth = (counts_b + lambda * p_parent_b) / (N + lambda)

Rewritten as a convex combination:

    theta_smooth = w * p_leaf + (1 - w) * p_parent,  w = N / (N + lambda)

Key properties:
  - w → 1 for well-populated leaves (N >> lambda): smoothing negligible.
  - w → 0 for sparse leaves (N << lambda): heavy shrinkage toward parent.
  - w = 0 exactly for empty leaves: falls back to the parent PMF.
  - Self-regulating: no explicit sparsity gate required; the formula
    provides a smooth, continuous transition.

Design
------
  - Applied AFTER tree building, BEFORE EVT fitting.
  - Stores smoothed PMFs as float64 in tree_data_["smoothed_pmf"].
  - Never modifies distribution_counts (raw empirical histogram preserved).
  - Zero C++ changes required.
  - Internal nodes store their own empirical PMF in smoothed_pmf (used as
    prior source candidates), but are never accessed during inference.
"""

import numpy as np


def smooth_leaf_pmf(
    leaf_counts: np.ndarray,
    parent_pmf: np.ndarray,
    prior_weight: float,
) -> np.ndarray:
    """
    Apply Dirichlet posterior mean shrinkage to a single leaf.

    Computes the Bayes-optimal PMF under a Dirichlet-Multinomial conjugate
    model with concentration parameters ``alpha_b = prior_weight * parent_pmf_b``.

    Parameters
    ----------
    leaf_counts : np.ndarray, shape (B,), dtype int32 or float
        Raw histogram counts for the leaf.  May be all-zero (empty leaf).
    parent_pmf : np.ndarray, shape (B,), dtype float64
        Empirical PMF of the prior source (parent or stable ancestor).
        Must sum to 1.0 and be non-negative.
    prior_weight : float
        Dirichlet concentration parameter (lambda >= 0).
        ``lambda = 0`` → pure data (no shrinkage).
        ``lambda → inf`` → pure prior (full shrinkage to parent_pmf).

    Returns
    -------
    np.ndarray, float64, shape (B,)
        Smoothed PMF summing to 1.0 with all entries >= 0.

    """
    N = float(leaf_counts.sum())

    # Degenerate: empty leaf → return parent PMF exactly
    if N == 0:
        return parent_pmf.copy()

    # Degenerate: zero prior weight → pure empirical leaf PMF
    if prior_weight == 0.0:
        pmf = leaf_counts.astype(np.float64) / N
        return pmf

    leaf_pmf = leaf_counts.astype(np.float64) / N

    # Convex combination weight: data weight w = N / (N + lambda)
    w = N / (N + prior_weight)
    smoothed = w * leaf_pmf + (1.0 - w) * parent_pmf

    # Numerical safety: clip and re-normalise for floating-point precision
    smoothed = np.maximum(smoothed, 0.0)
    total = smoothed.sum()
    if total > 0.0:
        smoothed /= total
    else:
        # Fallback: uniform (should never happen if parent_pmf is valid)
        B = len(smoothed)
        smoothed = np.ones(B, dtype=np.float64) / B

    return smoothed


def _build_parent_map(tree_data: dict) -> np.ndarray:
    """
    Build a parent-index map from the tree's child arrays.

    Parameters
    ----------
    tree_data : dict
        Serialised tree dict containing ``left_child_id``, ``right_child_id``,
        and ``is_leaf``.

    Returns
    -------
    np.ndarray, int32, shape (n_nodes,)
        ``parent_of[node_id]`` is the node_id of the parent.
        The root's entry is -1 (no parent).

    """
    n_nodes = len(tree_data["is_leaf"])
    parent_of = np.full(n_nodes, -1, dtype=np.int32)
    left_child = tree_data["left_child_id"]
    right_child = tree_data["right_child_id"]
    is_leaf = tree_data["is_leaf"]

    for nid in range(n_nodes):
        if not is_leaf[nid]:
            lc = int(left_child[nid])
            rc = int(right_child[nid])
            if lc >= 0:
                parent_of[lc] = nid
            if rc >= 0:
                parent_of[rc] = nid

    return parent_of


def _find_stable_ancestor_pmf(
    node_id: int,
    parent_of: np.ndarray,
    distribution_counts: np.ndarray,
    total_samples: np.ndarray,
    min_ancestor_samples: int,
) -> np.ndarray:
    """
    Walk up the tree from ``node_id`` to find a stable ancestor PMF.

    Returns the empirical PMF of the first ancestor that has at least
    ``min_ancestor_samples`` total samples.  If no such ancestor exists
    (including the root), the root's PMF is returned regardless.  If the
    root itself has zero samples (pathological edge case), a uniform PMF
    is returned.

    Parameters
    ----------
    node_id : int
        The node whose stable ancestor is needed.
    parent_of : np.ndarray, int32, shape (n_nodes,)
        Output of ``_build_parent_map``.
    distribution_counts : np.ndarray, int32, shape (n_nodes, B)
        Raw histogram counts for every node.
    total_samples : np.ndarray, shape (n_nodes,)
        Total sample count per node.
    min_ancestor_samples : int
        Minimum samples threshold an ancestor must meet to be used as prior.

    Returns
    -------
    np.ndarray, float64, shape (B,)
        Empirical PMF of the chosen ancestor, summing to 1.0.

    """
    current = int(parent_of[node_id])

    while current >= 0:
        n_samples = int(total_samples[current])
        if n_samples >= min_ancestor_samples:
            pmf = distribution_counts[current].astype(np.float64)
            pmf_sum = pmf.sum()
            if pmf_sum > 0:
                return pmf / pmf_sum
        current = int(parent_of[current])

    # Fell through: use root (node 0) as last resort
    root_counts = distribution_counts[0].astype(np.float64)
    root_total = root_counts.sum()
    if root_total > 0:
        return root_counts / root_total

    # Degenerate: root is empty → uniform prior
    B = distribution_counts.shape[1]
    return np.ones(B, dtype=np.float64) / B


def smooth_all_leaves(
    tree_data: dict,
    prior_weight: float,
    min_ancestor_samples: int,
) -> np.ndarray:
    """
    Smooth all terminal leaf PMFs via hierarchical Dirichlet shrinkage.

    For each terminal leaf, walks up the tree to find the first ancestor
    with ``>= min_ancestor_samples`` to use as the Dirichlet prior source,
    then computes the posterior-mean smoothed PMF.  Internal nodes store
    their own empirical PMF (not smoothed; used only as ancestor priors,
    never accessed during inference).

    This is the top-level entry point called by ``DDTRegressor.fit()`` when
    ``smooth_leaves=True``.

    Parameters
    ----------
    tree_data : dict
        Pybind11-serialised tree dict.  Must contain:
        ``is_leaf``, ``distribution_counts``, ``total_samples``,
        ``left_child_id``, ``right_child_id``, ``n_bins``.
    prior_weight : float
        Dirichlet concentration parameter (lambda).  Higher = stronger
        shrinkage toward ancestor.  Recommended default: ``min_samples_leaf``.
    min_ancestor_samples : int
        Minimum total_samples an ancestor must have to serve as the prior
        source.  Recommended default: ``4 * min_samples_leaf``.

    Returns
    -------
    np.ndarray, float64, shape (n_nodes, n_bins)
        Smoothed PMF array for every node.
        - Terminal leaves: Dirichlet posterior mean PMF (sums to 1.0).
        - Internal nodes: empirical PMF (sums to 1.0).
        - All entries are non-negative.

    """
    n_nodes = len(tree_data["is_leaf"])
    n_bins = int(tree_data["n_bins"])
    is_leaf = tree_data["is_leaf"]
    dist_counts = tree_data["distribution_counts"]  # (n_nodes, n_bins) int32
    total_samples = tree_data["total_samples"]

    parent_of = _build_parent_map(tree_data)

    smoothed_pmf = np.zeros((n_nodes, n_bins), dtype=np.float64)

    for nid in range(n_nodes):
        counts = dist_counts[nid]
        N = int(total_samples[nid])

        if is_leaf[nid]:
            # Terminal leaf: apply Dirichlet shrinkage toward stable ancestor
            ancestor_pmf = _find_stable_ancestor_pmf(
                nid,
                parent_of,
                dist_counts,
                total_samples,
                min_ancestor_samples,
            )
            smoothed_pmf[nid] = smooth_leaf_pmf(counts, ancestor_pmf, prior_weight)
        else:
            # Internal node: store its own empirical PMF as prior source
            # for descendant leaves.  Never accessed during inference.
            if N > 0:
                smoothed_pmf[nid] = counts.astype(np.float64) / float(N)
            else:
                # Degenerate empty internal node → uniform
                smoothed_pmf[nid] = np.ones(n_bins, dtype=np.float64) / n_bins

    return smoothed_pmf
