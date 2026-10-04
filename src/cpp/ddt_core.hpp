#pragma once
// =============================================================================
// Distributional Decision Trees (DDT) — Core Engine Header
// =============================================================================
// High-performance, non-parametric decision tree engine that preserves full
// Empirical Cumulative Distribution Functions (ECDF) in terminal leaves.
//
// Architecture:
//   - Data-Oriented Design (DOD): Struct of Arrays (SoA) for tree storage
//   - O(N + K*B) split evaluation via double-histogram quantization
//   - Zero std::vector allocations in split search
//   - Flat inline SIMD-friendly Wasserstein metric
//
// Copyright (c) 2026. All rights reserved.
// =============================================================================

#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace ddt {

// -----------------------------------------------------------------------------
// 1D Wasserstein Distance (Earth Mover's Distance)
// -----------------------------------------------------------------------------
// Closed-form solution for 1D discrete distributions on aligned bins:
//   W_1(P, Q) = sum_{b=0}^{B-1} |CDF_P(b) - CDF_Q(b)|
//
// Properties:
//   - O(B) computation on contiguous flat arrays
//   - Explicit flat for-loop designed for compiler auto-vectorization (SIMD)
inline double wasserstein_1d(const int* left_counts, int left_total, 
                             const int* right_counts, int right_total, int B) 
{
    if (left_total == 0 || right_total == 0) return 0.0;

    double w1 = 0.0;
    double cdf_left = 0.0;
    double cdf_right = 0.0;
    const double inv_left = 1.0 / static_cast<double>(left_total);
    const double inv_right = 1.0 / static_cast<double>(right_total);

    for (int b = 0; b < B; ++b) {
        cdf_left += left_counts[b] * inv_left;
        cdf_right += right_counts[b] * inv_right;
        w1 += std::abs(cdf_left - cdf_right);
    }
    return w1;
}

// -----------------------------------------------------------------------------
// 1D Weighted Wasserstein Distance (variable bin widths)
// -----------------------------------------------------------------------------
// Extends wasserstein_1d to non-uniform bins via a normalised width vector:
//   W_1w(P, Q) = sum_{b=0}^{B-1} delta_x_norm[b] * |CDF_P(b) - CDF_Q(b)|
//
// delta_x_norm[b] = bin_width[b] / mean(bin_widths)  — normalised so that
// equal-width bins produce delta_x_norm[b] = 1.0 for all b, and the gain
// magnitude is compatible with existing thresholds (min_divergence_decrease).
//
// Properties:
//   - O(B) computation; compatible with compiler auto-vectorization
//   - For equal-width bins with no consolidation, produces same value as wasserstein_1d
//   - Wider bins (tails, consolidated) carry proportionally more weight
inline double wasserstein_1d_weighted(
    const int* left_counts, int left_total,
    const int* right_counts, int right_total,
    const double* delta_x_norm, int B)
{
    if (left_total == 0 || right_total == 0) return 0.0;

    double w1 = 0.0;
    double cdf_left = 0.0;
    double cdf_right = 0.0;
    const double inv_left = 1.0 / static_cast<double>(left_total);
    const double inv_right = 1.0 / static_cast<double>(right_total);

    for (int b = 0; b < B; ++b) {
        cdf_left += left_counts[b] * inv_left;
        cdf_right += right_counts[b] * inv_right;
        w1 += delta_x_norm[b] * std::abs(cdf_left - cdf_right);
    }
    return w1;
}

// =============================================================================
// Tree (Data-Oriented Design: Struct of Arrays)
// =============================================================================
// Stores the complete tree state in parallel contiguous arrays.
// Zero-copy translatable to Python NumPy arrays via pybind11.

struct Tree {
    int n_bins;
    std::vector<uint8_t> is_leaf;
    std::vector<int32_t> split_feature_idx;
    std::vector<uint8_t> split_threshold;
    std::vector<int32_t> left_child_id;
    std::vector<int32_t> right_child_id;
    std::vector<double> wasserstein_gain;
    std::vector<int32_t> depth;
    std::vector<int32_t> total_samples;
    std::vector<int32_t> distribution_counts; // Flat array of size num_nodes * n_bins

    Tree(int b) : n_bins(b) {}
    Tree() : n_bins(0) {}

    // Pre-allocate estimated capacity
    void reserve(size_t cap) {
        is_leaf.reserve(cap);
        split_feature_idx.reserve(cap);
        split_threshold.reserve(cap);
        left_child_id.reserve(cap);
        right_child_id.reserve(cap);
        wasserstein_gain.reserve(cap);
        depth.reserve(cap);
        total_samples.reserve(cap);
        distribution_counts.reserve(cap * n_bins);
    }

    // Add a new node and return its index
    int add_node(int current_depth) {
        int id = static_cast<int>(is_leaf.size());
        is_leaf.push_back(1); // default to leaf
        split_feature_idx.push_back(-1);
        split_threshold.push_back(0);
        left_child_id.push_back(-1);
        right_child_id.push_back(-1);
        wasserstein_gain.push_back(0.0);
        depth.push_back(current_depth);
        total_samples.push_back(0);
        distribution_counts.resize(distribution_counts.size() + n_bins, 0);
        return id;
    }
};

// =============================================================================
// Tree View (Zero-Copy Inference Abstraction)
// =============================================================================
// Holds raw pointers to the tree's internal SoA arrays. This prevents any
// Python or JSON serialization objects from leaking into the core prediction
// engine, enabling pure bare-metal C++ inference.

struct TreeView {
    int n_bins;
    const uint8_t* is_leaf;
    const int32_t* split_feature_idx;
    const uint8_t* split_threshold;
    const int32_t* left_child_id;
    const int32_t* right_child_id;
    const int32_t* distribution_counts;
};

// =============================================================================
// Smooth Tree View (Phase 6: Smoothed PMF Inference)
// =============================================================================
// Like TreeView but reads from the float64 smoothed PMF array produced by
// smooth_tree(). Eliminates the int32→float cast in the quantile inner loop
// and reads the Dirichlet-blended PMF directly. Used by
// predict_quantiles_fast_smooth().

struct TreeViewSmooth {
    int n_bins;
    const uint8_t* is_leaf;
    const int32_t* split_feature_idx;
    const uint8_t* split_threshold;
    const int32_t* left_child_id;
    const int32_t* right_child_id;
    const double*  smoothed_pmf;  // float64, shape (n_nodes, n_bins), row-major
};

// =============================================================================
// Float-Split Tree View (Phase 5: Float-Split Inference)
// =============================================================================
// Like TreeView but uses pre-computed float64 split thresholds per node.
// This eliminates the bin_edges offset lookup in the traversal hot-loop,
// replacing it with a single direct comparison: val < split_threshold_float[node_id].

struct TreeViewFloat {
    int n_bins;
    const uint8_t* is_leaf;
    const int32_t* split_feature_idx;
    const double*  split_threshold_float;  // pre-computed float64 per node
    const int32_t* left_child_id;
    const int32_t* right_child_id;
    const int32_t* distribution_counts;
};

// =============================================================================
// Split Workspace
// =============================================================================
// Pre-allocated buffers to prevent std::vector allocations during search loop.
//
// // Instead of zeroing the full F*256*B feature_hists array before every split,
// we track only the (feature, bin_val) pairs that actually received samples
// and zero just those.  For F=200, B=64, N=50, this reduces write bandwidth
// from 12.5 MB to ~640 KB per node (≤20× reduction).

struct SplitWorkspace {
    std::vector<int> root_hist;         // Size B
    std::vector<int> feature_hists;     // Size F * K * B
    std::vector<int> feature_totals;    // Size F * K
    std::vector<int> left_hist;         // Size B
    std::vector<int> right_hist;        // Size B

    // Variable-bin fields (null for equal-width fast path)
    const double* delta_x_norm = nullptr; // Size B — normalised bin widths
    bool has_variable_bins = false;

    //     // dirty_pairs[0..dirty_count): (feature_idx, bin_val) pairs touched this node
    std::vector<std::pair<int,int>> dirty_pairs;
    int dirty_count = 0;
    int n_bins_ = 0;   // stored so reset_dirty() can compute the correct offset
    int n_features_ = 0; // stored for bounds safety (unused at runtime)

    SplitWorkspace(int F, int K, int B)
        : n_bins_(B), n_features_(F)
    {
        root_hist.resize(B, 0);
        feature_hists.resize(F * K * B, 0);
        feature_totals.resize(F * K, 0);
        left_hist.resize(B, 0);
        right_hist.resize(B, 0);
        // Worst-case dirty set: every (feature, bin_val) pair in the node sample.
        // In practice O(N * F) but at most F * K entries.
        dirty_pairs.resize(static_cast<size_t>(F) * K);
        dirty_count = 0;
    }

    // Mark a (feature, bin_val) pair as dirty on first use.
    // The check `feature_totals[f * K + k] == 0` is the sentinel: a pair is
    // new if its total is still zero when we first try to write to it.
    inline void mark_dirty(int f, int k) {
        constexpr int K = 256;
        if (feature_totals[f * K + k] == 0) {
            dirty_pairs[dirty_count++] = {f, k};
        }
    }

    // Zero only the previously-used (feature, bin_val) entries and reset count.
    // Called at the start of each split instead of full std::fill.
    inline void reset_dirty() {
        constexpr int K = 256;
        for (int d = 0; d < dirty_count; ++d) {
            auto [f, k] = dirty_pairs[d];
            int offset = (f * K + k) * n_bins_;
            std::fill(feature_hists.data() + offset,
                      feature_hists.data() + offset + n_bins_, 0);
            feature_totals[f * K + k] = 0;
        }
        dirty_count = 0;
    }
};

// =============================================================================
// Split Result
// =============================================================================

struct SplitResult {
    bool found;
    int feature_idx;
    uint8_t threshold;
    double gain;
    // We do NOT copy vectors here. We reconstruct the partition in build_node.

    SplitResult() : found(false), feature_idx(-1), threshold(0), gain(-1.0) {}
};

// =============================================================================
// Core Algorithm Functions
// =============================================================================

// Find best split, using strictly the pre-allocated workspace. No heap allocations.
// feature_mask: nullable char array of length n_features. When non-null, features
// where mask[f]==0 are skipped entirely (Phase 5: per-feature max splits).
SplitResult find_best_split(
    const uint8_t* X_data,
    const int32_t* y_data,
    const int* sample_indices,
    int N,
    int n_features,
    int n_bins,
    int min_samples_leaf,
    SplitWorkspace& ws,
    const char* feature_mask = nullptr
);

// Build a complete decision tree from quantized data using Struct of Arrays.
// max_splits_per_feature: optional int array of length n_features. When non-null,
// feature f may appear as a split variable at most max_splits_per_feature[f] times
// across the entire tree (global limit, Phase 5). Pass nullptr to disable.
Tree build_tree(
    const uint8_t* X_data,
    const int32_t* y_data,
    int n_samples,
    int n_features,
    int n_bins,
    int max_depth,
    int min_samples_leaf,
    double min_divergence_decrease,
    const std::string& divergence_name,
    const int* max_splits_per_feature = nullptr
);

// Build a tree with variable-width bin Wasserstein (weighted path).
// delta_x_norm must have length n_bins: normalised bin widths (mean = 1.0).
// max_splits_per_feature: same semantics as build_tree (Phase 5).
Tree build_tree_weighted(
    const uint8_t* X_data,
    const int32_t* y_data,
    int n_samples,
    int n_features,
    int n_bins,
    int max_depth,
    int min_samples_leaf,
    double min_divergence_decrease,
    const std::string& divergence_name,
    const double* delta_x_norm,
    const int* max_splits_per_feature = nullptr
);

// Predict the leaf node index for a single sample.
int predict_leaf(
    const Tree& tree,
    const uint8_t* sample_features,
    int n_features
);


// =============================================================================
// 
// =============================================================================
// QuantileGrid describes the target-space grid every inference path inverts a
// histogram CDF over.  All three arrays are required, have length B (the number
// of active target bins) and live in original target space.
//
//   bin_lo[b]     lower edge of bin b (the atom value for atom bins)
//   bin_width[b]  hi - lo (0 for atom bins)
//   bin_rep[b]    point representative (centroid / atom value); used for the mean
//                 fallback of empty leaves, never for quantile inversion
//
// `quantile_interpolation="snap"` is expressed purely as bin_lo = bin_rep and
// bin_width = 0 -- the kernel has no mode flag.  The pre-// mode was removed in Phase 4.
struct QuantileGrid {
    const double* bin_lo    = nullptr;
    const double* bin_width = nullptr;
    const double* bin_rep   = nullptr;
};

// Inverts a (possibly unnormalised) cumulative mass array under the
// uniform-within-bin assumption (.
//
//   cdf[0..B-1]  non-decreasing cumulative mass, T = cdf[B-1]; empty bins allowed
//   thr          q * T
//   bin_lo/width lower edge and width of each bin (width may be nullptr == all 0)
//
// Guarantees: non-decreasing in thr; thr <= 0 returns the lower edge of the first
// non-empty bin; thr >= T returns the upper edge of the last non-empty bin; empty
// bins are never returned as interior points.  All arithmetic is in double.
template <typename CdfT>
inline double invert_cdf(const CdfT* cdf, int B, double thr,
                         const double* bin_lo, const double* bin_width) noexcept
{
    if (B <= 0) return 0.0;

    int    b;
    double frac;
    if (thr <= 0.0) {
        // First bin carrying positive mass.
        b = static_cast<int>(std::upper_bound(cdf, cdf + B, 0.0) - cdf);
        if (b >= B) b = B - 1;  // all-zero input (caller normally filters this out)
        frac = 0.0;
    } else {
        b = static_cast<int>(std::lower_bound(cdf, cdf + B, thr) - cdf);
        if (b >= B) {           // floating-point overshoot of thr above T
            b = B - 1;
            frac = 1.0;
        } else {
            const double prev  = (b > 0) ? static_cast<double>(cdf[b - 1]) : 0.0;
            const double denom = static_cast<double>(cdf[b]) - prev;
            frac = (denom > 0.0) ? (thr - prev) / denom : 0.0;
            if (frac < 0.0) frac = 0.0;
            if (frac > 1.0) frac = 1.0;
        }
    }
    if (bin_width == nullptr) return bin_lo[b];
    return bin_lo[b] + frac * bin_width[b];
}

// Monolithic C++ inference function for fast quantile prediction.
// `grid` is required (: quantiles are produced by invert_cdf over
// (bin_lo, bin_width) and bin_rep is the point representative.  Each of the three
// arrays must have n_target_bins entries.
void predict_quantiles_fast(
    const TreeView& tree,
    const double* X_data,
    int n_samples,
    int n_features,
    const double* bin_edges,
    const int32_t* bin_offsets,
    int max_bins,
    const double* quantiles,
    int num_quantiles,
    const QuantileGrid& grid,
    int n_target_bins,
    double* out_results
);

// Fast-float inference: uses pre-computed float64 split thresholds per node.
// Eliminates bin_edges lookup from the traversal hot-loop.
// Caller must have previously called compute_float_thresholds() (Python-side)
// and stored the result in tree_data_["split_threshold_float"].
void predict_quantiles_fast_float(
    const TreeViewFloat& tree,
    const double* X_data,
    int n_samples,
    int n_features,
    const double* quantiles,
    int num_quantiles,
    const QuantileGrid& grid,
    int n_target_bins,
    double* out_results
);

// =============================================================================
// // =============================================================================
// Replaces 5 near-identical predict_quantiles_fast_* functions (~396 lines,
// ~300 of which were copy-paste) with a single template resolved at compile
// time via policy types.
//
// Three independent dimensions of variation, each a policy:
//   ThreshPolicy — how to traverse the tree (BinThreshold | FloatThreshold)
//   PMFPolicy    — which distribution to walk the CDF over (CountsPMF | SmoothPMF)
//   EVTPolicy    — whether/how to splice GPD tails (NoEVT | WithEVT)
//
// //   1. Pre-compute CDF array once per leaf (O(B)) then binary-search for each
//      quantile (O(log B)), reducing per-sample work from O(Q*B) to O(B + Q*log B).
//      For Q=9, B=64: 576 → 118 comparisons (4.9×).
//   2. Pre-multiply threshold to eliminate the division inside the search loop.
//
// MAX_BINS: stack buffer size for the precomputed CDF array.  100 covers all
// standard configurations (32/64/128 bins).  For n_target_bins > MAX_BINS the
// binary search still works correctly (just reads the heap-allocated cdf vec).

// ---- Threshold policies ----
struct BinThreshold {
    const double*  bin_edges;
    const int32_t* bin_offsets;

    inline int traverse(const uint8_t* is_leaf, const int32_t* split_feat,
                        const uint8_t* split_thresh, const int32_t* left_child,
                        const int32_t* right_child, const double* X_row,
                        int /*n_features*/) const noexcept {
        int node_id = 0;
        while (!is_leaf[node_id]) {
            int f = split_feat[node_id];
            double val = X_row[f];
            int start = bin_offsets[f];
            int end_  = bin_offsets[f + 1];
            if (end_ - start < 2) {
                node_id = left_child[node_id];
            } else {
                double thresh = bin_edges[start + split_thresh[node_id] + 1];
                node_id = (val < thresh) ? left_child[node_id] : right_child[node_id];
            }
        }
        return node_id;
    }
};

struct FloatThreshold {
    const double* split_threshold_float;

    inline int traverse(const uint8_t* is_leaf, const int32_t* split_feat,
                        const uint8_t* /*unused*/, const int32_t* left_child,
                        const int32_t* right_child, const double* X_row,
                        int /*n_features*/) const noexcept {
        int node_id = 0;
        while (!is_leaf[node_id]) {
            int f = split_feat[node_id];
            node_id = (X_row[f] < split_threshold_float[node_id])
                      ? left_child[node_id] : right_child[node_id];
        }
        return node_id;
    }
};

// ---- PMF policies ----
struct CountsPMF {
    const int32_t* distribution_counts;
    int n_bins;

    // Accumulates raw int32 counts into a cumulative CDF array (float64).
    // Returns the running total (== leaf sample count).
    inline double get_cdf(int node_id, int n_target_bins, double* cdf_out) const noexcept {
        const int32_t* counts = distribution_counts + node_id * n_bins;
        double running = 0.0;
        for (int b = 0; b < n_target_bins; ++b) {
            running += counts[b];
            cdf_out[b] = running;
        }
        return running;  // unnormalised — caller divides threshold by this total
    }

    // Population-weighted mean from root node (node 0); used as empty-leaf fallback.
    inline double root_mean(int n_target_bins, const double* bin_rep) const noexcept {
        const int32_t* root = distribution_counts;  // row 0
        double total = 0.0, mean = 0.0;
        for (int b = 0; b < n_target_bins; ++b) total += root[b];
        if (total > 0.0) {
            for (int b = 0; b < n_target_bins; ++b) mean += (root[b] / total) * bin_rep[b];
        } else {
            for (int b = 0; b < n_target_bins; ++b) mean += bin_rep[b];
            mean /= static_cast<double>(n_target_bins > 0 ? n_target_bins : 1);
        }
        return mean;
    }
};

struct SmoothPMF {
    const double* smoothed_pmf;
    int n_bins;

    // Accumulates float64 PMF values into a cumulative CDF array.
    // Returns the PMF sum (should be ~1.0 for a valid normalised PMF).
    inline double get_cdf(int node_id, int n_target_bins, double* cdf_out) const noexcept {
        const double* pmf = smoothed_pmf + node_id * n_bins;
        double running = 0.0;
        for (int b = 0; b < n_target_bins; ++b) {
            running += pmf[b];
            cdf_out[b] = running;
        }
        return running;
    }

    inline double root_mean(int n_target_bins, const double* bin_rep) const noexcept {
        const double* root = smoothed_pmf;  // row 0
        double pmf_sum = 0.0, mean = 0.0;
        for (int b = 0; b < n_target_bins; ++b) pmf_sum += root[b];
        if (pmf_sum > 0.0) {
            for (int b = 0; b < n_target_bins; ++b) mean += (root[b] / pmf_sum) * bin_rep[b];
        } else {
            for (int b = 0; b < n_target_bins; ++b) mean += bin_rep[b];
            mean /= static_cast<double>(n_target_bins > 0 ? n_target_bins : 1);
        }
        return mean;
    }
};

// ---- EVT policies ----
struct NoEVT {
    static constexpr bool has_evt = false;
    inline bool upper_active(int /*node*/) const noexcept { return false; }
    inline bool lower_active(int /*node*/) const noexcept { return false; }
    inline double f_u(int /*node*/)        const noexcept { return 2.0;  } // sentinel > 1
    inline double f_u_lower(int /*node*/)  const noexcept { return -1.0; } // sentinel < 0
    inline double query_upper(double /*q*/, int /*node*/) const noexcept { return 0.0; }
    inline double query_lower(double /*q*/, int /*node*/) const noexcept { return 0.0; }
};

// WithEVT and predict_quantiles_unified are defined after the GPD helpers below,
// because WithEVT::query_upper/query_lower call gpd_upper_quantile/gpd_lower_quantile.

// =============================================================================

// Phase 6: C++ Dirichlet Smoothing Engine
// =============================================================================
// Ports smooth_all_leaves() from Python/_smoothing.py into C++.
// Inputs: flat int32 distribution_counts (n_nodes x n_bins), total_samples,
//         is_leaf, left_child_id, right_child_id — all of length n_nodes.
// Returns a flat float64 array of shape (n_nodes x n_bins) where:
//   - terminal leaves hold the Dirichlet posterior mean PMF
//   - internal nodes hold their empirical PMF (used as ancestor priors)
// The caller (Python side) stores this as tree_data_["smoothed_pmf"].

std::vector<double> smooth_tree(
    const int32_t* distribution_counts,  // (n_nodes, n_bins) row-major
    const int32_t* total_samples,         // (n_nodes,)
    const uint8_t* is_leaf,               // (n_nodes,)
    const int32_t* left_child_id,         // (n_nodes,)
    const int32_t* right_child_id,        // (n_nodes,)
    int n_nodes,
    int n_bins,
    double prior_weight,
    int min_ancestor_samples
);

// Smooth-PMF fast inference: reads from float64 smoothed_pmf array.
// Tree traversal is identical to predict_quantiles_fast (uint8 bin thresholds);
// the only difference is the quantile inner loop which reads pre-normalised
// float64 PMFs instead of raw int32 counts. This enables peak-performance
// quantile prediction for smooth_leaves=True models without any C++ bypass.
void predict_quantiles_fast_smooth(
    const TreeViewSmooth& tree,
    const double* X_data,
    int n_samples,
    int n_features,
    const double* bin_edges,
    const int32_t* bin_offsets,
    int max_bins,
    const double* quantiles,
    int num_quantiles,
    const QuantileGrid& grid,
    int n_target_bins,
    double* out_results
);

// =============================================================================
// Phase 1a: TreeViewEVT — EVT-aware zero-copy inference struct
// =============================================================================
// Extends TreeView with per-node GPD tail parameters stored as flat numpy
// arrays (zero-copy: the Python dict keeps lifetime ownership via pybind11).
//
// EVT routing at inference time (per leaf per quantile q):
//   q >= evt_F_u[leaf]        → gpd_upper_quantile (upper tail)
//   q <= evt_lower_F_u[leaf]  → gpd_lower_quantile (lower tail, if enabled)
//   otherwise                 → standard CDF walk over distribution_counts
//
// Nullable pointers: evt_lower_* are nullptr when evt_tails_lower=False.
//                    evt_S_u is nullptr for models pickled before Phase 3
//                    (backward compat — fall back to 1 - F_u in that case).

struct TreeViewEVT {
    int n_bins;
    const uint8_t*  is_leaf;
    const int32_t*  split_feature_idx;
    const uint8_t*  split_threshold;
    const int32_t*  left_child_id;
    const int32_t*  right_child_id;
    const int32_t*  distribution_counts; // (n_nodes, n_bins) row-major int32

    // Upper-tail EVT parameters (per node; sentinel 0 for non-EVT nodes)
    const uint8_t*  evt_enabled;         // 0 or 1
    const double*   evt_threshold_u;     // GPD threshold (physical value)
    const double*   evt_F_u;             // Empirical CDF at threshold
    const double*   evt_S_u;             // Survival = P(Y > u), nullable (Phase 3)
    const double*   evt_gpd_shape;       // xi (GPD shape)
    const double*   evt_gpd_scale;       // sigma (GPD scale, > 0)

    // Lower-tail EVT parameters — nullable when evt_tails_lower=False
    const uint8_t*  evt_lower_enabled;
    const double*   evt_lower_threshold_u;
    const double*   evt_lower_F_u;
    const double*   evt_lower_gpd_shape;
    const double*   evt_lower_gpd_scale;
};

// =============================================================================
// Phase 1b: TreeViewSmoothEVT — smoothed PMF + EVT inference struct
// =============================================================================
// For models where both smooth_leaves=True and evt_tails=True.  Replaces
// distribution_counts with smoothed_pmf in the body CDF walk, otherwise
// identical routing logic to TreeViewEVT.

struct TreeViewSmoothEVT {
    int n_bins;
    const uint8_t*  is_leaf;
    const int32_t*  split_feature_idx;
    const uint8_t*  split_threshold;
    const int32_t*  left_child_id;
    const int32_t*  right_child_id;
    const double*   smoothed_pmf;        // (n_nodes, n_bins) float64 row-major

    // Upper-tail EVT parameters (same layout as TreeViewEVT)
    const uint8_t*  evt_enabled;
    const double*   evt_threshold_u;
    const double*   evt_F_u;
    const double*   evt_S_u;             // nullable (Phase 3)
    const double*   evt_gpd_shape;
    const double*   evt_gpd_scale;

    // Lower-tail EVT parameters — nullable when evt_tails_lower=False
    const uint8_t*  evt_lower_enabled;
    const double*   evt_lower_threshold_u;
    const double*   evt_lower_F_u;
    const double*   evt_lower_gpd_shape;
    const double*   evt_lower_gpd_scale;
};

// =============================================================================
// Phase 2a: GPD Quantile Helpers (inline — zero-overhead, header-only)
// =============================================================================
// Mathematical reference: _evt.py::query_gpd_quantile and
//                         _evt.py::query_gpd_lower_quantile.
// The Python functions are the specification; these must produce identical
// results within floating-point round-trip tolerance.
//
// Survival precision (Phase 3 integration):
//   When S_u > 0 (non-null and non-zero sentinel), the exceedance probability
//   is computed as (1 - q) / S_u instead of (1 - q) / (1 - F_u), avoiding
//   catastrophic cancellation when q is very close to 1 (e.g. P99.9+).

// Upper-tail GPD inverse CDF.  Caller guarantees q >= F_u.
// Returns +inf for q >= 1.0 (caller should clamp before calling).
inline double gpd_upper_quantile(double q, double u, double F_u,
                                 double S_u, double xi, double sigma) noexcept
{
    if (q >= 1.0) return std::numeric_limits<double>::infinity();

    // Compute exceedance probability with survival-precision branch (Phase 3).
    // S_u == 0.0 is the sentinel meaning "not yet computed" (old pickled model).
    double exceedance_prob;
    if (S_u > 0.0) {
        exceedance_prob = (1.0 - q) / S_u;
    } else {
        double denom = 1.0 - F_u;
        if (denom <= 0.0) return std::numeric_limits<double>::infinity();
        exceedance_prob = (1.0 - q) / denom;
    }

    if (std::abs(xi) < 1e-6) {
        // Exponential (Gumbel) limit: x_q = u - sigma * log(exceedance_prob)
        return u - sigma * std::log(exceedance_prob);
    }
    // General Pareto: x_q = u + (sigma/xi) * (exceedance_prob^(-xi) - 1)
    return u + (sigma / xi) * (std::pow(exceedance_prob, -xi) - 1.0);
}

// Lower-tail GPD inverse CDF.  Caller guarantees q <= F_u_lower.
// Returns -inf for q <= 0.0 (caller should clamp before calling).
inline double gpd_lower_quantile(double q, double u_lower, double F_u_lower,
                                 double xi, double sigma) noexcept
{
    if (q <= 0.0) return -std::numeric_limits<double>::infinity();
    if (F_u_lower <= 0.0) return -std::numeric_limits<double>::infinity();

    double exceedance_prob = q / F_u_lower;

    if (std::abs(xi) < 1e-6) {
        // Exponential limit: x_q = u_lower + sigma * log(exceedance_prob)
        // log(exceedance_prob) <= 0 for exceedance_prob in (0,1], so x_q <= u_lower
        return u_lower + sigma * std::log(exceedance_prob);
    }
    // General Pareto (leftward): x_q = u_lower - (sigma/xi) * (exceedance_prob^(-xi) - 1)
    return u_lower - (sigma / xi) * (std::pow(exceedance_prob, -xi) - 1.0);
}

// =============================================================================
// // =============================================================================
// Placed here (after gpd_upper/lower_quantile) because WithEVT::query_upper/lower
// call those inline helpers, which must already be defined.

struct WithEVT {
    static constexpr bool has_evt = true;
    const uint8_t* evt_enabled;
    const double*  evt_threshold_u;
    const double*  evt_F_u;
    const double*  evt_S_u;            // nullable (Phase 3 compat)
    const double*  evt_gpd_shape;
    const double*  evt_gpd_scale;
    const uint8_t* evt_lower_enabled;  // nullable
    const double*  evt_lower_threshold_u;
    const double*  evt_lower_F_u;
    const double*  evt_lower_gpd_shape;
    const double*  evt_lower_gpd_scale;

    inline bool upper_active(int n) const noexcept {
        return evt_enabled && evt_enabled[n];
    }
    inline bool lower_active(int n) const noexcept {
        return evt_lower_enabled && evt_lower_enabled[n];
    }
    inline double f_u(int n)       const noexcept { return evt_F_u[n]; }
    inline double f_u_lower(int n) const noexcept { return evt_lower_F_u[n]; }

    inline double query_upper(double q, int n) const noexcept {
        double s_u = (evt_S_u != nullptr) ? evt_S_u[n] : 0.0;
        return gpd_upper_quantile(q, evt_threshold_u[n], evt_F_u[n], s_u,
                                   evt_gpd_shape[n], evt_gpd_scale[n]);
    }
    inline double query_lower(double q, int n) const noexcept {
        return gpd_lower_quantile(q, evt_lower_threshold_u[n], evt_lower_F_u[n],
                                   evt_lower_gpd_shape[n], evt_lower_gpd_scale[n]);
    }
};

// ---- predict_quantiles_unified template ----
// MAX_BINS_UNIFIED: stack CDF buffer size; covers all standard bin counts.
static constexpr int MAX_BINS_UNIFIED = 100;

// Scratch CDF buffer: stack for B <= MAX_BINS_UNIFIED, heap otherwise.
struct CdfScratch {
    double stack_buf[MAX_BINS_UNIFIED];
    std::vector<double> heap_buf;

    inline double* get(int n_bins) {
        if (n_bins <= MAX_BINS_UNIFIED) return stack_buf;
        heap_buf.resize(static_cast<size_t>(n_bins));
        return heap_buf.data();
    }
};

// ---- // Everything predict_quantiles_unified used to do after tree traversal:
// EVT routing + body CDF inversion for ONE leaf and all requested quantiles.
// The traversal-based path and the leaf-id entry point share this function so
// calibration (which already knows the leaf ids) sees the exact same predictor.
//
// Output addressing: quantile qi is written to out[qi * out_stride].
//   traversal path : out = out_results + i,        out_stride = n_samples  -> (Q, N)
//   leaf-id path   : out = out_results + i * Q,    out_stride = 1          -> (N, Q)
//
// Body is invert_cdf over (bin_lo, bin_width); empty leaves return population_mean.
template <typename PMFPolicy, typename EVTPolicy>
inline void quantiles_for_leaf(
    const PMFPolicy&     pmf,
    const EVTPolicy&     evt,
    int                  leaf,
    const double*        quantiles,
    int                  num_quantiles,
    const QuantileGrid&  grid,
    int                  n_target_bins,
    double               population_mean,
    double*              cdf_buf,
    double*              out,
    std::ptrdiff_t       out_stride) noexcept
{
    bool up_active = false, lo_active = false;
    double fu = 2.0, fu_lo = -1.0;
    if constexpr (EVTPolicy::has_evt) {
        up_active = evt.upper_active(leaf);
        lo_active = evt.lower_active(leaf);
        if (up_active) fu    = evt.f_u(leaf);
        if (lo_active) fu_lo = evt.f_u_lower(leaf);
    }

    double total = pmf.get_cdf(leaf, n_target_bins, cdf_buf);
    if (total <= 0.0) {
        for (int qi = 0; qi < num_quantiles; ++qi)
            out[qi * out_stride] = population_mean;
        return;
    }

    for (int qi = 0; qi < num_quantiles; ++qi) {
        double q = quantiles[qi];
        if constexpr (EVTPolicy::has_evt) {
            if (up_active && q >= fu) {
                out[qi * out_stride] = evt.query_upper(q, leaf);
                continue;
            }
            if (lo_active && q <= fu_lo) {
                out[qi * out_stride] = evt.query_lower(q, leaf);
                continue;
            }
        }
        out[qi * out_stride] = invert_cdf(cdf_buf, n_target_bins, q * total,
                                          grid.bin_lo, grid.bin_width);
    }
}

template <typename ThreshPolicy, typename PMFPolicy, typename EVTPolicy>
void predict_quantiles_unified(
    const ThreshPolicy&  thresh,
    const PMFPolicy&     pmf,
    const EVTPolicy&     evt,
    const uint8_t*  is_leaf,
    const int32_t*  split_feat,
    const uint8_t*  split_thresh_uint8,
    const int32_t*  left_child,
    const int32_t*  right_child,
    const double*   X_data,
    int n_samples, int n_features,
    const double*   quantiles, int num_quantiles,
    const QuantileGrid& grid, int n_target_bins,
    double          population_mean,
    double*         out_results)
{
    CdfScratch scratch;
    double* cdf_buf = scratch.get(n_target_bins);

    for (int i = 0; i < n_samples; ++i) {
        const double* x_row = X_data + i * n_features;
        int node_id = thresh.traverse(is_leaf, split_feat, split_thresh_uint8,
                                       left_child, right_child, x_row, n_features);

        quantiles_for_leaf(pmf, evt, node_id, quantiles, num_quantiles,
                           grid, n_target_bins, population_mean, cdf_buf,
                           out_results + i, static_cast<std::ptrdiff_t>(n_samples));
    }
}

// ---- // Inverts the leaf CDFs for a caller-supplied list of leaf ids (no traversal).
// out_results is row-major (n_leaves, num_quantiles).  Caller (the binding) has
// already validated that every leaf id is in range and is a leaf node.
template <typename PMFPolicy, typename EVTPolicy>
void predict_quantiles_from_leaves(
    const PMFPolicy&     pmf,
    const EVTPolicy&     evt,
    const int32_t*       leaf_ids,
    int                  n_leaves,
    const double*        quantiles,
    int                  num_quantiles,
    const QuantileGrid&  grid,
    int                  n_target_bins,
    double               population_mean,
    double*              out_results)
{
    CdfScratch scratch;
    double* cdf_buf = scratch.get(n_target_bins);
    for (int i = 0; i < n_leaves; ++i) {
        quantiles_for_leaf(pmf, evt, static_cast<int>(leaf_ids[i]),
                           quantiles, num_quantiles, grid, n_target_bins,
                           population_mean, cdf_buf,
                           out_results + static_cast<std::ptrdiff_t>(i) * num_quantiles,
                           /*out_stride=*/1);
    }
}

// Builds a WithEVT policy from either TreeViewEVT or TreeViewSmoothEVT.
template <typename ViewT>
inline WithEVT make_with_evt(const ViewT& v) noexcept {
    return WithEVT{
        v.evt_enabled, v.evt_threshold_u, v.evt_F_u, v.evt_S_u,
        v.evt_gpd_shape, v.evt_gpd_scale,
        v.evt_lower_enabled, v.evt_lower_threshold_u, v.evt_lower_F_u,
        v.evt_lower_gpd_shape, v.evt_lower_gpd_scale
    };
}

// =============================================================================
// Phase 2b/c: EVT Fast Inference — Declarations
// =============================================================================
// Implementations live in ddt_core.cpp (Phase 2 execution).

// EVT fast inference: distribution_counts body + GPD tail splice.
void predict_quantiles_fast_evt(
    const TreeViewEVT& tree,
    const double* X_data,
    int n_samples,
    int n_features,
    const double* bin_edges,
    const int32_t* bin_offsets,
    int max_bins,
    const double* quantiles,
    int num_quantiles,
    const QuantileGrid& grid,
    int n_target_bins,
    double* out_results
);

// Smooth-EVT fast inference: smoothed_pmf body + GPD tail splice.
void predict_quantiles_fast_smooth_evt(
    const TreeViewSmoothEVT& tree,
    const double* X_data,
    int n_samples,
    int n_features,
    const double* bin_edges,
    const int32_t* bin_offsets,
    int max_bins,
    const double* quantiles,
    int num_quantiles,
    const QuantileGrid& grid,
    int n_target_bins,
    double* out_results
);


} // namespace ddt
