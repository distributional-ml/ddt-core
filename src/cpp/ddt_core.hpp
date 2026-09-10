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
// Copyright (c) 2026 Distributional ML Contributors. Licensed under Apache-2.0.
// =============================================================================

#include <vector>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace ddt {

// -----------------------------------------------------------------------------
// 1D Wasserstein Distance (Earth Mover's Distance) — pure integer math
// -----------------------------------------------------------------------------
// Closed-form solution for 1D discrete distributions on aligned bins:
//   W_1(P, Q) = sum_{b=0}^{B-1} |CDF_P(b) - CDF_Q(b)|
//
// Computed entirely in integer arithmetic via cross-multiplication to avoid
// any floating-point division or per-bin multipliers:
//   |cdf_left_accum * right_total - cdf_right_accum * left_total|
//
// Return value is the raw integer-scaled divergence. All comparisons against
// min_divergence_decrease are performed in the same integer domain.
//
// Properties:
//   - O(B) computation on contiguous flat arrays
//   - Zero floating-point multipliers
//   - Explicit flat for-loop designed for compiler auto-vectorization (SIMD)
inline int64_t wasserstein_1d(const int* left_counts, int left_total,
                              const int* right_counts, int right_total, int B)
{
    if (left_total == 0 || right_total == 0) return 0;

    int64_t w1 = 0;
    int64_t cdf_left  = 0;
    int64_t cdf_right = 0;
    const int64_t nl = static_cast<int64_t>(left_total);
    const int64_t nr = static_cast<int64_t>(right_total);

    for (int b = 0; b < B; ++b) {
        cdf_left  += left_counts[b];
        cdf_right += right_counts[b];
        // Cross-multiply to avoid division: compare cdf_left/nl vs cdf_right/nr
        int64_t diff = cdf_left * nr - cdf_right * nl;
        w1 += diff < 0 ? -diff : diff;
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
// Split Workspace
// =============================================================================
// Pre-allocated buffers to prevent std::vector allocations during search loop.
// All buffers are sized at construction and reused across every split search.

struct SplitWorkspace {
    std::vector<int> root_hist;         // Size B
    std::vector<int> feature_hists;     // Size F * K * B
    std::vector<int> feature_totals;    // Size F * K
    std::vector<int> left_hist;         // Size B
    std::vector<int> right_hist;        // Size B

    SplitWorkspace(int F, int K, int B) {
        root_hist.resize(B, 0);
        feature_hists.resize(F * K * B, 0);
        feature_totals.resize(F * K, 0);
        left_hist.resize(B, 0);
        right_hist.resize(B, 0);
    }
};

// =============================================================================
// Split Result
// =============================================================================

struct SplitResult {
    bool found;
    int feature_idx;
    uint8_t threshold;
    int64_t gain;  // Integer-scaled Wasserstein divergence (cross-multiplied)
    // We do NOT copy vectors here. We reconstruct the partition in build_node.

    SplitResult() : found(false), feature_idx(-1), threshold(0), gain(-1) {}
};

// =============================================================================
// Core Algorithm Functions
// =============================================================================

// Find best split, using strictly the pre-allocated workspace. No heap allocations.
SplitResult find_best_split(
    const uint8_t* X_data,
    const int32_t* y_data,
    const int* sample_indices,
    int N,
    int n_features,
    int n_bins,
    int min_samples_leaf,
    SplitWorkspace& ws
);

// Build a complete decision tree from quantized data using Struct of Arrays.
Tree build_tree(
    const uint8_t* X_data,
    const int32_t* y_data,
    int n_samples,
    int n_features,
    int n_bins,
    int max_depth,
    int min_samples_leaf,
    double min_divergence_decrease,
    const std::string& divergence_name
);

// Predict the leaf node index for a single sample.
int predict_leaf(
    const Tree& tree,
    const uint8_t* sample_features,
    int n_features
);

// Monolithic C++ inference function for fast quantile prediction.
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
    const double* bin_centers,
    int n_target_bins,
    double* out_results
);

} // namespace ddt
