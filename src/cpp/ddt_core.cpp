// =============================================================================
// Distributional Decision Trees (DDT) — Core Engine Implementation
// =============================================================================
// O(N + K*B) split evaluation via double-histogram quantization.
// Zero std::vector allocations in search loops. DOD Struct of Arrays memory.
// =============================================================================

#include "ddt_core.hpp"

#include <numeric>
#include <cassert>
#include <cstring> // For memset

namespace ddt {

// =============================================================================
// find_best_split
// =============================================================================

SplitResult find_best_split(
    const uint8_t* X_data,
    const int32_t* y_data,
    const int* sample_indices,
    int N,
    int n_features,
    int n_bins,
    int min_samples_leaf,
    SplitWorkspace& ws,
    const char* feature_mask)  // Phase 5: nullable; when non-null, skip features where mask[f]==0
{
    SplitResult best;
    
    if (N < 2 * min_samples_leaf) {
        return best;
    }

    const int B = n_bins;
    const int K = 256;

    //     // root_hist is small (B entries) — full fill is fine.
    std::fill(ws.root_hist.begin(), ws.root_hist.end(), 0);
    // feature_hists and feature_totals: zero only previously-touched entries.
    ws.reset_dirty();

    // Step 1: Build root histogram and all feature histograms in a SINGLE pass over N
    // This maximizes cache reuse!
    for (int i = 0; i < N; ++i) {
        int idx = sample_indices[i];
        int bin = y_data[idx];
        ws.root_hist[bin]++;
        
        for (int f = 0; f < n_features; ++f) {
            uint8_t fval = X_data[static_cast<size_t>(idx) * n_features + f];
            ws.mark_dirty(f, fval);  //             int offset = (f * K + fval) * B + bin;
            ws.feature_hists[offset]++;
            ws.feature_totals[f * K + fval]++;
        }
    }

    // Step 2: Sweep thresholds for each feature
    for (int f = 0; f < n_features; ++f) {
        // Phase 5: skip this feature if the mask says it is exhausted.
        if (feature_mask != nullptr && !feature_mask[f]) continue;

        std::fill(ws.left_hist.begin(), ws.left_hist.end(), 0);
        int n_left = 0;

        for (int k = 0; k < K - 1; ++k) {
            // Accumulate into left child
            int fval_total = ws.feature_totals[f * K + k];
            n_left += fval_total;
            
            if (fval_total > 0) {
                int offset = (f * K + k) * B;
                for (int b = 0; b < B; ++b) {
                    ws.left_hist[b] += ws.feature_hists[offset + b];
                }
            }

            int n_right = N - n_left;

            if (n_left < min_samples_leaf || n_right < min_samples_leaf) {
                continue;
            }

            // Derive right histogram
            for (int b = 0; b < B; ++b) {
                ws.right_hist[b] = ws.root_hist[b] - ws.left_hist[b];
            }

            // Compute Divergence
            double gain = wasserstein_1d(ws.left_hist.data(), n_left, ws.right_hist.data(), n_right, B);

            if (gain > best.gain) {
                best.gain = gain;
                best.feature_idx = f;
                best.threshold = static_cast<uint8_t>(k);
                best.found = true;
            }
        }
    }

    return best;
}

// =============================================================================
// // =============================================================================
// Replaces the former build_node + build_node_weighted pair (which were
// ~300 lines of near-identical code, each heap-allocating two vectors per
// internal node).
//
// Key changes vs the old implementation:
//   1. In-place partition: instead of creating left_indices / right_indices,
//      we partition sample_buf[start..end) in-place using a two-pointer swap.
//      The shared buffer is allocated ONCE in build_tree() and reused by every
//      recursive call — zero per-node heap allocations for partitioning.
//   2. Template<bool Weighted>: the ONLY run-time difference between the
//      unweighted and weighted paths is the Wasserstein gain call.  The
//      if constexpr branch is resolved at compile time with zero overhead.
//   3. The weighted path inlines the split search (delta_x_norm via ws);
//      the unweighted path calls find_best_split() which is unchanged and
//      also uses ws for the histogram buffers.

namespace {

// Helper: build feature eligibility mask into mask_storage, return raw ptr
// (or nullptr when feature_split_counts is null).
static const char* make_feature_mask(
    std::vector<char>& mask_storage,
    int n_features,
    const int* feature_split_counts,
    const int* max_splits_per_feature)
{
    if (feature_split_counts == nullptr || max_splits_per_feature == nullptr)
        return nullptr;
    mask_storage.resize(n_features);
    for (int f = 0; f < n_features; ++f)
        mask_storage[f] = (feature_split_counts[f] < max_splits_per_feature[f]) ? 1 : 0;
    return mask_storage.data();
}

template <bool Weighted>
int build_node_impl(
    Tree& tree,
    const uint8_t* X_data,
    const int32_t* y_data,
    int* sample_buf,   // shared buffer; this call owns [start, end)
    int start, int end,
    int n_features,
    int max_depth,
    int min_samples_leaf,
    double min_divergence_decrease,
    SplitWorkspace& ws,
    int current_depth,
    int* feature_split_counts,         // Phase 5: nullable
    const int* max_splits_per_feature) // Phase 5: nullable
{
    const int N = end - start;
    int node_id = tree.add_node(current_depth);

    // Build distribution for this node
    tree.total_samples[node_id] = N;
    int* dist_ptr = &tree.distribution_counts[node_id * tree.n_bins];
    for (int i = start; i < end; ++i)
        dist_ptr[y_data[sample_buf[i]]]++;

    // Stopping conditions
    if ((max_depth > 0 && current_depth >= max_depth) || (N < 2 * min_samples_leaf)) {
        tree.is_leaf[node_id] = 1;
        return node_id;
    }

    // Phase 5: compute feature eligibility mask from split counters.
    // Use std::vector<char> not std::vector<bool> (bool is a bitset: no .data()).
    std::vector<char> mask_storage;
    const char* feature_mask = make_feature_mask(
        mask_storage, n_features, feature_split_counts, max_splits_per_feature);

    // ---- Find best split ----
    SplitResult best;
    if constexpr (!Weighted) {
        // Unweighted path: delegate to find_best_split (uses workspace internally).
        best = find_best_split(
            X_data, y_data, sample_buf + start, N,
            n_features, tree.n_bins, min_samples_leaf, ws, feature_mask
        );
    } else {
        // Weighted path: inline split search (ws.delta_x_norm already set by caller).
        const int B = tree.n_bins;
        const int K = 256;

        // .
        std::fill(ws.root_hist.begin(), ws.root_hist.end(), 0);
        ws.reset_dirty();

        for (int i = start; i < end; ++i) {
            int idx = sample_buf[i];
            int bin = y_data[idx];
            ws.root_hist[bin]++;
            for (int f = 0; f < n_features; ++f) {
                uint8_t fval = X_data[static_cast<size_t>(idx) * n_features + f];
                ws.mark_dirty(f, fval);  //                 int offset = (f * K + fval) * B + bin;
                ws.feature_hists[offset]++;
                ws.feature_totals[f * K + fval]++;
            }
        }

        for (int f = 0; f < n_features; ++f) {
            // Phase 5: skip masked features.
            if (feature_mask != nullptr && !feature_mask[f]) continue;

            std::fill(ws.left_hist.begin(), ws.left_hist.end(), 0);
            int n_left = 0;

            for (int k = 0; k < K - 1; ++k) {
                int fval_total = ws.feature_totals[f * K + k];
                n_left += fval_total;

                if (fval_total > 0) {
                    int offset = (f * K + k) * B;
                    for (int b = 0; b < B; ++b)
                        ws.left_hist[b] += ws.feature_hists[offset + b];
                }

                int n_right = N - n_left;
                if (n_left < min_samples_leaf || n_right < min_samples_leaf) continue;

                for (int b = 0; b < B; ++b)
                    ws.right_hist[b] = ws.root_hist[b] - ws.left_hist[b];

                double gain = wasserstein_1d_weighted(
                    ws.left_hist.data(), n_left,
                    ws.right_hist.data(), n_right,
                    ws.delta_x_norm, B
                );

                if (gain > best.gain) {
                    best.gain = gain;
                    best.feature_idx = f;
                    best.threshold = static_cast<uint8_t>(k);
                    best.found = true;
                }
            }
        }
    }

    if (!best.found || best.gain < min_divergence_decrease) {
        tree.is_leaf[node_id] = 1;
        return node_id;
    }

    // Phase 5: increment the counter for the chosen feature before recursing.
    if (feature_split_counts != nullptr)
        feature_split_counts[best.feature_idx]++;

    // ---- In-place partition: sample_buf[start..end) around the split threshold ----
    // After this loop: [start..mid) → left, [mid..end) → right.
    // Relative order within each half is not preserved (not needed for split quality).
    int mid = start;
    for (int i = start; i < end; ++i) {
        uint8_t fval = X_data[static_cast<size_t>(sample_buf[i]) * n_features + best.feature_idx];
        if (fval <= best.threshold) {
            std::swap(sample_buf[mid], sample_buf[i]);
            ++mid;
        }
    }

    // Populate internal node
    tree.is_leaf[node_id] = 0;
    tree.split_feature_idx[node_id] = best.feature_idx;
    tree.split_threshold[node_id] = best.threshold;
    tree.wasserstein_gain[node_id] = best.gain;

    // Recurse — both children share the same contiguous sample_buf.
    // Left child owns [start..mid), right child owns [mid..end).
    int left_id = build_node_impl<Weighted>(
        tree, X_data, y_data, sample_buf, start, mid,
        n_features, max_depth, min_samples_leaf,
        min_divergence_decrease, ws, current_depth + 1,
        feature_split_counts, max_splits_per_feature
    );
    int right_id = build_node_impl<Weighted>(
        tree, X_data, y_data, sample_buf, mid, end,
        n_features, max_depth, min_samples_leaf,
        min_divergence_decrease, ws, current_depth + 1,
        feature_split_counts, max_splits_per_feature
    );

    tree.left_child_id[node_id] = left_id;
    tree.right_child_id[node_id] = right_id;
    return node_id;
}

} // anonymous namespace

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
    const int* max_splits_per_feature)  // Phase 5: nullable
{
    if (n_samples <= 0 || n_features <= 0 || n_bins <= 0 || min_samples_leaf <= 0) {
        throw std::invalid_argument("Invalid positive argument requirement.");
    }
    if (divergence_name != "wasserstein" && !divergence_name.empty()) {
        throw std::invalid_argument("Only 'wasserstein' is supported for now.");
    }

    Tree tree(n_bins);
    tree.reserve(static_cast<size_t>(2 * n_samples / min_samples_leaf + 1));

    //     std::vector<int> sample_buf(n_samples);
    std::iota(sample_buf.begin(), sample_buf.end(), 0);

    // Phase 5: allocate per-feature split counter (heap-once, shared through recursion).
    std::vector<int> split_counts;
    int* split_counts_ptr = nullptr;
    if (max_splits_per_feature != nullptr) {
        split_counts.assign(n_features, 0);
        split_counts_ptr = split_counts.data();
    }

    // Allocate workspace exactly once
    SplitWorkspace ws(n_features, 256, n_bins);

    build_node_impl</*Weighted=*/false>(
        tree, X_data, y_data, sample_buf.data(), 0, n_samples,
        n_features, max_depth, min_samples_leaf,
        min_divergence_decrease, ws, 0,
        split_counts_ptr, max_splits_per_feature
    );

    return tree;
}

// =============================================================================
// build_tree_weighted — variable-width bin Wasserstein (weighted path)
// =============================================================================
// // the same in-place partitioning as the unweighted path.  ws.delta_x_norm
// is set on the workspace so the Weighted branch can access it without
// introducing any additional function parameters.

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
    const int* max_splits_per_feature)  // Phase 5: nullable
{
    if (n_samples <= 0 || n_features <= 0 || n_bins <= 0 || min_samples_leaf <= 0) {
        throw std::invalid_argument("Invalid positive argument requirement.");
    }
    if (delta_x_norm == nullptr) {
        throw std::invalid_argument("delta_x_norm must be non-null for build_tree_weighted.");
    }
    if (divergence_name != "wasserstein" && !divergence_name.empty()) {
        throw std::invalid_argument("Only 'wasserstein' is supported for now.");
    }

    Tree tree(n_bins);
    tree.reserve(static_cast<size_t>(2 * n_samples / min_samples_leaf + 1));

    //     std::vector<int> sample_buf(n_samples);
    std::iota(sample_buf.begin(), sample_buf.end(), 0);

    // Phase 5: allocate per-feature split counter.
    std::vector<int> split_counts_w;
    int* split_counts_ptr_w = nullptr;
    if (max_splits_per_feature != nullptr) {
        split_counts_w.assign(n_features, 0);
        split_counts_ptr_w = split_counts_w.data();
    }

    SplitWorkspace ws(n_features, 256, n_bins);
    ws.delta_x_norm = delta_x_norm;
    ws.has_variable_bins = true;

    build_node_impl</*Weighted=*/true>(
        tree, X_data, y_data, sample_buf.data(), 0, n_samples,
        n_features, max_depth, min_samples_leaf,
        min_divergence_decrease, ws, 0,
        split_counts_ptr_w, max_splits_per_feature
    );

    return tree;
}


// =============================================================================
// predict_leaf
// =============================================================================

int predict_leaf(
    const Tree& tree,
    const uint8_t* sample_features,
    int /* n_features */)
{
    int node_id = 0; 
    while (!tree.is_leaf[node_id]) {
        uint8_t fval = sample_features[tree.split_feature_idx[node_id]];
        if (fval <= tree.split_threshold[node_id]) {
            node_id = tree.left_child_id[node_id];
        } else {
            node_id = tree.right_child_id[node_id];
        }
    }
    return node_id;
}

// =============================================================================
// predict_quantiles_fast — // =============================================================================
// BinThreshold (uint8 bin-edges traversal) + CountsPMF + NoEVT.
// The unified template applies the CDF binary-search optimization (Task 3)
// automatically: O(B + Q*log B) per sample instead of the old O(Q*B).

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
)
{
    (void)max_bins;  // Parameter kept for API symmetry; clamping done Python-side

    BinThreshold thresh{bin_edges, bin_offsets};
    CountsPMF    pmf{tree.distribution_counts, tree.n_bins};
    NoEVT        evt{};

    double pop_mean = pmf.root_mean(n_target_bins, grid.bin_rep);

    predict_quantiles_unified(
        thresh, pmf, evt,
        tree.is_leaf, tree.split_feature_idx, tree.split_threshold,
        tree.left_child_id, tree.right_child_id,
        X_data, n_samples, n_features,
        quantiles, num_quantiles,
        grid, n_target_bins,
        pop_mean, out_results
    );
}


// =============================================================================
// predict_quantiles_fast_float (Phase 5: Float-Split Inference)
// =============================================================================
// // FloatThreshold + CountsPMF + NoEVT.
// Single float comparison per tree level — zero bin_edges/offset indirection.

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
)
{
    FloatThreshold thresh{tree.split_threshold_float};
    CountsPMF      pmf{tree.distribution_counts, tree.n_bins};
    NoEVT          evt{};

    double pop_mean = pmf.root_mean(n_target_bins, grid.bin_rep);

    // FloatThreshold.traverse() ignores split_thresh_uint8 — pass null safely.
    predict_quantiles_unified(
        thresh, pmf, evt,
        tree.is_leaf, tree.split_feature_idx, /*split_thresh_uint8=*/nullptr,
        tree.left_child_id, tree.right_child_id,
        X_data, n_samples, n_features,
        quantiles, num_quantiles,
        grid, n_target_bins,
        pop_mean, out_results
    );
}


// =============================================================================
// smooth_tree (Phase 6: C++ Dirichlet Smoothing Engine)
// =============================================================================
// Ports _smoothing.py::smooth_all_leaves() to C++ for peak fit-time performance.
//
// Algorithm:
//   1. Build parent_of[n] = parent index (root has -1) from left/right child ids.
//   2. For each internal node: store its own empirical PMF in smoothed_pmf[n].
//   3. For each terminal leaf:
//      a. Walk up the tree from the parent until an ancestor with
//         >= min_ancestor_samples is found.  Fall through to root if none qualifies.
//      b. Apply the Dirichlet posterior mean:
//           theta_smooth = w * p_leaf + (1-w) * p_ancestor, w = N/(N+lambda)
//      c. Clip to [0,1] and re-normalise for floating-point safety.
//
// Matches the mathematical contract of smooth_all_leaves() exactly.

std::vector<double> smooth_tree(
    const int32_t* distribution_counts,
    const int32_t* total_samples,
    const uint8_t* is_leaf,
    const int32_t* left_child_id,
    const int32_t* right_child_id,
    int n_nodes,
    int n_bins,
    double prior_weight,
    int min_ancestor_samples)
{
    // ---- 1. Build parent map ----
    std::vector<int32_t> parent_of(n_nodes, -1);
    for (int nid = 0; nid < n_nodes; ++nid) {
        if (!is_leaf[nid]) {
            int lc = left_child_id[nid];
            int rc = right_child_id[nid];
            if (lc >= 0) parent_of[lc] = nid;
            if (rc >= 0) parent_of[rc] = nid;
        }
    }

    // ---- 2. Allocate output ----
    std::vector<double> smoothed_pmf(static_cast<size_t>(n_nodes) * n_bins, 0.0);

    // Precompute empirical PMFs for all nodes (used both for internal storage
    // and as candidates during the ancestor walk).  We store them temporarily
    // in smoothed_pmf; leaves will be overwritten below.
    for (int nid = 0; nid < n_nodes; ++nid) {
        const int32_t* counts = distribution_counts + nid * n_bins;
        int N = total_samples[nid];
        double* pmf_row = smoothed_pmf.data() + nid * n_bins;
        if (N > 0) {
            double inv_N = 1.0 / static_cast<double>(N);
            for (int b = 0; b < n_bins; ++b) {
                pmf_row[b] = counts[b] * inv_N;
            }
        } else {
            // Degenerate empty node → uniform
            double inv_B = 1.0 / static_cast<double>(n_bins);
            for (int b = 0; b < n_bins; ++b) {
                pmf_row[b] = inv_B;
            }
        }
    }

    // ---- 3. Smooth terminal leaves ----
    for (int nid = 0; nid < n_nodes; ++nid) {
        if (!is_leaf[nid]) continue;

        // -- 3a. Find stable ancestor PMF --
        const double* ancestor_pmf = nullptr;
        int current = parent_of[nid];
        while (current >= 0) {
            if (total_samples[current] >= min_ancestor_samples) {
                ancestor_pmf = smoothed_pmf.data() + current * n_bins;
                break;
            }
            current = parent_of[current];
        }
        if (ancestor_pmf == nullptr) {
            // Fell through: use root (node 0) as last resort
            ancestor_pmf = smoothed_pmf.data();  // node 0 row
        }

        // -- 3b. Dirichlet posterior mean --
        const int32_t* counts = distribution_counts + nid * n_bins;
        int N = total_samples[nid];
        double* pmf_row = smoothed_pmf.data() + nid * n_bins;

        if (N == 0) {
            // Empty leaf → copy ancestor PMF
            for (int b = 0; b < n_bins; ++b) {
                pmf_row[b] = ancestor_pmf[b];
            }
        } else if (prior_weight == 0.0) {
            // Zero prior weight → pure empirical (already set above)
            // pmf_row already holds the empirical PMF; nothing to do.
        } else {
            double w = static_cast<double>(N) / (static_cast<double>(N) + prior_weight);
            double one_minus_w = 1.0 - w;
            double inv_N = 1.0 / static_cast<double>(N);
            double total_prob = 0.0;
            for (int b = 0; b < n_bins; ++b) {
                double leaf_prob = counts[b] * inv_N;
                double blended = w * leaf_prob + one_minus_w * ancestor_pmf[b];
                if (blended < 0.0) blended = 0.0;
                pmf_row[b] = blended;
                total_prob += blended;
            }
            // Re-normalise for floating-point safety
            if (total_prob > 0.0) {
                double inv_total = 1.0 / total_prob;
                for (int b = 0; b < n_bins; ++b) {
                    pmf_row[b] *= inv_total;
                }
            } else {
                // Fallback: uniform (should never happen if ancestor PMF is valid)
                double inv_B = 1.0 / static_cast<double>(n_bins);
                for (int b = 0; b < n_bins; ++b) {
                    pmf_row[b] = inv_B;
                }
            }
        }
    }

    return smoothed_pmf;
}

// =============================================================================
// predict_quantiles_fast_smooth (Phase 6: Smoothed-PMF Fast Inference)
// =============================================================================
// // BinThreshold + SmoothPMF + NoEVT.

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
    double* out_results)
{
    (void)max_bins;

    BinThreshold thresh{bin_edges, bin_offsets};
    SmoothPMF    pmf{tree.smoothed_pmf, tree.n_bins};
    NoEVT        evt{};

    double pop_mean = pmf.root_mean(n_target_bins, grid.bin_rep);

    predict_quantiles_unified(
        thresh, pmf, evt,
        tree.is_leaf, tree.split_feature_idx, tree.split_threshold,
        tree.left_child_id, tree.right_child_id,
        X_data, n_samples, n_features,
        quantiles, num_quantiles,
        grid, n_target_bins,
        pop_mean, out_results
    );
}


// =============================================================================
// Phase 2b: predict_quantiles_fast_evt
// =============================================================================
// // BinThreshold + CountsPMF + WithEVT.
// GPD helpers from ddt_core.hpp are called through the WithEVT policy.

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
    double* out_results)
{
    (void)max_bins;

    BinThreshold thresh{bin_edges, bin_offsets};
    CountsPMF    pmf{tree.distribution_counts, tree.n_bins};
    WithEVT      evt = make_with_evt(tree);

    double pop_mean = pmf.root_mean(n_target_bins, grid.bin_rep);

    predict_quantiles_unified(
        thresh, pmf, evt,
        tree.is_leaf, tree.split_feature_idx, tree.split_threshold,
        tree.left_child_id, tree.right_child_id,
        X_data, n_samples, n_features,
        quantiles, num_quantiles,
        grid, n_target_bins,
        pop_mean, out_results
    );
}


// =============================================================================
// Phase 2c: predict_quantiles_fast_smooth_evt
// =============================================================================
// // BinThreshold + SmoothPMF + WithEVT.

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
    double* out_results)
{
    (void)max_bins;

    BinThreshold thresh{bin_edges, bin_offsets};
    SmoothPMF    pmf{tree.smoothed_pmf, tree.n_bins};
    WithEVT      evt = make_with_evt(tree);

    double pop_mean = pmf.root_mean(n_target_bins, grid.bin_rep);

    predict_quantiles_unified(
        thresh, pmf, evt,
        tree.is_leaf, tree.split_feature_idx, tree.split_threshold,
        tree.left_child_id, tree.right_child_id,
        X_data, n_samples, n_features,
        quantiles, num_quantiles,
        grid, n_target_bins,
        pop_mean, out_results
    );
}



} // namespace ddt
