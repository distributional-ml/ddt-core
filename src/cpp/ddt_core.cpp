// =============================================================================
// Distributional Decision Trees (DDT) — Core Engine Implementation
// =============================================================================
// O(N + K*B) split evaluation via double-histogram quantization.
// Zero std::vector allocations in search loops. DOD Struct of Arrays memory.
// Pure integer Wasserstein inner loop — no floating-point multipliers.
// =============================================================================

#include "ddt_core.hpp"

#include <numeric>
#include <cassert>
#include <cstring>

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
    SplitWorkspace& ws)
{
    SplitResult best;
    
    if (N < 2 * min_samples_leaf) {
        return best;
    }

    const int B = n_bins;
    const int K = 256;

    // Zero out workspace arrays we are about to use
    std::fill(ws.root_hist.begin(), ws.root_hist.end(), 0);
    std::fill(ws.feature_hists.begin(), ws.feature_hists.end(), 0);
    std::fill(ws.feature_totals.begin(), ws.feature_totals.end(), 0);

    // Step 1: Build root histogram and all feature histograms in a SINGLE pass over N
    // This maximizes cache reuse!
    for (int i = 0; i < N; ++i) {
        int idx = sample_indices[i];
        int bin = y_data[idx];
        ws.root_hist[bin]++;
        
        for (int f = 0; f < n_features; ++f) {
            uint8_t fval = X_data[static_cast<size_t>(idx) * n_features + f];
            int offset = (f * K + fval) * B + bin;
            ws.feature_hists[offset]++;
            ws.feature_totals[f * K + fval]++;
        }
    }

    // Step 2: Sweep thresholds for each feature
    for (int f = 0; f < n_features; ++f) {
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

            // Compute integer-scaled Wasserstein divergence (no fp multipliers)
            int64_t gain = wasserstein_1d(ws.left_hist.data(), n_left, ws.right_hist.data(), n_right, B);

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
// build_tree (recursive implementation)
// =============================================================================

namespace {

int build_node(
    Tree& tree,
    const uint8_t* X_data,
    const int32_t* y_data,
    std::vector<int>& sample_indices,
    int n_features,
    int max_depth,
    int min_samples_leaf,
    double min_divergence_decrease,
    SplitWorkspace& ws,
    int current_depth)
{
    const int N = static_cast<int>(sample_indices.size());
    int node_id = tree.add_node(current_depth);

    // Build distribution for this node
    tree.total_samples[node_id] = N;
    int* dist_ptr = &tree.distribution_counts[node_id * tree.n_bins];
    for (int idx : sample_indices) {
        dist_ptr[y_data[idx]]++;
    }

    // Stopping conditions
    if ((max_depth > 0 && current_depth >= max_depth) || (N < 2 * min_samples_leaf)) {
        tree.is_leaf[node_id] = 1;
        return node_id;
    }

    // Find best split
    SplitResult split = find_best_split(
        X_data, y_data, sample_indices.data(), N,
        n_features, tree.n_bins, min_samples_leaf, ws
    );

    // min_divergence_decrease is stored as double; scale comparison into integer
    // domain: gain is cross-multiplied by (n_left * n_right), threshold is
    // multiplied by the same product implicitly — so we compare directly after
    // casting the threshold to int64_t scaled by n_left * n_right.
    const int64_t gain_threshold = static_cast<int64_t>(min_divergence_decrease
        * static_cast<double>(static_cast<int64_t>(N) * N));
    if (!split.found || split.gain < gain_threshold) {
        tree.is_leaf[node_id] = 1;
        return node_id;
    }

    // Partition samples
    std::vector<int> left_indices;
    std::vector<int> right_indices;
    left_indices.reserve(N);
    right_indices.reserve(N);

    for (int idx : sample_indices) {
        uint8_t fval = X_data[static_cast<size_t>(idx) * n_features + split.feature_idx];
        if (fval <= split.threshold) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }

    // Clear parent indices
    sample_indices.clear();
    sample_indices.shrink_to_fit();

    // Populate internal node details
    tree.is_leaf[node_id] = 0;
    tree.split_feature_idx[node_id] = split.feature_idx;
    tree.split_threshold[node_id] = split.threshold;
    tree.wasserstein_gain[node_id] = static_cast<double>(split.gain);

    // Recurse
    int left_id = build_node(
        tree, X_data, y_data, left_indices,
        n_features, max_depth, min_samples_leaf,
        min_divergence_decrease, ws, current_depth + 1
    );

    int right_id = build_node(
        tree, X_data, y_data, right_indices,
        n_features, max_depth, min_samples_leaf,
        min_divergence_decrease, ws, current_depth + 1
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
    const std::string& divergence_name)
{
    if (n_samples <= 0 || n_features <= 0 || n_bins <= 0 || min_samples_leaf <= 0) {
        throw std::invalid_argument("Invalid positive argument requirement.");
    }
    if (divergence_name != "wasserstein" && !divergence_name.empty()) {
        throw std::invalid_argument("Only 'wasserstein' is supported for now.");
    }

    Tree tree(n_bins);
    tree.reserve(static_cast<size_t>(2 * n_samples / min_samples_leaf + 1));

    std::vector<int> sample_indices(n_samples);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);

    // Allocate workspace exactly once
    SplitWorkspace ws(n_features, 256, n_bins);

    build_node(
        tree, X_data, y_data, sample_indices,
        n_features, max_depth, min_samples_leaf,
        min_divergence_decrease, ws, 0
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
// predict_quantiles_fast
// =============================================================================

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
)
{
    double mean_center = 0.0;
    for (int i = 0; i < n_target_bins; ++i) {
        mean_center += bin_centers[i];
    }
    mean_center /= static_cast<double>(n_target_bins > 0 ? n_target_bins : 1);

    for (int i = 0; i < n_samples; ++i) {
        int node_id = 0;
        
        while (!tree.is_leaf[node_id]) {
            int f = tree.split_feature_idx[node_id];
            double val = X_data[i * n_features + f];
            
            int start_idx = bin_offsets[f];
            int end_idx = bin_offsets[f + 1];
            int num_edges = end_idx - start_idx;
            
            if (num_edges < 2) {
                node_id = tree.left_child_id[node_id];
            } else {
                int thresh_uint8 = tree.split_threshold[node_id];
                double raw_thresh = bin_edges[start_idx + thresh_uint8 + 1];
                if (val < raw_thresh) {
                    node_id = tree.left_child_id[node_id];
                } else {
                    node_id = tree.right_child_id[node_id];
                }
            }
        }
        
        int total = 0;
        for (int b = 0; b < n_target_bins; ++b) {
            total += tree.distribution_counts[node_id * tree.n_bins + b];
        }
        
        if (total == 0) {
            for (int qi = 0; qi < num_quantiles; ++qi) {
                out_results[qi * n_samples + i] = mean_center;
            }
        } else {
            double safe_total = static_cast<double>(total);
            for (int qi = 0; qi < num_quantiles; ++qi) {
                double q = quantiles[qi];
                int current_sum = 0;
                int target_bin = n_target_bins - 1;
                for (int b = 0; b < n_target_bins; ++b) {
                    current_sum += tree.distribution_counts[node_id * tree.n_bins + b];
                    double cdf = current_sum / safe_total;
                    if (cdf >= q) {
                        target_bin = b;
                        break;
                    }
                }
                // NOTE (debt): This snaps to the bin centre of the first bin
                // whose CDF >= q. The Python _predict_raw_quantiles() path
                // linearly interpolates between adjacent bin centres, which
                // produces fractionally different values. Calibration residuals
                // are computed via the Python path, introducing a small
                // systematic bias. Future work: add linear interpolation here
                // or unify both paths through a single C++ implementation.
                out_results[qi * n_samples + i] = bin_centers[target_bin];
            }
        }
    }
}

} // namespace ddt
