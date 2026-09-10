// =============================================================================
// Distributional Decision Trees (DDT) — Pybind11 Module
// =============================================================================
// Zero-copy bridge between Python (NumPy arrays) and C++ core engine.
// Returns a dictionary of NumPy arrays representing the Struct of Arrays Tree.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ddt_core.hpp"

namespace py = pybind11;

// =============================================================================
// Tree serialization helpers (Zero-Copy)
// =============================================================================

template <typename T>
py::array_t<T> make_array(std::vector<T>&& vec) {
    auto* ptr = new std::vector<T>(std::move(vec));
    auto capsule = py::capsule(ptr, [](void* p) { delete reinterpret_cast<std::vector<T>*>(p); });
    return py::array_t<T>(ptr->size(), ptr->data(), capsule);
}

py::array_t<int32_t> make_2d_array(std::vector<int32_t>&& vec, int rows, int cols) {
    auto* ptr = new std::vector<int32_t>(std::move(vec));
    auto capsule = py::capsule(ptr, [](void* p) { delete reinterpret_cast<std::vector<int32_t>*>(p); });
    return py::array_t<int32_t>(
        {rows, cols}, 
        {static_cast<py::ssize_t>(cols * sizeof(int32_t)), static_cast<py::ssize_t>(sizeof(int32_t))}, 
        ptr->data(), 
        capsule
    );
}

static py::dict tree_to_py(ddt::Tree&& tree) {
    py::dict d;
    int num_nodes = static_cast<int>(tree.is_leaf.size());
    int n_bins = tree.n_bins;

    d["is_leaf"] = make_array(std::move(tree.is_leaf));
    d["split_feature_idx"] = make_array(std::move(tree.split_feature_idx));
    d["split_threshold"] = make_array(std::move(tree.split_threshold));
    d["left_child_id"] = make_array(std::move(tree.left_child_id));
    d["right_child_id"] = make_array(std::move(tree.right_child_id));
    d["wasserstein_gain"] = make_array(std::move(tree.wasserstein_gain));
    d["depth"] = make_array(std::move(tree.depth));
    d["total_samples"] = make_array(std::move(tree.total_samples));
    
    d["distribution_counts"] = make_2d_array(std::move(tree.distribution_counts), num_nodes, n_bins);
    d["n_bins"] = n_bins;

    return d;
}

static ddt::TreeView py_to_tree_view(const py::dict& d) {
    ddt::TreeView tree;
    tree.n_bins = d["n_bins"].cast<int>();
    tree.is_leaf = d["is_leaf"].cast<py::array_t<uint8_t>>().data(0);
    tree.split_feature_idx = d["split_feature_idx"].cast<py::array_t<int32_t>>().data(0);
    tree.split_threshold = d["split_threshold"].cast<py::array_t<uint8_t>>().data(0);
    tree.left_child_id = d["left_child_id"].cast<py::array_t<int32_t>>().data(0);
    tree.right_child_id = d["right_child_id"].cast<py::array_t<int32_t>>().data(0);
    tree.distribution_counts = d["distribution_counts"].cast<py::array_t<int32_t>>().data(0, 0);
    return tree;
}

// =============================================================================
// Python-facing functions
// =============================================================================

static py::dict py_build_tree(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> y,
    int n_bins,
    int max_depth,
    int min_samples_leaf)
{
    if (X.ndim() != 2) throw std::invalid_argument("X must be a 2D array (N x F)");
    if (y.ndim() != 1) throw std::invalid_argument("y must be a 1D array (N,)");
    if (n_bins <= 0)           throw std::invalid_argument("n_bins must be positive");
    if (max_depth <= 0)        throw std::invalid_argument("max_depth must be positive");
    if (min_samples_leaf <= 0) throw std::invalid_argument("min_samples_leaf must be positive");

    const int n_samples = static_cast<int>(X.shape(0));
    const int n_features = static_cast<int>(X.shape(1));

    if (y.shape(0) != n_samples) throw std::invalid_argument("X and y sample mismatch");

    auto X_acc = X.unchecked<2>();
    auto y_acc = y.unchecked<1>();

    // min_divergence_decrease is fixed at 0.0 — the public API does not expose
    // this tuning knob; tree size is bounded by max_depth and min_samples_leaf.
    auto tree = ddt::build_tree(
        X_acc.data(0, 0), y_acc.data(0), n_samples, n_features, n_bins,
        max_depth, min_samples_leaf, 0.0, "wasserstein"
    );

    return tree_to_py(std::move(tree));
}

static py::array_t<int32_t> py_predict_leaves(
    const py::dict& tree_data,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X)
{
    if (X.ndim() != 2) throw std::invalid_argument("X must be a 2D array (N x F)");

    const int n_samples = static_cast<int>(X.shape(0));
    
    // Extract raw pointers from the Python dict for O(1) inference routing
    auto is_leaf_arr = tree_data["is_leaf"].cast<py::array_t<uint8_t>>();
    auto split_feat_arr = tree_data["split_feature_idx"].cast<py::array_t<int32_t>>();
    auto split_thresh_arr = tree_data["split_threshold"].cast<py::array_t<uint8_t>>();
    auto left_child_arr = tree_data["left_child_id"].cast<py::array_t<int32_t>>();
    auto right_child_arr = tree_data["right_child_id"].cast<py::array_t<int32_t>>();

    auto is_leaf = is_leaf_arr.unchecked<1>();
    auto split_feat = split_feat_arr.unchecked<1>();
    auto split_thresh = split_thresh_arr.unchecked<1>();
    auto left_child = left_child_arr.unchecked<1>();
    auto right_child = right_child_arr.unchecked<1>();

    auto X_acc = X.unchecked<2>();

    py::array_t<int32_t> result(n_samples);
    auto result_mut = result.mutable_unchecked<1>();

    for (int i = 0; i < n_samples; ++i) {
        const uint8_t* row = X_acc.data(i, 0);
        int node_id = 0;
        
        while (!is_leaf(node_id)) {
            uint8_t fval = row[split_feat(node_id)];
            if (fval <= split_thresh(node_id)) {
                node_id = left_child(node_id);
            } else {
                node_id = right_child(node_id);
            }
        }
        result_mut(i) = node_id;
    }

    return result;
}

static py::array_t<int32_t> py_get_node_distribution(
    const py::dict& tree_data,
    int node_idx)
{
    auto dist_arr = tree_data["distribution_counts"].cast<py::array_t<int32_t>>();
    auto dist = dist_arr.unchecked<2>();

    if (node_idx < 0 || node_idx >= dist.shape(0)) {
        throw std::out_of_range("node_idx out of range");
    }

    int n_bins = static_cast<int>(dist.shape(1));
    py::array_t<int32_t> result(n_bins);
    auto result_mut = result.mutable_unchecked<1>();
    
    for (int i = 0; i < n_bins; ++i) {
        result_mut(i) = dist(node_idx, i);
    }

    return result;
}

static int64_t py_calculate_wasserstein(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> left_counts,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> right_counts)
{
    auto l = left_counts.unchecked<1>();
    auto r = right_counts.unchecked<1>();
    if (l.shape(0) != r.shape(0))
        throw std::invalid_argument("left_counts and right_counts must have equal length");
    int B = static_cast<int>(l.shape(0));

    int l_total = 0, r_total = 0;
    for (int i = 0; i < B; ++i) {
        l_total += l(i);
        r_total += r(i);
    }

    return ddt::wasserstein_1d(l.data(0), l_total, r.data(0), r_total, B);
}

static py::array_t<uint8_t> py_quantize_features(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> bin_edges,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> bin_offsets,
    int max_bins)
{
    if (X.ndim() != 2) throw std::invalid_argument("X must be a 2D array (N x F)");
    if (bin_edges.ndim() != 1) throw std::invalid_argument("bin_edges must be 1D");
    if (bin_offsets.ndim() != 1) throw std::invalid_argument("bin_offsets must be 1D");

    const int n_samples = static_cast<int>(X.shape(0));
    const int n_features = static_cast<int>(X.shape(1));

    if (bin_offsets.shape(0) != n_features + 1) {
        throw std::invalid_argument("bin_offsets must have length n_features + 1");
    }

    auto X_acc = X.unchecked<2>();
    auto edges = bin_edges.unchecked<1>();
    auto offsets = bin_offsets.unchecked<1>();

    py::array_t<uint8_t> result({n_samples, n_features});
    auto result_mut = result.mutable_unchecked<2>();

    #pragma omp parallel for
    for (int i = 0; i < n_samples; ++i) {
        for (int f = 0; f < n_features; ++f) {
            double val = X_acc(i, f);
            int start_idx = offsets(f);
            int end_idx = offsets(f + 1);
            int num_edges = end_idx - start_idx;
            
            if (num_edges < 2) {
                result_mut(i, f) = 0;
                continue;
            }

            const double* feature_edges = edges.data(start_idx);
            
            auto it = std::upper_bound(feature_edges + 1, feature_edges + num_edges - 1, val);
            int bin_idx = static_cast<int>(std::distance(feature_edges, it)) - 1;
            
            if (bin_idx < 0) bin_idx = 0;
            if (bin_idx >= max_bins) bin_idx = max_bins - 1;
            
            int max_idx_for_feature = num_edges - 2;
            if (bin_idx > max_idx_for_feature) bin_idx = max_idx_for_feature;

            result_mut(i, f) = static_cast<uint8_t>(bin_idx);
        }
    }

    return result;
}

static py::array_t<double> py_predict_quantiles_fast(
    const py::dict& tree_data,
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> bin_edges,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> bin_offsets,
    int max_bins,
    py::array_t<double, py::array::c_style | py::array::forcecast> quantiles,
    py::array_t<double, py::array::c_style | py::array::forcecast> bin_centers)
{
    if (X.ndim() != 2) throw std::invalid_argument("X must be a 2D array (N x F)");
    if (bin_edges.ndim() != 1) throw std::invalid_argument("bin_edges must be 1D");
    if (bin_offsets.ndim() != 1) throw std::invalid_argument("bin_offsets must be 1D");
    if (quantiles.ndim() != 1) throw std::invalid_argument("quantiles must be 1D");
    if (bin_centers.ndim() != 1) throw std::invalid_argument("bin_centers must be 1D");

    const int n_samples = static_cast<int>(X.shape(0));
    const int n_features = static_cast<int>(X.shape(1));
    const int num_quantiles = static_cast<int>(quantiles.shape(0));
    const int n_target_bins = static_cast<int>(bin_centers.shape(0));

    ddt::TreeView tree = py_to_tree_view(tree_data);

    auto X_acc = X.unchecked<2>();
    auto edges = bin_edges.unchecked<1>();
    auto offsets = bin_offsets.unchecked<1>();
    auto q_acc = quantiles.unchecked<1>();
    auto centers = bin_centers.unchecked<1>();

    py::array_t<double> result({num_quantiles, n_samples});
    auto result_mut = result.mutable_unchecked<2>();

    ddt::predict_quantiles_fast(
        tree,
        X_acc.data(0, 0),
        n_samples,
        n_features,
        edges.data(0),
        offsets.data(0),
        max_bins,
        q_acc.data(0),
        num_quantiles,
        centers.data(0),
        n_target_bins,
        result_mut.mutable_data(0, 0)
    );

    return result;
}

// =============================================================================
// Module Definition
// =============================================================================

PYBIND11_MODULE(_ddt_core, m) {
    m.doc() = "DDT Core Engine — C++17 / Pybind11 bridge for Distributional Decision Trees";

    m.def("build_tree", &py_build_tree,
        py::arg("X"), py::arg("y"), py::arg("n_bins"),
        py::arg("max_depth") = 10,
        py::arg("min_samples_leaf") = 20,
        R"doc(
        Build a DDT tree using the pure integer Wasserstein split criterion.

        Parameters
        ----------
        X             : ndarray uint8  (N, F)  Quantized feature matrix
        y             : ndarray int32  (N,)    Quantized target in [0, n_bins-1]
        n_bins        : int                    Number of target histogram bins
        max_depth     : int                    Maximum tree depth (default 10)
        min_samples_leaf : int                 Minimum samples per leaf (default 20)

        Returns
        -------
        dict of NumPy arrays representing the Struct-of-Arrays tree.
        )doc");

    m.def("predict_leaves", &py_predict_leaves,
        py::arg("tree_data"), py::arg("X"));

    m.def("get_node_distribution", &py_get_node_distribution,
        py::arg("tree_data"), py::arg("node_idx"));

    m.def("calculate_wasserstein", &py_calculate_wasserstein,
        py::arg("left_counts"), py::arg("right_counts"));

    m.def("quantize_features", &py_quantize_features,
        py::arg("X"), py::arg("bin_edges"), py::arg("bin_offsets"), py::arg("max_bins"));

    m.def("predict_quantiles_fast", &py_predict_quantiles_fast,
        py::arg("tree_data"), py::arg("X"), py::arg("bin_edges"), 
        py::arg("bin_offsets"), py::arg("max_bins"), 
        py::arg("quantiles"), py::arg("bin_centers"));
}
