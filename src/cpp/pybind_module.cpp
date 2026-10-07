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

static ddt::TreeViewFloat py_to_tree_view_float(const py::dict& d) {
    ddt::TreeViewFloat tree;
    tree.n_bins = d["n_bins"].cast<int>();
    tree.is_leaf = d["is_leaf"].cast<py::array_t<uint8_t>>().data(0);
    tree.split_feature_idx = d["split_feature_idx"].cast<py::array_t<int32_t>>().data(0);
    tree.split_threshold_float = d["split_threshold_float"].cast<py::array_t<double>>().data(0);
    tree.left_child_id = d["left_child_id"].cast<py::array_t<int32_t>>().data(0);
    tree.right_child_id = d["right_child_id"].cast<py::array_t<int32_t>>().data(0);
    tree.distribution_counts = d["distribution_counts"].cast<py::array_t<int32_t>>().data(0, 0);
    return tree;
}

static ddt::TreeViewSmooth py_to_tree_view_smooth(const py::dict& d,
                                                   const py::array_t<double>& pmf_arr) {
    ddt::TreeViewSmooth tree;
    tree.n_bins = d["n_bins"].cast<int>();
    tree.is_leaf = d["is_leaf"].cast<py::array_t<uint8_t>>().data(0);
    tree.split_feature_idx = d["split_feature_idx"].cast<py::array_t<int32_t>>().data(0);
    tree.split_threshold = d["split_threshold"].cast<py::array_t<uint8_t>>().data(0);
    tree.left_child_id = d["left_child_id"].cast<py::array_t<int32_t>>().data(0);
    tree.right_child_id = d["right_child_id"].cast<py::array_t<int32_t>>().data(0);
    tree.smoothed_pmf = pmf_arr.data(0, 0);
    return tree;
}

// =============================================================================
// Phase 1c: EVT Deserialisers (zero-copy, nullable lower-tail and S_u)
// =============================================================================
// Extracts raw pointers from the Python tree_data dict into the C++ EVT
// struct views.  The Python dict (and its numpy arrays) must remain alive for
// the lifetime of the returned struct — this invariant is enforced by pybind11
// keeping the dict alive through the call stack.
//
// Nullable rules:
//   evt_S_u            — nullptr if key "evt_S_u" is absent (pre-Phase-3 model)
//   evt_lower_*        — nullptr if key "evt_lower_enabled" is absent
//                        (model fitted without evt_tails_lower=True)

// Helper: returns data pointer for a double array key, or nullptr if absent.
static const double* get_optional_double_ptr(const py::dict& d, const char* key) {
    if (!d.contains(key)) return nullptr;
    auto arr = d[key].cast<py::array_t<double>>();
    return arr.data(0);
}

// Helper: returns data pointer for a uint8 array key, or nullptr if absent.
static const uint8_t* get_optional_uint8_ptr(const py::dict& d, const char* key) {
    if (!d.contains(key)) return nullptr;
    auto arr = d[key].cast<py::array_t<uint8_t>>();
    return arr.data(0);
}

static ddt::TreeViewEVT py_to_tree_view_evt(const py::dict& d) {
    ddt::TreeViewEVT tree;
    tree.n_bins              = d["n_bins"].cast<int>();
    tree.is_leaf             = d["is_leaf"].cast<py::array_t<uint8_t>>().data(0);
    tree.split_feature_idx   = d["split_feature_idx"].cast<py::array_t<int32_t>>().data(0);
    tree.split_threshold     = d["split_threshold"].cast<py::array_t<uint8_t>>().data(0);
    tree.left_child_id       = d["left_child_id"].cast<py::array_t<int32_t>>().data(0);
    tree.right_child_id      = d["right_child_id"].cast<py::array_t<int32_t>>().data(0);
    tree.distribution_counts = d["distribution_counts"].cast<py::array_t<int32_t>>().data(0, 0);

    // Upper-tail EVT — required keys (caller ensures evt_tails=True was set)
    tree.evt_enabled     = d["evt_enabled"].cast<py::array_t<uint8_t>>().data(0);
    tree.evt_threshold_u = d["evt_threshold_u"].cast<py::array_t<double>>().data(0);
    tree.evt_F_u         = d["evt_F_u"].cast<py::array_t<double>>().data(0);
    tree.evt_gpd_shape   = d["evt_gpd_shape"].cast<py::array_t<double>>().data(0);
    tree.evt_gpd_scale   = d["evt_gpd_scale"].cast<py::array_t<double>>().data(0);

    // Survival field — nullable (backward compat: nullptr -> fall back to 1-F_u)
    tree.evt_S_u = get_optional_double_ptr(d, "evt_S_u");

    // Lower-tail EVT — nullable (nullptr when evt_tails_lower=False)
    tree.evt_lower_enabled     = get_optional_uint8_ptr(d, "evt_lower_enabled");
    tree.evt_lower_threshold_u = get_optional_double_ptr(d, "evt_lower_threshold_u");
    tree.evt_lower_F_u         = get_optional_double_ptr(d, "evt_lower_F_u");
    tree.evt_lower_gpd_shape   = get_optional_double_ptr(d, "evt_lower_gpd_shape");
    tree.evt_lower_gpd_scale   = get_optional_double_ptr(d, "evt_lower_gpd_scale");

    return tree;
}

static ddt::TreeViewSmoothEVT py_to_tree_view_smooth_evt(const py::dict& d,
                                                          const py::array_t<double>& pmf_arr) {
    ddt::TreeViewSmoothEVT tree;
    tree.n_bins            = d["n_bins"].cast<int>();
    tree.is_leaf           = d["is_leaf"].cast<py::array_t<uint8_t>>().data(0);
    tree.split_feature_idx = d["split_feature_idx"].cast<py::array_t<int32_t>>().data(0);
    tree.split_threshold   = d["split_threshold"].cast<py::array_t<uint8_t>>().data(0);
    tree.left_child_id     = d["left_child_id"].cast<py::array_t<int32_t>>().data(0);
    tree.right_child_id    = d["right_child_id"].cast<py::array_t<int32_t>>().data(0);
    tree.smoothed_pmf      = pmf_arr.data(0, 0);

    // Upper-tail EVT — required keys
    tree.evt_enabled     = d["evt_enabled"].cast<py::array_t<uint8_t>>().data(0);
    tree.evt_threshold_u = d["evt_threshold_u"].cast<py::array_t<double>>().data(0);
    tree.evt_F_u         = d["evt_F_u"].cast<py::array_t<double>>().data(0);
    tree.evt_gpd_shape   = d["evt_gpd_shape"].cast<py::array_t<double>>().data(0);
    tree.evt_gpd_scale   = d["evt_gpd_scale"].cast<py::array_t<double>>().data(0);

    // Survival field — nullable
    tree.evt_S_u = get_optional_double_ptr(d, "evt_S_u");

    // Lower-tail EVT — nullable
    tree.evt_lower_enabled     = get_optional_uint8_ptr(d, "evt_lower_enabled");
    tree.evt_lower_threshold_u = get_optional_double_ptr(d, "evt_lower_threshold_u");
    tree.evt_lower_F_u         = get_optional_double_ptr(d, "evt_lower_F_u");
    tree.evt_lower_gpd_shape   = get_optional_double_ptr(d, "evt_lower_gpd_shape");
    tree.evt_lower_gpd_scale   = get_optional_double_ptr(d, "evt_lower_gpd_scale");

    return tree;
}

// =============================================================================
// 
// =============================================================================
// Every quantile binding takes bin_lo, bin_width and bin_rep (float64, shape (B,),
// original target space) in place of the legacy bin_centers argument (removed in
// .  Quantiles are produced by edge-based inversion (spec 3.2).
// B is derived from the arrays and must be >= 1.

using DoubleArr = py::array_t<double, py::array::c_style | py::array::forcecast>;

struct GridHolder {
    DoubleArr lo, width, rep;           // keep the buffers alive for the call
    ddt::QuantileGrid grid{};
    int B = 0;
};

// max_B : largest B the tree/pmf rows can serve (n_bins); -1 disables the check
static GridHolder parse_grid(const DoubleArr& bin_lo, const DoubleArr& bin_width,
                             const DoubleArr& bin_rep, int max_B)
{
    if (bin_lo.ndim() != 1 || bin_width.ndim() != 1 || bin_rep.ndim() != 1)
        throw std::invalid_argument("bin_lo, bin_width and bin_rep must be 1D");
    const int B = static_cast<int>(bin_lo.shape(0));
    if (static_cast<int>(bin_width.shape(0)) != B || static_cast<int>(bin_rep.shape(0)) != B)
        throw std::invalid_argument("bin_lo, bin_width and bin_rep must have the same length");
    if (B < 1)
        throw std::invalid_argument("bin_lo/bin_width/bin_rep must have at least one bin");
    if (max_B >= 0 && B > max_B)
        throw std::invalid_argument(
            "bin_lo/bin_width/bin_rep are longer than the tree's n_bins");
    const double* w = bin_width.data();
    for (int b = 0; b < B; ++b)
        if (!(w[b] >= 0.0))   // also rejects NaN
            throw std::invalid_argument("bin_width must be >= 0 (NaN and negative values are rejected)");

    GridHolder h{bin_lo, bin_width, bin_rep, ddt::QuantileGrid{}, B};
    h.grid.bin_lo    = h.lo.data();
    h.grid.bin_width = h.width.data();
    h.grid.bin_rep   = h.rep.data();
    return h;
}

// Grid validated once and owning internal copies of the three arrays; passed to the
// predict_quantiles_* bindings to skip per-call array conversion / validation.
// Non-copyable: grid pointers refer to the owned vectors.
class CompiledGrid {
public:
    CompiledGrid(const DoubleArr& bin_lo, const DoubleArr& bin_width, const DoubleArr& bin_rep)
    {
        GridHolder h = parse_grid(bin_lo, bin_width, bin_rep, /*max_B=*/-1);
        B = h.B;
        lo.assign(h.lo.data(), h.lo.data() + B);
        width.assign(h.width.data(), h.width.data() + B);
        rep.assign(h.rep.data(), h.rep.data() + B);
        grid.bin_lo = lo.data();
        grid.bin_width = width.data();
        grid.bin_rep = rep.data();
    }
    CompiledGrid(const CompiledGrid&) = delete;
    CompiledGrid& operator=(const CompiledGrid&) = delete;

    void check(int max_B) const
    {
        if (max_B >= 0 && B > max_B)
            throw std::invalid_argument(
                "bin_lo/bin_width/bin_rep are longer than the tree's n_bins");
    }

    std::vector<double> lo, width, rep;
    ddt::QuantileGrid grid{};
    int B = 0;
};

// -----------------------------------------------------------------------------
// native-boundary validation shared by every fast-inference binding.
// All checks run before the GIL is released; they raise instead of reading out
// of bounds.
// -----------------------------------------------------------------------------

// quantiles in [0, 1] (NaN rejected) and, for binned paths, bin_offsets/bin_edges
// consistent with X.shape[1].
static void check_fast_inputs(
    int n_features,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& quantiles,
    const py::array_t<double, py::array::c_style | py::array::forcecast>* bin_edges,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>* bin_offsets,
    int max_bins)
{
    const double* q = quantiles.data();
    for (py::ssize_t i = 0; i < quantiles.shape(0); ++i)
        if (!(q[i] >= 0.0 && q[i] <= 1.0))
            throw std::invalid_argument("quantiles must lie in [0, 1]");

    if (bin_offsets == nullptr || bin_edges == nullptr) return;
    if (max_bins < 1)
        throw std::invalid_argument("max_bins must be >= 1");
    if (bin_offsets->shape(0) != static_cast<py::ssize_t>(n_features) + 1)
        throw std::invalid_argument(
            "bin_offsets must have length X.shape[1] + 1 (got " +
            std::to_string(bin_offsets->shape(0)) + " for " +
            std::to_string(n_features) + " features)");
    const int32_t* off = bin_offsets->data();
    const py::ssize_t n_edges = bin_edges->shape(0);
    if (off[0] < 0)
        throw std::invalid_argument("bin_offsets must be non-negative");
    for (int f = 0; f < n_features; ++f)
        if (off[f + 1] < off[f])
            throw std::invalid_argument("bin_offsets must be non-decreasing");
    if (off[n_features] > n_edges)
        throw std::invalid_argument("bin_offsets exceed the length of bin_edges");
}

// Internal nodes must reference a split feature in [0, n_feat_limit) and children
// in [0, n_nodes).
static void check_tree_topology(const py::dict& d, int n_feat_limit)
{
    auto is_leaf = d["is_leaf"].cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
    auto sfeat   = d["split_feature_idx"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    auto lch     = d["left_child_id"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    auto rch     = d["right_child_id"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    const py::ssize_t n_nodes = is_leaf.shape(0);
    if (sfeat.shape(0) < n_nodes || lch.shape(0) < n_nodes || rch.shape(0) < n_nodes)
        throw std::invalid_argument("tree arrays have inconsistent lengths");
    const uint8_t* il = is_leaf.data();
    const int32_t* sf = sfeat.data();
    const int32_t* lc = lch.data();
    const int32_t* rc = rch.data();
    for (py::ssize_t i = 0; i < n_nodes; ++i) {
        if (il[i]) continue;
        if (sf[i] < 0 || sf[i] >= n_feat_limit)
            throw std::invalid_argument(
                "split_feature_idx[" + std::to_string(i) + "] = " + std::to_string(sf[i]) +
                " is out of range for X with " + std::to_string(n_feat_limit) + " features");
        if (lc[i] < 0 || lc[i] >= n_nodes || rc[i] < 0 || rc[i] >= n_nodes)
            throw std::invalid_argument(
                "child id of node " + std::to_string(i) + " is out of range");
    }
}

static void py_check_forest_entry(const py::dict& d, const int32_t* feat_idx,
                                  int n_tree_features, int n_features_global)
{
    if (feat_idx != nullptr) {
        for (int i = 0; i < n_tree_features; ++i)
            if (feat_idx[i] < 0 || feat_idx[i] >= n_features_global)
                throw std::invalid_argument(
                    "feature_indices[" + std::to_string(i) + "] = " +
                    std::to_string(feat_idx[i]) + " is out of range for X with " +
                    std::to_string(n_features_global) + " features");
        check_tree_topology(d, n_tree_features);
    } else {
        check_tree_topology(d, n_features_global);
    }
}

static py::dict py_build_tree(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> y,
    int n_bins,
    int max_depth,
    int min_samples_leaf,
    double min_divergence_decrease,
    const std::string& divergence_name,
    py::object max_splits_obj,  // Phase 5: optional int32 array, or None
    const std::string& split_weighting_name)  // 'none' | 'sqrt' | 'crps'
{
    const ddt::SplitWeighting split_weighting = ddt::parse_split_weighting(split_weighting_name);
    if (X.ndim() != 2) throw std::invalid_argument("X must be a 2D array (N x F)");
    if (y.ndim() != 1) throw std::invalid_argument("y must be a 1D array (N,)");

    const int n_samples = static_cast<int>(X.shape(0));
    const int n_features = static_cast<int>(X.shape(1));

    if (y.shape(0) != n_samples) throw std::invalid_argument("X and y sample mismatch");

    // Phase 5: resolve optional max_splits_per_feature array.
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> max_splits_arr;
    const int* max_splits_ptr = nullptr;
    if (!max_splits_obj.is_none()) {
        max_splits_arr = max_splits_obj.cast<
            py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        if (max_splits_arr.ndim() != 1)
            throw std::invalid_argument("max_splits_per_feature must be a 1D array");
        if (max_splits_arr.shape(0) != n_features)
            throw std::invalid_argument(
                "max_splits_per_feature length must equal n_features");
        max_splits_ptr = max_splits_arr.data(0);
    }

    auto X_acc = X.unchecked<2>();
    auto y_acc = y.unchecked<1>();

    // GIL is released for the pure-C++ build; all Python buffers are held by
    // the local arrays above and no Python API is touched inside the scope.
    auto tree = [&]() {
        py::gil_scoped_release release;
        return ddt::build_tree(
            X_acc.data(0, 0), y_acc.data(0), n_samples, n_features, n_bins,
            max_depth, min_samples_leaf, min_divergence_decrease,
            divergence_name, max_splits_ptr, split_weighting
        );
    }();

    return tree_to_py(std::move(tree));
}

static py::dict py_build_tree_weighted(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> delta_x_norm,
    int n_bins,
    int max_depth,
    int min_samples_leaf,
    double min_divergence_decrease,
    const std::string& divergence_name,
    py::object max_splits_obj,  // Phase 5: optional int32 array, or None
    const std::string& split_weighting_name)  // 'none' | 'sqrt' | 'crps'
{
    const ddt::SplitWeighting split_weighting = ddt::parse_split_weighting(split_weighting_name);
    if (X.ndim() != 2) throw std::invalid_argument("X must be a 2D array (N x F)");
    if (y.ndim() != 1) throw std::invalid_argument("y must be a 1D array (N,)");
    if (delta_x_norm.ndim() != 1) throw std::invalid_argument("delta_x_norm must be 1D");
    if (delta_x_norm.shape(0) != n_bins)
        throw std::invalid_argument("delta_x_norm length must equal n_bins");

    const int n_samples = static_cast<int>(X.shape(0));
    const int n_features = static_cast<int>(X.shape(1));

    if (y.shape(0) != n_samples) throw std::invalid_argument("X and y sample mismatch");

    // Phase 5: resolve optional max_splits_per_feature array.
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> max_splits_arr_w;
    const int* max_splits_ptr_w = nullptr;
    if (!max_splits_obj.is_none()) {
        max_splits_arr_w = max_splits_obj.cast<
            py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        if (max_splits_arr_w.ndim() != 1)
            throw std::invalid_argument("max_splits_per_feature must be a 1D array");
        if (max_splits_arr_w.shape(0) != n_features)
            throw std::invalid_argument(
                "max_splits_per_feature length must equal n_features");
        max_splits_ptr_w = max_splits_arr_w.data(0);
    }

    auto X_acc = X.unchecked<2>();
    auto y_acc = y.unchecked<1>();
    auto dx_acc = delta_x_norm.unchecked<1>();

    auto tree = [&]() {
        py::gil_scoped_release release;
        return ddt::build_tree_weighted(
            X_acc.data(0, 0), y_acc.data(0), n_samples, n_features, n_bins,
            max_depth, min_samples_leaf, min_divergence_decrease,
            divergence_name, dx_acc.data(0), max_splits_ptr_w, split_weighting
        );
    }();

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

    {
        py::gil_scoped_release release;
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

static double py_calculate_wasserstein(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> left_counts,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> right_counts)
{
    auto l = left_counts.unchecked<1>();
    auto r = right_counts.unchecked<1>();
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

    {
        py::gil_scoped_release release;
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
    const CompiledGrid& cg)
{
    if (X.ndim() != 2) throw std::invalid_argument("X must be a 2D array (N x F)");
    if (bin_edges.ndim() != 1) throw std::invalid_argument("bin_edges must be 1D");
    if (bin_offsets.ndim() != 1) throw std::invalid_argument("bin_offsets must be 1D");
    if (quantiles.ndim() != 1) throw std::invalid_argument("quantiles must be 1D");

    const int n_samples = static_cast<int>(X.shape(0));
    const int n_features = static_cast<int>(X.shape(1));
    const int num_quantiles = static_cast<int>(quantiles.shape(0));

    check_fast_inputs(n_features, quantiles, &bin_edges, &bin_offsets, max_bins);
    check_tree_topology(tree_data, n_features);
    ddt::TreeView tree = py_to_tree_view(tree_data);
    cg.check(tree.n_bins);
    const ddt::QuantileGrid& grid = cg.grid;
    const int n_target_bins = cg.B;

    auto X_acc = X.unchecked<2>();
    auto edges = bin_edges.unchecked<1>();
    auto offsets = bin_offsets.unchecked<1>();
    auto q_acc = quantiles.unchecked<1>();

    py::array_t<double> result({num_quantiles, n_samples});
    auto result_mut = result.mutable_unchecked<2>();

    {
        py::gil_scoped_release release;
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
            grid,
            n_target_bins,
            result_mut.mutable_data(0, 0)
        );
    }

    return result;
}

static py::array_t<double> py_predict_quantiles_fast_float(
    const py::dict& tree_data,
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> quantiles,
    const CompiledGrid& gh)
{
    if (X.ndim() != 2) throw std::invalid_argument("X must be a 2D array (N x F)");
    if (quantiles.ndim() != 1) throw std::invalid_argument("quantiles must be 1D");
    if (!tree_data.contains("split_threshold_float"))
        throw std::runtime_error(
            "tree_data is missing 'split_threshold_float'. "
            "Call model.compute_float_thresholds() before using fast_inference=True.");

    const int n_samples = static_cast<int>(X.shape(0));
    const int n_features = static_cast<int>(X.shape(1));
    const int num_quantiles = static_cast<int>(quantiles.shape(0));

    check_fast_inputs(n_features, quantiles, nullptr, nullptr, 1);
    check_tree_topology(tree_data, n_features);
    ddt::TreeViewFloat tree = py_to_tree_view_float(tree_data);
    gh.check(tree.n_bins);
    const int n_target_bins = gh.B;

    auto X_acc = X.unchecked<2>();
    auto q_acc = quantiles.unchecked<1>();

    py::array_t<double> result({num_quantiles, n_samples});
    auto result_mut = result.mutable_unchecked<2>();

    {
        py::gil_scoped_release release;
        ddt::predict_quantiles_fast_float(
            tree,
            X_acc.data(0, 0),
            n_samples,
            n_features,
            q_acc.data(0),
            num_quantiles,
            gh.grid,
            n_target_bins,
            result_mut.mutable_data(0, 0)
        );
    }

    return result;
}


// =============================================================================
// Phase 6: C++ Dirichlet Smoothing Engine — Pybind Wrapper
// =============================================================================

static py::array_t<double> py_smooth_tree(
    const py::dict& tree_data,
    double prior_weight,
    int min_ancestor_samples)
{
    // Pipeline stage guard
    if (tree_data.contains("_pipeline_stage")) {
        int stage = tree_data["_pipeline_stage"].cast<int>();
        if (stage >= 2) {  // SMOOTHED or later
            throw std::runtime_error(
                "Cannot smooth a tree at pipeline stage >= SMOOTHED. "
                "Strict order: FITTED -> PRUNED -> SMOOTHED -> EVT_FITTED -> CALIBRATED.");
        }
    }

    auto dist_arr  = tree_data["distribution_counts"].cast<py::array_t<int32_t>>();
    auto tot_arr   = tree_data["total_samples"].cast<py::array_t<int32_t>>();
    auto leaf_arr  = tree_data["is_leaf"].cast<py::array_t<uint8_t>>();
    auto left_arr  = tree_data["left_child_id"].cast<py::array_t<int32_t>>();
    auto right_arr = tree_data["right_child_id"].cast<py::array_t<int32_t>>();

    if (dist_arr.ndim() != 2)
        throw std::invalid_argument("distribution_counts must be 2D (n_nodes, n_bins)");

    const int n_nodes = static_cast<int>(dist_arr.shape(0));
    const int n_bins  = static_cast<int>(dist_arr.shape(1));

    auto flat = [&]() {
        py::gil_scoped_release release;
        return ddt::smooth_tree(
            dist_arr.data(0, 0),
            tot_arr.data(0),
            leaf_arr.data(0),
            left_arr.data(0),
            right_arr.data(0),
            n_nodes,
            n_bins,
            prior_weight,
            min_ancestor_samples
        );
    }();

    // Return as (n_nodes, n_bins) float64 array (zero-copy via capsule)
    auto* vec_ptr = new std::vector<double>(std::move(flat));
    auto capsule = py::capsule(vec_ptr, [](void* p) {
        delete reinterpret_cast<std::vector<double>*>(p);
    });
    return py::array_t<double>(
        {n_nodes, n_bins},
        {static_cast<py::ssize_t>(n_bins * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        vec_ptr->data(),
        capsule
    );
}

// =============================================================================
// Phase 6: Smooth-PMF Fast Inference — Pybind Wrapper
// =============================================================================

static py::array_t<double> py_predict_quantiles_fast_smooth(
    const py::dict& tree_data,
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> bin_edges,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> bin_offsets,
    int max_bins,
    py::array_t<double, py::array::c_style | py::array::forcecast> quantiles,
    const CompiledGrid& gh)
{
    if (X.ndim() != 2)           throw std::invalid_argument("X must be 2D (N x F)");
    if (bin_edges.ndim() != 1)   throw std::invalid_argument("bin_edges must be 1D");
    if (bin_offsets.ndim() != 1) throw std::invalid_argument("bin_offsets must be 1D");
    if (quantiles.ndim() != 1)   throw std::invalid_argument("quantiles must be 1D");

    // Accept smoothed_pmf (float64) or inference_pmf (float16/float32 upcast)
    py::array_t<double, py::array::c_style | py::array::forcecast> pmf_arr;
    if (tree_data.contains("smoothed_pmf")) {
        pmf_arr = tree_data["smoothed_pmf"].cast<
            py::array_t<double, py::array::c_style | py::array::forcecast>>();
    } else if (tree_data.contains("inference_pmf")) {
        pmf_arr = tree_data["inference_pmf"].cast<
            py::array_t<double, py::array::c_style | py::array::forcecast>>();
    } else {
        throw std::runtime_error(
            "tree_data contains neither 'smoothed_pmf' nor 'inference_pmf'. "
            "Call smooth_tree() before predict_quantiles_fast_smooth().");
    }

    if (pmf_arr.ndim() != 2)
        throw std::invalid_argument("smoothed_pmf / inference_pmf must be 2D (n_nodes, n_bins)");

    const int n_samples     = static_cast<int>(X.shape(0));
    const int n_features    = static_cast<int>(X.shape(1));
    const int num_quantiles = static_cast<int>(quantiles.shape(0));

    check_fast_inputs(n_features, quantiles, &bin_edges, &bin_offsets, max_bins);
    check_tree_topology(tree_data, n_features);
    ddt::TreeViewSmooth tree = py_to_tree_view_smooth(tree_data, pmf_arr);
    gh.check(tree.n_bins);
    const int n_target_bins = gh.B;

    auto X_acc   = X.unchecked<2>();
    auto edges   = bin_edges.unchecked<1>();
    auto offsets = bin_offsets.unchecked<1>();
    auto q_acc   = quantiles.unchecked<1>();

    py::array_t<double> result({num_quantiles, n_samples});
    auto result_mut = result.mutable_unchecked<2>();

    {
        py::gil_scoped_release release;
        ddt::predict_quantiles_fast_smooth(
            tree,
            X_acc.data(0, 0),
            n_samples,
            n_features,
            edges.data(0),
            offsets.data(0),
            max_bins,
            q_acc.data(0),
            num_quantiles,
            gh.grid,
            n_target_bins,
            result_mut.mutable_data(0, 0)
        );
    }

    return result;
}

// =============================================================================
// // =============================================================================
// Inverts the stored leaf distributions for caller-supplied leaf ids, skipping
// tree traversal.  It instantiates exactly the quantiles_for_leaf<PMF, EVT>
// specialisation the corresponding traversal binding would use, so calibration
// (which already knows each calibration sample's leaf) is computed with the very
// same predictor it later corrects.
//
// All arrays are validated before the GIL is released; no unchecked indexing:
//   - every leaf id must be in [0, n_nodes) and refer to a leaf node,
//   - the grid must have B >= 1 bins and B <= the PMF row width,
//   - per-node PMF / EVT arrays must cover all n_nodes nodes.

static py::array_t<double> py_predict_quantiles_from_leaves(
    const py::dict& tree_data,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> leaf_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> quantiles,
    py::array_t<double, py::array::c_style | py::array::forcecast> bin_lo,
    py::array_t<double, py::array::c_style | py::array::forcecast> bin_width,
    py::array_t<double, py::array::c_style | py::array::forcecast> bin_rep,
    bool use_smooth,
    bool use_evt)
{
    if (leaf_ids.ndim() != 1)  throw std::invalid_argument("leaf_ids must be 1D");
    if (quantiles.ndim() != 1) throw std::invalid_argument("quantiles must be 1D");

    GridHolder gh = parse_grid(bin_lo, bin_width, bin_rep, /*max_B=*/-1);
    const int B = gh.B;

    const int n_leaves = static_cast<int>(leaf_ids.shape(0));
    const int Q        = static_cast<int>(quantiles.shape(0));
    for (int qi = 0; qi < Q; ++qi) {
        const double q = quantiles.data()[qi];
        if (!(q >= 0.0 && q <= 1.0))
            throw std::invalid_argument("quantiles must lie in [0, 1]");
    }

    // ---- tree structure: leaf-id validation ----
    auto is_leaf_arr = tree_data["is_leaf"].cast<py::array_t<uint8_t>>();
    const int n_nodes = static_cast<int>(is_leaf_arr.shape(0));
    const uint8_t* is_leaf = is_leaf_arr.data();
    const int32_t* ids = leaf_ids.data();
    for (int i = 0; i < n_leaves; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || id >= n_nodes)
            throw std::invalid_argument(
                "leaf_ids[" + std::to_string(i) + "] = " + std::to_string(id) +
                " is out of range [0, " + std::to_string(n_nodes) + ")");
        if (!is_leaf[id])
            throw std::invalid_argument(
                "leaf_ids[" + std::to_string(i) + "] = " + std::to_string(id) +
                " is not a leaf node");
    }

    // ---- body distribution ----
    py::array_t<double, py::array::c_style | py::array::forcecast> pmf_arr;
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> counts_arr;
    int row_width = 0;
    if (use_smooth) {
        if (tree_data.contains("smoothed_pmf")) {
            pmf_arr = tree_data["smoothed_pmf"].cast<
                py::array_t<double, py::array::c_style | py::array::forcecast>>();
        } else if (tree_data.contains("inference_pmf")) {
            pmf_arr = tree_data["inference_pmf"].cast<
                py::array_t<double, py::array::c_style | py::array::forcecast>>();
        } else {
            throw std::runtime_error(
                "use_smooth=True but tree_data has neither 'smoothed_pmf' nor 'inference_pmf'.");
        }
        if (pmf_arr.ndim() != 2)
            throw std::invalid_argument("smoothed_pmf / inference_pmf must be 2D (n_nodes, n_bins)");
        if (static_cast<int>(pmf_arr.shape(0)) < n_nodes)
            throw std::invalid_argument("smoothed_pmf / inference_pmf has fewer rows than is_leaf has nodes");
        row_width = static_cast<int>(pmf_arr.shape(1));
    } else {
        counts_arr = tree_data["distribution_counts"].cast<
            py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        if (counts_arr.ndim() != 2)
            throw std::invalid_argument("distribution_counts must be 2D (n_nodes, n_bins)");
        if (static_cast<int>(counts_arr.shape(0)) < n_nodes)
            throw std::invalid_argument("distribution_counts has fewer rows than is_leaf has nodes");
        row_width = static_cast<int>(counts_arr.shape(1));
    }
    if (B > row_width)
        throw std::invalid_argument("bin_lo/bin_width/bin_rep are longer than the tree's n_bins");

    // ---- EVT arrays must cover every node ----
    if (use_evt) {
        static const char* const evt_keys[] = {
            "evt_enabled", "evt_threshold_u", "evt_F_u", "evt_gpd_shape", "evt_gpd_scale",
            "evt_S_u", "evt_lower_enabled", "evt_lower_threshold_u", "evt_lower_F_u",
            "evt_lower_gpd_shape", "evt_lower_gpd_scale"};
        for (const char* key : evt_keys) {
            if (!tree_data.contains(key)) continue;
            py::array a = tree_data[key].cast<py::array>();
            if (a.ndim() != 1 || static_cast<int>(a.shape(0)) < n_nodes)
                throw std::invalid_argument(
                    std::string("'") + key + "' must be a 1D array covering all nodes");
        }
    }

    py::array_t<double> result({n_leaves, Q});
    if (n_leaves == 0 || Q == 0) return result;
    double* out = result.mutable_data();
    const double* q_ptr = quantiles.data();
    const ddt::QuantileGrid& grid = gh.grid;

    auto run = [&](const auto& pmf, const auto& evt) {
        const double pop_mean = pmf.root_mean(B, grid.bin_rep);
        py::gil_scoped_release release;
        ddt::predict_quantiles_from_leaves(
            pmf, evt, ids, n_leaves, q_ptr, Q, grid, B, pop_mean, out);
    };

    if (use_smooth) {
        ddt::SmoothPMF pmf{pmf_arr.data(), row_width};
        if (use_evt) {
            ddt::TreeViewSmoothEVT v = py_to_tree_view_smooth_evt(tree_data, pmf_arr);
            run(pmf, ddt::make_with_evt(v));
        } else {
            run(pmf, ddt::NoEVT{});
        }
    } else {
        ddt::CountsPMF pmf{counts_arr.data(), row_width};
        if (use_evt) {
            ddt::TreeViewEVT v = py_to_tree_view_evt(tree_data);
            run(pmf, ddt::make_with_evt(v));
        } else {
            run(pmf, ddt::NoEVT{});
        }
    }
    return result;
}

// =============================================================================
// Module Definition
// =============================================================================

PYBIND11_MODULE(_ddt_core, m) {
    m.doc() = "DDT Core Engine — C++17 / Pybind11 bridge for Distributional Decision Trees";

    m.def("build_tree", &py_build_tree,
        py::arg("X"), py::arg("y"), py::arg("n_bins"),
        py::arg("max_depth") = 10, py::arg("min_samples_leaf") = 20,
        py::arg("min_divergence_decrease") = 0.01,
        py::arg("divergence") = "wasserstein",
        py::arg("max_splits_per_feature") = py::none(),
        py::arg("split_weighting") = "none");

    m.def("build_tree_weighted", &py_build_tree_weighted,
        py::arg("X"), py::arg("y"), py::arg("delta_x_norm"), py::arg("n_bins"),
        py::arg("max_depth") = 10, py::arg("min_samples_leaf") = 20,
        py::arg("min_divergence_decrease") = 0.01,
        py::arg("divergence") = "wasserstein",
        py::arg("max_splits_per_feature") = py::none(),
        py::arg("split_weighting") = "none",
        R"doc(
        Build a DDT tree using variable-width bin Wasserstein (weighted path).

        Parameters
        ----------
        X : ndarray uint8 (N, F)   Quantized features
        y : ndarray int32 (N,)     Quantized target in [0, n_bins_active-1]
        delta_x_norm : ndarray float64 (n_bins,)  Normalised bin widths (mean=1.0)
        n_bins : int               Number of active bins
        max_splits_per_feature : ndarray int32 (n_features,) or None
            Per-feature upper bound on the number of splits across the entire tree.
            When None (default), all features are unconstrained.
        )doc");

    m.def("predict_leaves", &py_predict_leaves,
        py::arg("tree_data"), py::arg("X"));

    m.def("get_node_distribution", &py_get_node_distribution,
        py::arg("tree_data"), py::arg("node_idx"));

    m.def("calculate_wasserstein", &py_calculate_wasserstein,
        py::arg("left_counts"), py::arg("right_counts"));

    m.def("quantize_features", &py_quantize_features,
        py::arg("X"), py::arg("bin_edges"), py::arg("bin_offsets"), py::arg("max_bins"));

    py::class_<CompiledGrid>(m, "CompiledGrid",
        "Quantile grid (bin_lo, bin_width, bin_rep) validated once; owns internal copies.")
        .def(py::init<const DoubleArr&, const DoubleArr&, const DoubleArr&>(),
             py::arg("bin_lo"), py::arg("bin_width"), py::arg("bin_rep"))
        .def_property_readonly("n_bins", [](const CompiledGrid& g) { return g.B; });

    // Prepared-grid overload (hot path) is registered first so it is tried first.
    m.def("predict_quantiles_fast", &py_predict_quantiles_fast,
        py::arg("tree_data"), py::arg("X"), py::arg("bin_edges"),
        py::arg("bin_offsets"), py::arg("max_bins"),
        py::arg("quantiles"), py::arg("grid"));
    m.def("predict_quantiles_fast",
        [](const py::dict& td, DoubleArr X, DoubleArr be,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bo,
           int mb, DoubleArr q, DoubleArr lo, DoubleArr w, DoubleArr rep) {
            CompiledGrid cg(lo, w, rep);
            return py_predict_quantiles_fast(td, X, be, bo, mb, q, cg);
        },
        py::arg("tree_data"), py::arg("X"), py::arg("bin_edges"), 
        py::arg("bin_offsets"), py::arg("max_bins"), 
        py::arg("quantiles"), py::arg("bin_lo"), py::arg("bin_width"), py::arg("bin_rep"),
        R"doc(
        Fast quantile prediction (uint8 bin traversal, raw-count PMF).

        , all
        required) define the quantile grid; quantiles are obtained by edge-based
        inversion (uniform within bin).
        )doc");

    m.def("predict_quantiles_fast_float", &py_predict_quantiles_fast_float,
        py::arg("tree_data"), py::arg("X"),
        py::arg("quantiles"), py::arg("grid"));
    m.def("predict_quantiles_fast_float",
        [](const py::dict& td, DoubleArr X, DoubleArr q,
           DoubleArr lo, DoubleArr w, DoubleArr rep) {
            CompiledGrid cg(lo, w, rep);
            return py_predict_quantiles_fast_float(td, X, q, cg);
        },
        py::arg("tree_data"), py::arg("X"),
        py::arg("quantiles"), py::arg("bin_lo"), py::arg("bin_width"), py::arg("bin_rep"),
        R"doc(
        Fast quantile prediction using pre-computed float64 split thresholds.

        Requires tree_data to contain 'split_threshold_float' (float64 array of
        shape (n_nodes,)) produced by DDTRegressor.compute_float_thresholds().
        The traversal loop uses a single float comparison per level — no bin_edges
        or bin_offsets arrays are needed, minimising memory indirection.

        Parameters
        ----------
        tree_data : dict   Tree dict with 'split_threshold_float' populated.
        X : ndarray float64 (N, F)   Raw (unquantized) feature matrix.
        quantiles : ndarray float64 (Q,)   Target quantiles.
        bin_lo, bin_width, bin_rep : ndarray float64 (B,)  Quantile grid (original space).

        Returns
        -------
        ndarray float64 (Q, N)   Predicted quantiles.
        )doc");

    m.def("smooth_tree", &py_smooth_tree,
        py::arg("tree_data"), py::arg("prior_weight"), py::arg("min_ancestor_samples"),
        R"doc(
        Phase 6: C++ Dirichlet smoothing engine.

        Applies hierarchical Empirical Bayes (Dirichlet-Multinomial) smoothing
        to all terminal leaf PMFs.  For each leaf, walks up the tree to find the
        first ancestor with >= min_ancestor_samples and blends:

            theta_smooth = w * p_leaf + (1-w) * p_ancestor,  w = N / (N + lambda)

        Parameters
        ----------
        tree_data : dict   Serialised tree dict (must contain distribution_counts,
                           total_samples, is_leaf, left_child_id, right_child_id).
        prior_weight : float   Dirichlet concentration lambda (>= 0).
        min_ancestor_samples : int  Minimum samples for an ancestor to qualify as prior.

        Returns
        -------
        ndarray float64 (n_nodes, n_bins)  Smoothed PMF array.
        )doc");

    m.def("predict_quantiles_fast_smooth", &py_predict_quantiles_fast_smooth,
        py::arg("tree_data"), py::arg("X"), py::arg("bin_edges"),
        py::arg("bin_offsets"), py::arg("max_bins"),
        py::arg("quantiles"), py::arg("grid"));
    m.def("predict_quantiles_fast_smooth",
        [](const py::dict& td, DoubleArr X, DoubleArr be,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bo,
           int mb, DoubleArr q, DoubleArr lo, DoubleArr w, DoubleArr rep) {
            CompiledGrid cg(lo, w, rep);
            return py_predict_quantiles_fast_smooth(td, X, be, bo, mb, q, cg);
        },
        py::arg("tree_data"), py::arg("X"), py::arg("bin_edges"),
        py::arg("bin_offsets"), py::arg("max_bins"),
        py::arg("quantiles"), py::arg("bin_lo"), py::arg("bin_width"), py::arg("bin_rep"),
        R"doc(
        Phase 6: Fast quantile prediction from smoothed float64 PMF.

        Reads from tree_data['smoothed_pmf'] or tree_data['inference_pmf'].
        Traversal is identical to predict_quantiles_fast; the quantile inner
        loop reads float64 PMFs directly (no int32->float cast, no division).
        This restores C++ fast inference for smooth_leaves=True models.

        Parameters
        ----------
        tree_data : dict   Tree dict with 'smoothed_pmf' or 'inference_pmf'.
        X : ndarray float64 (N, F)   Raw feature matrix.
        bin_edges : ndarray float64 (total_edges,)   Flat bin edges array.
        bin_offsets : ndarray int32 (n_features+1,)  Per-feature edge offsets.
        max_bins : int   Maximum number of bins per feature.
        quantiles : ndarray float64 (Q,)   Target quantiles in [0, 1].
        bin_lo, bin_width, bin_rep : ndarray float64 (B,)  Quantile grid.

        Returns
        -------
        ndarray float64 (Q, N)   Predicted quantiles.
        )doc");

    // =========================================================================
    // Phase 4a: EVT fast-inference wrappers
    // =========================================================================

    auto evt_fn =
        [](const py::dict& tree_data,
           py::array_t<double, py::array::c_style | py::array::forcecast> X,
           py::array_t<double, py::array::c_style | py::array::forcecast> bin_edges,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bin_offsets,
           int max_bins,
           py::array_t<double, py::array::c_style | py::array::forcecast> quantiles,
           const CompiledGrid& gh)
           -> py::array_t<double>
        {
            if (X.ndim() != 2)           throw std::invalid_argument("X must be 2D (N x F)");
            if (bin_edges.ndim() != 1)   throw std::invalid_argument("bin_edges must be 1D");
            if (bin_offsets.ndim() != 1) throw std::invalid_argument("bin_offsets must be 1D");
            if (quantiles.ndim() != 1)   throw std::invalid_argument("quantiles must be 1D");

            const int n_samples     = static_cast<int>(X.shape(0));
            const int n_features    = static_cast<int>(X.shape(1));
            const int num_quantiles = static_cast<int>(quantiles.shape(0));

            check_fast_inputs(n_features, quantiles, &bin_edges, &bin_offsets, max_bins);
            check_tree_topology(tree_data, n_features);
            ddt::TreeViewEVT tree = py_to_tree_view_evt(tree_data);
            gh.check(tree.n_bins);
            const int n_target_bins = gh.B;

            auto X_acc   = X.unchecked<2>();
            auto edges   = bin_edges.unchecked<1>();
            auto offsets = bin_offsets.unchecked<1>();
            auto q_acc   = quantiles.unchecked<1>();

            py::array_t<double> result({num_quantiles, n_samples});
            auto result_mut = result.mutable_unchecked<2>();

            {
                py::gil_scoped_release release;
                ddt::predict_quantiles_fast_evt(
                    tree,
                    X_acc.data(0, 0),
                    n_samples, n_features,
                    edges.data(0), offsets.data(0), max_bins,
                    q_acc.data(0), num_quantiles,
                    gh.grid, n_target_bins,
                    result_mut.mutable_data(0, 0));
            }

            return result;
        };
    m.def("predict_quantiles_fast_evt", evt_fn,
        py::arg("tree_data"), py::arg("X"), py::arg("bin_edges"),
        py::arg("bin_offsets"), py::arg("max_bins"),
        py::arg("quantiles"), py::arg("grid"));
    m.def("predict_quantiles_fast_evt",
        [evt_fn](const py::dict& td, DoubleArr X, DoubleArr be,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bo,
           int mb, DoubleArr q, DoubleArr lo, DoubleArr w, DoubleArr rep) {
            CompiledGrid cg(lo, w, rep);
            return evt_fn(td, X, be, bo, mb, q, cg);
        },
        py::arg("tree_data"), py::arg("X"), py::arg("bin_edges"),
        py::arg("bin_offsets"), py::arg("max_bins"),
        py::arg("quantiles"), py::arg("bin_lo"), py::arg("bin_width"), py::arg("bin_rep"),
        R"doc(
        Phase 2b: Fast quantile prediction with EVT tail splice (distribution_counts body).

        Routes each (sample, quantile) to:
          - GPD upper tail  when q >= evt_F_u[leaf] and evt_enabled[leaf]
          - GPD lower tail  when q <= evt_lower_F_u[leaf] and lower arrays present
          - Standard CDF walk otherwise

        Requires tree_data to contain evt_enabled, evt_threshold_u, evt_F_u,
        evt_gpd_shape, evt_gpd_scale.  Optional: evt_S_u (survival precision),
        evt_lower_* arrays (lower tail).

        Returns ndarray float64 (Q, N).
        )doc");

    auto smooth_evt_fn =
        [](const py::dict& tree_data,
           py::array_t<double, py::array::c_style | py::array::forcecast> X,
           py::array_t<double, py::array::c_style | py::array::forcecast> bin_edges,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bin_offsets,
           int max_bins,
           py::array_t<double, py::array::c_style | py::array::forcecast> quantiles,
           const CompiledGrid& gh)
           -> py::array_t<double>
        {
            if (X.ndim() != 2)           throw std::invalid_argument("X must be 2D (N x F)");
            if (bin_edges.ndim() != 1)   throw std::invalid_argument("bin_edges must be 1D");
            if (bin_offsets.ndim() != 1) throw std::invalid_argument("bin_offsets must be 1D");
            if (quantiles.ndim() != 1)   throw std::invalid_argument("quantiles must be 1D");

            // Accept smoothed_pmf or inference_pmf (forcecast to float64)
            py::array_t<double, py::array::c_style | py::array::forcecast> pmf_arr;
            if (tree_data.contains("smoothed_pmf")) {
                pmf_arr = tree_data["smoothed_pmf"].cast<
                    py::array_t<double, py::array::c_style | py::array::forcecast>>();
            } else if (tree_data.contains("inference_pmf")) {
                pmf_arr = tree_data["inference_pmf"].cast<
                    py::array_t<double, py::array::c_style | py::array::forcecast>>();
            } else {
                throw std::runtime_error(
                    "tree_data contains neither 'smoothed_pmf' nor 'inference_pmf'. "
                    "Call smooth_tree() before predict_quantiles_fast_smooth_evt().");
            }
            if (pmf_arr.ndim() != 2)
                throw std::invalid_argument(
                    "smoothed_pmf / inference_pmf must be 2D (n_nodes, n_bins)");

            const int n_samples     = static_cast<int>(X.shape(0));
            const int n_features    = static_cast<int>(X.shape(1));
            const int num_quantiles = static_cast<int>(quantiles.shape(0));

            check_fast_inputs(n_features, quantiles, &bin_edges, &bin_offsets, max_bins);
            check_tree_topology(tree_data, n_features);
            ddt::TreeViewSmoothEVT tree = py_to_tree_view_smooth_evt(tree_data, pmf_arr);
            gh.check(tree.n_bins);
            const int n_target_bins = gh.B;

            auto X_acc   = X.unchecked<2>();
            auto edges   = bin_edges.unchecked<1>();
            auto offsets = bin_offsets.unchecked<1>();
            auto q_acc   = quantiles.unchecked<1>();

            py::array_t<double> result({num_quantiles, n_samples});
            auto result_mut = result.mutable_unchecked<2>();

            {
                py::gil_scoped_release release;
                ddt::predict_quantiles_fast_smooth_evt(
                    tree,
                    X_acc.data(0, 0),
                    n_samples, n_features,
                    edges.data(0), offsets.data(0), max_bins,
                    q_acc.data(0), num_quantiles,
                    gh.grid, n_target_bins,
                    result_mut.mutable_data(0, 0));
            }

            return result;
        };
    m.def("predict_quantiles_fast_smooth_evt", smooth_evt_fn,
        py::arg("tree_data"), py::arg("X"), py::arg("bin_edges"),
        py::arg("bin_offsets"), py::arg("max_bins"),
        py::arg("quantiles"), py::arg("grid"));
    m.def("predict_quantiles_fast_smooth_evt",
        [smooth_evt_fn](const py::dict& td, DoubleArr X, DoubleArr be,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bo,
           int mb, DoubleArr q, DoubleArr lo, DoubleArr w, DoubleArr rep) {
            CompiledGrid cg(lo, w, rep);
            return smooth_evt_fn(td, X, be, bo, mb, q, cg);
        },
        py::arg("tree_data"), py::arg("X"), py::arg("bin_edges"),
        py::arg("bin_offsets"), py::arg("max_bins"),
        py::arg("quantiles"), py::arg("bin_lo"), py::arg("bin_width"), py::arg("bin_rep"),
        R"doc(
        Phase 2c: Fast quantile prediction with EVT tail splice (smoothed PMF body).

        Same routing as predict_quantiles_fast_evt but reads float64 smoothed_pmf
        (or inference_pmf) for the body CDF walk.  Enables full C++ inference for
        the smooth_leaves=True + evt_tails=True combination.

        Requires tree_data to contain smoothed_pmf or inference_pmf plus the EVT
        arrays listed in predict_quantiles_fast_evt.

        Returns ndarray float64 (Q, N).
        )doc");


    // =========================================================================
    //     // =========================================================================

    m.def("predict_quantiles_from_leaves", &py_predict_quantiles_from_leaves,
        py::arg("tree_data"), py::arg("leaf_ids"), py::arg("quantiles"),
        py::arg("bin_lo"), py::arg("bin_width"), py::arg("bin_rep"),
        py::arg("use_smooth"), py::arg("use_evt"),
        R"doc(
        .

        Runs the same per-leaf routine as predict_quantiles_fast / _smooth / _evt /
        _smooth_evt (selected by use_smooth and use_evt), so calibration computed
        through this entry point uses exactly the predictor it later corrects.

        Parameters
        ----------
        tree_data : dict
            Standard tree dict.  Needs distribution_counts (use_smooth=False) or
            smoothed_pmf / inference_pmf (use_smooth=True); and the evt_* arrays
            when use_evt=True.
        leaf_ids : ndarray int32 (n,)
            Node indices; every id must be in range and refer to a leaf node
            (otherwise ValueError -- nothing is read out of bounds).
        quantiles : ndarray float64 (Q,)   Quantile levels in [0, 1].
        bin_lo, bin_width, bin_rep : ndarray float64 (B,)
            Quantile grid (see predict_quantiles_fast).  All required.
        use_smooth, use_evt : bool
            Which PMF / EVT policy to apply.

        Returns
        -------
        ndarray float64 (n, Q)   Note the (n, Q) layout, unlike the (Q, N)
        layout of the traversal bindings.
        )doc");
}
