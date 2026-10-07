import numpy as np
import pytest
from scipy.stats import t as student_t

from ddt import DDTRegressor, _ddt_core

# q=0 is excluded from the snap-grid membership checks on purpose: the kernel
# returns the first NON-EMPTY bin at q=0 (spec section 3.2).
QS_BODY = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0])
QS_FULL = np.concatenate([[0.0], QS_BODY])


# ---------------------------------------------------------------------------
# Hand-built single-leaf trees
# ---------------------------------------------------------------------------
def _leaf_tree(counts):
    """A one-node tree (root is a leaf) holding ``counts`` as its distribution."""
    counts = np.asarray(counts, dtype=np.int32)
    B = counts.shape[0]
    return {
        "is_leaf": np.array([1], dtype=np.uint8),
        "split_feature_idx": np.array([0], dtype=np.int32),
        "split_threshold": np.array([0], dtype=np.uint8),
        "left_child_id": np.array([-1], dtype=np.int32),
        "right_child_id": np.array([-1], dtype=np.int32),
        "distribution_counts": counts.reshape(1, B),
        "n_bins": B,
    }


def _invert(counts, lo, width, qs, rep=None):
    lo = np.asarray(lo, dtype=np.float64)
    width = np.asarray(width, dtype=np.float64)
    rep = lo + 0.5 * width if rep is None else np.asarray(rep, dtype=np.float64)
    td = _leaf_tree(counts)
    out = _ddt_core.predict_quantiles_from_leaves(
        td,
        np.zeros(1, dtype=np.int32),
        np.asarray(qs, dtype=np.float64),
        lo,
        width,
        rep,
        False,
        False,
    )
    assert out.shape == (1, len(qs))
    return out[0]


def _invert_ref(counts, lo, width, q):
    """Independent pure-Python implementation of spec section 3.2."""
    C = np.cumsum(np.asarray(counts, dtype=np.float64))
    T = C[-1]
    thr = q * T
    B = len(C)
    if thr <= 0:
        b = int(np.argmax(C > 0))
        frac = 0.0
    else:
        b = int(np.searchsorted(C, thr, side="left"))
        if b >= B:
            b, frac = B - 1, 1.0
        else:
            prev = C[b - 1] if b > 0 else 0.0
            denom = C[b] - prev
            frac = (thr - prev) / denom if denom > 0 else 0.0
            frac = min(max(frac, 0.0), 1.0)
    return lo[b] + frac * width[b]


class TestInversionFormula:
    def test_linear_interpolation_and_empty_bin(self):
        counts = [10, 10, 0, 20]  # C = [10, 20, 20, 40]
        lo = [0.0, 1.0, 2.0, 3.0]
        width = [1.0, 1.0, 1.0, 1.0]
        got = _invert(counts, lo, width, [0.0, 0.25, 0.375, 0.5, 0.75, 1.0])
        # q=0.5 lands exactly on C[1]=20: upper edge of bin 1, NOT inside empty bin 2.
        np.testing.assert_allclose(got, [0.0, 1.0, 1.5, 2.0, 3.5, 4.0], atol=1e-12)

    def test_leading_empty_bins(self):
        got = _invert([0, 0, 5, 5], [0, 1, 2, 3], [1, 1, 1, 1], [0.0, 1.0])
        np.testing.assert_allclose(got, [2.0, 4.0], atol=1e-12)  # lo of first non-empty, hi of last

    def test_trailing_empty_bins(self):
        got = _invert([5, 5, 0, 0], [0, 1, 2, 3], [1, 1, 1, 1], [0.0, 1.0])
        np.testing.assert_allclose(got, [0.0, 2.0], atol=1e-12)  # q=1 -> hi of bin 1, not bin 3

    def test_single_nonempty_bin(self):
        got = _invert([0, 7, 0], [10.0, 20.0, 30.0], [4.0, 4.0, 4.0], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(got, [20.0, 22.0, 24.0], atol=1e-12)

    def test_zero_width_bins_snap(self):
        lo = [0.0, 10.0]
        got = _invert([5, 5], lo, [0.0, 0.0], [0.0, 0.5, 0.51, 1.0], rep=lo)
        np.testing.assert_allclose(got, [0.0, 0.0, 10.0, 10.0], atol=1e-12)

    def test_single_bin_grid(self):
        got = _invert([3], [5.0], [2.0], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(got, [5.0, 6.0, 7.0], atol=1e-12)

    def test_matches_reference_and_monotone(self):
        rng = np.random.default_rng(0)
        qs = np.linspace(0.0, 1.0, 101)
        for _ in range(60):
            B = int(rng.integers(2, 40))
            counts = rng.integers(0, 6, size=B)
            counts[rng.random(B) < 0.35] = 0
            if counts.sum() == 0:
                counts[rng.integers(0, B)] = 3
            edges = np.sort(rng.uniform(-5, 5, B + 1))
            lo, width = edges[:-1], np.diff(edges)
            got = _invert(counts, lo, width, qs)
            ref = np.array([_invert_ref(counts, lo, width, q) for q in qs])
            np.testing.assert_allclose(got, ref, atol=1e-12)
            assert np.all(np.diff(got) >= -1e-12), "quantiles must be non-decreasing"

    def test_wide_grid_uses_heap_scratch(self):
        # B > MAX_BINS_UNIFIED (100) exercises the heap CDF buffer.
        B = 150
        counts = np.arange(1, B + 1)
        edges = np.arange(B + 1, dtype=np.float64)
        got = _invert(counts, edges[:-1], np.diff(edges), QS_FULL)
        ref = np.array([_invert_ref(counts, edges[:-1], np.diff(edges), q) for q in QS_FULL])
        np.testing.assert_allclose(got, ref, atol=1e-12)


# ---------------------------------------------------------------------------
# Fitted models for each branch
# ---------------------------------------------------------------------------
def _data(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 3))
    y = student_t.rvs(df=3, size=n, random_state=rng) + 2.0 * X[:, 0]
    return X, y


@pytest.fixture(scope="module")
def data():
    return _data()


BRANCHES = {
    "plain": dict(),
    "float": dict(fast_inference=True),
    "smooth": dict(smooth_leaves=True),
    "evt": dict(evt_tails=True, evt_tails_lower=True, evt_min_samples=15),
    "smooth_evt": dict(smooth_leaves=True, evt_tails=True, evt_tails_lower=True, evt_min_samples=15),
}


@pytest.fixture(scope="module")
def models(data):
    X, y = data
    out = {}
    for name, kw in BRANCHES.items():
        out[name] = DDTRegressor(max_depth=3, min_samples_leaf=100, quantize_engine="cpp", **kw).fit(X, y)
    return out


def _grid(centers, kind):
    centers = np.asarray(centers, dtype=np.float64)
    if kind == "snap":
        return dict(bin_lo=centers.copy(), bin_width=np.zeros_like(centers), bin_rep=centers.copy())
    # A valid partition of the line: bins tile [e_0, e_B] with edges at centre midpoints.
    mid = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    edges = np.concatenate([[first], mid, [last]])
    return dict(bin_lo=edges[:-1], bin_width=np.diff(edges), bin_rep=centers + 0.01)


def _traverse(name, m, X, q, grid):
    """Call the traversal binding for ``name``. Returns (Q, N)."""
    td = m.tree_data_
    fq = m.feature_quantizer_
    if name == "plain":
        return _ddt_core.predict_quantiles_fast(td, X, fq._flat_bin_edges_, fq._bin_offsets_, fq.n_bins, q, **grid)
    if name == "float":
        assert "split_threshold_float" in td, "fast_inference model must carry float thresholds"
        return _ddt_core.predict_quantiles_fast_float(td, X, q, **grid)
    if name == "smooth":
        return _ddt_core.predict_quantiles_fast_smooth(
            td, X, fq._flat_bin_edges_, fq._bin_offsets_, fq.n_bins, q, **grid
        )
    if name == "evt":
        return _ddt_core.predict_quantiles_fast_evt(td, X, fq._flat_bin_edges_, fq._bin_offsets_, fq.n_bins, q, **grid)
    if name == "smooth_evt":
        return _ddt_core.predict_quantiles_fast_smooth_evt(
            td, X, fq._flat_bin_edges_, fq._bin_offsets_, fq.n_bins, q, **grid
        )
    raise AssertionError(name)


def _from_leaves(name, m, X, q, grid):
    leaves = _ddt_core.predict_leaves(m.tree_data_, m._preprocess_X(X)).astype(np.int32)
    use_smooth = name in ("smooth", "smooth_evt")
    use_evt = name in ("evt", "smooth_evt")
    return _ddt_core.predict_quantiles_from_leaves(
        m.tree_data_, leaves, q, grid["bin_lo"], grid["bin_width"], grid["bin_rep"], use_smooth, use_evt
    )


@pytest.mark.parametrize("name", list(BRANCHES))
@pytest.mark.parametrize("kind", ["snap", "linear"])
class TestLeafEntryPoint:
    def test_from_leaves_equals_traversal(self, models, data, name, kind):
        X, _ = data
        m = models[name]
        centers = m.target_binner_.inverse_transform_bin_centers().astype(np.float64)
        grid = _grid(centers, kind)
        Xs = X[:300]
        trav = _traverse(name, m, Xs, QS_FULL, grid)  # (Q, N)
        leaf = _from_leaves(name, m, Xs, QS_FULL, grid)  # (N, Q)
        np.testing.assert_allclose(leaf.T, trav, rtol=0, atol=1e-12)

    def test_monotone_in_q(self, models, data, name, kind):
        X, _ = data
        m = models[name]
        centers = m.target_binner_.inverse_transform_bin_centers().astype(np.float64)
        grid = _grid(centers, kind)
        # q in (0, 1): EVT tails are +/-inf at the open ends by design.
        out = _from_leaves(name, m, X[:200], np.linspace(0.005, 0.995, 41), grid)
        assert np.all(np.isfinite(out))
        if name in ("evt", "smooth_evt"):
            return  # body/GPD splice monotonicity is not a property of the inversion kernel
        assert np.all(np.diff(out, axis=1) >= -1e-9)


@pytest.mark.parametrize("name", ["plain", "float", "smooth"])
def test_snap_grid_returns_exact_bin_centres(models, data, name):
    """Lo = rep = centres, width = 0 yields only exact bin centres (body, no EVT)."""
    X, _ = data
    m = models[name]
    centers = m.target_binner_.inverse_transform_bin_centers().astype(np.float64)
    out = _traverse(name, m, X[:400], QS_BODY, _grid(centers, "snap"))
    assert np.all(np.isin(out, centers))


def test_grid_is_required(models, data):
    """The legacy bin_centers call (no grid arrays) no longer exists."""
    X, _ = data
    m = models["plain"]
    fq = m.feature_quantizer_
    with pytest.raises(TypeError):
        _ddt_core.predict_quantiles_fast(
            m.tree_data_,
            X[:50],
            fq._flat_bin_edges_,
            fq._bin_offsets_,
            fq.n_bins,
            QS_BODY,
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    def _args(self, counts=(3, 4, 5), **over):
        td = over.pop("td", _leaf_tree(counts))
        B = td["distribution_counts"].shape[1]
        lo = np.arange(B, dtype=np.float64)
        d = dict(
            tree_data=td,
            leaf_ids=np.zeros(1, np.int32),
            quantiles=np.array([0.5]),
            bin_lo=lo,
            bin_width=np.ones(B),
            bin_rep=lo + 0.5,
            use_smooth=False,
            use_evt=False,
        )
        d.update(over)
        return d

    def test_ok_baseline(self):
        out = _ddt_core.predict_quantiles_from_leaves(**self._args())
        assert out.shape == (1, 1)

    @pytest.mark.parametrize("bad", [-1, 1, 7])
    def test_bad_leaf_id_raises_not_reads(self, bad):
        with pytest.raises(ValueError, match="leaf"):
            _ddt_core.predict_quantiles_from_leaves(**self._args(leaf_ids=np.array([bad], np.int32)))

    def test_non_leaf_id_raises(self):
        td = _leaf_tree([3, 4, 5])
        td["is_leaf"] = np.array([1, 0], dtype=np.uint8)  # node 1 exists but is internal
        for k in ("split_feature_idx", "left_child_id", "right_child_id"):
            td[k] = np.concatenate([td[k], td[k]])
        td["split_threshold"] = np.zeros(2, np.uint8)
        td["distribution_counts"] = np.vstack([td["distribution_counts"]] * 2)
        with pytest.raises(ValueError, match="not a leaf"):
            _ddt_core.predict_quantiles_from_leaves(**self._args(td=td, leaf_ids=np.array([1], np.int32)))

    def test_empty_inputs(self):
        out = _ddt_core.predict_quantiles_from_leaves(**self._args(leaf_ids=np.zeros(0, np.int32)))
        assert out.shape == (0, 1)
        out = _ddt_core.predict_quantiles_from_leaves(**self._args(quantiles=np.zeros(0)))
        assert out.shape == (1, 0)

    def test_grid_length_mismatch(self):
        with pytest.raises(ValueError):
            _ddt_core.predict_quantiles_from_leaves(**self._args(bin_width=np.ones(2)))

    def test_grid_longer_than_tree_bins(self):
        a = self._args()
        a.update(bin_lo=np.arange(4.0), bin_width=np.ones(4), bin_rep=np.arange(4.0))
        with pytest.raises(ValueError, match="n_bins"):
            _ddt_core.predict_quantiles_from_leaves(**a)

    def test_negative_or_nan_width(self):
        for w in (np.array([1.0, -1.0, 1.0]), np.array([1.0, np.nan, 1.0])):
            with pytest.raises(ValueError, match="bin_width"):
                _ddt_core.predict_quantiles_from_leaves(**self._args(bin_width=w))

    @pytest.mark.parametrize("q", [-0.1, 1.1, np.nan])
    def test_quantile_out_of_range(self, q):
        with pytest.raises(ValueError, match="quantiles"):
            _ddt_core.predict_quantiles_from_leaves(**self._args(quantiles=np.array([q])))

    def test_missing_smooth_pmf(self):
        with pytest.raises(RuntimeError, match="smoothed_pmf"):
            _ddt_core.predict_quantiles_from_leaves(**self._args(use_smooth=True))

    def test_partial_grid_on_traversal_binding(self, models, data):
        X, _ = data
        m = models["plain"]
        fq = m.feature_quantizer_
        centers = m.target_binner_.inverse_transform_bin_centers().astype(np.float64)
        with pytest.raises(TypeError):  # bin_width / bin_rep are mandatory
            _ddt_core.predict_quantiles_fast(
                m.tree_data_,
                X[:5],
                fq._flat_bin_edges_,
                fq._bin_offsets_,
                fq.n_bins,
                QS_BODY,
                bin_lo=centers,
            )

    def test_grid_longer_than_tree_bins_on_traversal_binding(self, models, data):
        X, _ = data
        m = models["plain"]
        fq = m.feature_quantizer_
        c = m.target_binner_.inverse_transform_bin_centers().astype(np.float64)
        c = np.append(c, c[-1] + 1.0)
        with pytest.raises(ValueError, match="n_bins"):
            _ddt_core.predict_quantiles_fast(
                m.tree_data_,
                X[:5],
                fq._flat_bin_edges_,
                fq._bin_offsets_,
                fq.n_bins,
                QS_BODY,
                bin_lo=c,
                bin_width=np.zeros_like(c),
                bin_rep=c,
            )

    def test_empty_grid_on_traversal_binding(self, models, data):
        X, _ = data
        m = models["plain"]
        fq = m.feature_quantizer_
        e = np.zeros(0)
        with pytest.raises(ValueError, match="at least one bin"):
            _ddt_core.predict_quantiles_fast(
                m.tree_data_,
                X[:5],
                fq._flat_bin_edges_,
                fq._bin_offsets_,
                fq.n_bins,
                QS_BODY,
                bin_lo=e,
                bin_width=e,
                bin_rep=e,
            )
