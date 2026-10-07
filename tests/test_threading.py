"""Verify that the native bindings release the GIL during C++ compute."""

import threading
import time

import numpy as np
import pytest

from ddt import _ddt_core


def _build_inputs(n_samples=400_000, n_features=16, n_bins=64, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 32, size=(n_samples, n_features)).astype(np.uint8)
    y = ((X[:, 0].astype(np.int32) * 2 + rng.randint(0, 8, n_samples)) % n_bins).astype(np.int32)
    return X, y, n_bins


def _count_ticks_while_running(fn):
    """Count observer wake-ups strictly inside the worker's call interval."""
    box = {}
    start = threading.Event()

    def worker():
        start.wait()
        box["started"] = time.perf_counter()
        try:
            box["result"] = fn()
        except BaseException as exc:
            box["error"] = exc
        finally:
            box["finished"] = time.perf_counter()

    thread = threading.Thread(target=worker)
    wakeups = []
    thread.start()
    start.set()
    while thread.is_alive():
        time.sleep(0.001)
        wakeups.append(time.perf_counter())
    thread.join()
    if "error" in box:
        raise box["error"]
    ticks = sum(box["started"] < tick < box["finished"] for tick in wakeups)
    return ticks, box["finished"] - box["started"], box["result"]


def test_build_tree_releases_gil():
    """A Python thread must keep running while build_tree executes in another."""
    X, y, n_bins = _build_inputs()

    def build():
        return _ddt_core.build_tree(X, y, n_bins, max_depth=12, min_samples_leaf=5)

    ticks, elapsed, tree = _count_ticks_while_running(build)

    assert elapsed > 0.05, "workload too small to measure GIL release"
    # Require repeated Python progress during the call, not a fixed wake-up rate.
    # Windows timers and shared CI runners can substantially overshoot sleep(0.001).
    # Holding the GIL prevents sustained observer progress during native compute.
    assert ticks >= 3, (
        f"main thread only ticked {ticks} times during a {elapsed:.2f}s native call; the GIL appears to be held"
    )
    assert tree["is_leaf"].shape[0] > 1


def test_concurrent_builds_are_consistent():
    """Concurrent GIL-free builds on shared read-only inputs give identical trees."""
    X, y, n_bins = _build_inputs(n_samples=50_000)
    reference = _ddt_core.build_tree(X, y, n_bins, max_depth=8, min_samples_leaf=10)

    results = [None] * 4

    def run(i):
        results[i] = _ddt_core.build_tree(X, y, n_bins, max_depth=8, min_samples_leaf=10)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for res in results:
        assert res is not None
        for key in ("is_leaf", "split_feature_idx", "split_threshold", "distribution_counts"):
            np.testing.assert_array_equal(res[key], reference[key])


def test_concurrent_predict_leaves_is_consistent():
    """predict_leaves from several threads matches the single-threaded result."""
    X, y, n_bins = _build_inputs(n_samples=50_000)
    tree = _ddt_core.build_tree(X, y, n_bins, max_depth=8, min_samples_leaf=10)
    expected = _ddt_core.predict_leaves(tree, X)

    results = [None] * 4

    def run(i):
        results[i] = _ddt_core.predict_leaves(tree, X)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for res in results:
        np.testing.assert_array_equal(res, expected)


@pytest.mark.parametrize("name", ["omp parallel"])
def test_no_dead_openmp_pragmas(name):
    """OpenMP is not enabled in any build, so no pragma may remain in the sources."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "cpp"
    for path in list(src.glob("*.cpp")) + list(src.glob("*.hpp")):
        assert name not in path.read_text(encoding="utf-8"), f"{name!r} found in {path.name}"
