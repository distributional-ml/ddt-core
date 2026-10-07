from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class TreeState:
    """
    Typed, validated container for a fitted tree's SoA arrays.

    Replaces the untyped `tree_data_: dict` with explicit fields, dtype
    contracts, and a unified PMF access API.
    """

    # ---- Core SoA (always present after fit) ----
    is_leaf: np.ndarray  # uint8,   (n_nodes,)
    split_feature_idx: np.ndarray  # int32,   (n_nodes,)
    split_threshold: np.ndarray  # uint8,   (n_nodes,)
    left_child_id: np.ndarray  # int32,   (n_nodes,)
    right_child_id: np.ndarray  # int32,   (n_nodes,)
    distribution_counts: np.ndarray | None  # int32,   (n_nodes, n_bins)
    n_bins: int

    # ---- Optional: metadata (may be stripped by __getstate__) ----
    wasserstein_gain: np.ndarray | None = None  # float64, (n_nodes,)
    depth: np.ndarray | None = None  # int32,   (n_nodes,)
    total_samples: np.ndarray | None = None  # int32,   (n_nodes,)

    # ---- Optional: float-split fast path ----
    split_threshold_float: np.ndarray | None = None  # float64, (n_nodes,)

    # ---- Optional: smoothed PMF (mutually exclusive storage) ----
    smoothed_pmf: np.ndarray | None = None  # float64, (n_nodes, n_bins)
    inference_pmf: np.ndarray | None = None  # float16/32, (n_nodes, n_bins)

    # ---- Optional: EVT upper tail ----
    evt_enabled: np.ndarray | None = None  # uint8,   (n_nodes,)
    evt_threshold_u: np.ndarray | None = None  # float64, (n_nodes,)
    evt_F_u: np.ndarray | None = None  # float64, (n_nodes,)
    evt_S_u: np.ndarray | None = None  # float64, (n_nodes,)
    evt_gpd_shape: np.ndarray | None = None  # float64, (n_nodes,)
    evt_gpd_scale: np.ndarray | None = None  # float64, (n_nodes,)

    # ---- Optional: EVT lower tail ----
    evt_lower_enabled: np.ndarray | None = None  # uint8, (n_nodes,)
    evt_lower_threshold_u: np.ndarray | None = None  # float64
    evt_lower_F_u: np.ndarray | None = None  # float64
    evt_lower_gpd_shape: np.ndarray | None = None  # float64
    evt_lower_gpd_scale: np.ndarray | None = None  # float64

    # ---- Pipeline stage ----
    pipeline_stage: int = 0  # PipelineStage enum value

    @property
    def n_nodes(self) -> int:
        return len(self.is_leaf)

    @property
    def has_smooth_pmf(self) -> bool:
        return self.smoothed_pmf is not None or self.inference_pmf is not None

    @property
    def has_upper_evt(self) -> bool:
        return self.evt_enabled is not None

    @property
    def has_lower_evt(self) -> bool:
        return self.evt_lower_enabled is not None

    @property
    def has_float_threshold(self) -> bool:
        return self.split_threshold_float is not None

    def get_active_pmf(self) -> np.ndarray | None:
        """Return the active PMF array (smoothed or inference), or None."""
        if self.smoothed_pmf is not None:
            return self.smoothed_pmf
        if self.inference_pmf is not None:
            return self.inference_pmf.astype(np.float64)
        return None

    def leaf_pmf(self, ids) -> np.ndarray:
        """Get the normalized PMF for the given leaf IDs."""
        pmf = self.get_active_pmf()
        if pmf is not None:
            return pmf[ids]
        counts = self.distribution_counts[ids]
        if counts.ndim == 1:
            s = counts.sum()
            return (counts / s).astype(np.float64) if s > 0 else np.zeros_like(counts, dtype=np.float64)
        sums = counts.sum(axis=1, keepdims=True)
        safe_sums = np.where(sums > 0, sums, 1)
        return (counts / safe_sums).astype(np.float64)

    def leaf_counts(self, ids) -> np.ndarray:
        """
        Get the raw empirical bin counts for the given leaf IDs.

        Unlike :meth:`leaf_pmf`, this never returns the smoothed PMF. Counts
        may be float after compaction. Raises if raw counts are not stored.
        """
        if self.distribution_counts is None:
            raise ValueError("Raw distribution counts are not available (compact_inference=True).")
        return self.distribution_counts[ids]

    def leaf_n(self, ids) -> np.ndarray:
        """Get the total samples for the given leaf IDs (as floats)."""
        if self.total_samples is not None:
            return self.total_samples[ids].astype(np.float64)
        # Fallback if total_samples is missing (e.g. legacy model)
        if self.distribution_counts is not None:
            counts = self.distribution_counts[ids]
            if counts.ndim == 1:
                return float(counts.sum())
            return counts.sum(axis=1).astype(np.float64)
        raise ValueError(
            "Cannot determine leaf sample counts: total_samples is stripped and distribution_counts is compacted."
        )

    def to_dict(self) -> dict:
        """
        Convert to the legacy dict format for pybind11 compatibility.

        The C++ side expects a py::dict with specific string keys.
        This method bridges the typed Python world with the untyped C++ interface.
        """
        d = {
            "is_leaf": self.is_leaf,
            "split_feature_idx": self.split_feature_idx,
            "split_threshold": self.split_threshold,
            "left_child_id": self.left_child_id,
            "right_child_id": self.right_child_id,
            "distribution_counts": self.distribution_counts,
            "n_bins": self.n_bins,
        }
        # Add optional fields
        for field in [
            "wasserstein_gain",
            "depth",
            "total_samples",
            "split_threshold_float",
            "smoothed_pmf",
            "inference_pmf",
            "evt_enabled",
            "evt_threshold_u",
            "evt_F_u",
            "evt_S_u",
            "evt_gpd_shape",
            "evt_gpd_scale",
            "evt_lower_enabled",
            "evt_lower_threshold_u",
            "evt_lower_F_u",
            "evt_lower_gpd_shape",
            "evt_lower_gpd_scale",
        ]:
            val = getattr(self, field, None)
            if val is not None:
                d[field] = val
        if self.pipeline_stage > 0:
            d["_pipeline_stage"] = self.pipeline_stage
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TreeState:
        """
        Construct a TreeState from the legacy dict format.

        Used to wrap dicts returned by _ddt_core.build_tree().
        """
        return cls(
            is_leaf=np.asarray(d["is_leaf"], dtype=np.uint8),
            split_feature_idx=np.asarray(d["split_feature_idx"], dtype=np.int32),
            split_threshold=np.asarray(d["split_threshold"], dtype=np.uint8),
            left_child_id=np.asarray(d["left_child_id"], dtype=np.int32),
            right_child_id=np.asarray(d["right_child_id"], dtype=np.int32),
            distribution_counts=np.asarray(d["distribution_counts"], dtype=np.int32)
            if "distribution_counts" in d
            else None,
            n_bins=int(d["n_bins"]),
            wasserstein_gain=d.get("wasserstein_gain"),
            depth=d.get("depth"),
            total_samples=d.get("total_samples"),
            split_threshold_float=d.get("split_threshold_float"),
            smoothed_pmf=d.get("smoothed_pmf"),
            inference_pmf=d.get("inference_pmf"),
            evt_enabled=d.get("evt_enabled"),
            evt_threshold_u=d.get("evt_threshold_u"),
            evt_F_u=d.get("evt_F_u"),
            evt_S_u=d.get("evt_S_u"),
            evt_gpd_shape=d.get("evt_gpd_shape"),
            evt_gpd_scale=d.get("evt_gpd_scale"),
            evt_lower_enabled=d.get("evt_lower_enabled"),
            evt_lower_threshold_u=d.get("evt_lower_threshold_u"),
            evt_lower_F_u=d.get("evt_lower_F_u"),
            evt_lower_gpd_shape=d.get("evt_lower_gpd_shape"),
            evt_lower_gpd_scale=d.get("evt_lower_gpd_scale"),
            pipeline_stage=d.get("_pipeline_stage", 0),
        )

    def validate(self):
        """Runtime validation of array shapes and dtypes."""
        n = self.n_nodes
        b = self.n_bins

        def _check(name, arr, expected_shape, expected_dtype=None):
            if arr is None:
                return
            arr = np.asarray(arr)
            if arr.shape != expected_shape:
                raise ValueError(f"TreeState.{name}: expected shape {expected_shape}, got {arr.shape}")
            if expected_dtype and arr.dtype != np.dtype(expected_dtype):
                raise ValueError(f"TreeState.{name}: expected dtype {expected_dtype}, got {arr.dtype}")

        _check("is_leaf", self.is_leaf, (n,), np.uint8)
        _check("split_feature_idx", self.split_feature_idx, (n,), np.int32)
        _check("split_threshold", self.split_threshold, (n,), np.uint8)
        _check("left_child_id", self.left_child_id, (n,), np.int32)
        _check("right_child_id", self.right_child_id, (n,), np.int32)
        _check("distribution_counts", self.distribution_counts, (n, b), np.int32)
        _check("smoothed_pmf", self.smoothed_pmf, (n, b), np.float64)
        _check("evt_enabled", self.evt_enabled, (n,), np.uint8)
