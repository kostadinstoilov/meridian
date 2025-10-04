import os
import numpy as np
import pytest

from meridian_ml_service.clustering import ClusterParams, run_clustering


def test_single_sample_explicit_noise():
    X = np.random.randn(1, 16).astype(np.float32)
    params = ClusterParams()
    reduced, labels, probs, umap_2d, meta, umap_info = run_clustering(X, params)

    # Reduced vectors should be explicitly None when reduction is skipped
    assert reduced is None
    assert labels.shape == (1,)
    assert labels[0] == -1
    assert probs.shape == (1,)
    assert probs[0] == 0.0
    assert umap_2d is None
    assert meta == {}
    assert umap_info["status"] == "skipped_insufficient_points"


@pytest.mark.parametrize("env_val,expected", [("1", 1), ("2", 2), ("0", 1), ("notint", None)])
def test_umap_n_jobs_env_override(env_val, expected, monkeypatch):
    # Save prior env
    old = os.environ.get("UMAP_N_JOBS")
    try:
        monkeypatch.setenv("UMAP_N_JOBS", env_val)
        params = ClusterParams()
        if expected is None:
            # invalid value should fall back to bounded default
            assert isinstance(params.n_jobs, int) and params.n_jobs >= 1
        else:
            assert params.n_jobs == expected
    finally:
        if old is None:
            monkeypatch.delenv("UMAP_N_JOBS", raising=False)
        else:
            monkeypatch.setenv("UMAP_N_JOBS", old)


def test_two_samples_skip_reduction():
    X = np.random.randn(2, 16).astype(np.float32)
    params = ClusterParams()
    reduced, labels, probs, umap_2d, meta, umap_info = run_clustering(X, params)

    # For n=2 we intentionally skip producing reduced_vectors and 2D viz
    assert reduced is None
    assert umap_2d is None
    assert umap_info["status"] == "skipped_insufficient_points"
    # Clustering should still run on the original feature space
    assert labels.shape == (2,)
    assert probs.shape == (2,)
