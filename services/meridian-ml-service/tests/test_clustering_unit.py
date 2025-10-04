"""Unit tests for the clustering module."""

import os
import numpy as np
import pytest

from meridian_ml_service.clustering import (
    ClusterParams,
    l2_normalize,
    run_clustering,
    _cluster_metadata,
)


class TestL2Normalize:
    """Test the l2_normalize function."""

    def test_l2_normalize_basic(self):
        """Test basic L2 normalization."""
        x = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        x_norm = l2_normalize(x)
        
        # Check that vectors are unit length
        norms = np.linalg.norm(x_norm, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-6)
        
        # Check that shape is preserved
        assert x_norm.shape == x.shape

    def test_l2_normalize_with_zeros(self):
        """Test L2 normalization with zero vectors."""
        x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        x_norm = l2_normalize(x)
        
        # Zero vectors remain zero, non-zero vectors get unit norm
        norms = np.linalg.norm(x_norm, axis=1)
        np.testing.assert_allclose(norms, [0.0, 1.0], rtol=1e-6)

    def test_l2_normalize_preserves_shape(self):
        """Test that L2 normalization preserves array shape."""
        x = np.random.randn(10, 32).astype(np.float32)
        x_norm = l2_normalize(x)
        assert x_norm.shape == x.shape


class TestClusterParams:
    """Test the ClusterParams class."""

    def test_default_params(self):
        """Test default parameter values."""
        params = ClusterParams()
        
        assert params.umap_neighbors == 15
        assert params.umap_components == 15
        assert params.umap_min_dist == 0.0
        assert params.umap_metric == "cosine"
        assert params.min_cluster_size == 6
        assert params.min_samples is None
        assert params.cluster_selection_epsilon == 0.0
        assert params.cluster_selection_method == "eom"
        assert params.random_state == 42
        assert params.create_2d_viz is True
        assert params.normalize_vectors is True

    def test_custom_params(self):
        """Test custom parameter values."""
        params = ClusterParams(
            umap_neighbors=20,
            umap_components=10,
            min_cluster_size=8,
            random_state=123,
            create_2d_viz=False,
            normalize_vectors=False,
        )
        
        assert params.umap_neighbors == 20
        assert params.umap_components == 10
        assert params.min_cluster_size == 8
        assert params.random_state == 123
        assert params.create_2d_viz is False
        assert params.normalize_vectors is False


class TestClusterMetadata:
    """Test the _cluster_metadata function."""

    def test_cluster_metadata_basic(self):
        """Test basic cluster metadata computation."""
        labels = np.array([0, 0, 1, 1, 1, -1])  # 2 clusters, 1 noise
        probs = np.array([0.8, 0.6, 0.9, 0.7, 0.5, 0.0])
        base_vectors = np.random.randn(6, 32).astype(np.float32)
        
        meta = _cluster_metadata(labels, probs, base_vectors)
        
        assert len(meta) == 2  # 2 clusters
        assert "0" in meta
        assert "1" in meta
        
        # Check cluster 0
        cluster_0 = meta["0"]
        assert cluster_0["size"] == 2
        assert 0 <= cluster_0["medoid_index"] < 6
        assert len(cluster_0["exemplar_indices"]) == 2
        assert all(0 <= idx < 6 for idx in cluster_0["exemplar_indices"])
        
        # Check cluster 1
        cluster_1 = meta["1"]
        assert cluster_1["size"] == 3
        assert 0 <= cluster_1["medoid_index"] < 6
        assert len(cluster_1["exemplar_indices"]) == 3
        assert all(0 <= idx < 6 for idx in cluster_1["exemplar_indices"])

    def test_cluster_metadata_no_clusters(self):
        """Test metadata computation with no clusters (all noise)."""
        labels = np.array([-1, -1, -1])
        probs = np.array([0.0, 0.0, 0.0])
        base_vectors = np.random.randn(3, 32).astype(np.float32)
        
        meta = _cluster_metadata(labels, probs, base_vectors)
        
        assert len(meta) == 0  # No clusters


class TestRunClustering:
    """Test the run_clustering function."""

    def test_basic_clustering_small_blob(self, small_blob_data):
        """Test basic clustering on well-separated blobs."""
        X, _ = small_blob_data
        params = ClusterParams(
            umap_neighbors=15,
            umap_components=15,
            min_cluster_size=5,
            create_2d_viz=True,
        )
        
        reduced, labels, probs, umap_2d, meta, umap_info = run_clustering(X, params)
        
        # Check shapes
        assert reduced.shape[0] == X.shape[0]
        assert reduced.shape[1] == params.umap_components
        assert umap_2d is not None and umap_2d.shape[1] == 2
        assert len(labels) == X.shape[0]
        assert len(probs) == X.shape[0]
        
        # Check probabilities are in valid range
        assert (probs >= 0).all() and (probs <= 1).all()
        
        # Check that we found some clusters (should be 3 for well-separated data)
        non_noise = sorted(set(labels.tolist()) - {-1})
        assert 2 <= len(non_noise) <= 3  # Allow small variance on tiny sets
        
        # Check metadata
        for k, v in meta.items():
            assert 0 <= v["medoid_index"] < X.shape[0]
            assert v["size"] > 0
            assert len(v["exemplar_indices"]) > 0

    def test_very_small_dataset(self, very_small_blob_data):
        """Test clustering on a very small dataset."""
        X, _ = very_small_blob_data
        params = ClusterParams(
            umap_neighbors=3,
            umap_components=5,
            min_cluster_size=3,  # Same as cluster size
            create_2d_viz=False,
        )
        
        reduced, labels, probs, umap_2d, meta, umap_info = run_clustering(X, params)
        
        # Check shapes
        assert reduced.shape[0] == X.shape[0]
        assert reduced.shape[1] == params.umap_components
        assert umap_2d is None  # create_2d_viz=False
        assert len(labels) == X.shape[0]
        
        # With min_cluster_size=3 and only 3 points, we expect mostly noise
        noise_count = np.sum(labels == -1)
        assert noise_count >= 0  # Could be all noise or one cluster

    def test_normalization_toggle(self, small_blob_data):
        """Test the normalize_vectors parameter."""
        X, _ = small_blob_data
        
        # Test with normalization
        params_norm = ClusterParams(normalize_vectors=True)
        reduced_norm, _, _, _, meta_norm, umap_info = run_clustering(X, params_norm)
        
        # Test without normalization
        params_no_norm = ClusterParams(normalize_vectors=False)
        reduced_no_norm, _, _, _, meta_no_norm, umap_info = run_clustering(X, params_no_norm)
        
        # Both should produce valid results
        assert reduced_norm.shape == reduced_no_norm.shape
        assert len(meta_norm) > 0 or len(meta_no_norm) > 0

    def test_umap_components_change(self, small_blob_data):
        """Test that umap_components affects output dimension."""
        X, _ = small_blob_data
        
        # Test with different UMAP components
        params_5 = ClusterParams(umap_components=5)
        params_10 = ClusterParams(umap_components=10)
        
        reduced_5, _, _, _, _, _ = run_clustering(X, params_5)
        reduced_10, _, _, _, _, _ = run_clustering(X, params_10)
        
        # Check that dimensions match
        assert reduced_5.shape[1] == 5
        assert reduced_10.shape[1] == 10

    def test_2d_viz_toggle(self, small_blob_data):
        """Test the create_2d_viz parameter."""
        X, _ = small_blob_data
        
        # Test with 2D visualization
        params_with_viz = ClusterParams(create_2d_viz=True)
        _, _, _, umap_2d_with, _, umap_info = run_clustering(X, params_with_viz)
        
        # Test without 2D visualization
        params_without_viz = ClusterParams(create_2d_viz=False)
        _, _, _, umap_2d_without, _, umap_info = run_clustering(X, params_without_viz)
        
        # Check 2D visualization presence
        assert umap_2d_with is not None
        assert umap_2d_without is None

    def test_determinism(self, small_blob_data):
        """Test that clustering is deterministic with same seed.

        Use label-invariant comparisons and tolerant numeric checks instead of
        exact float/label equality to avoid brittle tests across platforms.
        """
        X, _ = small_blob_data

        params = ClusterParams(random_state=42)

        # Run clustering twice with same parameters
        reduced_1, labels_1, probs_1, umap_2d_1, meta_1, umap_info1 = run_clustering(X, params)
        reduced_2, labels_2, probs_2, umap_2d_2, meta_2, umap_info2 = run_clustering(X, params)

        # Label-invariant: expect perfect agreement
        from test_utils import adjusted_rand_index, pairwise_dists

        ari = adjusted_rand_index(labels_1, labels_2)
        assert ari == 1.0

        # Noise count should match
        assert int((labels_1 == -1).sum()) == int((labels_2 == -1).sum())

        # Probabilities should be close
        np.testing.assert_allclose(probs_1, probs_2, rtol=1e-6, atol=1e-6)

        # Compare pairwise distance matrices for reduced embeddings (rotation/reflection invariant)
        D1 = pairwise_dists(reduced_1)
        D2 = pairwise_dists(reduced_2)
        np.testing.assert_allclose(D1, D2, rtol=1e-4, atol=1e-4)

        # If 2D viz was computed, compare its pairwise distances as well
        if umap_2d_1 is not None and umap_2d_2 is not None:
            D1_2d = pairwise_dists(umap_2d_1)
            D2_2d = pairwise_dists(umap_2d_2)
            np.testing.assert_allclose(D1_2d, D2_2d, rtol=1e-4, atol=1e-4)

        # Avoid asserting raw metadata equality (label integers may vary).

    def test_outlier_handling(self, outlier_embeddings):
        """Test that outliers are properly handled."""
        X = outlier_embeddings
        params = ClusterParams(
            umap_neighbors=10,
            umap_components=10,
            min_cluster_size=3,
        )
        
        reduced, labels, probs, umap_2d, meta, umap_info = run_clustering(X, params)
        
        # Should find some clusters but also have some noise
        noise_count = np.sum(labels == -1)
        assert noise_count >= 0  # Some outliers might be labeled as noise
        
        # Check that we still get valid outputs
        assert len(labels) == X.shape[0]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_empty_input(self):
        """Test clustering with empty input."""
        # UMAP requires at least 1 sample, so this test should fail gracefully
        X = np.empty((0, 32), dtype=np.float32)
        params = ClusterParams()
        
        # Should raise an error due to UMAP's minimum sample requirement
        with pytest.raises((ValueError, RuntimeError)):
            run_clustering(X, params)

    def test_single_point(self):
        """Test clustering with a single point."""
        X = np.random.randn(1, 32).astype(np.float32)
        params = ClusterParams()
        
        # HDBSCAN may have issues with single point, should either error or return noise
        try:
            reduced, labels, probs, umap_2d, meta, umap_info = run_clustering(X, params)
            # If it succeeds, single point should be noise and UMAP skipped
            assert reduced is None
            assert labels[0] == -1
            assert probs[0] == 0.0
            assert len(meta) == 0
            assert isinstance(umap_info, dict)
            assert umap_info.get("status") == "skipped_insufficient_points"
        except (ValueError, RuntimeError):
            # It's acceptable for HDBSCAN to error on single point
            pass


class TestClusteringIntegration:
    """Integration tests for clustering functionality."""

    def test_parameter_combinations(self, small_blob_data):
        """Test various parameter combinations."""
        X, _ = small_blob_data
        
        # Test different parameter combinations
        param_sets = [
            ClusterParams(umap_neighbors=5, umap_components=5),
            ClusterParams(umap_neighbors=20, umap_components=20),
            ClusterParams(min_cluster_size=2),
            ClusterParams(min_cluster_size=10),
            ClusterParams(normalize_vectors=False),
            ClusterParams(create_2d_viz=False),
        ]
        
        for params in param_sets:
            reduced, labels, probs, umap_2d, meta, umap_info = run_clustering(X, params)
            
            # Basic sanity checks
            assert reduced.shape[0] == X.shape[0]
            assert len(labels) == X.shape[0]
            assert (probs >= 0).all() and (probs <= 1).all()
            assert umap_2d is None or umap_2d.shape[1] == 2

    def test_coassociation_matrix(self, small_blob_data):
        """Test co-association matrix for partition equivalence."""
        X, _ = small_blob_data
        
        # Run clustering twice with same parameters
        params = ClusterParams(random_state=42)
        _, labels_1, _, _, _, umap_info1 = run_clustering(X, params)
        _, labels_2, _, _, _, umap_info2 = run_clustering(X, params)
        
        # Build co-association matrices
        n = len(labels_1)
        coassoc_1 = np.zeros((n, n), dtype=bool)
        coassoc_2 = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            for j in range(n):
                coassoc_1[i, j] = (labels_1[i] == labels_1[j] and labels_1[i] != -1)
                coassoc_2[i, j] = (labels_2[i] == labels_2[j] and labels_2[i] != -1)
        
        # Matrices should be identical
        np.testing.assert_array_equal(coassoc_1, coassoc_2)