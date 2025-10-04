"""Integration tests for the /cluster endpoint."""

import pytest
from fastapi.testclient import TestClient


class TestClusterEndpoint:
    """Integration tests for the /cluster endpoint."""

    def test_cluster_with_embeddings(self, app_client, small_blob_data):
        """Test clustering with pre-computed embeddings."""
        X, _ = small_blob_data
        payload = {
            "embeddings": X.tolist(),
            "params": {
                "umap_neighbors": 15,
                "umap_components": 12,
                "min_cluster_size": 5,
                "create_2d_viz": True,
            }
        }
        
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 200
        data = r.json()
        
        # Check response structure
        assert "labels" in data
        assert "probabilities" in data
        assert "reduced_vectors" in data
        assert "umap_2d" in data
        assert "cluster_metadata" in data
        assert "model_name" in data
        
        # Check shapes and types
        assert len(data["labels"]) == X.shape[0]
        assert len(data["probabilities"]) == X.shape[0]
        assert len(data["reduced_vectors"]) == X.shape[0]
        assert len(data["reduced_vectors"][0]) == 12
        assert data["umap_2d"] is not None
        assert len(data["umap_2d"]) == X.shape[0]
        assert len(data["umap_2d"][0]) == 2
        assert isinstance(data["cluster_metadata"], dict)
        assert isinstance(data["model_name"], str)

    def test_cluster_with_texts(self, app_client, mock_embed_texts):
        """Test clustering with texts (using mocked embeddings)."""
        texts = [
            "This is the first text about technology",
            "This is the second text about computers",
            "This is a text about cooking recipes",
            "Another recipe text about food preparation",
            "A third recipe text about cooking",
            "Sports news about the game last night",
            "Latest sports scores and highlights",
            "More sports coverage and analysis",
        ]
        
        payload = {
            "texts": texts,
            "params": {
                "umap_neighbors": 3,  # Further reduced to avoid eigenvalue issues
                "umap_components": 5,  # Reduced to match small dataset
                "min_cluster_size": 2,
                "create_2d_viz": False,
            }
        }
        
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 200
        data = r.json()
        
        # Check response structure
        assert "labels" in data
        assert "probabilities" in data
        assert "reduced_vectors" in data
        assert "umap_2d" in data
        assert "cluster_metadata" in data
        assert "model_name" in data
        
        # Check shapes and types
        assert len(data["labels"]) == len(texts)
        assert len(data["probabilities"]) == len(texts)
        assert len(data["reduced_vectors"]) == len(texts)
        assert len(data["reduced_vectors"][0]) == 5  # Updated to match umap_components=5
        assert data["umap_2d"] is None  # create_2d_viz=False
        assert isinstance(data["cluster_metadata"], dict)
        assert isinstance(data["model_name"], str)

    def test_cluster_with_texts_prefix_handling(self, app_client, mock_embed_texts):
        """Test that 'passage:' prefix is properly handled."""
        texts = [
            "plain text without prefix",
            "passage: text with prefix",
            "query: text with query prefix",
            "another text to make it work",
        ]
        
        payload = {
            "texts": texts,
            "params": {
                "umap_neighbors": 2,  # Must be < n_samples (4)
                "umap_components": 2,
                "min_cluster_size": 2,
                "create_2d_viz": False,
            }
        }
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 200
        
        # Should work without errors
        data = r.json()
        assert len(data["labels"]) == len(texts)

    def test_cluster_missing_inputs(self, app_client):
        """Test error when neither texts nor embeddings are provided."""
        payload = {
            "params": {
                "umap_neighbors": 15,
                "umap_components": 10,
            }
        }
        
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 400
        assert "Provide either texts or embeddings" in r.json()["detail"]

    def test_cluster_both_inputs(self, app_client, small_blob_data):
        """Test error when both texts and embeddings are provided."""
        X, _ = small_blob_data
        payload = {
            "texts": ["some text"],
            "embeddings": X.tolist(),
            "params": {"umap_neighbors": 15},
        }
        
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 400
        assert "Provide either texts or embeddings" in r.json()["detail"]

    def test_cluster_invalid_params(self, app_client):
        """Test error with invalid parameters."""
        payload = {
            "embeddings": [[1.0, 2.0], [3.0, 4.0]],
            "params": {
                "umap_neighbors": -1,  # Invalid: negative
                "min_cluster_size": 0,  # Invalid: zero
            }
        }
        
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 422  # Pydantic validation error

    def test_cluster_invalid_embeddings(self, app_client):
        """Test error with invalid embeddings format."""
        payload = {
            "embeddings": "not a list",  # Invalid type
            "params": {"umap_neighbors": 15},
        }
        
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 422  # Pydantic validation error

    def test_cluster_params_toggle_2d_viz(self, app_client, small_blob_data):
        """Test that create_2d_viz parameter works correctly."""
        X, _ = small_blob_data
        
        # Test with 2D visualization
        payload_with_viz = {
            "embeddings": X.tolist(),
            "params": {"create_2d_viz": True}
        }
        r = app_client.post("/cluster", json=payload_with_viz)
        assert r.status_code == 200
        data = r.json()
        assert data["umap_2d"] is not None
        
        # Test without 2D visualization
        payload_without_viz = {
            "embeddings": X.tolist(),
            "params": {"create_2d_viz": False}
        }
        r = app_client.post("/cluster", json=payload_without_viz)
        assert r.status_code == 200
        data = r.json()
        assert data["umap_2d"] is None

    def test_cluster_params_umap_components(self, app_client, small_blob_data):
        """Test that umap_components parameter affects output dimension."""
        X, _ = small_blob_data
        
        # Test with different UMAP components
        payload = {
            "embeddings": X.tolist(),
            "params": {"umap_components": 5}
        }
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert len(data["reduced_vectors"][0]) == 5

    def test_cluster_empty_embeddings(self, app_client):
        """Test clustering with empty embeddings list."""
        payload = {
            "embeddings": [],
            "params": {"umap_neighbors": 15}
        }
        
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 200
        data = r.json()
        
        # Should return empty results
        assert len(data["labels"]) == 0
        assert len(data["probabilities"]) == 0
        assert len(data["reduced_vectors"]) == 0

    def test_cluster_small_dataset(self, app_client, mock_embed_texts):
        """Test clustering with a small but valid dataset."""
        payload = {
            "texts": ["first text", "second text", "third text", "fourth text", "fifth text", "sixth text", "seventh text", "eighth text", "ninth text", "tenth text", "eleventh text", "twelfth text", "thirteenth text", "fourteenth text", "fifteenth text", "sixteenth text"],
            "params": {"min_cluster_size": 2}  # Use default parameters
        }
        
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 200
        data = r.json()
        
        # Sixteen points should be handled
        assert len(data["labels"]) == 16
        assert len(data["probabilities"]) == 16

    @pytest.mark.slow
    def test_cluster_large_payload(self, app_client, random_embeddings):
        """Test clustering with a larger payload (marked as slow)."""
        X = random_embeddings
        
        payload = {
            "embeddings": X.tolist(),
            "params": {
                "umap_neighbors": 15,
                "umap_components": 15,
                "min_cluster_size": 5,
            }
        }
        
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 200
        data = r.json()
        
        # Should handle larger input
        assert len(data["labels"]) == X.shape[0]
        assert len(data["reduced_vectors"]) == X.shape[0]


class TestClusterEndpointAuth:
    """Authentication tests for the /cluster endpoint."""

    def test_cluster_without_auth_when_token_set(self, app_client_with_token):
        """Test that requests fail without auth when API_TOKEN is set."""
        payload = {
            "embeddings": [[1.0, 2.0], [3.0, 4.0]],
            "params": {"umap_neighbors": 15}
        }
        
        r = app_client_with_token.post("/cluster", json=payload)
        assert r.status_code == 403  # Forbidden

    def test_cluster_with_invalid_token(self, app_client_with_token):
        """Test that requests fail with invalid token when API_TOKEN is set."""
        payload = {
            "embeddings": [[1.0, 2.0], [3.0, 4.0]],
            "params": {"umap_neighbors": 15}
        }
        
        r = app_client_with_token.post("/cluster", json=payload, headers={
            "Authorization": "Bearer invalid_token"
        })
        assert r.status_code == 403  # Forbidden

    def test_cluster_with_valid_token(self, app_client_with_token):
        """Test that requests succeed with valid token when API_TOKEN is set."""
        payload = {
            "embeddings": [[1.0, 2.0], [3.0, 4.0]],
            "params": {"umap_neighbors": 15}
        }
        
        r = app_client_with_token.post("/cluster", json=payload, headers={
            "Authorization": "Bearer devtoken123"
        })
        assert r.status_code == 200

    def test_cluster_without_auth_when_no_token(self, app_client):
        """Test that requests succeed without auth when API_TOKEN is not set."""
        payload = {
            "embeddings": [[1.0, 2.0], [3.0, 4.0]],
            "params": {"umap_neighbors": 15}
        }
        
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 200

    def test_cluster_with_auth_when_no_token(self, app_client):
        """Test that requests succeed with auth even when API_TOKEN is not set."""
        payload = {
            "embeddings": [[1.0, 2.0], [3.0, 4.0]],
            "params": {"umap_neighbors": 15}
        }
        
        r = app_client.post("/cluster", json=payload, headers={
            "Authorization": "Bearer any_token"
        })
        assert r.status_code == 200


class TestClusterEndpointRegression:
    """Regression tests for the /cluster endpoint."""

    def test_cluster_metadata_consistency(self, app_client, small_blob_data):
        """Test that cluster metadata is consistent with labels."""
        X, _ = small_blob_data
        payload = {
            "embeddings": X.tolist(),
            "params": {"min_cluster_size": 3}
        }
        
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 200
        data = r.json()
        
        # Check that metadata matches labels
        labels = data["labels"]
        metadata = data["cluster_metadata"]
        
        # All clusters in metadata should exist in labels
        label_set = set(labels)
        for cluster_id in metadata:
            assert int(cluster_id) in label_set
            assert cluster_id != "-1"  # No noise in metadata
        
        # Check that metadata sizes match label counts
        for cluster_id, cluster_info in metadata.items():
            count_in_labels = labels.count(int(cluster_id))
            assert cluster_info["size"] == count_in_labels

    def test_cluster_probability_bounds(self, app_client, small_blob_data):
        """Test that probabilities are within valid bounds."""
        X, _ = small_blob_data
        payload = {
            "embeddings": X.tolist(),
            "params": {"min_cluster_size": 3}
        }
        
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 200
        data = r.json()
        
        # Check that all probabilities are in [0, 1]
        for prob in data["probabilities"]:
            assert 0.0 <= prob <= 1.0

    def test_cluster_medoid_indices_valid(self, app_client, small_blob_data):
        """Test that medoid indices are within valid range."""
        X, _ = small_blob_data
        payload = {
            "embeddings": X.tolist(),
            "params": {"min_cluster_size": 3}
        }
        
        r = app_client.post("/cluster", json=payload)
        assert r.status_code == 200
        data = r.json()
        
        # Check that medoid indices are valid
        for cluster_info in data["cluster_metadata"].values():
            medoid_idx = cluster_info["medoid_index"]
            assert 0 <= medoid_idx < len(data["labels"])

    def test_determinism_with_endpoint(self, app_client, small_blob_data):
        """Test that endpoint is deterministic with same inputs.

        Use label-invariant and tolerant numeric comparisons rather than exact
        floating-point or label-integer equality.
        """
        X, _ = small_blob_data
        payload = {
            "embeddings": X.tolist(),
            "params": {"random_state": 42}
        }

        # Make two identical requests
        r1 = app_client.post("/cluster", json=payload)
        r2 = app_client.post("/cluster", json=payload)

        assert r1.status_code == 200
        assert r2.status_code == 200

        data1 = r1.json()
        data2 = r2.json()

        # Labels should be equivalent up to relabeling
        from test_utils import adjusted_rand_index, pairwise_dists
        import numpy as np

        labels1 = np.array(data1["labels"])
        labels2 = np.array(data2["labels"])
        ari = adjusted_rand_index(labels1, labels2)
        assert ari == 1.0

        # Probabilities should be close
        probs1 = np.array(data1["probabilities"], dtype=float)
        probs2 = np.array(data2["probabilities"], dtype=float)
        np.testing.assert_allclose(probs1, probs2, rtol=1e-6, atol=1e-6)

        # Reduced vectors: compare pairwise distances (UMAP may be rotated/reflected)
        reduced1 = np.array(data1["reduced_vectors"], dtype=float)
        reduced2 = np.array(data2["reduced_vectors"], dtype=float)
        D1 = pairwise_dists(reduced1)
        D2 = pairwise_dists(reduced2)
        np.testing.assert_allclose(D1, D2, rtol=1e-4, atol=1e-4)

        # If UMAP 2D exists, compare pairwise distances as well
        if data1.get("umap_2d") is not None and data2.get("umap_2d") is not None:
            umap1 = np.array(data1["umap_2d"], dtype=float)
            umap2 = np.array(data2["umap_2d"], dtype=float)
            D1u = pairwise_dists(umap1)
            D2u = pairwise_dists(umap2)
            np.testing.assert_allclose(D1u, D2u, rtol=1e-4, atol=1e-4)

        # Metadata should be consistent with labels (sizes match)
        meta1 = data1["cluster_metadata"]
        for cid, info in meta1.items():
            assert info["size"] == data1["labels"].count(int(cid))