"""Test configuration and fixtures for meridian-ml-service."""

import os
import numpy as np
import pytest
from fastapi.testclient import TestClient


def make_blobs(n_per=20, d=32, sep=5.0, seed=42):
    """Generate synthetic blob data with clear cluster structure."""
    rng = np.random.default_rng(seed)
    centers = np.stack([
        np.pad([+sep], (0, d-1)),
        np.pad([-sep], (0, d-1)),
        np.pad([0.0], (0, d-1)) + sep * np.roll(np.eye(d)[0], 1)
    ], axis=0)  # 3 centers in d-dim
    blobs = []
    labels = []
    for i, c in enumerate(centers):
        x = rng.normal(loc=c, scale=1.0, size=(n_per, d))
        blobs.append(x)
        labels.extend([i] * n_per)
    X = np.vstack(blobs).astype(np.float32)
    y = np.array(labels, dtype=int)
    return X, y


@pytest.fixture
def small_blob_data():
    """Small synthetic dataset with 3 well-separated clusters."""
    X, y = make_blobs(n_per=10, d=32, sep=6.0, seed=123)
    return X, y


@pytest.fixture
def very_small_blob_data():
    """Very small synthetic dataset for edge case testing."""
    X, y = make_blobs(n_per=3, d=16, sep=6.0, seed=7)
    return X, y


@pytest.fixture
def app_client():
    """Create a test client for the FastAPI app."""
    # Import here so tests don't import app at collection time if deps missing
    from meridian_ml_service.main import app
    client = TestClient(app)
    return client


@pytest.fixture
def app_client_with_token(monkeypatch):
    """Create a test client with API token authentication."""
    monkeypatch.setenv("API_TOKEN", "devtoken123")
    from importlib import reload
    from meridian_ml_service import main
    reload(main)
    from meridian_ml_service.main import app
    client = TestClient(app)
    return client


@pytest.fixture
def mock_embed_texts(monkeypatch, small_blob_data):
    """Mock the compute_embeddings function to return synthetic embeddings."""
    X, _ = small_blob_data
    def _mock(texts, model_components, batch_size=32, normalize=True, e5_prefix=None):
        # Return as many rows as texts, cycling through X
        k = len(texts)
        reps = (k + len(X) - 1) // len(X)
        out = np.vstack([X] * reps)[:k]
        return out
    
    from meridian_ml_service import embeddings
    monkeypatch.setattr(embeddings, "compute_embeddings", _mock)


@pytest.fixture
def random_embeddings():
    """Generate random embeddings for testing."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(50, 32)).astype(np.float32)


@pytest.fixture
def outlier_embeddings():
    """Generate embeddings with clear outliers."""
    rng = np.random.default_rng(42)
    # Most points in a tight cluster
    normal_points = rng.normal(loc=0.0, scale=0.5, size=(20, 32))
    # A few clear outliers
    outliers = rng.normal(loc=10.0, scale=1.0, size=(5, 32))
    X = np.vstack([normal_points, outliers]).astype(np.float32)
    return X