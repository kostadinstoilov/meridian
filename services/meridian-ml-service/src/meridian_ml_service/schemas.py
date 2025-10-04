from typing import Optional, List, Dict
from enum import Enum

from pydantic import BaseModel, Field


class UMAPStatus(str, Enum):
    computed = "computed"
    skipped_insufficient_points = "skipped_insufficient_points"
    skipped_disabled = "skipped_disabled"
    failed = "failed"


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of texts to embed")


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of computed embeddings")
    model_name: str = Field(..., description="Name of the model used")


class ClusterParamsModel(BaseModel):
    umap_neighbors: int = Field(15, ge=2, description="Number of neighbors for UMAP")
    umap_components: int = Field(15, ge=1, le=100, description="Number of components for UMAP")
    umap_min_dist: float = Field(0.0, ge=0.0, le=1.0, description="Minimum distance for UMAP")
    umap_metric: str = Field("cosine", description="Distance metric for UMAP")
    min_cluster_size: int = Field(6, ge=1, description="Minimum cluster size for HDBSCAN")
    min_samples: Optional[int] = Field(None, ge=1, description="Minimum samples for HDBSCAN")
    cluster_selection_epsilon: float = Field(0.0, ge=0.0, description="Cluster selection epsilon")
    cluster_selection_method: str = Field("eom", description="Cluster selection method")
    random_state: int = Field(42, ge=0, description="Random state for reproducibility")
    create_2d_viz: bool = Field(True, description="Create 2D visualization")
    normalize_vectors: bool = Field(True, description="Normalize input vectors")
    # Optional: allow controlling UMAP/HDBSCAN cpu usage
    n_jobs: Optional[int] = Field(None, description="Number of CPU threads to use; if None a bounded default is chosen")


class ClusterRequest(BaseModel):
    texts: Optional[list[str]] = Field(
        default=None, description="Raw texts to embed and cluster"
    )
    embeddings: Optional[List[List[float]]] = Field(
        default=None, description="Precomputed vectors to cluster"
    )
    params: ClusterParamsModel = Field(
        default_factory=ClusterParamsModel, description="Clustering parameters"
    )


class ClusterInfo(BaseModel):
    size: int
    medoid_index: int
    exemplar_indices: List[int]


class UMAPInfo(BaseModel):
    status: UMAPStatus
    reason: Optional[str] = None


class ClusterResponse(BaseModel):
    labels: List[int]
    probabilities: List[float]
    reduced_vectors: Optional[List[List[float]]] = None
    umap_2d: Optional[List[List[float]]] = None
    umap_info: UMAPInfo
    cluster_metadata: Dict[str, ClusterInfo]
    model_name: str