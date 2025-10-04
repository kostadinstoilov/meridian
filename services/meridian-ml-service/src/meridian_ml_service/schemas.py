from typing import Optional

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of texts to embed")


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]] = Field(
        ..., description="List of computed embeddings"
    )
    model_name: str = Field(..., description="Name of the model used")


class ClusterParamsModel(BaseModel):
    umap_neighbors: int = 15
    umap_components: int = 15
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    min_cluster_size: int = 6
    min_samples: Optional[int] = None
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "eom"
    random_state: int = 42
    create_2d_viz: bool = True
    normalize_vectors: bool = True


class ClusterRequest(BaseModel):
    texts: Optional[list[str]] = Field(
        default=None, description="Raw texts to embed and cluster"
    )
    embeddings: Optional[list[list[float]]] = Field(
        default=None, description="Precomputed vectors to cluster"
    )
    params: ClusterParamsModel = Field(
        default_factory=ClusterParamsModel, description="Clustering parameters"
    )


class ClusterResponse(BaseModel):
    labels: list[int]
    probabilities: list[float]
    reduced_vectors: list[list[float]]
    umap_2d: Optional[list[list[float]]] = None
    cluster_metadata: dict[str, dict]
    model_name: str
