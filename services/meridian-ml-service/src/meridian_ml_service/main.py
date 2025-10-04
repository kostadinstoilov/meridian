import numpy as np
from fastapi import Depends, FastAPI, HTTPException

from .clustering import ClusterParams, run_clustering
from .config import settings
from .dependencies import (
    ModelDep,
    verify_token,
    get_embedding_model,
)
from .embeddings import compute_embeddings
from .schemas import (
    ClusterRequest,
    ClusterResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)

app = FastAPI(
    title="Meridian ML Service",
    description="Handles ML tasks like embeddings and clustering.",
    version="0.1.0",
)


# Simple root endpoint for health check
@app.get("/")
async def read_root():
    return {"status": "ok", "service": "Meridian ML Service"}


@app.get("/ping")
async def ping():
    return {"pong": True}


@app.post("/embeddings", response_model=EmbeddingResponse)
async def api_compute_embeddings(
    request: EmbeddingRequest,
    model_components: ModelDep,
    _: None = Depends(verify_token),
):
    """
    Computes embeddings for the provided list of texts.
    """
    print(f"Received request to embed {len(request.texts)} texts.")
    try:
        embeddings_np: np.ndarray = compute_embeddings(
            texts=request.texts,
            model_components=model_components,
        )

        embeddings_list: list[list[float]] = embeddings_np.tolist()

        return EmbeddingResponse(
            embeddings=embeddings_list, model_name=settings.embedding_model_name
        )
    except Exception as e:
        print(f"ERROR during embedding computation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during embedding computation: {str(e)}",
        ) from e


def _to_params(p) -> ClusterParams:
    return ClusterParams(
        umap_neighbors=p.umap_neighbors,
        umap_components=p.umap_components,
        umap_min_dist=p.umap_min_dist,
        umap_metric=p.umap_metric,
        min_cluster_size=p.min_cluster_size,
        min_samples=p.min_samples,
        cluster_selection_epsilon=p.cluster_selection_epsilon,
        cluster_selection_method=p.cluster_selection_method,
        random_state=p.random_state,
        create_2d_viz=p.create_2d_viz,
        normalize_vectors=p.normalize_vectors,
    )


@app.post("/cluster", response_model=ClusterResponse)
async def api_cluster(
    req: ClusterRequest,
    model_components: ModelDep,
    _: None = Depends(verify_token),
):
    """
    Clusters texts or precomputed embeddings using UMAP + HDBSCAN.
    """
    if not req.texts and not req.embeddings:
        raise HTTPException(
            status_code=400, detail="Provide either texts or embeddings."
        )

    model_name = settings.embedding_model_name

    if req.embeddings:
        X = np.array(req.embeddings, dtype=np.float32)
    elif req.texts:
        texts = [
            t if t.startswith("passage:") or t.startswith("query:") else f"passage: {t}"
            for t in req.texts
        ]
        print(f"Received request to cluster {len(texts)} texts.")
        embeddings_np: np.ndarray = compute_embeddings(
            texts=texts,
            model_components=model_components,
        )
        X = embeddings_np
    else:
        raise HTTPException(
            status_code=400, detail="Provide either texts or embeddings."
        )

    params = _to_params(req.params)
    reduced, labels, probs, umap_2d, meta = run_clustering(X, params)

    return ClusterResponse(
        labels=[int(x) for x in labels.tolist()],
        probabilities=[float(x) for x in probs.tolist()],
        reduced_vectors=reduced.astype(float).tolist(),
        umap_2d=None if umap_2d is None else umap_2d.astype(float).tolist(),
        cluster_metadata=meta,
        model_name=model_name,
    )
