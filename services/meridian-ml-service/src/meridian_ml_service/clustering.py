from typing import Optional

import hdbscan
import numpy as np
import umap


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


class ClusterParams:
    def __init__(
        self,
        umap_neighbors: int = 15,
        umap_components: int = 15,
        umap_min_dist: float = 0.0,
        umap_metric: str = "cosine",
        min_cluster_size: int = 6,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
        random_state: int = 42,
        create_2d_viz: bool = True,
        normalize_vectors: bool = True,
    ):
        self.umap_neighbors = umap_neighbors
        self.umap_components = umap_components
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.random_state = random_state
        self.create_2d_viz = create_2d_viz
        self.normalize_vectors = normalize_vectors


def _cluster_metadata(
    labels: np.ndarray,
    probs: np.ndarray,
    base_vectors: np.ndarray,
) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    unique_labels = sorted([int(x) for x in set(labels.tolist()) if x != -1])
    for cid in unique_labels:
        idxs = np.where(labels == cid)[0]
        centroid = base_vectors[idxs].mean(axis=0, keepdims=True)
        dists = np.linalg.norm(base_vectors[idxs] - centroid, axis=1)
        medoid_local = int(np.argmin(dists))
        medoid_index = int(idxs[medoid_local])
        meta[str(cid)] = {
            "size": int(len(idxs)),
            "medoid_index": medoid_index,
            "exemplar_indices": [
                int(i) for i in idxs[np.argsort(probs[idxs])[-3:]][::-1].tolist()
            ],
        }
    return meta


def run_clustering(
    embeddings: np.ndarray,
    params: ClusterParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], dict[str, dict]]:
    X = embeddings.astype(np.float32, copy=False)
    if params.normalize_vectors:
        Xn = l2_normalize(X)
    else:
        Xn = X

    umap_model = umap.UMAP(
        n_neighbors=params.umap_neighbors,
        n_components=params.umap_components,
        min_dist=params.umap_min_dist,
        metric=params.umap_metric,
        random_state=params.random_state,
        n_jobs=-1,
        verbose=False,
    )
    reduced = umap_model.fit_transform(Xn)

    umap_2d = None
    if params.create_2d_viz:
        umap2 = umap.UMAP(
            n_neighbors=params.umap_neighbors,
            n_components=2,
            min_dist=0.0,
            metric=params.umap_metric,
            random_state=params.random_state,
            n_jobs=-1,
            verbose=False,
        )
        umap_2d = umap2.fit_transform(Xn)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        cluster_selection_epsilon=params.cluster_selection_epsilon,
        cluster_selection_method=params.cluster_selection_method,
        metric="euclidean",
        prediction_data=True,
        core_dist_n_jobs=-1,
    )
    clusterer.fit(reduced)
    labels = clusterer.labels_
    probs = clusterer.probabilities_

    meta = _cluster_metadata(labels, probs, Xn)

    return reduced, labels, probs, umap_2d, meta
