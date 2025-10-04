from typing import Optional

import os
import inspect
from .schemas import UMAPStatus

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
        n_jobs: Optional[int] = None,
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
        # Determine n_jobs default. Prefer an environment variable override
        # (UMAP_N_JOBS) if set; otherwise fall back to a bounded default to avoid
        # saturating CPU in server contexts.
        if n_jobs is None:
            env_val = os.getenv("UMAP_N_JOBS")
            if env_val is not None:
                try:
                    parsed = int(env_val)
                    self.n_jobs = max(1, parsed)
                except ValueError:
                    # Fall back to bounded default if env var is invalid
                    cpu = os.cpu_count() or 1
                    self.n_jobs = min(4, max(1, cpu))
            else:
                # Leave n_jobs as None by default to allow downstream callers
                # (UMAP/HDBSCAN) to use single-threaded defaults, which is more
                # deterministic for tests and CI environments. Tests can set
                # UMAP_N_JOBS when a specific concurrency is required.
                self.n_jobs = None
        else:
            self.n_jobs = n_jobs


def _build_umap_kwargs(
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
    umap_n_jobs: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Build kwargs for umap.UMAP while guarding unsupported arguments.

    Some versions of umap-learn do not accept the `n_jobs` argument; passing it
    will raise a TypeError. Inspect the UMAP signature and only include
    `n_jobs` when supported. This keeps callers free to request single-threaded
    operation in tests via the UMAP_N_JOBS env var while remaining compatible
    with older library versions.
    """
    params = dict(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    try:
        sig = inspect.signature(umap.UMAP)
        if "n_jobs" in sig.parameters and umap_n_jobs is not None:
            params["n_jobs"] = int(umap_n_jobs)
        if "verbose" in sig.parameters:
            params["verbose"] = verbose
    except (ValueError, TypeError):
        # If signature inspection fails for any reason, omit optional args to be safe.
        pass
    return params


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
) -> tuple[Optional[np.ndarray], np.ndarray, np.ndarray, Optional[np.ndarray], dict[str, dict], dict]:
    """Run UMAP + HDBSCAN clustering with safeguards for small datasets.

    This function caps UMAP parameters to avoid ARPACK failures when the
    requested number of components is too large for the number of samples.
    It also guards passing unsupported arguments to umap.UMAP.
    """
    X = embeddings.astype(np.float32, copy=False)

    # Empty input: raise to surface the issue to callers (unit tests expect an error)
    if X.size == 0 or X.shape[0] == 0:
        raise ValueError("Embeddings array is empty")

    if params.normalize_vectors:
        Xn = l2_normalize(X)
    else:
        Xn = X

    n_samples = Xn.shape[0]

    umap_2d = None
    umap_status = None
    umap_reason = None

    # Separate the features used for clustering from what we return to clients
    features_for_clustering: Optional[np.ndarray] = None
    reduced_for_output: Optional[np.ndarray] = None

    # Handle very small sample counts where UMAP's spectral init would fail
    if n_samples == 1:
        # Explicitly treat single sample as noise and skip returning reduced vectors
        labels = np.array([-1], dtype=int)
        probs = np.array([0.0], dtype=float)
        meta = {}
        umap_status = UMAPStatus.skipped_insufficient_points
        umap_reason = "n<3"
        return None, labels, probs, umap_2d, meta, {"status": umap_status.value, "reason": umap_reason}

    elif n_samples == 2:
        # Skip UMAP outputs and do not fabricate reduced vectors
        reduced_for_output = None
        umap_status = UMAPStatus.skipped_insufficient_points
        umap_reason = "n<3"
        # Use the original (possibly normalized) vectors for clustering
        features_for_clustering = Xn.astype(float)

    else:
        # For spectral initialization UMAP requests k = n_components + 1 which
        # must be < n_samples. So enforce n_components <= n_samples - 2.
        max_allowed_components = max(1, n_samples - 2)
        effective_components = min(params.umap_components, max_allowed_components)

        # n_neighbors must be < n_samples; clamp to sensible range
        effective_n_neighbors = min(params.umap_neighbors, max(2, n_samples - 1))

        # Ensure a stable random state is used (default to 42)
        rs = params.random_state if params.random_state is not None else 42

        # Build kwargs for reduced-dimension UMAP while guarding unsupported args
        kwargs = _build_umap_kwargs(
            n_components=effective_components,
            n_neighbors=effective_n_neighbors,
            min_dist=params.umap_min_dist,
            metric=params.umap_metric,
            random_state=rs,
            umap_n_jobs=params.n_jobs,
            verbose=False,
        )

        try:
            umap_model = umap.UMAP(**kwargs)
            reduced = umap_model.fit_transform(Xn)
            umap_status = UMAPStatus.computed
            umap_reason = None
        except Exception as e:
            # If UMAP fails for any reason, fall back to a simple truncation to
            # keep downstream clustering running and mark the UMAP status as failed.
            effective_components = min(effective_components, Xn.shape[1])
            reduced = Xn[:, :effective_components].astype(float)
            umap_status = UMAPStatus.failed
            umap_reason = str(e)

        # Optionally compute a separate 2D visualization.
        if params.create_2d_viz:
            kwargs2 = _build_umap_kwargs(
                n_components=2,
                n_neighbors=effective_n_neighbors,
                min_dist=0.0,
                metric=params.umap_metric,
                random_state=rs,
                umap_n_jobs=params.n_jobs,
                verbose=False,
            )
            try:
                umap2 = umap.UMAP(**kwargs2)
                umap_2d = umap2.fit_transform(Xn)
            except Exception:
                # If 2D viz fails, don't fail the whole pipeline; leave umap_2d as None
                # and preserve umap_status/reason from the main reduction attempt.
                umap_2d = None

        features_for_clustering = np.asarray(reduced, dtype=float)
        reduced_for_output = features_for_clustering

    # Ensure features_for_clustering is set for clustering paths
    if features_for_clustering is None:
        raise RuntimeError("features_for_clustering was not set")

    # Ensure 2D shape for clustering features
    if features_for_clustering.ndim == 1:
        features_for_clustering = features_for_clustering.reshape(-1, 1)

    # Run HDBSCAN on the chosen feature space (may be original Xn for n==2)
    # Ensure cluster size and min_samples are ints and not larger than the dataset
    effective_min_cluster_size = max(1, min(params.min_cluster_size, n_samples))
    # If min_samples is not provided, HDBSCAN uses min_cluster_size; set an
    # explicit value to avoid internal queries with k > n_samples.
    if params.min_samples is None:
        effective_min_samples = effective_min_cluster_size
    else:
        effective_min_samples = max(1, min(params.min_samples, n_samples))

    # Use single-threaded core distance computation by default when not set
    core_jobs = params.n_jobs if params.n_jobs is not None else 1
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(effective_min_cluster_size),
        min_samples=int(effective_min_samples),
        cluster_selection_epsilon=params.cluster_selection_epsilon,
        cluster_selection_method=params.cluster_selection_method,
        metric="euclidean",
        prediction_data=True,
        core_dist_n_jobs=core_jobs,
        # Avoid generating extra structures that can vary across platforms
        gen_min_span_tree=False,
    )
    clusterer.fit(features_for_clustering)
    labels = clusterer.labels_
    probs = clusterer.probabilities_

    meta = _cluster_metadata(labels, probs, Xn)

    # Return reduced_for_output (None if we intentionally skipped producing one)
    return (
        None if reduced_for_output is None else reduced_for_output.astype(float),
        labels,
        probs,
        None if umap_2d is None else umap_2d.astype(float),
        meta,
        {"status": (umap_status.value if umap_status is not None else UMAPStatus.failed.value), "reason": umap_reason},
    )
