"""Small test utilities for clustering determinism checks.

These helpers are intentionally dependency-free (no sklearn/scipy) so tests
can run in CI without extra packages.
"""
from __future__ import annotations

import numpy as np
from math import comb


def relabel_by_reference(pred_labels: np.ndarray, ref_labels: np.ndarray) -> np.ndarray:
    """Map predicted labels to best-matching reference labels.

    Noise label (-1) remains -1.
    """
    pred = np.asarray(pred_labels)
    ref = np.asarray(ref_labels)
    mapped = pred.copy()

    for pl in np.unique(pred):
        if pl == -1:
            continue
        idx = pred == pl
        if idx.sum() == 0:
            continue
        ref_vals, counts = np.unique(ref[idx], return_counts=True)
        # prefer non-noise, then larger counts, then smaller label for stability
        order = np.argsort([(- (rv != -1), -cnt, rv) for rv, cnt in zip(ref_vals, counts)])
        best_ref = ref_vals[order[0]]
        mapped[idx] = best_ref
    return mapped


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Compact ARI implementation for small test sizes.

    Keeps -1 (noise) out of the clustering comparison.
    """
    la = np.asarray(labels_a)
    lb = np.asarray(labels_b)

    def canonize(x):
        vals = [v for v in np.unique(x) if v != -1]
        mapping = {v: i for i, v in enumerate(vals)}
        return np.array([mapping.get(v, -1) for v in x])

    a = canonize(la)
    b = canonize(lb)
    n = len(a)
    if n == 0:
        return 1.0

    ua = [v for v in np.unique(a) if v != -1]
    ub = [v for v in np.unique(b) if v != -1]

    if len(ua) == 0 and len(ub) == 0:
        return 1.0

    # contingency table
    table = np.zeros((len(ua), len(ub)), dtype=int)
    for i, va in enumerate(ua):
        for j, vb in enumerate(ub):
            table[i, j] = int(np.sum((a == va) & (b == vb)))

    sum_comb = sum(comb(int(x), 2) for x in table.flat)
    a_marg = table.sum(axis=1)
    b_marg = table.sum(axis=0)
    sum_comb_a = sum(comb(int(x), 2) for x in a_marg)
    sum_comb_b = sum(comb(int(x), 2) for x in b_marg)
    total_comb = comb(n, 2) if n >= 2 else 0

    expected = (sum_comb_a * sum_comb_b) / total_comb if total_comb else 0.0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return (sum_comb - expected) / denom


def coords_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-5, rtol: float = 1e-6) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, atol=atol, rtol=rtol)


def pairwise_dists(a: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix without scipy.

    a should be shape (n_samples, n_features).
    """
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    diff = a[:, None, :] - a[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    return D
