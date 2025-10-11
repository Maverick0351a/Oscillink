from __future__ import annotations

from typing import Optional

import numpy as np


def mutual_knn_adj(
    Y: np.ndarray,
    k: int,
    *,
    deterministic: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Build a symmetric mutual-kNN adjacency by cosine similarity; nonnegative weights.

    Parameters
    ----------
    Y : (N,D) float32 array
    k : int
        Number of neighbors.
    deterministic : bool, optional
        If True, performs a full argsort with stable tie-breaking on (similarity desc, index asc)
        to ensure reproducible neighbor sets. Slightly slower than argpartition.
    seed : int, optional
        If provided (and deterministic=False), adds a tiny jitter (stable per seed) only to break
        similarity ties deterministically without a full sort.
    """
    N = Y.shape[0]
    if N <= 1:
        # Degenerate case: no neighbors possible
        return np.zeros((N, N), dtype=np.float32)
    # Clamp k to valid range [1, N-1]
    k = int(max(1, min(k, N - 1)))
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    S = Yn @ Yn.T
    np.fill_diagonal(S, -np.inf)

    if deterministic:
        # Full deterministic ordering: similarity descending, then index ascending.
        # np.lexsort uses last key as primary, so provide (index, -similarity).
        A = np.zeros((N, N), dtype=np.float32)
        indices = np.arange(N)
        for i in range(N):
            row = S[i]
            order = np.lexsort(
                (indices, -row)
            )  # primary: -row (similarity desc), tie-break: index asc
            keep = order[:k]
            for j in keep:
                if row[j] > 0:
                    A[i, j] = float(max(row[j], 0.0))
    else:
        if seed is not None:
            rng = np.random.default_rng(seed)
            # Add minimal jitter to break exact ties reproducibly
            jitter = rng.uniform(-1e-8, 1e-8, size=S.shape)
            S = S + jitter
        idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
        A = np.zeros((N, N), dtype=np.float32)
        rows = np.arange(N)[:, None]
        A[rows, idx] = S[rows, idx].astype(np.float32).clip(min=0.0)

    M = ((A > 0) & (A.T > 0)).astype(np.float32)
    A = np.maximum(A * M, (A * M).T)
    return A


def row_sum_cap(A: np.ndarray, cap: float) -> np.ndarray:
    """Cap row sums while preserving (approximate) symmetry.

    Original implementation scaled rows independently which could break symmetry
    of an already symmetrized mutual-kNN adjacency. For SPD guarantees we prefer
    to keep A as close to symmetric as possible. We compute per-row scale then
    apply the geometric mean of scale_i and scale_j to each edge weight.
    """
    sums = A.sum(axis=1, keepdims=True) + 1e-12
    scale = np.minimum(1.0, cap / sums).astype(np.float32)
    # geometric mean scaling for symmetry preservation
    gs = np.sqrt(scale * scale.T)
    A2 = A * gs
    # final symmetrization guard (numerical drift)
    return 0.5 * (A2 + A2.T)


def normalized_laplacian(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = A.sum(axis=1)
    sqrt_deg = np.sqrt(np.maximum(d, 1e-12))
    Dm12 = 1.0 / sqrt_deg
    W = (A * Dm12[:, None]) * Dm12[None, :]
    N = A.shape[0]
    L = np.eye(N, dtype=np.float32) - W.astype(np.float32)
    return L, sqrt_deg


def build_path_laplacian(
    N: int,
    chain: list[int],
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        weights = [1.0] * (max(0, len(chain) - 1))
    A = np.zeros((N, N), dtype=np.float32)
    for k in range(len(chain) - 1):
        i, j = int(chain[k]), int(chain[k + 1])
        w = float(weights[k])
        if 0 <= i < N and 0 <= j < N:
            A[i, j] = max(A[i, j], w)
            A[j, i] = max(A[j, i], w)
    L, _ = normalized_laplacian(A)
    return L, A


def mmr_diversify(Y: np.ndarray, scores: np.ndarray, k: int, lambda_div: float = 0.5) -> list[int]:
    """Simple MMR over cosine similarities using anchors Y as representers."""
    if k <= 0:
        return []
    N = Y.shape[0]
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    sims = Yn @ Yn.T
    chosen: list[int] = []
    cand = set(range(N))
    while len(chosen) < min(k, N):
        best_idx, best_val = -1, -1e9
        for i in cand:
            rep = scores[i]
            div = 0.0 if not chosen else max(float(sims[i, j]) for j in chosen)
            val = (1 - lambda_div) * rep - lambda_div * div
            if val > best_val:
                best_val, best_idx = val, i
        chosen.append(best_idx)
        cand.remove(best_idx)
    return chosen
