"""Screened diffusion based gate preprocessor.

This module provides an optional physics‑inspired preprocessing step that derives
adaptive gating weights for the query attraction term via a *screened diffusion*
process over the anchor graph.

Mathematical model
------------------
Solve the linear system:
    (L_sym + gamma I) h = beta * s
where
    L_sym : normalized graph Laplacian (I - D^{-1/2} A D^{-1/2})
    s     : non‑negative source strengths (e.g. similarity to query embedding)
    beta  : injection scaling
    gamma : screening / dissipation (>0 ensures locality & SPD)

The solution h is scaled to [0,1] and can be passed as `gates` to
`OscillinkLattice.set_query(psi, gates=h)`.

Design notes
------------
- Uses the existing mutual kNN adjacency for structural consistency with the lattice.
- Falls back to identity gating (uniform) if numerical issues (rare) are detected.
- Keeps dependencies minimal (pure NumPy); system is SPD so direct solve via `np.linalg.solve` is stable
  for moderate N (phase‑1 target sizes). For very large N, an iterative CG could be substituted.
"""

from __future__ import annotations  # noqa: I001
from typing import Optional  # noqa: I001
import numpy as np  # noqa: I001
from ..core.graph import mutual_knn_adj, normalized_laplacian, row_sum_cap  # noqa: I001
from ..core.solver import cg_solve  # local import to avoid adding a new dependency


def compute_diffusion_gates(
    Y: np.ndarray,
    psi: np.ndarray,
    *,
    kneighbors: int = 6,
    row_cap_val: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.1,
    similarity: str = "cosine",
    deterministic_k: bool = False,
    neighbor_seed: Optional[int] = None,
    clamp: bool = True,
    method: str = "direct",
    tol: float = 1e-4,
    max_iters: int = 256,
) -> np.ndarray:
    """Compute screened diffusion gates.

    Parameters
    ----------
    Y : (N,D) float32 array
        Anchor embeddings.
    psi : (D,) float32 array
        Query embedding.
    kneighbors : int, default 6
        Number of nearest neighbors for the structural graph.
    row_cap_val : float, default 1.0
        Row sum cap applied to adjacency for stability (mirrors lattice construction).
    beta : float, default 1.0
        Source injection scaling.
    gamma : float, default 0.1
        Dissipation / screening (must be >0 for strict SPD).
    similarity : {"cosine"}, default "cosine"
        Similarity measure for source strengths. (Future: allow dot / exp kernels.)
    deterministic_k : bool, default False
        If True, builds fully deterministic neighbor sets (slower for large N).
    neighbor_seed : int, optional
        Seed to break ties deterministically without full sort when deterministic_k=False.
    clamp : bool, default True
        If True, normalizes resulting h to [0,1].

    Returns
    -------
    h : (N,) float32 array
        Normalized gating weights suitable for `OscillinkLattice.set_query(..., gates=h)`.
    """
    if Y.ndim != 2:
        raise ValueError("Y must be 2D")
    N, D = Y.shape
    if psi.shape[0] != D:
        raise ValueError("psi dimension mismatch")
    if gamma <= 0:
        raise ValueError("gamma must be > 0 for SPD")
    if kneighbors < 1:
        raise ValueError("kneighbors must be >=1")

    # Explicit annotations to satisfy mypy var-annotated check
    Yf: np.ndarray = Y.astype(np.float32, copy=False)
    psif: np.ndarray = psi.astype(np.float32, copy=False)

    # 1. Build structural adjacency consistent with lattice defaults.
    A = mutual_knn_adj(
        Yf,
        k=kneighbors,
        deterministic=deterministic_k,
        seed=neighbor_seed,
    )
    A = row_sum_cap(A, row_cap_val)
    L_sym, _ = normalized_laplacian(A)

    # 2. Source strengths from similarity to query.
    if similarity == "cosine":
        Yn = Yf / (np.linalg.norm(Yf, axis=1, keepdims=True) + 1e-12)
        psi_n = psif / (np.linalg.norm(psif) + 1e-12)
        s = (Yn @ psi_n).astype(np.float32)
    else:
        raise ValueError("unsupported similarity metric")
    # Positive injection only (restrict to forward influence) & scale beta.
    s = beta * np.maximum(0.0, s)

    # 3. Solve (L_sym + gamma I) h = s (screened diffusion / Poisson).
    h = _solve_screened_diffusion(L_sym, gamma, s, method=method, tol=tol, max_iters=max_iters)

    # 4. Clamp / normalize.
    if clamp:
        h_min = float(np.min(h))
        h_max = float(np.max(h))
        h = np.ones(N, dtype=np.float32) if h_max - h_min < 1e-12 else (h - h_min) / (h_max - h_min)
    h = np.clip(h, 0.0, 1.0).astype(np.float32)
    return h


__all__ = ["compute_diffusion_gates"]


def _solve_screened_diffusion(
    L_sym: np.ndarray, gamma: float, s: np.ndarray, *, method: str, tol: float, max_iters: int
) -> np.ndarray:
    """Solve (L_sym + gamma I) h = s using either direct or CG method.

    Returns h as float32. On failure, returns uniform ones.
    """
    N = L_sym.shape[0]
    if method == "cg":
        # Jacobi preconditioner: diag(L_sym) + gamma
        M_diag = np.diag(L_sym).astype(np.float32) + float(gamma)

        def A_mul(x: np.ndarray) -> np.ndarray:
            x2 = x if x.ndim == 2 else x[:, None]
            out = (L_sym @ x2) + gamma * x2
            return out if x.ndim == 2 else out.squeeze()

        try:
            h, _iters, _res = cg_solve(
                A_mul, s.astype(np.float32), x0=None, M_diag=M_diag, tol=tol, max_iters=max_iters
            )
            return h.astype(np.float32)
        except Exception:
            return np.ones(N, dtype=np.float32)
    # direct
    M = L_sym + gamma * np.eye(N, dtype=np.float32)
    try:
        return np.linalg.solve(M, s).astype(np.float32)
    except np.linalg.LinAlgError:
        M_pert = M + 1e-6 * np.eye(N, dtype=np.float32)
        try:
            return np.linalg.solve(M_pert, s).astype(np.float32)
        except np.linalg.LinAlgError:
            return np.ones(N, dtype=np.float32)
