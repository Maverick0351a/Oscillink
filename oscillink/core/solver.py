from __future__ import annotations

import numpy as np


def cg_solve(
    A_mul,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    M_diag: np.ndarray | None = None,
    tol: float = 1e-3,
    max_iters: int = 100,
):
    """Conjugate Gradient for (symmetric) linear operator A_mul. Works for multiple RHS (N x D)."""
    if b.ndim == 1:
        b = b[:, None]
    n, m = b.shape
    x = np.zeros((n, m), dtype=b.dtype) if x0 is None else x0.copy()
    r = b - A_mul(x)
    z = r if M_diag is None else r / (M_diag[:, None] + 1e-12)
    p = z.copy()
    rz_old = (r * z).sum(axis=0)
    for it in range(1, max_iters + 1):
        Ap = A_mul(p)
        denom = (p * Ap).sum(axis=0) + 1e-18
        alpha = rz_old / denom
        x = x + p * alpha
        r = r - Ap * alpha
        res = float(np.linalg.norm(r, axis=0).max())
        if res <= tol:
            return (x.squeeze() if m == 1 else x, it, res)
        z = r if M_diag is None else r / (M_diag[:, None] + 1e-12)
        rz_new = (r * z).sum(axis=0)
        beta = rz_new / (rz_old + 1e-18)
        p = z + p * beta
        rz_old = rz_new
    return (x.squeeze() if m == 1 else x, it, res)
