from __future__ import annotations

import numpy as np
import hmac
import hashlib
import json


def deltaH_trace(
    U: np.ndarray,
    Ustar: np.ndarray,
    lamG: float,
    lamC: float,
    Lsym: np.ndarray,
    lamQ: float,
    Bdiag: np.ndarray,
    lamP: float = 0.0,
    Lpath: np.ndarray | None = None,
) -> float:
    diff = (U - Ustar).astype(np.float32)
    term = lamG * diff + lamC * (Lsym @ diff) + lamQ * (Bdiag[:, None] * diff)
    if Lpath is not None and lamP > 0.0:
        term = term + lamP * (Lpath @ diff)
    return float(np.sum(diff * term))


def per_node_components(
    Y: np.ndarray,
    Ustar: np.ndarray,
    A: np.ndarray,
    Lsym: np.ndarray,
    sqrt_deg: np.ndarray,
    lamG: float,
    lamC: float,
    lamQ: float,
    Bdiag: np.ndarray,
    psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Yn = Y / (sqrt_deg[:, None] + 1e-12)
    Un = Ustar / (sqrt_deg[:, None] + 1e-12)
    N = Y.shape[0]
    coh_drop = np.zeros(N, dtype=np.float32)
    for i in range(N):
        row = A[i]
        for j, w in enumerate(row):
            if w <= 0.0:
                continue
            ydiff = Yn[i] - Yn[j]
            udiff = Un[i] - Un[j]
            coh_drop[i] += 0.5 * lamC * w * (float(ydiff @ ydiff) - float(udiff @ udiff))
    anchor_pen = lamG * np.sum((Ustar - Y) ** 2, axis=1).astype(np.float32)
    qp = Ustar - psi[None, :]
    query_term = lamQ * Bdiag * np.sum(qp * qp, axis=1).astype(np.float32)
    return coh_drop, anchor_pen, query_term


def null_points(
    Ustar: np.ndarray,
    A: np.ndarray,
    sqrt_deg: np.ndarray,
    lamC: float,
    z_th: float = 3.0,
):
    Un = Ustar / (sqrt_deg[:, None] + 1e-12)
    diffs = Un[:, None, :] - Un[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    R = lamC * A * d2.astype(np.float32)
    mu = R.mean(axis=1, keepdims=True)
    sigma = R.std(axis=1, keepdims=True) + 1e-12
    Z = (R - mu) / sigma
    nulls = []
    N = Ustar.shape[0]
    for i in range(N):
        j = int(np.argmax(Z[i]))
        if R[i, j] > 0 and Z[i, j] > z_th:
            nulls.append({"edge": [i, j], "z": float(Z[i, j]), "residual": float(R[i, j])})
    return nulls


def verify_receipt(receipt: dict, secret: bytes | str) -> bool:
    """Verify an HMAC-SHA256 signed receipt produced by OscillinkLattice.

    Expects receipt['meta']['signature'] block with fields:
      - algorithm: 'HMAC-SHA256'
      - payload: {...}
      - signature: hex digest
    Returns True if signature matches, else False. Does NOT raise.
    """
    try:
        sig_block = receipt.get("meta", {}).get("signature")
        if not sig_block or sig_block.get("algorithm") != "HMAC-SHA256":
            return False
        payload = sig_block.get("payload")
        claimed = sig_block.get("signature")
        if payload is None or claimed is None:
            return False
        if isinstance(secret, str):
            secret = secret.encode("utf-8")
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        calc = hmac.new(secret, raw, hashlib.sha256).hexdigest()
        # constant time compare
        return hmac.compare_digest(calc, str(claimed))
    except Exception:
        return False
