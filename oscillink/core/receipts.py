from __future__ import annotations

import hashlib
import hmac
import json

import numpy as np


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
        nz = np.nonzero(A[i])[0]
        if nz.size == 0:
            continue
        yi = Yn[i]
        ui = Un[i]
        for j in nz:
            w = float(A[i, j])
            if w <= 0.0:
                continue
            ydiff = yi - Yn[j]
            udiff = ui - Un[j]
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


def verify_receipt_mode(
    receipt: dict,
    secret: bytes | str,
    require_mode: str | None = None,
    minimal_subset: bool = False,
    required_sig_v: int | None = None,
) -> tuple[bool, dict | None]:
    """Enhanced verification supporting minimal vs extended modes.

    Parameters
    ----------
    receipt : dict
        Receipt object produced by OscillinkLattice.receipt().
    secret : bytes | str
        HMAC secret.
    require_mode : {'minimal','extended',None}
        If set, verification fails when payload['mode'] != require_mode.
    minimal_subset : bool
        When True and the payload mode is 'extended', compute an alternative
        verification pass over the minimal subset fields (state_sig, deltaH_total, mode='minimal').
        This allows consumers that only care about the core claim to down-scope trust.

    Returns
    -------
    (ok, payload_or_none)
        ok True when signature valid (and mode requirement satisfied). payload contains the signed payload.

    Notes
    -----
    - Always prefer trusting the full payload when available; minimal_subset is a compatibility bridge.
    - If minimal_subset is used and the recomputed minimal HMAC does not match, ok=False.
    """
    try:
        sig_block = receipt.get("meta", {}).get("signature")
        if not sig_block or sig_block.get("algorithm") != "HMAC-SHA256":
            return False, None
        payload = sig_block.get("payload")
        sig_hex = sig_block.get("signature")
        if payload is None or sig_hex is None:
            return False, None
        mode = payload.get("mode")
        if require_mode and mode != require_mode:
            return False, None
        if required_sig_v is not None and payload.get("sig_v") != required_sig_v:
            return False, None
        secret_b = secret.encode("utf-8") if isinstance(secret, str) else secret
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        calc = hmac.new(secret_b, raw, hashlib.sha256).hexdigest()
        if hmac.compare_digest(calc, str(sig_hex)):
            return True, payload
        # Optionally attempt minimal subset verification from an extended payload
        if minimal_subset and mode == "extended":
            minimal_payload = {
                "sig_v": payload.get("sig_v"),  # preserve version for subset validation
                "mode": "minimal",
                "state_sig": payload.get("state_sig"),
                "deltaH_total": payload.get("deltaH_total"),
            }
            raw_min = json.dumps(minimal_payload, sort_keys=True).encode("utf-8")
            calc_min = hmac.new(secret_b, raw_min, hashlib.sha256).hexdigest()
            # If the signature was created in minimal mode originally, extended mismatch will not validate.
            # We only accept if the current signature matches the minimal structure (meaning original was minimal).
            if hmac.compare_digest(calc_min, str(sig_hex)) and (require_mode in (None, "minimal")):
                return True, minimal_payload
        return False, None
    except Exception:
        return False, None
