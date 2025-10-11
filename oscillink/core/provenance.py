from __future__ import annotations

import hashlib
from typing import Any, Dict

import numpy as np

from .lattice import OscillinkLattice


def compare_provenance(a: OscillinkLattice, b: OscillinkLattice) -> Dict[str, Any]:
    """Return a structured diff of two lattices' core provenance inputs.

    Components compared:
      - params (lamG, lamC, lamQ, lamP)
      - shape (N, D)
      - adjacency fingerprint (subset hash used in signature)
      - chain length & presence
      - query embedding psi hash (rounded) & gating vector hash (rounded)

    Hashing approach keeps payload small while giving high collision resistance for debugging.
    """

    def hash_array(arr: np.ndarray, round_decimals: int = 6) -> str:
        r = np.round(arr.astype(float), round_decimals)
        h = hashlib.sha256(r.tobytes()).hexdigest()
        return h

    def adj_fingerprint(lat: OscillinkLattice) -> str:
        nz = np.argwhere(lat.A > 0)[:2048]
        return hashlib.sha256(nz.tobytes()).hexdigest()

    pa = {"lamG": a.lamG, "lamC": a.lamC, "lamQ": a.lamQ, "lamP": a.lamP}
    pb = {"lamG": b.lamG, "lamC": b.lamC, "lamQ": b.lamQ, "lamP": b.lamP}

    out: Dict[str, Any] = {
        "same": True,
        "params_equal": pa == pb,
        "shape_equal": (a.N, a.D) == (b.N, b.D),
        "adj_equal": adj_fingerprint(a) == adj_fingerprint(b),
        "chain_equal": (a._chain_nodes is not None) == (b._chain_nodes is not None)
        and (len(a._chain_nodes or []) == len(b._chain_nodes or [])),
        "psi_equal": hash_array(a.psi) == hash_array(b.psi),
        "gates_equal": hash_array(a.B_diag) == hash_array(b.B_diag),
    }
    out["same"] = all(out[k] for k in list(out.keys()) if k.endswith("_equal"))

    if not out["same"]:
        out["detail"] = {
            "params_a": pa,
            "params_b": pb,
            "shape_a": (a.N, a.D),
            "shape_b": (b.N, b.D),
        }
    return out


__all__ = ["compare_provenance"]
