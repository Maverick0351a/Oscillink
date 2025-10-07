from __future__ import annotations

import hashlib

import numpy as np


def simple_text_embed(texts: list[str], d: int = 384) -> np.ndarray:
    """Deterministic hash-based embeddings (placeholder).

    Replace with sentence-transformers / CLIP (or other real embedding model) in production.
    """
    out = np.zeros((len(texts), d), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode("utf-8")).digest()
        rs = np.random.RandomState(int.from_bytes(h[:8], "little", signed=False) % (2**31-1))
        v = rs.randn(d).astype(np.float32)
        out[i] = v / (np.linalg.norm(v) + 1e-12)
    return out
