from __future__ import annotations

import hashlib
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Iterable, List

import numpy as np


def simple_text_embed(texts: list[str], d: int = 384) -> np.ndarray:
    """Deterministic hash-based embeddings (placeholder).

    Replace with sentence-transformers / CLIP (or other real embedding model) in production.
    """
    out = np.zeros((len(texts), d), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode("utf-8")).digest()
        rs = np.random.RandomState(int.from_bytes(h[:8], "little", signed=False) % (2**31 - 1))
        v = rs.randn(d).astype(np.float32)
        out[i] = v / (np.linalg.norm(v) + 1e-12)
    return out


@lru_cache(maxsize=2)
def _load_st_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    # Dynamically import sentence_transformers only if available to avoid hard import
    # errors during static type checking or when the optional dependency isn't installed.
    try:  # pragma: no cover - optional dependency
        if find_spec("sentence_transformers") is None:
            return None
        st_mod = import_module("sentence_transformers")
        SentenceTransformer = getattr(st_mod, "SentenceTransformer", None)
        if SentenceTransformer is None:
            return None
        return SentenceTransformer(model_name)
    except Exception:
        return None


def embed_texts(
    texts: Iterable[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    fallback_dim: int = 384,
    normalize: bool = True,
) -> np.ndarray:
    """Flexible embedding helper.

    Tries sentence-transformers; falls back to deterministic hash embeddings.

    Parameters
    ----------
    texts : iterable of str
        Input texts.
    model_name : str
        Name of sentence-transformers model to attempt.
    fallback_dim : int
        Dimensionality for hash fallback.
    normalize : bool
        L2 normalize output rows when True.
    """
    texts_list: List[str] = list(texts)
    if not texts_list:
        return np.zeros((0, fallback_dim), dtype=np.float32)

    model = _load_st_model(model_name)
    if model is not None:
        try:
            vecs = model.encode(
                texts_list,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )
            if not normalize:
                # Model may have normalized already; if not, we accept raw.
                pass
            return vecs.astype(np.float32)
        except Exception:
            pass  # fall back

    # Fallback
    emb = simple_text_embed(texts_list, d=fallback_dim)
    if not normalize:
        return emb * np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


__all__ = [
    "simple_text_embed",
    "embed_texts",
]
