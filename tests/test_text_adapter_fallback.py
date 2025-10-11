import numpy as np

from oscillink.adapters import text as text_mod
from oscillink.adapters.text import embed_texts, simple_text_embed


def test_empty_input_returns_zero_rows():
    out = embed_texts([])
    assert isinstance(out, np.ndarray)
    assert out.shape == (0, 384)


def test_fallback_shapes_and_determinism_and_norms(monkeypatch):
    # Force fallback path regardless of optional sentence-transformers availability
    monkeypatch.setattr(text_mod, "_load_st_model", lambda *a, **k: None, raising=False)
    texts = [
        "hello world",
        "Oscillink coherence",
        "teh quick brown fox",  # typo shouldn't matter for embeddings
    ]
    e1 = embed_texts(texts, fallback_dim=128, normalize=True)
    e2 = embed_texts(texts, fallback_dim=128, normalize=True)
    assert e1.shape == (len(texts), 128)
    assert np.allclose(e1, e2)  # deterministic hash-based fallback
    norms = np.linalg.norm(e1, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_fallback_without_normalize_has_nonzero_norms(monkeypatch):
    monkeypatch.setattr(text_mod, "_load_st_model", lambda *a, **k: None, raising=False)
    texts = ["a", "b", "c"]
    e = embed_texts(texts, fallback_dim=64, normalize=False)
    assert e.shape == (3, 64)
    norms = np.linalg.norm(e, axis=1)
    # our simple fallback yields unit vectors scaled by their norm; still > 0
    assert np.all(norms > 0)


def test_simple_text_embed_is_deterministic():
    texts = ["same text", "another one"]
    a = simple_text_embed(texts, d=32)
    b = simple_text_embed(texts, d=32)
    assert np.allclose(a, b)
    # normalized rows
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
