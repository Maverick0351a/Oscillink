import numpy as np

from oscillink import compute_diffusion_gates


def test_diffusion_gates_direct_and_cg_agree_small():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(16, 8)).astype(np.float32)
    psi = rng.normal(size=(8,)).astype(np.float32)
    h_direct = compute_diffusion_gates(
        Y, psi, kneighbors=4, gamma=0.2, method="direct", clamp=False
    )
    h_cg = compute_diffusion_gates(
        Y, psi, kneighbors=4, gamma=0.2, method="cg", tol=1e-6, max_iters=1024, clamp=False
    )
    # Relative L2 error should be small
    num = float(np.linalg.norm(h_direct - h_cg))
    den = float(np.linalg.norm(h_direct)) + 1e-12
    rel = num / den
    assert rel < 5e-3


def test_diffusion_gates_validation_and_clamp():
    Y = np.eye(4, dtype=np.float32)
    psi = np.ones(4, dtype=np.float32)
    h = compute_diffusion_gates(Y, psi, kneighbors=1, gamma=0.5, clamp=True)
    assert np.all(h >= 0) and np.all(h <= 1)
    # invalid gamma
    try:
        compute_diffusion_gates(Y, psi, gamma=0.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_diffusion_gates_invalid_similarity():
    Y = np.eye(3, dtype=np.float32)
    psi = np.ones(3, dtype=np.float32)
    try:
        compute_diffusion_gates(Y, psi, similarity="dot")
        raise AssertionError("expected ValueError for unsupported similarity")
    except ValueError:
        pass


def test_diffusion_direct_fallback_on_linalg(monkeypatch):
    Y = np.eye(4, dtype=np.float32)
    psi = np.ones(4, dtype=np.float32)
    # Force np.linalg.solve to raise LinAlgError to exercise fallback
    original = np.linalg.solve

    def raise_lin_alg(*args, **kwargs):
        raise np.linalg.LinAlgError("bad")

    monkeypatch.setattr(np.linalg, "solve", raise_lin_alg)
    h = compute_diffusion_gates(Y, psi, gamma=0.5, method="direct")
    # Fallback should return ones
    assert np.allclose(h, 1.0)
    # restore (pytest will also undo monkeypatch)
    monkeypatch.setattr(np.linalg, "solve", original)


def test_diffusion_cg_fallback_to_ones(monkeypatch):
    # Force cg_solve to raise via monkeypatching the internal helper
    import oscillink.preprocess.diffusion as diff

    Y = np.eye(3, dtype=np.float32)
    psi = np.ones(3, dtype=np.float32)

    def bad_cg(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(diff, "cg_solve", bad_cg)
    h = diff.compute_diffusion_gates(Y, psi, method="cg")
    assert np.allclose(h, np.ones(3, dtype=np.float32))
