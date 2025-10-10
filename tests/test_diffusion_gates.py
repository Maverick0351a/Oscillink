import numpy as np

from oscillink import compute_diffusion_gates


def test_diffusion_gates_basic_properties():
    rng = np.random.default_rng(42)
    N, D = 60, 32
    Y = rng.normal(size=(N, D)).astype(np.float32)
    psi = rng.normal(size=(D,)).astype(np.float32)

    gates = compute_diffusion_gates(Y, psi, kneighbors=5, beta=1.2, gamma=0.15, neighbor_seed=123)

    assert gates.shape == (N,)
    # In [0,1]
    assert float(gates.min()) >= 0.0 - 1e-6
    assert float(gates.max()) <= 1.0 + 1e-6
    # Not all identical (unless degenerate similarity); variance positive
    assert float(np.var(gates)) > 0.0


def test_diffusion_gates_similarity_monotonicity():
    # Construct Y where first half more aligned with psi than second half.
    rng = np.random.default_rng(7)
    D = 16
    base = rng.normal(size=(D,)).astype(np.float32)
    psi = base / (np.linalg.norm(base) + 1e-12)
    # First 10 near psi, next 10 roughly orthogonal noise
    Y_close = psi + 0.01 * rng.normal(size=(10, D)).astype(np.float32)
    Y_far = rng.normal(size=(10, D)).astype(np.float32)
    Y = np.vstack([Y_close, Y_far]).astype(np.float32)

    gates = compute_diffusion_gates(Y, psi, kneighbors=4, gamma=0.2, beta=1.0, deterministic_k=True)

    # Expect average gate weight of close group > far group
    g_close = float(gates[:10].mean())
    g_far = float(gates[10:].mean())
    assert g_close > g_far, (g_close, g_far)


def test_diffusion_gates_determinism_seed():
    rng = np.random.default_rng(99)
    N, D = 40, 24
    Y = rng.normal(size=(N, D)).astype(np.float32)
    psi = rng.normal(size=(D,)).astype(np.float32)

    g1 = compute_diffusion_gates(Y, psi, kneighbors=5, neighbor_seed=555)
    g2 = compute_diffusion_gates(Y, psi, kneighbors=5, neighbor_seed=555)
    # Seeded tie-breaking should yield identical outputs
    assert np.allclose(g1, g2)
