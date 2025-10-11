import numpy as np

from oscillink import OscillinkLattice, compute_diffusion_gates


def test_receipt_gating_stats_uniform():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(50, 16)).astype(np.float32)
    psi = rng.normal(size=(16,)).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=5)
    lat.set_query(psi)
    lat.settle()
    rec = lat.receipt()
    meta = rec["meta"]
    assert "gates_min" in meta and "gates_max" in meta and "gates_uniform" in meta
    assert meta["gates_min"] == meta["gates_max"] == meta["gates_mean"]
    assert meta["gates_uniform"] is True


def test_receipt_gating_stats_diffusion():
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(60, 24)).astype(np.float32)
    psi = rng.normal(size=(24,)).astype(np.float32)
    gates = compute_diffusion_gates(Y, psi, kneighbors=6, beta=1.0, gamma=0.15, neighbor_seed=42)
    lat = OscillinkLattice(Y, kneighbors=6)
    lat.set_query(psi, gates=gates)
    lat.settle()
    rec = lat.receipt()
    meta = rec["meta"]
    assert meta["gates_min"] >= 0.0
    assert meta["gates_max"] <= 1.0 + 1e-6
    assert meta["gates_uniform"] is False
    # Diffusion should produce variance
    assert meta["gates_max"] > meta["gates_min"]
