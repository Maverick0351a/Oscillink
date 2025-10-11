import numpy as np

from oscillink import OscillinkLattice, compare_provenance


def test_repr_contains_core_fields():
    Y = np.random.RandomState(123).randn(10, 4).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=3, deterministic_k=True)
    lat.add_chain([0, 1, 2], lamP=0.2)
    r = repr(lat)
    assert "N=10" in r and "D=4" in r and "k=3" in r and "chain_len=3" in r


def test_compare_provenance_basic():
    rng = np.random.RandomState(0)
    Y = rng.randn(12, 6).astype(np.float32)
    lat1 = OscillinkLattice(Y, kneighbors=4, deterministic_k=True)
    lat1.set_query(rng.randn(6).astype(np.float32))
    lat2 = OscillinkLattice(Y.copy(), kneighbors=4, deterministic_k=True)
    lat2.set_query(lat1.psi.copy())
    diff = compare_provenance(lat1, lat2)
    assert diff["same"]
    # mutate gating -> diff
    lat2.set_gates(np.ones(lat2.N, dtype=np.float32) * 0.5)
    diff2 = compare_provenance(lat1, lat2)
    assert not diff2["same"]
    assert diff2["gates_equal"] is False
