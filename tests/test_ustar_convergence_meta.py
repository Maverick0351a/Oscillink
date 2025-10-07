import numpy as np

from oscillink import OscillinkLattice


def test_ustar_convergence_meta_present():
    Y = np.random.RandomState(0).randn(10, 5).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=3, deterministic_k=True, neighbor_seed=0)
    lat.set_query(np.zeros(5, dtype=np.float32))
    r = lat.receipt()
    meta = r["meta"]
    assert "ustar_converged" in meta
    assert "ustar_res" in meta
    assert "ustar_iters" in meta
    # residual should be non-negative
    assert meta["ustar_res"] >= 0.0
    assert meta["ustar_iters"] >= 0
