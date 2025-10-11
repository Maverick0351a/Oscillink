import numpy as np

from oscillink.core.lattice import OscillinkLattice


def test_deltaH_nonnegative_and_spd():
    N, D = 80, 64
    Y = np.random.randn(N, D).astype(np.float32)
    psi = (Y[:20].mean(axis=0) / (np.linalg.norm(Y[:20].mean(axis=0)) + 1e-12)).astype(np.float32)

    lat = OscillinkLattice(Y, kneighbors=6, lamG=1.0, lamC=0.5, lamQ=4.0)
    lat.set_query(psi=psi)
    lat.add_chain([1, 3, 5, 7], lamP=0.2)

    lat.settle(dt=1.0, max_iters=8, tol=1e-3)
    rec = lat.receipt()
    assert rec["deltaH_total"] >= -1e-5  # numeric tolerance
