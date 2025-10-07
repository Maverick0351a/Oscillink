import numpy as np

from oscillink.core.lattice import OscillinkLattice


def test_api_surface_bundle_and_receipts():
    N, D = 60, 48
    Y = np.random.randn(N, D).astype(np.float32)
    psi = (Y[:10].mean(axis=0) / (np.linalg.norm(Y[:10].mean(axis=0)) + 1e-12)).astype(np.float32)

    lat = OscillinkLattice(Y)
    lat.set_query(psi=psi)
    lat.settle()
    r = lat.receipt()
    assert "deltaH_total" in r and "null_points" in r

    out = lat.bundle(k=5)
    assert isinstance(out, list) and len(out) <= 5
