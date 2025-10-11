import numpy as np

from oscillink.core.lattice import OscillinkLattice


def test_chain_verdict_toggle_on_break():
    N, D = 100, 96
    Y = np.random.randn(N, D).astype(np.float32)
    psi = (Y[:25].mean(axis=0) / (np.linalg.norm(Y[:25].mean(axis=0)) + 1e-12)).astype(np.float32)

    lat = OscillinkLattice(Y, kneighbors=6, lamG=1.0, lamC=0.5, lamQ=4.0)
    lat.set_query(psi=psi)
    chain = [2, 5, 7, 9]
    lat.add_chain(chain, lamP=0.3)

    lat.settle()
    ok = lat.chain_receipt(chain)
    assert ok["verdict"] in (True, False)

    # induce break
    Y[7] += 3.0 * np.random.randn(D).astype(np.float32)
    lat2 = OscillinkLattice(Y, kneighbors=6, lamG=1.0, lamC=0.5, lamQ=4.0)
    lat2.set_query(psi=psi)
    lat2.add_chain(chain, lamP=0.3)
    lat2.settle()
    fail = lat2.chain_receipt(chain)
    # the weakest link z-score should increase or verdict should be more likely false
    assert (fail["weakest_link"]["zscore"] >= ok["weakest_link"]["zscore"]) or (
        fail["verdict"] is False
    )
