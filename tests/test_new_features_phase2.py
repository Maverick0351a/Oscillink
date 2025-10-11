import numpy as np

from oscillink.core.lattice import OscillinkLattice


def test_deterministic_neighbors_reproducible():
    rng = np.random.default_rng(42)
    Y = rng.standard_normal((40, 24), dtype=np.float32)
    lat1 = OscillinkLattice(Y, kneighbors=5, deterministic_k=True)
    lat2 = OscillinkLattice(Y, kneighbors=5, deterministic_k=True)
    # identical adjacency
    assert np.allclose(lat1.A, lat2.A)


def test_logger_captures_events():
    Y = np.random.randn(30, 16).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=4, deterministic_k=True)
    events = []

    def logger(ev, payload):
        events.append(ev)

    lat.set_logger(logger)
    psi = (Y[:5].mean(axis=0) / (np.linalg.norm(Y[:5].mean(axis=0)) + 1e-12)).astype(np.float32)
    lat.set_query(psi)
    lat.settle(max_iters=3)
    lat.receipt()
    assert (
        "settle" in events and "receipt" in events and any(e.startswith("ustar_") for e in events)
    )


def test_receipt_signing_changes_on_state_mutation():
    Y = np.random.randn(25, 12).astype(np.float32)
    lat = OscillinkLattice(Y, deterministic_k=True)
    lat.set_receipt_secret("secret-key")
    psi = (Y[:4].mean(axis=0) / (np.linalg.norm(Y[:4].mean(axis=0)) + 1e-12)).astype(np.float32)
    lat.set_query(psi)
    r1 = lat.receipt()
    sig1 = r1["meta"]["signature"]["signature"]
    # mutate by adding chain -> signature should change
    lat.add_chain([0, 1, 2, 3], lamP=0.3)
    r2 = lat.receipt()
    sig2 = r2["meta"]["signature"]["signature"]
    assert sig1 != sig2
