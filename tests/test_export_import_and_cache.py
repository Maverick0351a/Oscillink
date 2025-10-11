import numpy as np

from oscillink.core.lattice import OscillinkLattice


def test_export_import_roundtrip_deltaH_close():
    N, D = 40, 32
    Y = np.random.randn(N, D).astype(np.float32)
    psi = (Y[:10].mean(axis=0) / (np.linalg.norm(Y[:10].mean(axis=0)) + 1e-12)).astype(np.float32)

    lat = OscillinkLattice(Y, kneighbors=4)
    lat.set_query(psi)
    lat.add_chain([1, 3, 5], lamP=0.2)
    lat.settle(max_iters=6)
    r1 = lat.receipt()

    state = lat.export_state()
    lat2 = OscillinkLattice.from_state(state)
    lat2.settle(max_iters=6)
    r2 = lat2.receipt()

    assert (
        abs(r1["deltaH_total"] - r2["deltaH_total"]) < 1e-2
    )  # loose tolerance (randomness differences)


def test_cached_ustar_reuse():
    N, D = 30, 24
    Y = np.random.randn(N, D).astype(np.float32)
    psi = (Y[:8].mean(axis=0) / (np.linalg.norm(Y[:8].mean(axis=0)) + 1e-12)).astype(np.float32)
    lat = OscillinkLattice(Y)
    lat.set_query(psi)

    # first receipt triggers a solve
    lat.receipt()
    solves_after_first = lat.stats["ustar_solves"]
    cache_hits_after_first = lat.stats["ustar_cache_hits"]

    # second receipt-related call should reuse cache
    lat.bundle(k=5)
    solves_after_second = lat.stats["ustar_solves"]
    cache_hits_after_second = lat.stats["ustar_cache_hits"]

    assert solves_after_second == solves_after_first  # no new solve
    assert cache_hits_after_second == cache_hits_after_first + 1


def test_parameter_validation_errors():
    import pytest

    Y = np.random.randn(10, 8).astype(np.float32)
    with pytest.raises(ValueError):
        OscillinkLattice(Y, kneighbors=0)
    with pytest.raises(ValueError):
        OscillinkLattice(Y, lamG=0.0)
    with pytest.raises(ValueError):
        OscillinkLattice(Y, lamC=-0.1)
