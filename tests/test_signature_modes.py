import numpy as np

from oscillink.core.lattice import OscillinkLattice
from oscillink.core.receipts import verify_receipt


def make_lat(seed=None):
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng(0)
    Y = rng.normal(size=(18, 6)).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=5, deterministic_k=True, neighbor_seed=123)
    lat.set_query(rng.normal(size=6).astype(np.float32))
    return lat


def test_minimal_signature_mode():
    secret = b"test-secret"
    lat = make_lat()
    lat.set_receipt_secret(secret)
    # default mode minimal
    r = lat.receipt()
    sig_block = r["meta"]["signature"]
    assert sig_block["payload"]["mode"] == "minimal"
    assert "params" not in sig_block["payload"]
    assert verify_receipt(r, secret)


def test_extended_signature_mode():
    secret = b"test-secret"
    lat = make_lat()
    lat.set_receipt_secret(secret)
    lat.set_signature_mode("extended")
    r = lat.receipt()
    sig_block = r["meta"]["signature"]
    payload = sig_block["payload"]
    assert payload["mode"] == "extended"
    for k in ["params", "graph", "ustar_iters", "ustar_res", "ustar_converged"]:
        assert k in payload
    assert verify_receipt(r, secret)


def test_neighbor_seed_repro():
    # neighbor_seed should influence adjacency signature if deterministic_k False but with seed; here we enforce determinism
    rng = np.random.default_rng(42)
    Y = rng.normal(size=(25, 5)).astype(np.float32)
    lat1 = OscillinkLattice(Y, kneighbors=7, deterministic_k=True, neighbor_seed=777)
    lat2 = OscillinkLattice(Y, kneighbors=7, deterministic_k=True, neighbor_seed=777)
    # same seed -> same internal signature
    sig1 = lat1._signature()
    sig2 = lat2._signature()
    assert sig1 == sig2
    # different seed should still match because deterministic_k enforces tie-breaks independent of seed
    lat3 = OscillinkLattice(Y, kneighbors=7, deterministic_k=True, neighbor_seed=42)
    sig3 = lat3._signature()
    assert sig1 == sig3
    # if we disable deterministic_k, seed likely changes ordering (best-effort probabilistic check)
    lat4 = OscillinkLattice(Y, kneighbors=7, deterministic_k=False, neighbor_seed=1)
    lat5 = OscillinkLattice(Y, kneighbors=7, deterministic_k=False, neighbor_seed=2)
    # At least allow the possibility they differ; not strictly required but if equal that's fine
    # So we don't assert inequality; instead ensure both are valid hex and length 64
    for s in (lat4._signature(), lat5._signature()):
        assert isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)
