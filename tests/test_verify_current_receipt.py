import numpy as np

from oscillink import OscillinkLattice, verify_receipt


def test_verify_current_receipt_matches_helper():
    Y = np.random.RandomState(123).randn(12, 6).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=4, deterministic_k=True, neighbor_seed=11)
    lat.set_query(np.zeros(6, dtype=np.float32))
    secret = b"mykey"
    lat.set_receipt_secret(secret)
    rec = lat.receipt()  # compute once
    assert verify_receipt(rec, secret) is True
    # verify via convenience (will recompute receipt but reuse cached U*)
    assert lat.verify_current_receipt(secret) is True


def test_verify_current_receipt_wrong_secret():
    Y = np.random.RandomState(42).randn(8, 4).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=3, deterministic_k=True, neighbor_seed=5)
    lat.set_query(np.zeros(4, dtype=np.float32))
    lat.set_receipt_secret("secret")
    rec_valid = lat.verify_current_receipt("secret")
    assert rec_valid is True
    assert lat.verify_current_receipt("other") is False
