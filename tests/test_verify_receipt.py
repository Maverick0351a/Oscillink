import numpy as np

from oscillink import OscillinkLattice, verify_receipt


def test_verify_receipt_roundtrip():
    Y = np.random.RandomState(0).randn(6, 4).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=3, deterministic_k=True, neighbor_seed=42)
    lat.set_query(np.zeros(4, dtype=np.float32))
    secret = b"test-secret"
    lat.set_receipt_secret(secret)
    r = lat.receipt()
    assert verify_receipt(r, secret) is True


def test_verify_receipt_invalid_secret():
    Y = np.random.RandomState(1).randn(6, 3).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=2, deterministic_k=True, neighbor_seed=7)
    lat.set_query(np.zeros(3, dtype=np.float32))
    lat.set_receipt_secret("abc123")
    r = lat.receipt()
    # wrong secret
    assert verify_receipt(r, b"wrong") is False


def test_verify_receipt_tamper_payload():
    Y = np.random.RandomState(2).randn(5, 3).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=2, deterministic_k=True, neighbor_seed=5)
    lat.set_query(np.zeros(3, dtype=np.float32))
    lat.set_receipt_secret(b"s3c")
    r = lat.receipt()
    # mutate payload value
    sig = r["meta"].get("signature")
    assert sig is not None
    sig["payload"]["deltaH_total"] = sig["payload"]["deltaH_total"] + 1.0
    assert verify_receipt(r, b"s3c") is False
