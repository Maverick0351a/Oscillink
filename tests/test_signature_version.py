import numpy as np

from oscillink import OscillinkLattice, verify_receipt_mode


def test_signature_version_minimal():
    Y = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
    lat = OscillinkLattice(Y, deterministic_k=True)
    lat.set_query(np.random.default_rng(1).normal(size=4).astype(np.float32))
    lat.set_receipt_secret(b"secret")
    rec = lat.receipt()
    sig_block = rec["meta"]["signature"]
    assert sig_block["payload"]["sig_v"] == 1
    ok, payload = verify_receipt_mode(rec, b"secret", required_sig_v=1)
    assert ok and payload["sig_v"] == 1


def test_signature_version_extended():
    Y = np.random.default_rng(2).normal(size=(12, 5)).astype(np.float32)
    lat = OscillinkLattice(Y, deterministic_k=True)
    lat.set_query(np.random.default_rng(3).normal(size=5).astype(np.float32))
    lat.set_receipt_secret(b"secret")
    lat.set_signature_mode("extended")
    rec = lat.receipt()
    sig_block = rec["meta"]["signature"]
    assert sig_block["payload"]["sig_v"] == 1
    ok, payload = verify_receipt_mode(rec, b"secret", require_mode="extended", required_sig_v=1)
    assert ok and payload["mode"] == "extended"
    # Requiring a mismatched version should fail
    bad_ok, _ = verify_receipt_mode(rec, b"secret", required_sig_v=999)
    assert not bad_ok
