import numpy as np

from oscillink import OscillinkLattice, verify_receipt_mode


def test_verify_receipt_mode_import_and_usage():
    Y = np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32)
    lat = OscillinkLattice(Y, deterministic_k=True)
    lat.set_query(np.random.default_rng(1).normal(size=4).astype(np.float32))
    lat.set_receipt_secret(b"secret")
    lat.set_signature_mode("extended")
    rec = lat.receipt()
    ok, payload = verify_receipt_mode(rec, b"secret", require_mode="extended")
    assert ok and payload["mode"] == "extended"
