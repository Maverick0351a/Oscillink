import hashlib
import hmac
import json

from oscillink.core.receipts import verify_receipt_mode


def _sign(payload: dict, secret: bytes) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hmac.new(secret, raw, hashlib.sha256).hexdigest()


def test_minimal_subset_accepts_when_signature_matches_minimal():
    secret = b"k"
    # Simulate a receipt that was signed in minimal mode originally
    minimal_payload = {"sig_v": 1, "mode": "minimal", "state_sig": "abc", "deltaH_total": 1.23}
    sig = _sign(minimal_payload, secret)
    receipt = {
        "meta": {
            "signature": {"algorithm": "HMAC-SHA256", "payload": minimal_payload, "signature": sig}
        }
    }
    ok, payload = verify_receipt_mode(
        receipt, secret, require_mode=None, minimal_subset=True, required_sig_v=1
    )
    assert ok and payload == minimal_payload


def test_minimal_subset_accepts_with_extended_payload_when_signature_is_minimal():
    secret = b"k2"
    extended = {
        "sig_v": 1,
        "mode": "extended",
        "state_sig": "xyz",
        "deltaH_total": 0.5,
        "ustar_iters": 3,
        "params": {"lamG": 1.0},
    }
    # Sign the minimal subset of the extended payload; primary check will fail,
    # but minimal_subset fallback should succeed.
    minimal_subset = {
        "sig_v": extended["sig_v"],
        "mode": "minimal",
        "state_sig": extended["state_sig"],
        "deltaH_total": extended["deltaH_total"],
    }
    sig_min = _sign(minimal_subset, secret)
    receipt = {
        "meta": {
            "signature": {"algorithm": "HMAC-SHA256", "payload": extended, "signature": sig_min}
        }
    }
    ok, payload = verify_receipt_mode(
        receipt, secret, require_mode=None, minimal_subset=True, required_sig_v=1
    )
    assert ok and payload == minimal_subset


def test_minimal_subset_rejects_when_neither_signature_matches():
    secret = b"k3x"
    extended = {"sig_v": 1, "mode": "extended", "state_sig": "s", "deltaH_total": 0.1}
    # Sign a different payload so it matches neither extended nor its minimal subset
    wrong = {"sig_v": 1, "mode": "minimal", "state_sig": "DIFF", "deltaH_total": 9.9}
    bad_sig = _sign(wrong, secret)
    receipt = {
        "meta": {
            "signature": {"algorithm": "HMAC-SHA256", "payload": extended, "signature": bad_sig}
        }
    }
    ok, _ = verify_receipt_mode(
        receipt, secret, require_mode=None, minimal_subset=True, required_sig_v=1
    )
    assert not ok


def test_require_mode_and_version_filters():
    secret = b"k3"
    payload = {"sig_v": 2, "mode": "extended", "state_sig": "s", "deltaH_total": 0.0}
    sig = _sign(payload, secret)
    r = {"meta": {"signature": {"algorithm": "HMAC-SHA256", "payload": payload, "signature": sig}}}
    ok1, _ = verify_receipt_mode(r, secret, require_mode="extended", required_sig_v=2)
    ok2, _ = verify_receipt_mode(r, secret, require_mode="minimal", required_sig_v=2)
    ok3, _ = verify_receipt_mode(r, secret, require_mode="extended", required_sig_v=1)
    assert ok1 and not ok2 and not ok3
