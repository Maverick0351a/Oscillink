import hashlib
import hmac
import json

from oscillink.core.receipts import verify_receipt, verify_receipt_mode


def _signed(sig_v=1, mode="extended"):
    payload = {"sig_v": sig_v, "mode": mode, "state_sig": "abc", "deltaH_total": -1.23}
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    secret = b"s"
    sig = hmac.new(secret, raw, hashlib.sha256).hexdigest()
    return {
        "meta": {
            "signature": {
                "algorithm": "HMAC-SHA256",
                "payload": payload,
                "signature": sig,
            }
        }
    }, secret


def test_verify_receipt_ok():
    rec, secret = _signed()
    assert verify_receipt(rec, secret) is True


def test_verify_receipt_mode_strict_and_minimal_subset():
    rec, secret = _signed(sig_v=2, mode="extended")
    ok, payload = verify_receipt_mode(
        rec, secret, require_mode=None, minimal_subset=False, required_sig_v=2
    )
    assert ok and payload is not None and payload["mode"] == "extended"

    ok2, payload2 = verify_receipt_mode(rec, secret, require_mode="minimal", minimal_subset=True)
    # minimal subset path only passes if the original signature was created for minimal payload; here it won't
    assert ok2 is False and payload2 is None


def test_verify_receipt_mode_failures():
    rec, secret = _signed()
    # tamper with payload
    rec["meta"]["signature"]["payload"]["state_sig"] = "zzz"
    ok, payload = verify_receipt_mode(rec, secret)
    assert ok is False and payload is None


def test_verify_receipt_mode_minimal_success():
    # Build a receipt that was originally signed in minimal mode
    payload = {"sig_v": 1, "mode": "minimal", "state_sig": "abc", "deltaH_total": -1.0}
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    secret = b"s"
    sig = hmac.new(secret, raw, hashlib.sha256).hexdigest()
    rec = {
        "meta": {"signature": {"algorithm": "HMAC-SHA256", "payload": payload, "signature": sig}}
    }
    ok, pl = verify_receipt_mode(rec, secret, require_mode="minimal", minimal_subset=True)
    assert ok and pl is not None and pl["mode"] == "minimal"


def test_verify_receipt_bad_algorithm_and_missing_fields():
    # bad algorithm
    rec = {"meta": {"signature": {"algorithm": "SHA1", "payload": {}, "signature": "x"}}}
    assert verify_receipt(rec, b"s") is False
    ok, payload = verify_receipt_mode(rec, b"s")
    assert ok is False and payload is None
    # missing block
    rec2 = {"meta": {}}
    assert verify_receipt(rec2, b"s") is False
    ok2, payload2 = verify_receipt_mode(rec2, b"s")
    assert ok2 is False and payload2 is None
    # mode/required_sig_v mismatch
    proper, secret = _signed(sig_v=1, mode="extended")
    ok3, payload3 = verify_receipt_mode(proper, secret, require_mode="minimal")
    assert ok3 is False and payload3 is None
    ok4, payload4 = verify_receipt_mode(proper, secret, required_sig_v=2)
    assert ok4 is False and payload4 is None


def test_verify_receipt_exception_paths():
    # payload not JSON-serializable triggers exception
    rec = {
        "meta": {
            "signature": {"algorithm": "HMAC-SHA256", "payload": {"x": object()}, "signature": "x"}
        }
    }
    assert verify_receipt(rec, b"s") is False
    ok, payload = verify_receipt_mode(rec, b"s")
    assert ok is False and payload is None
