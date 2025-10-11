from __future__ import annotations

import json

from fastapi.testclient import TestClient

from cloud.app.main import app

client = TestClient(app)

EVENT = {
    "id": "evt_hash_test_1",
    "type": "customer.subscription.created",
    "data": {
        "object": {
            "metadata": {"api_key": "key_hash"},
            "items": {"data": [{"price": {"id": "price_free"}}]},
        }
    },
}


def test_webhook_payload_hash_present(monkeypatch):
    monkeypatch.setenv("OSCILLINK_STRIPE_PRICE_MAP", "price_free:free")
    r = client.post(
        "/stripe/webhook",
        content=json.dumps(EVENT),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "payload_sha256" in body and len(body["payload_sha256"]) == 64
    # Duplicate call returns duplicate record (short-circuit) - should still have id but *not* recompute new hash path changes
    r2 = client.post(
        "/stripe/webhook",
        content=json.dumps(EVENT),
        headers={"content-type": "application/json"},
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2.get("duplicate") is True
    assert "payload_sha256" not in body2  # duplicate fast path does not include new hash
