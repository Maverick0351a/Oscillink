from __future__ import annotations

import json

from fastapi.testclient import TestClient

from cloud.app.main import app

client = TestClient(app)

# Minimal synthetic Stripe subscription event payload
BASE_EVENT = {
    "id": "evt_test_sub_1",
    "type": "customer.subscription.created",
    "data": {
        "object": {
            "metadata": {"api_key": "user_key_test"},
            "items": {"data": [{"price": {"id": "price_free"}}]},
        }
    },
}


def test_webhook_first_then_duplicate(monkeypatch):
    # Ensure price map maps price_free to free
    monkeypatch.setenv("OSCILLINK_STRIPE_PRICE_MAP", "price_free:free")
    # Post first time
    r1 = client.post(
        "/stripe/webhook",
        content=json.dumps(BASE_EVENT),
        headers={"content-type": "application/json"},
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["processed"] is True
    assert body1.get("duplicate") is None

    # Second post (duplicate id) should short-circuit
    r2 = client.post(
        "/stripe/webhook",
        content=json.dumps(BASE_EVENT),
        headers={"content-type": "application/json"},
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["processed"] is False
    assert body2.get("duplicate") is True
    assert body2["id"] == BASE_EVENT["id"]

    # Different event id -> processed again
    evt2 = dict(BASE_EVENT)
    evt2["id"] = "evt_test_sub_2"
    r3 = client.post(
        "/stripe/webhook",
        content=json.dumps(evt2),
        headers={"content-type": "application/json"},
    )
    assert r3.status_code == 200
    body3 = r3.json()
    assert body3["processed"] is True
    assert body3.get("duplicate") is None
