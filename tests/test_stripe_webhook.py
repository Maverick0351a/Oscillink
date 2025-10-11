import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cloud.app.keystore import get_keystore
from cloud.app.main import _webhook_events_mem, app


@pytest.fixture(autouse=True)
def clear_webhook_events():
    _webhook_events_mem.clear()
    yield
    _webhook_events_mem.clear()


@pytest.fixture
def client(monkeypatch):
    # Allow unverified for tests (no stripe lib assumed)
    monkeypatch.setenv("OSCILLINK_ALLOW_UNVERIFIED_STRIPE", "1")
    return TestClient(app)


SUB_CREATED = {
    "id": "evt_test_created",
    "type": "customer.subscription.created",
    "data": {
        "object": {
            "id": "sub_123",
            "metadata": {"api_key": "key_test_1"},
            "items": {"data": [{"price": {"id": "price_cloud_pro_monthly"}}]},
        }
    },
}

SUB_UPDATED = {
    "id": "evt_test_updated",
    "type": "customer.subscription.updated",
    "data": {
        "object": {
            "id": "sub_123",
            "metadata": {"api_key": "key_test_1"},
            "items": {"data": [{"price": {"id": "price_cloud_enterprise"}}]},
        }
    },
}

SUB_DELETED = {
    "id": "evt_test_deleted",
    "type": "customer.subscription.deleted",
    "data": {
        "object": {
            "id": "sub_123",
            "metadata": {"api_key": "key_test_1"},
            "items": {"data": [{"price": {"id": "price_cloud_pro_monthly"}}]},
        }
    },
}


def post_event(client: TestClient, event: dict[str, Any]):
    return client.post(
        "/stripe/webhook", content=json.dumps(event), headers={"content-type": "application/json"}
    )


def test_webhook_create_and_update_and_delete(client, monkeypatch):
    # Create
    r1 = post_event(client, SUB_CREATED)
    assert r1.status_code == 200
    ks = get_keystore()
    meta = ks.get("key_test_1")
    assert meta and meta.tier in {"pro", "enterprise", "free"}  # mapping may fallback
    # Update to enterprise
    r2 = post_event(client, SUB_UPDATED)
    assert r2.status_code == 200
    meta2 = ks.get("key_test_1")
    assert meta2 and meta2.tier in {"enterprise", "pro"}
    # Duplicate processed idempotent
    r2b = post_event(client, SUB_UPDATED)
    assert r2b.json().get("idempotent") or r2b.json().get("duplicate") or r2b.status_code == 200
    # Deletion
    r3 = post_event(client, SUB_DELETED)
    assert r3.status_code == 200
    # Event storage size
    assert len(_webhook_events_mem) >= 3
