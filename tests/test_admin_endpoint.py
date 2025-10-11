import pytest
from fastapi.testclient import TestClient

from cloud.app.main import app


@pytest.fixture(autouse=True)
def set_admin_secret(monkeypatch):
    monkeypatch.setenv("OSCILLINK_ADMIN_SECRET", "test-admin")
    yield


def test_admin_create_and_get_key():
    client = TestClient(app)
    # Create key
    resp = client.put(
        "/admin/keys/testkey123",
        json={"tier": "pro", "status": "active"},
        headers={"x-admin-secret": "test-admin"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["api_key"] == "testkey123"
    assert data["tier"] == "pro"
    # Fetch
    resp2 = client.get("/admin/keys/testkey123", headers={"x-admin-secret": "test-admin"})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["tier"] == "pro"


def test_admin_requires_secret():
    client = TestClient(app)
    resp = client.get("/admin/keys/doesnotmatter")
    assert resp.status_code in (401, 503)


def test_admin_invalid_secret():
    client = TestClient(app)
    resp = client.get("/admin/keys/somekey", headers={"x-admin-secret": "wrong"})
    assert resp.status_code in (401, 503)
