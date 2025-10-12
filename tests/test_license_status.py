from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from cloud.app.main import app


def _client() -> TestClient:
    return TestClient(app)


def test_license_status_ok(tmp_path, monkeypatch):
    ent_path = tmp_path / "entitlements.json"
    data = {
        "iss": "test-issuer",
        "sub": "acct_123",
        "tier": "pro",
        "exp": int(time.time()) + 3600,
    }
    ent_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("OSCILLINK_ENTITLEMENTS_PATH", str(ent_path))
    # Ensure not required to avoid interference
    monkeypatch.delenv("OSCILLINK_LICENSE_REQUIRED", raising=False)
    c = _client()
    r = c.get("/license/status")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["sub"] == "acct_123"
    assert body["tier"] == "pro"
    assert isinstance(body["exp"], int)


def test_license_status_expired_not_required(tmp_path, monkeypatch):
    ent_path = tmp_path / "entitlements.json"
    data = {
        "iss": "test-issuer",
        "sub": "acct_123",
        "tier": "pro",
        "exp": int(time.time()) - 10,
    }
    ent_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("OSCILLINK_ENTITLEMENTS_PATH", str(ent_path))
    # Explicitly not required
    monkeypatch.setenv("OSCILLINK_LICENSE_REQUIRED", "0")
    c = _client()
    r = c.get("/license/status")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"stale", "ok"}  # ok if within leeway, else stale


def test_license_status_expired_required(tmp_path, monkeypatch):
    ent_path = tmp_path / "entitlements.json"
    # Set exp far enough in the past to bypass leeway
    data = {
        "iss": "test-issuer",
        "sub": "acct_123",
        "tier": "pro",
        "exp": int(time.time()) - 3600,
    }
    ent_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("OSCILLINK_ENTITLEMENTS_PATH", str(ent_path))
    monkeypatch.setenv("OSCILLINK_LICENSE_REQUIRED", "1")
    # Use small leeway to ensure expiry triggers
    monkeypatch.setenv("OSCILLINK_JWT_LEEWAY", "0")
    c = _client()
    r = c.get("/license/status")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "expired"


def test_license_status_missing_required(tmp_path, monkeypatch):
    # Point to a non-existent file
    ent_path = tmp_path / "missing.json"
    monkeypatch.setenv("OSCILLINK_ENTITLEMENTS_PATH", str(ent_path))
    monkeypatch.setenv("OSCILLINK_LICENSE_REQUIRED", "true")
    c = _client()
    r = c.get("/license/status")
    assert r.status_code == 503
    assert r.json()["status"] in {"unlicensed", "expired"}


def test_license_status_leeway(tmp_path, monkeypatch):
    ent_path = tmp_path / "entitlements.json"
    # Expired by 100s, but leeway 300s should consider it ok
    data = {
        "iss": "test-issuer",
        "sub": "acct_123",
        "tier": "pro",
        "exp": int(time.time()) - 100,
    }
    ent_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("OSCILLINK_ENTITLEMENTS_PATH", str(ent_path))
    monkeypatch.setenv("OSCILLINK_JWT_LEEWAY", "300")
    monkeypatch.setenv("OSCILLINK_LICENSE_REQUIRED", "0")
    c = _client()
    r = c.get("/license/status")
    # Within leeway: treat as ok
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
