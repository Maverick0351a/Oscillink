from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from cloud.app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok" and "version" in data


def test_settle_small():
    Y = np.random.RandomState(0).randn(12, 6).astype(float).tolist()
    req = {
        "Y": Y,
        "psi": [0.0] * 6,
        "params": {"kneighbors": 4, "lamG": 1.0, "lamC": 0.5, "lamQ": 4.0},
        "options": {"max_iters": 4, "tol": 1e-2, "include_receipt": True, "bundle_k": 3},
    }
    r = client.post("/v1/settle", json=req)
    assert r.status_code == 200
    out = r.json()
    assert "state_sig" in out and out["receipt"] is not None
    assert len(out["bundle"]) == 3


def test_api_key_required(monkeypatch):
    # simulate enabling API key auth
    monkeypatch.setenv("OSCILLINK_API_KEYS", "k1,k2")
    # Need to clear cached settings so new env applies
    from cloud.app import config as cfg

    cfg.get_settings.cache_clear()  # type: ignore
    # Re-import app dependency or recreate client
    from importlib import reload

    reload(cfg)
    from cloud.app import main as mainmod

    reload(mainmod)
    client2 = TestClient(mainmod.app)

    Y = np.random.RandomState(0).randn(4, 3).astype(float).tolist()
    body = {"Y": Y, "psi": [0.0] * 3}
    # Missing key -> 401
    r1 = client2.post("/v1/settle", json=body)
    assert r1.status_code == 401
    # Wrong key -> 401
    r2 = client2.post("/v1/settle", headers={"x-api-key": "bad"}, json=body)
    assert r2.status_code == 401
    # Valid key
    r3 = client2.post("/v1/settle", headers={"x-api-key": "k2"}, json=body)
    assert r3.status_code == 200
    assert "state_sig" in r3.json()


def test_request_id_propagation():
    from importlib import reload

    from cloud.app import main as mainmod

    reload(mainmod)
    client3 = TestClient(mainmod.app)
    Y = np.random.RandomState(1).randn(5, 4).astype(float).tolist()
    supplied = "RID123"
    r = client3.post("/v1/settle", headers={"x-request-id": supplied}, json={"Y": Y})
    assert r.status_code == 200
    assert r.headers.get("x-request-id") == supplied
    assert r.json()["meta"]["request_id"] == supplied


def test_metrics_after_settle():
    from importlib import reload

    from cloud.app import main as mainmod

    reload(mainmod)
    client4 = TestClient(mainmod.app)
    Y = np.random.RandomState(2).randn(6, 4).astype(float).tolist()
    r = client4.post("/v1/settle", json={"Y": Y})
    assert r.status_code == 200
    m = client4.get("/metrics")
    assert m.status_code == 200
    body = m.text
    assert "oscillink_settle_requests_total" in body
    assert "oscillink_settle_latency_seconds_bucket" in body


def test_receipt_endpoint():
    Y = np.random.RandomState(3).randn(10, 5).astype(float).tolist()
    req = {"Y": Y, "options": {"max_iters": 3}}
    r = client.post("/v1/receipt", json=req)
    assert r.status_code == 200
    data = r.json()
    assert "receipt" in data and "bundle" not in data


def test_bundle_endpoint():
    Y = np.random.RandomState(4).randn(9, 5).astype(float).tolist()
    req = {"Y": Y, "options": {"max_iters": 3, "bundle_k": 4}}
    r = client.post("/v1/bundle", json=req)
    assert r.status_code == 200
    data = r.json()
    assert "bundle" in data and len(data["bundle"]) == 4


def test_chain_receipt_endpoint():
    Y = np.random.RandomState(5).randn(11, 6).astype(float).tolist()
    chain = [0, 2, 4]
    req = {"Y": Y, "chain": chain, "options": {"max_iters": 3}}
    r = client.post("/v1/chain/receipt", json=req)
    assert r.status_code == 200
    data = r.json()
    assert "chain_receipt" in data and "verdict" in data["chain_receipt"]


def test_rate_limiting(monkeypatch):
    # Configure very low limit
    monkeypatch.setenv("OSCILLINK_RATE_LIMIT", "2")
    monkeypatch.setenv("OSCILLINK_RATE_WINDOW", "30")
    # Reload main module to pick up env changes
    from importlib import reload

    from cloud.app import main as mainmod

    reload(mainmod)
    test_client = TestClient(mainmod.app)
    Y = np.random.RandomState(6).randn(4, 3).astype(float).tolist()
    body = {"Y": Y}
    # First two should pass
    r1 = test_client.post("/v1/receipt", json=body)
    r2 = test_client.post("/v1/receipt", json=body)
    assert r1.status_code == 200 and r2.status_code == 200
    # Third exceeds limit
    r3 = test_client.post("/v1/receipt", json=body)
    assert r3.status_code == 429
    assert "X-RateLimit-Limit" in r3.headers and r3.headers["X-RateLimit-Remaining"] == "0"
