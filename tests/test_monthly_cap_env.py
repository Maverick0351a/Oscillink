from __future__ import annotations

from importlib import reload

import numpy as np
from fastapi.testclient import TestClient

from cloud.app.main import app


def _client() -> TestClient:
    return TestClient(app)


def test_monthly_cap_single_request_exceeds(monkeypatch):
    # Set a very small monthly cap and ensure a single request over cap is rejected with 413
    monkeypatch.setenv("OSCILLINK_API_KEYS", "k1")
    monkeypatch.setenv("OSCILLINK_MONTHLY_CAP", "100")
    # Reload keystore to pick up new env
    import cloud.app.keystore as ks

    reload(ks)
    c = _client()
    # Create Y with units N*D > 100, e.g., N=11, D=10 => 110
    Y = np.random.RandomState(0).randn(11, 10).astype(float).tolist()
    r = c.post(
        "/v1/receipt", headers={"X-API-Key": "k1"}, json={"Y": Y, "options": {"max_iters": 1}}
    )
    assert r.status_code == 413
    assert "monthly cap" in r.json()["detail"]


def test_monthly_cap_cumulative_exceeds(monkeypatch):
    # Cap 120 units; two requests of 60 pass, third of 10 should exceed with 429
    monkeypatch.setenv("OSCILLINK_API_KEYS", "k1")
    monkeypatch.setenv("OSCILLINK_MONTHLY_CAP", "120")
    import cloud.app.keystore as ks

    reload(ks)
    c = _client()
    Y60 = np.random.RandomState(1).randn(12, 5).astype(float).tolist()  # 60 units
    r1 = c.post(
        "/v1/receipt", headers={"X-API-Key": "k1"}, json={"Y": Y60, "options": {"max_iters": 1}}
    )
    r2 = c.post(
        "/v1/receipt", headers={"X-API-Key": "k1"}, json={"Y": Y60, "options": {"max_iters": 1}}
    )
    assert r1.status_code == 200 and r2.status_code == 200
    Y10 = np.random.RandomState(2).randn(5, 2).astype(float).tolist()  # 10 units
    r3 = c.post(
        "/v1/receipt", headers={"X-API-Key": "k1"}, json={"Y": Y10, "options": {"max_iters": 1}}
    )
    assert r3.status_code == 429
    # Check headers expose remaining
    assert r3.headers.get("X-MonthCap-Limit") == "120"
