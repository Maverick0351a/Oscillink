from importlib import reload

import numpy as np
from fastapi.testclient import TestClient


def _reload_app():
    from cloud.app import main as mainmod

    reload(mainmod)
    return mainmod.app


def test_rate_limit_headers(monkeypatch):
    # Enable API key and low rate limit of 2 per window
    monkeypatch.setenv("OSCILLINK_API_KEYS", "kR")
    monkeypatch.setenv("OSCILLINK_RATE_LIMIT", "2")
    monkeypatch.setenv("OSCILLINK_RATE_WINDOW", "60")
    app = _reload_app()
    client = TestClient(app)
    Y = np.random.RandomState(0).randn(3, 3).astype(float).tolist()
    body = {"Y": Y}
    # First request
    r1 = client.post("/v1/receipt", headers={"x-api-key": "kR"}, json=body)
    assert r1.status_code == 200
    assert "X-RateLimit-Limit" in r1.headers
    assert "X-RateLimit-Remaining" in r1.headers
    # Second request should still succeed (remaining should drop)
    r2 = client.post("/v1/receipt", headers={"x-api-key": "kR"}, json=body)
    assert r2.status_code == 200
    # Third request should 429
    r3 = client.post("/v1/receipt", headers={"x-api-key": "kR"}, json=body)
    assert r3.status_code == 429
    assert r3.json().get("detail") == "rate limit exceeded"
    assert r3.headers.get("X-RateLimit-Limit") == r1.headers.get("X-RateLimit-Limit")
    assert r3.headers.get("X-RateLimit-Remaining") == "0"
    assert "X-RateLimit-Reset" in r3.headers
