from importlib import reload

import numpy as np
from fastapi.testclient import TestClient


def _reload_app():
    from cloud.app import main as mainmod

    reload(mainmod)
    return mainmod.app


def test_per_ip_rate_limit(monkeypatch):
    # Enable per-IP limiter: 2 requests per window
    monkeypatch.setenv("OSCILLINK_IP_RATE_LIMIT", "2")
    monkeypatch.setenv("OSCILLINK_IP_RATE_WINDOW", "60")
    app = _reload_app()
    client = TestClient(app)
    Y = np.random.RandomState(0).randn(3, 3).astype(float).tolist()
    body = {"Y": Y}
    r1 = client.post("/v1/receipt", json=body)
    assert r1.status_code == 200
    assert "X-IPLimit-Limit" in r1.headers
    r2 = client.post("/v1/receipt", json=body)
    assert r2.status_code == 200
    r3 = client.post("/v1/receipt", json=body)
    assert r3.status_code == 429
    assert r3.json().get("detail") == "ip rate limit exceeded"
    assert r3.headers.get("X-IPLimit-Remaining") == "0"
    assert "X-IPLimit-Reset" in r3.headers
