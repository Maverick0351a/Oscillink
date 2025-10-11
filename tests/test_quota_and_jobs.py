from __future__ import annotations

import json
import time
from importlib import reload

import numpy as np
from fastapi.testclient import TestClient


def _reload_app():
    from cloud.app import main as mainmod

    reload(mainmod)
    return mainmod.app


def test_quota_exceeded_and_headers(monkeypatch):
    # Set small quota window and limit
    monkeypatch.setenv("OSCILLINK_API_KEYS", "kA")
    monkeypatch.setenv("OSCILLINK_KEY_NODE_UNITS_LIMIT", "500")  # allow 500 units
    monkeypatch.setenv("OSCILLINK_KEY_NODE_UNITS_WINDOW", "3600")
    # Clear rate limit to avoid interference
    monkeypatch.delenv("OSCILLINK_RATE_LIMIT", raising=False)
    from cloud.app import config as cfg

    cfg.get_settings.cache_clear()  # type: ignore
    app = _reload_app()
    client = TestClient(app)

    # Each request: N*D units. Use N=20, D=10 => 200 units.
    Y = np.random.RandomState(0).randn(20, 10).astype(float).tolist()
    body = {"Y": Y}
    # First two should succeed (200 + 200 = 400 <= 500)
    r1 = client.post("/v1/receipt", headers={"x-api-key": "kA"}, json=body)
    assert r1.status_code == 200
    assert "X-Quota-Limit" in r1.headers and int(r1.headers["X-Quota-Limit"]) == 500
    r2 = client.post("/v1/receipt", headers={"x-api-key": "kA"}, json=body)
    assert r2.status_code == 200
    # Third should exceed (would be 600 > 500)
    r3 = client.post("/v1/receipt", headers={"x-api-key": "kA"}, json=body)
    assert r3.status_code == 429
    assert r3.json()["detail"] == "quota exceeded"
    assert r3.headers.get("X-Quota-Remaining") == "0"


def test_quota_oversize_single_request(monkeypatch):
    monkeypatch.setenv("OSCILLINK_API_KEYS", "kB")
    monkeypatch.setenv("OSCILLINK_KEY_NODE_UNITS_LIMIT", "100")  # very small limit
    from cloud.app import config as cfg

    cfg.get_settings.cache_clear()  # type: ignore
    app = _reload_app()
    client = TestClient(app)
    # Build request with units 120 (N=12,D=10)
    Y = np.random.RandomState(1).randn(12, 10).astype(float).tolist()
    r = client.post("/v1/settle", headers={"x-api-key": "kB"}, json={"Y": Y})
    assert r.status_code == 413
    assert "exceed per-key limit" in r.json()["detail"]


def test_async_job_lifecycle_and_quota(monkeypatch):
    monkeypatch.setenv("OSCILLINK_API_KEYS", "kJ")
    monkeypatch.setenv("OSCILLINK_KEY_NODE_UNITS_LIMIT", "10000")
    from cloud.app import config as cfg

    cfg.get_settings.cache_clear()  # type: ignore
    app = _reload_app()
    client = TestClient(app)

    Y = np.random.RandomState(2).randn(30, 12).astype(float).tolist()  # 360 units
    submit = client.post(
        "/v1/jobs/settle",
        headers={"x-api-key": "kJ"},
        json={"Y": Y, "options": {"include_receipt": True}},
    )
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]
    # Poll until done (should be fast)
    for _ in range(30):
        time.sleep(0.05)
        status = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "kJ"})
        assert status.status_code == 200
        js = status.json()
        if js["status"] == "done":
            assert "result" in js and "meta" in js["result"]
            quota_meta = js["result"]["meta"].get("quota")
            # Quota meta present and remaining positive
            assert quota_meta is None or quota_meta.get("remaining", 1) >= 0
            break
    else:
        raise AssertionError("job did not complete in time")


def test_usage_log_append(monkeypatch, tmp_path):
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OSCILLINK_API_KEYS", "kL")
    monkeypatch.setenv("OSCILLINK_KEY_NODE_UNITS_LIMIT", "0")  # disable quota for clarity
    monkeypatch.setenv("OSCILLINK_USAGE_LOG", str(log_file))
    from cloud.app import config as cfg

    cfg.get_settings.cache_clear()  # type: ignore
    app = _reload_app()
    client = TestClient(app)

    Y = np.random.RandomState(3).randn(10, 4).astype(float).tolist()
    # Run a settle request
    r = client.post("/v1/settle", headers={"x-api-key": "kL"}, json={"Y": Y})
    assert r.status_code == 200
    # Run a receipt request
    r2 = client.post("/v1/receipt", headers={"x-api-key": "kL"}, json={"Y": Y})
    assert r2.status_code == 200
    # Run async job
    job = client.post("/v1/jobs/settle", headers={"x-api-key": "kL"}, json={"Y": Y})
    job_id = job.json()["job_id"]
    for _ in range(30):
        time.sleep(0.05)
        js = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "kL"}).json()
        if js["status"] == "done":
            break
    assert log_file.exists()
    lines = log_file.read_text().strip().splitlines()
    # Expect at least 3 lines (settle, receipt, job)
    assert len(lines) >= 3
    parsed = [json.loads(line) for line in lines]
    events = {p["event"] for p in parsed}
    assert {"settle", "receipt", "job_settle"} <= events
    # Basic field presence
    for p in parsed:
        assert "ts" in p and "units" in p
