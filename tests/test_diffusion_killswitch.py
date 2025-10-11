import importlib
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient  # type: ignore

# Skip if fastapi/cloud app not available
pytestmark = (
    pytest.mark.skipif(
        not os.environ.get("CI") and "fastapi" not in importlib.util.find_spec("fastapi").name,
        reason="fastapi not installed",
    )
    if importlib.util.find_spec("fastapi") is None
    else []  # type: ignore
)

# Import after setting env in individual tests


def _client():
    from cloud.app.main import app

    return TestClient(app)


PAYLOAD_BASE = {
    "Y": np.random.randn(6, 4).astype(float).tolist(),
    "psi": np.random.randn(4).astype(float).tolist(),
    "params": {"kneighbors": 3},
    "options": {"include_receipt": True},
}


def test_diffusion_gates_disabled():
    os.environ["OSCILLINK_DIFFUSION_GATES_ENABLED"] = "0"
    # Ensure kill-switch engaged
    client = _client()
    payload = dict(PAYLOAD_BASE)
    payload["gates"] = [1, 1, 1, 1, 1, 1]
    r = client.post("/v1/settle", json=payload)
    assert r.status_code == 403
    assert "temporarily disabled" in r.text
    # cleanup
    os.environ["OSCILLINK_DIFFUSION_GATES_ENABLED"] = "1"


def test_diffusion_gates_enabled_without_key():
    os.environ["OSCILLINK_DIFFUSION_GATES_ENABLED"] = "1"
    # No API key needed if auth open; but feature requires tier entitlement (free tier denies) -> 403
    client = _client()
    payload = dict(PAYLOAD_BASE)
    payload["gates"] = [1, 1, 1, 1, 1, 1]
    r = client.post("/v1/settle", json=payload)
    assert r.status_code == 403
    assert "not enabled for this tier" in r.text


def test_diffusion_gates_pro_tier_env_key():
    # Provide env key with tier override pro so gating allowed
    os.environ["OSCILLINK_API_KEYS"] = "testpro"
    os.environ["OSCILLINK_KEY_TIERS"] = "testpro:pro"
    os.environ["OSCILLINK_DIFFUSION_GATES_ENABLED"] = "1"
    # Re-import runtime config to pick up env changes for keys list
    from cloud.app import runtime_config

    importlib.reload(runtime_config)
    from cloud.app.main import app

    client = TestClient(app)
    payload = dict(PAYLOAD_BASE)
    payload["gates"] = [1, 1, 1, 1, 1, 1]
    r = client.post("/v1/settle", json=payload, headers={"x-api-key": "testpro"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "receipt" in body
    assert body["meta"]["N"] == 6
    # cleanup
    for var in ["OSCILLINK_API_KEYS", "OSCILLINK_KEY_TIERS", "OSCILLINK_DIFFUSION_GATES_ENABLED"]:
        os.environ.pop(var, None)
