from __future__ import annotations
import numpy as np
from fastapi.testclient import TestClient
from cloud.app.main import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'ok' and 'version' in data

def test_settle_small():
    Y = np.random.RandomState(0).randn(12, 6).astype(float).tolist()
    req = {
        "Y": Y,
        "psi": [0.0]*6,
        "params": {"kneighbors": 4, "lamG":1.0, "lamC":0.5, "lamQ":4.0},
        "options": {"max_iters": 4, "tol": 1e-2, "include_receipt": True, "bundle_k": 3}
    }
    r = client.post('/v1/settle', json=req)
    assert r.status_code == 200
    out = r.json()
    assert 'state_sig' in out and out['receipt'] is not None
    assert len(out['bundle']) == 3
