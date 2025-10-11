import json

import numpy as np
from fastapi.testclient import TestClient

from cloud.app.main import app
from oscillink import OscillinkLattice


def test_state_sig_in_receipt_meta():
    Y = np.random.randn(16, 8).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=6)
    lat.settle(max_iters=2)
    rec = lat.receipt()
    assert (
        "meta" in rec and "state_sig" in rec["meta"] and isinstance(rec["meta"]["state_sig"], str)
    )
    assert len(rec["meta"]["state_sig"]) >= 16


def test_row_sum_cap_preserves_symmetry():
    Y = np.random.randn(24, 12).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=5, deterministic_k=True)
    A = lat.A
    assert np.allclose(A, A.T, atol=1e-6), "Adjacency must remain symmetric after row cap"


def test_deterministic_knn_tie_break():
    # Construct duplicate vectors to force identical cosine similarities
    base = np.ones((10, 4), dtype=np.float32)
    Y = base.copy()
    lat = OscillinkLattice(Y, kneighbors=4, deterministic_k=True)
    # Build another lattice and ensure identical adjacency ordering fingerprint
    lat2 = OscillinkLattice(Y, kneighbors=4, deterministic_k=True)

    # Fingerprint: row -> indices of top-k neighbors (nonzero entries)
    def topk_indices(A):
        return [tuple(np.where(row > 0)[0].tolist()) for row in A]

    assert topk_indices(lat.A) == topk_indices(lat2.A)


def test_kneighbors_clamp_effective_vs_requested():
    Y = np.random.randn(6, 3).astype(np.float32)
    requested = 10  # > N-1
    lat = OscillinkLattice(Y, kneighbors=requested)
    assert lat._kneighbors == Y.shape[0] - 1


def test_webhook_unverified_override_allows_processing(monkeypatch):
    client = TestClient(app)
    # Enable override to allow processing without verification
    monkeypatch.setenv("OSCILLINK_ALLOW_UNVERIFIED_STRIPE", "1")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "test_secret")
    event = {
        "id": "evt_test_unverified_override",
        "type": "customer.subscription.created",
        "data": {
            "object": {
                "metadata": {"api_key": "key_xyz"},
                "items": {"data": [{"price": {"id": "price_free"}}]},
            }
        },
    }
    monkeypatch.setenv("OSCILLINK_STRIPE_PRICE_MAP", "price_free:free")
    r = client.post(
        "/stripe/webhook",
        content=json.dumps(event),
        headers={"stripe-signature": "t=1,v1=test", "content-type": "application/json"},
    )
    assert r.status_code == 200
    body = r.json()
    # With override, should process even though not verified
    assert body["processed"] is True
    assert body.get("verified") is False
    assert body.get("allow_unverified_override") is True
