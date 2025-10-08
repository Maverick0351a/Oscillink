from importlib import reload

from fastapi.testclient import TestClient

from cloud.app import main as mainmod


def test_kneighbors_clamp_receipt():
    reload(mainmod)
    client = TestClient(mainmod.app)
    # N=5, request kneighbors=10 (>=N) so effective should be N-1=4
    Y = [[float(i + j) for j in range(3)] for i in range(5)]
    payload = {"Y": Y, "params": {"kneighbors": 10}}
    r = client.post("/v1/receipt", json=payload)
    assert r.status_code == 200, r.text
    meta = r.json()["meta"]
    assert meta["kneighbors_requested"] == 10
    assert meta["kneighbors_effective"] == 4
