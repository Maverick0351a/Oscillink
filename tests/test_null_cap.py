import numpy as np

from oscillink.core.lattice import OscillinkLattice


def build_lat(N=40, D=32):
    Y = np.random.randn(N, D).astype(np.float32)
    psi = (Y[:5].mean(0) / (np.linalg.norm(Y[:5].mean(0)) + 1e-9)).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=6)
    lat.set_query(psi)
    lat.settle()
    return lat


def test_null_points_no_cap(monkeypatch):
    monkeypatch.delenv("OSCILLINK_RECEIPT_NULL_CAP", raising=False)
    lat = build_lat()
    r = lat.receipt()
    meta = r["meta"].get("null_points_summary", {})
    assert meta.get("null_cap_applied") is False
    assert meta.get("total_null_points") == meta.get("returned_null_points")
    assert isinstance(r["null_points"], list)


def test_null_points_with_cap(monkeypatch):
    monkeypatch.setenv("OSCILLINK_RECEIPT_NULL_CAP", "3")
    lat = build_lat()
    r = lat.receipt()
    meta = r["meta"].get("null_points_summary", {})
    assert meta.get("null_cap_applied") is True
    assert meta.get("returned_null_points") == 3
    assert meta.get("total_null_points") >= 3
    assert len(r["null_points"]) == 3
