import numpy as np

from oscillink.core.lattice import OscillinkLattice


def _tiny_lat(N: int = 6, D: int = 8) -> OscillinkLattice:
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(N, D)).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=2, deterministic_k=True, neighbor_seed=42)
    lat.set_query(rng.normal(size=(D,), loc=0.0, scale=0.1).astype(np.float32))
    return lat


def test_receipt_detail_light_and_full_and_null_cap(monkeypatch):
    lat = _tiny_lat()
    # ensure a first settle to populate last timings
    lat.settle(max_iters=4, tol=1e-3, warm_start=False)

    # light mode: should skip heavy diagnostics
    lat.set_receipt_detail("light")
    r_light = lat.receipt()
    assert r_light["meta"]["receipt_detail"] == "light"
    # allow env var to cap null points aggressively
    monkeypatch.setenv("OSCILLINK_RECEIPT_NULL_CAP", "1")
    r_light2 = lat.receipt()
    assert r_light2["meta"]["null_points_summary"]["null_cap_applied"] in (True, False)

    # full mode
    lat.set_receipt_detail("full")
    monkeypatch.setenv("OSCILLINK_RECEIPT_NULL_CAP", "0")
    r_full = lat.receipt()
    assert r_full["meta"]["receipt_detail"] == "full"
    assert "null_points" in r_full


def test_signature_modes_and_verify(monkeypatch):
    lat = _tiny_lat()
    secret = b"s3cret"
    lat.set_receipt_secret(secret)
    # minimal
    lat.set_signature_mode("minimal")
    r1 = lat.receipt()
    assert "signature" in r1["meta"]
    # extended
    lat.set_signature_mode("extended")
    r2 = lat.receipt()
    sig = r2["meta"]["signature"]
    assert sig["payload"]["mode"] == "extended"
    # verify helper should succeed
    assert lat.verify_current_receipt(secret)


def test_warm_start_and_inertia_change_x0(monkeypatch):
    lat = _tiny_lat()
    # first settle will update U
    lat.settle(max_iters=2, tol=1e-3, warm_start=False)
    U_before = lat.U.copy()
    # inertia warm start blend path
    lat.settle(max_iters=2, tol=1e-3, warm_start=True, inertia=0.5)
    assert not np.allclose(lat.U, U_before)


def test_rebuild_graph_and_cache_invalidation():
    lat = _tiny_lat()
    _ = lat.solve_Ustar()  # populate cache
    assert lat.stats["ustar_solves"] >= 1
    lat.rebuild_graph(kneighbors=3, deterministic_k=True, neighbor_seed=7, row_cap_val=0.8)
    # cache invalidated; subsequent solve should increase solves count
    prev_solves = lat.stats["ustar_solves"]
    _ = lat.solve_Ustar()
    assert lat.stats["ustar_solves"] == prev_solves + 1
