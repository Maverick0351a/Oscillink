import numpy as np

from oscillink.core.lattice import OscillinkLattice


def test_receipt_meta_and_version():
    Y = np.random.randn(20, 16).astype(np.float32)
    psi = (Y[:5].mean(axis=0) / (np.linalg.norm(Y[:5].mean(axis=0)) + 1e-12)).astype(np.float32)
    lat = OscillinkLattice(Y)
    lat.set_query(psi)
    lat.settle(max_iters=4)
    r = lat.receipt()
    assert "version" in r and "meta" in r and isinstance(r["meta"], dict)
    assert "ustar_solves" in r["meta"] and r["meta"]["ustar_solves"] >= 1


def test_refresh_Ustar_forces_new_solve():
    Y = np.random.randn(25, 12).astype(np.float32)
    psi = (Y[:4].mean(axis=0) / (np.linalg.norm(Y[:4].mean(axis=0)) + 1e-12)).astype(np.float32)
    lat = OscillinkLattice(Y)
    lat.set_query(psi)
    lat.receipt()  # compute & cache
    solves_before = lat.stats["ustar_solves"]
    lat.refresh_Ustar()
    assert lat.stats["ustar_solves"] == solves_before + 1


def test_settle_callback_invoked():
    Y = np.random.randn(18, 10).astype(np.float32)
    psi = (Y[:3].mean(axis=0) / (np.linalg.norm(Y[:3].mean(axis=0)) + 1e-12)).astype(np.float32)
    lat = OscillinkLattice(Y)
    lat.set_query(psi)
    called = {}

    def cb(lattice, stats):
        called["ok"] = True
        called["iters"] = stats["iters"]

    lat.add_settle_callback(cb)
    lat.settle(max_iters=3)
    assert called.get("ok") is True and "iters" in called
    lat.remove_settle_callback(cb)
    # ensure removal doesn't break anything
    lat.settle(max_iters=2)
