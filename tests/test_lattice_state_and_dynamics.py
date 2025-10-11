import io
import json

import numpy as np

from oscillink.core.lattice import OscillinkLattice, json_line_logger


def _mk_lat(N: int = 8, D: int = 6) -> OscillinkLattice:
    rng = np.random.default_rng(123)
    Y = rng.normal(size=(N, D)).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=3, deterministic_k=True, neighbor_seed=11)
    psi = rng.normal(size=(D,), scale=0.1).astype(np.float32)
    gates = np.clip(rng.random(size=(N,), dtype=np.float32), 0.1, 1.0)
    lat.set_query(psi, gates=gates)
    lat.add_chain(list(range(0, N, max(1, N // 4)))[:4], lamP=0.3)
    return lat


def test_export_import_json_npz_roundtrip(tmp_path):
    lat = _mk_lat()
    # ensure some computations happened
    lat.settle(max_iters=3, tol=1e-3, warm_start=False)

    # JSON roundtrip via export_state/from_state
    state = lat.export_state(include_graph=True, include_chain=True)
    data = json.dumps(state)
    state2 = json.loads(data)
    lat2 = OscillinkLattice.from_state(state2)
    assert lat2.N == lat.N and lat2.D == lat.D
    assert lat2.A.shape == lat.A.shape

    # NPZ roundtrip via save_state/from_npz
    p = tmp_path / "lat_state.npz"
    lat.save_state(str(p), format="npz", include_graph=True, include_chain=True)
    lat3 = OscillinkLattice.from_npz(str(p))
    assert lat3.N == lat.N and lat3.D == lat.D
    # adjacency restored if present
    assert lat3.A.shape == lat.A.shape


def test_invalid_modes_and_errors():
    lat = _mk_lat()
    # invalid signature mode
    try:
        lat.set_signature_mode("invalid")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # invalid receipt detail
    try:
        lat.set_receipt_detail("bad")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # set_gates wrong length
    import numpy as np

    try:
        lat.set_gates(np.ones(lat.N + 1, dtype=np.float32))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # set_query with gates wrong length
    try:
        lat.set_query(np.ones(lat.D, dtype=np.float32), gates=np.ones(lat.N + 2, dtype=np.float32))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_dynamics_flag_and_bundle(monkeypatch):
    lat = _mk_lat()
    monkeypatch.setenv("OSCILLINK_RECEIPT_DYNAMICS", "1")
    lat.settle(max_iters=3, tol=1e-3, warm_start=False)
    r = lat.receipt()
    assert "dynamics" in r["meta"]
    dyn = r["meta"]["dynamics"]
    for k in ("temperature", "step_deltaH", "viscosity_step", "flow_total", "radius"):
        assert k in dyn
    bundle = lat.bundle(k=3, alpha=0.6)
    assert len(bundle) == 3
    assert {"id", "score", "align"}.issubset(bundle[0].keys())


def test_chain_receipt_structure():
    lat = _mk_lat(N=10, D=5)
    chain = [0, 2, 5, 9]
    lat.add_chain(chain, lamP=0.25)
    cr = lat.chain_receipt(chain, z_th=10.0)
    assert isinstance(cr["verdict"], bool)
    assert len(cr["edges"]) == len(chain) - 1


def test_json_line_logger_captures_events():
    lat = _mk_lat()
    buf = io.StringIO()
    lat.set_logger(json_line_logger(stream=buf))
    lat.settle(max_iters=2, tol=1e-3, warm_start=False)
    _ = lat.receipt()
    content = buf.getvalue().strip().splitlines()
    # should have at least settle and receipt events
    assert any('"event":"settle"' in line for line in content)
    assert any('"event":"receipt"' in line for line in content)
