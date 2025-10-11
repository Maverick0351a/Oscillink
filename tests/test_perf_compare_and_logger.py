import io

import numpy as np

from oscillink.core.lattice import OscillinkLattice, json_line_logger
from oscillink.core.perf import compare_perf


def test_compare_perf_detects_no_failures():
    baseline = {
        "aggregates": {
            "build_ms": {"mean": 1.0},
            "settle_ms": {"mean": 2.0},
            "receipt_ms": {"mean": 3.0},
        }
    }
    current = {
        "aggregates": {
            "build_ms": {"mean": 1.1},
            "settle_ms": {"mean": 2.1},
            "receipt_ms": {"mean": 3.1},
        }
    }
    res = compare_perf(baseline, current, tolerance_pct=20.0)
    assert not res["failures"]


def test_json_line_logger_emits_events():
    buf = io.StringIO()
    logger = json_line_logger(stream=buf)
    Y = np.random.RandomState(0).randn(10, 4).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=3, deterministic_k=True, neighbor_seed=4)
    lat.set_logger(logger)
    lat.set_query(np.zeros(4, dtype=np.float32))
    lat.settle(max_iters=2, tol=1e-2)
    lat.receipt()
    lines = buf.getvalue().strip().splitlines()
    assert any('"event":"settle"' in ln for ln in lines)
    assert any('"event":"receipt"' in ln for ln in lines)
