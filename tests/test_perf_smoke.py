import time

import numpy as np

from oscillink.core.lattice import OscillinkLattice


def test_perf_smoke_small():
    # Not a strict performance assertion: just ensures settle + receipt runs under a loose ceiling.
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(64, 16)).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=6, deterministic_k=True)
    lat.set_query(rng.normal(size=16).astype(np.float32))
    t0 = time.time()
    lat.settle(max_iters=6)
    rec = lat.receipt()
    elapsed_ms = (time.time() - t0) * 1000
    # Very generous ceiling to avoid CI flakiness.
    assert rec["deltaH_total"] is not None
    assert elapsed_ms < 1500, f"Perf smoke exceeded ceiling: {elapsed_ms:.1f} ms"
