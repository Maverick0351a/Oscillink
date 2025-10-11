import json
import subprocess
import sys

import numpy as np

from oscillink import OscillinkLattice


def test_provenance_export_import_roundtrip():
    Y = np.random.RandomState(0).randn(20, 8).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=4, deterministic_k=True, neighbor_seed=5)
    lat.set_query(np.zeros(8, dtype=np.float32))
    lat.add_chain([0, 1, 2, 3], lamP=0.2)
    state = lat.export_state()
    assert "provenance" in state
    prov = state["provenance"]
    lat2 = OscillinkLattice.from_state(state)
    state2 = lat2.export_state()
    assert state2["provenance"] == prov


def test_benchmark_json_mode_runs():
    # invoke benchmark in json mode with tiny config
    cmd = [
        sys.executable,
        "scripts/benchmark.py",
        "--json",
        "--N",
        "50",
        "--D",
        "16",
        "--kneighbors",
        "4",
        "--trials",
        "1",
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    assert "aggregates" in data and "build_ms" in data["aggregates"]
