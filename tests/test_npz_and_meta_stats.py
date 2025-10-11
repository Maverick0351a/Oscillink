import os
import tempfile

import numpy as np

from oscillink import OscillinkLattice


def test_receipt_contains_adjacency_stats():
    Y = np.random.RandomState(0).randn(30, 10).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=5, deterministic_k=True, neighbor_seed=7)
    lat.set_query(np.zeros(10, dtype=np.float32))
    rec = lat.receipt()
    meta = rec["meta"]
    assert "avg_degree" in meta and meta["avg_degree"] > 0
    assert "edge_density" in meta and 0 < meta["edge_density"] < 1


def test_npz_roundtrip(tmp_path=None):
    if tmp_path is None:
        tmp_path = tempfile.TemporaryDirectory().name
        os.makedirs(tmp_path, exist_ok=True)
    Y = np.random.RandomState(1).randn(25, 6).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=4, deterministic_k=True, neighbor_seed=3)
    lat.set_query(np.zeros(6, dtype=np.float32))
    lat.add_chain([0, 1, 2, 3], lamP=0.2)
    p = os.path.join(tmp_path, "state.npz")
    lat.save_state(p, format="npz")
    lat2 = OscillinkLattice.from_npz(p)
    # verify provenance and chain restoration
    s1 = lat.export_state()
    s2 = lat2.export_state()
    assert s1["provenance"] == s2["provenance"]
    assert s2.get("chain_nodes") == s1.get("chain_nodes")
