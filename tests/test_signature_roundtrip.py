import numpy as np

from oscillink import OscillinkLattice


def test_export_import_signature_stable():
    Y = np.random.RandomState(0).randn(20, 5).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=4, deterministic_k=True, neighbor_seed=42)
    lat.set_query(np.zeros(5, dtype=np.float32))
    lat.add_chain([0, 1, 2, 3], lamP=0.3)
    sig_before = lat._signature()
    state = lat.export_state(include_graph=True, include_chain=True)
    lat2 = OscillinkLattice.from_state(state)
    sig_after = lat2._signature()
    assert sig_before == sig_after, "Signature mismatch after roundtrip with graph & chain"
