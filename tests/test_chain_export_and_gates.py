import numpy as np
import pytest

from oscillink.core.lattice import OscillinkLattice


def test_chain_order_preserved_export_import():
    Y = np.random.randn(25, 12).astype(np.float32)
    chain = [3, 5, 9, 14]
    lat = OscillinkLattice(Y, kneighbors=4, deterministic_k=True)
    lat.set_query(np.random.randn(12).astype(np.float32))
    lat.add_chain(chain, lamP=0.25)
    state = lat.export_state(include_chain=True)
    assert state.get("chain_nodes") == chain
    lat2 = OscillinkLattice.from_state(state)
    # exporting again should keep same chain order
    state2 = lat2.export_state(include_chain=True)
    assert state2.get("chain_nodes") == chain


def test_add_chain_weight_length_validation():
    Y = np.random.randn(10, 8).astype(np.float32)
    lat = OscillinkLattice(Y)
    with pytest.raises(ValueError):
        lat.add_chain([1], lamP=0.2)  # too short
    with pytest.raises(ValueError):
        lat.add_chain([0, 2, 4], lamP=0.2, weights=[0.5, 0.5, 0.5])  # mismatch length


def test_set_gates_and_invalid_length():
    Y = np.random.randn(12, 6).astype(np.float32)
    lat = OscillinkLattice(Y)
    gates = np.linspace(0.2, 1.0, Y.shape[0]).astype(np.float32)
    lat.set_gates(gates)
    assert np.allclose(lat.B_diag, gates)
    with pytest.raises(ValueError):
        lat.set_gates(np.ones(5, dtype=np.float32))  # wrong length
