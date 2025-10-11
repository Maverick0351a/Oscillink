import numpy as np
import pytest

from oscillink.preprocess.diffusion import compute_diffusion_gates


def test_diffusion_validation_errors():
    Y = np.eye(3, dtype=np.float32)
    psi = np.ones(3, dtype=np.float32)
    # Y must be 2D
    with pytest.raises(ValueError):
        compute_diffusion_gates(Y[0], psi)
    # psi dim mismatch
    with pytest.raises(ValueError):
        compute_diffusion_gates(Y, np.ones(2, dtype=np.float32))
    # gamma must be > 0
    with pytest.raises(ValueError):
        compute_diffusion_gates(Y, psi, gamma=0.0)
    # kneighbors must be >=1
    with pytest.raises(ValueError):
        compute_diffusion_gates(Y, psi, kneighbors=0)
