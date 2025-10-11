import numpy as np

from oscillink.core.graph import mmr_diversify, mutual_knn_adj


def test_mutual_knn_adj_degenerate():
    Y = np.zeros((1, 4), dtype=np.float32)
    A = mutual_knn_adj(Y, k=6)
    assert A.shape == (1, 1)
    assert float(A.sum()) == 0.0


def test_mmr_diversify_k_zero():
    Y = np.random.RandomState(0).randn(3, 2).astype(np.float32)
    scores = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = mmr_diversify(Y, scores, k=0)
    assert out == []
