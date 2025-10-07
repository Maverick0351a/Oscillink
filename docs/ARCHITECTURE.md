# Architecture (Phase 1)

- `core/graph.py`: mutual‑kNN, row‑cap, normalized Laplacian, path Laplacian, MMR.
- `core/solver.py`: Jacobi‑preconditioned CG (pure NumPy).
- `core/receipts.py`: ΔH via trace identity, per‑node attributions, null points.
- `core/lattice.py`: OscillinkLattice (settle, receipt, chain_receipt, bundle).
- `adapters/text.py`: deterministic hash embedder (placeholder).

All math uses normalized Laplacian; system matrix is SPD with λ_G > 0.
