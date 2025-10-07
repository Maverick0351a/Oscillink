# API

```python
from oscillink.core.lattice import OscillinkLattice
```

## `OscillinkLattice(Y, kneighbors=6, row_cap_val=1.0, lamG=1.0, lamC=0.5, lamQ=4.0)`
Create a lattice from anchor vectors `Y (N x D)`.

## `set_query(psi, gates=None)`
Set the focus vector and optional per‑node gates `b`.

## `add_chain(chain, lamP=0.2, weights=None)` / `clear_chain()`
Attach an SPD chain prior (path Laplacian) or remove it.

## `settle(dt=1.0, max_iters=12, tol=1e-3)`
Implicit step using CG (Jacobi preconditioner). Diagnostics returned.

## `receipt()`
Exact `ΔH` + per‑node components + null points + diagnostics.

## `chain_receipt(chain, z_th=2.5)`
Pass/fail verdict, weakest link, per‑edge z‑scores, chain coherence gain.

## `bundle(k=8, alpha=0.5)`
Return top‑k items ranked by blended coherence + alignment, MMR‑diversified.
