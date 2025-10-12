This document has moved.

New location: [docs/reference/RECEIPT_SCHEMA.md](./reference/RECEIPT_SCHEMA.md)
# Receipt Schema

Authoritative description of the JSON object returned by `OscillinkLattice.receipt()` and `chain_receipt()`.

Version field (`version`) will increment if breaking layout changes occur. Current: `1.0`.

## Top-Level Fields (receipt)
| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Schema version string (e.g. `"1.0"`). |
| `deltaH_total` | float | Total ΔH energy improvement from initial anchors to stationary solution. |
| `deltaH_anchor` | float | Anchor alignment component of ΔH (λ_G term). |
| `deltaH_graph` | float | Graph coherence component (λ_C Laplacian term). |
| `deltaH_query` | float | Query pull / gating component (λ_Q term). |
| `deltaH_chain` | float | Chain prior component (λ_P term, 0 if no chain). |
| `components` | object | Per-node component arrays (coherence_drop, anchor_penalty, query_term). |
| `null_points` | list | List of anomalous edges (if any) with `edge`, `z`, `residual`. |
| `meta` | object | Metadata & provenance (see below). |

### `components`
| Field | Type | Notes |
|-------|------|-------|
| `coherence_drop` | float[] | Per-node contribution from graph coherence term (positive means improved coherence). |
| `anchor_penalty` | float[] | Per-node anchor penalty residual energy. |
| `query_term` | float[] | Per-node query alignment penalty. |

### `null_points`
Each object: `{ "edge": [i, j], "z": float, "residual": float }`. An entry indicates node `i` has an anomalously large residual with neighbor `j` (Z-score threshold configurable via `chain_receipt` / receipt internals).

### `meta`
| Field | Type | Description |
|-------|------|-------------|
| `state_sig` | string | Stable signature hash (parameters + adjacency fingerprint + chain metadata). |
| `ustar_cached` | bool | Whether stationary solution U* reused prior cache. |
| `ustar_solves` | int | Total number of stationary solves performed. |
| `ustar_iters` | int | Iterations used by last stationary solve (CG). |
| `ustar_res` | float | Final residual of stationary solve. |
| `ustar_converged` | bool | Residual ≤ tolerance. |
| `ustar_solve_ms` | float | Milliseconds spent in last stationary solve. |
| `graph_build_ms` | float | Time to build neighbor graph at initialization. |
| `last_settle_ms` | float | Milliseconds of most recent `settle()` step. |
| `kneighbors` | int | k used for mutual kNN graph. |
| `avg_degree` | float | Average degree of adjacency (after mutual + cap). |
| `edge_density` | float | Density = edges / (N^2). |
| `chain_length` | int | Length of active chain (0 if none). |
| `lamG` | float | Anchor weight. |
| `lamC` | float | Graph coherence weight. |
| `lamQ` | float | Query pull weight. |
| `lamP` | float | Chain prior weight (0 if none). |
| `signature` | object? | Present only if receipt signing enabled (see below). |

#### `signature` (optional)
| Field | Type | Description |
|-------|------|-------------|
| `algorithm` | string | Currently `HMAC-SHA256`. |
| `payload` | object | `{ "state_sig": str, "deltaH_total": float }` |
| `signature` | string | Hex digest of HMAC(secret, canonical_json(payload)). |

## Chain Receipt (`chain_receipt`)
Adds focused chain-specific diagnostics:
| Field | Type | Description |
|-------|------|-------------|
| `verdict` | string | "pass" if chain internally coherent else "weak" / other label (implementation detail). |
| `weakest_link` | int | Index in chain with largest coherence deficit. |
| `z_scores` | float[] | Per-link Z-scores used for verdict. |
| `deltaH_chain` | float | Same as in standard receipt (chain component). |
| `meta` | object | Same structure as primary receipt (shared reference / recomputed). |

## Stability & Backwards Compatibility
Minor additive fields will increment a patch version (e.g. 1.0 -> 1.1). Removals / renames bump the major version.

## Example (abridged)
```json
{
  "version": "1.0",
  "deltaH_total": 12.34,
  "deltaH_anchor": 5.67,
  "deltaH_graph": 3.21,
  "deltaH_query": 2.11,
  "deltaH_chain": 1.35,
  "components": {
    "coherence_drop": [0.1, 0.05, ...],
    "anchor_penalty": [0.9, 0.8, ...],
    "query_term": [0.2, 0.1, ...]
  },
  "null_points": [ { "edge": [4, 17], "z": 3.4, "residual": 0.123 } ],
  "meta": {
    "state_sig": "1d9f...",
    "ustar_cached": false,
    "ustar_solves": 1,
    "ustar_iters": 23,
    "ustar_res": 0.00091,
    "ustar_converged": true,
    "ustar_solve_ms": 4.72,
    "graph_build_ms": 1.88,
    "last_settle_ms": 0.53,
    "kneighbors": 6,
    "avg_degree": 5.9,
    "edge_density": 0.049,
    "chain_length": 4,
    "lamG": 1.0,
    "lamC": 0.5,
    "lamQ": 4.0,
    "lamP": 0.2
  }
}
```

## Notes
- `deltaH_total` equals sum of its component deltas.
- Per-node arrays in `components` align with row indices of `Y`.
- Signed receipts add the `signature` block under `meta`; verification does not mutate the object.
# Receipt Schema

This document describes the structure and semantics of the dictionary returned by `OscillinkLattice.receipt()`.

## Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Package version (`oscillink.__version__`). |
| `deltaH_total` | float | Total energy decrease (ΔH). |
| `coh_drop_sum` | float | Sum of per-node coherence drop term. |
| `anchor_pen_sum` | float | Sum of anchor penalty term (λ_G). |
| `query_term_sum` | float | Sum of query penalty term (λ_Q, gated). |
| `cg_iters` | int | Iterations used in the last `settle()` (implicit Euler CG). |
| `residual` | float or null | Residual from the last `settle()` solve. |
| `t_ms` | float or null | Wall-clock ms of last settle. |
| `null_points` | list<object> | High residual structural outliers. |
| `meta` | object | Auxiliary diagnostic & cache metadata (see below). |

## Meta Fields

| Field | Type | Description |
|-------|------|-------------|
| `ustar_cached` | bool | Whether stationary solution U* was served from cache. |
| `ustar_solves` | int | Number of distinct stationary solves performed. |
| `ustar_cache_hits` | int | How many times the cached U* was reused. |
| `ustar_converged` | bool | CG convergence flag for latest stationary solve. |
| `ustar_res` | float | Final residual for latest stationary solve. |
| `ustar_iters` | int | Iterations consumed for latest stationary solve. |
| `signature` | object? | Present only when signing secret is configured (see below). |
| `avg_degree` | float | Average number of non-zero neighbors per node (A > 0 count / N). |
| `edge_density` | float | Non-zero adjacency fraction over N*(N-1). |

### Signature Object (Optional)

Emitted only if `set_receipt_secret(secret)` was called with non-null secret.

| Field | Type | Description |
|-------|------|-------------|
| `algorithm` | string | Always `HMAC-SHA256`. |
| `payload` | object | Canonical JSON structure hashed. |
| `signature` | string | Hex digest of HMAC(payload). |

Payload currently contains:

```jsonc
{
  "state_sig": "<hex>",     // hash of lattice-defining state
  "deltaH_total": <float>
}
```

### Null Points Entry

Each element of `null_points`:

| Field | Type | Description |
|-------|------|-------------|
| `edge` | [int,int] | Node pair (i,j) forming a high residual candidate. |
| `z` | float | Z-score of residual vs node i's distribution. |
| `residual` | float | Raw residual value. |

## State Signature (`state_sig`)

A stable hash covering:
- Rounded query vector ψ
- Gating vector B_diag
- Scalars λ_G, λ_C, λ_Q, λ_P
- Chain presence & length
- k-neighbor parameters (k, deterministic flag)
- Adjacency fingerprint (subset of nonzero indices)

Changing any of the above invalidates the cache.

## Verification

Use:

```python
from oscillink import verify_receipt
assert verify_receipt(receipt, secret)
```

Or convenience:

```python
lat.verify_current_receipt(secret)
```

Returns False if:
- Signature block missing / algorithm mismatch
- Payload mutated
- Secret incorrect

## Convergence Metadata

The stationary CG solve (`solve_Ustar`) records residual, iteration count, and a convergence boolean (residual <= tolerance). Included mainly for operational monitoring and regression detection.

## Versioning

Additive changes to the schema will be documented here. Removals / renames will bump the minor version. Signature payload extensions (additional fields) will preserve verification (existing fields unchanged).

## Future Extensions (Possible)
- Deterministic seed capture for full reproducibility lineage
- Per-term normalized contributions
- Structured performance counters (adjacency build millis, etc.)

---
Apache-2.0
