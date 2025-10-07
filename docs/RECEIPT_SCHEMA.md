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
