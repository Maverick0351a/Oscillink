This document has moved.

New location: [docs/cloud/CLOUD_ADVANCED_DIFFUSION_ENDPOINT.md](./cloud/CLOUD_ADVANCED_DIFFUSION_ENDPOINT.md)
# Cloud Advanced Diffusion Settle Endpoint (Draft)

Status: Proposal (Phase 2)
Tier: Pro / Enterprise (feature-gated)

## Summary
Adds an optional endpoint that performs a screened diffusion preprocessing step to derive adaptive query gates before performing the standard settle + optional receipt/bundle generation.

Motivation:
- Automates spatial weighting around the query
- Enhances relevance without user-supplied gates
- Provides tunable locality (gamma) & injection (beta)
- Differentiates paid cloud tier

## Route
```
POST /v1/settle/diffusion
```
(Version prefix uses existing `_API_VERSION` resolution; final path: `/{api_version}/settle/diffusion`)

## Request Schema (JSON)
```jsonc
{
  "Y": [[...], ...],        // required (N x D float list matrix)
  "psi": [...],             // optional; if omitted behaves as uniform query (zeros handled as today)
  "params": {               // optional lattice parameter overrides
    "kneighbors": 8,
    "lamG": 1.0,
    "lamC": 0.5,
    "lamQ": 4.0,
    "lamP": 0.0,
    "deterministic_k": false,
    "neighbor_seed": null
  },
  "chain": [1,3,5],         // optional chain prior
  "options": {
    "include_receipt": true,
    "bundle_k": 8,
    "dt": 1.0,
    "max_iters": 12,
    "tol": 0.001
  },
  "diffusion": {            // NEW (strongly optional)
    "enable": true,
    "beta": 1.0,
    "gamma": 0.15,
    "kneighbors": 6,        // (optional override for diffusion graph only)
    "deterministic_k": false,
    "neighbor_seed": null
  }
}
```

Notes:
- If `diffusion.enable` is false or block omitted → identical to base `/settle` behavior.
- `diffusion.kneighbors` defaults to lattice `kneighbors` when omitted.
- Validation ensures `gamma > 0`.

## Response Schema
Same shape as existing `/settle` endpoint:
```jsonc
{
  "state_sig": "...",
  "receipt": { ... },
  "bundle": [ ... ],
  "timings_ms": { "total_settle_ms": 7.4, "diffusion_gate_ms": 3.1 },
  "meta": {
    "N": 800,
    "D": 128,
    "kneighbors_requested": 8,
    "kneighbors_effective": 8,
    "diffusion": {              // NEW nested meta
       "beta": 1.0,
       "gamma": 0.15,
       "gate_min": 0.0,
       "gate_max": 1.0,
       "gate_mean": 0.37,
       "graph_k": 6
    },
    "lam": {"G":1.0,"C":0.5,"Q":4.0,"P":0.0},
    "request_id": "...",
    "usage": {"nodes":800, "node_dim_units":102400},
    "quota": { ... }
  }
}
```
Fields added:
- `timings_ms.diffusion_gate_ms`: milliseconds spent computing gates (absent if disabled).
- `meta.diffusion`: absent if disabled.

## Pseudocode (Server Side)
```python
if body.get('diffusion', {}).get('enable'):
    diff_cfg = body['diffusion']
    gates = compute_diffusion_gates(
        Y, psi,
        kneighbors=diff_cfg.get('kneighbors', params.kneighbors),
        beta=diff_cfg.get('beta', 1.0),
        gamma=diff_cfg.get('gamma', 0.15),
        deterministic_k=diff_cfg.get('deterministic_k', False),
        neighbor_seed=diff_cfg.get('neighbor_seed'),
    )
    diffusion_meta = {
       'beta': diff_cfg.get('beta', 1.0),
       'gamma': diff_cfg.get('gamma', 0.15),
       'gate_min': float(gates.min()),
       'gate_max': float(gates.max()),
       'gate_mean': float(gates.mean()),
       'graph_k': int(diff_cfg.get('kneighbors', params.kneighbors)),
    }
else:
    gates = None
    diffusion_meta = None

lat = OscillinkLattice(Y, ...)
lat.set_query(psi, gates=gates if gates is not None else None)
...
```

## Tier / Auth Considerations
- Require API key (already enforced).
- Add middleware/flag: if key not in PRO tier list and `diffusion.enable` -> 402/403 with explanatory message.
- Environment variable proposal: `OSCILLINK_PRO_KEYS="k1,k2"` for feature gating (similar dynamic reload pattern).

## Metrics & Observability
Add Prometheus gauge & histogram (optional):
- `oscillink_diffusion_gate_ms` (histogram)
- `oscillink_diffusion_requests_total{status="ok|error"}`
Potential counter: `oscillink_diffusion_enabled_total` to monitor adoption.

## Quota Interaction
- Gate computation uses same `N*D` units as a regular settle (no surcharge) OR introduce multiplier (e.g., 1.1x) for pricing differentiation—TBD.
- Optional: record `diffusion:true` flag in usage log JSON lines.

## Failure Modes & Handling
| Failure | Handling | Response |
|---------|----------|----------|
| gamma <= 0 | 400 validation | `{detail:"gamma must be > 0"}` |
| solve singular | fallback gates=uniform + meta.diffusion.note="fallback_uniform" | 200 |
| memory pressure (large N) | recommend raising 413 with guidance | 413 |

## Backwards Compatibility
- Additive only; no changes to existing endpoints.
- SDK unaffected (feature already available locally via `compute_diffusion_gates`).

## Open Questions
1. Should diffusion graph reuse lattice adjacency for speed? (Flag: `reuse_lattice_graph=true`)
2. Apply row sum capping separately or share param? (Current: separate; simpler mental model.)
3. Billable unit multiplier? (Decide after measuring % overhead in typical workloads.)

## Next Steps
1. Implement tier gating helper (`get_pro_keys()` similar to existing API key loader).
2. Add endpoint with feature flag.
3. Add tests: basic success, disabled tier rejection, meta presence, timing field.
4. Add metrics & usage log `diffusion` flag.
5. Document in README cloud section.

---
This draft is intentionally lean to minimize integration friction while clearly framing extensibility (reuse graph, billing multiplier, metrics). Modify as pricing & product decisions solidify.
