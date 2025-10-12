This document has moved.

New location: [docs/reference/API.md](./reference/API.md)

Please update bookmarks to the canonical path above.
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

### Cloud-specific headers and flags

When using the cloud API server, responses may include these headers when enabled via env flags:

- Adaptive profiles: `X-Profile-Id` (also present as `meta.profile_id`), gated by `OSCILLINK_ADAPTIVE_PROFILES` and `OSCILLINK_ADAPTIVE_LEARN`.
- Bundle caching (/bundle): `X-Cache: HIT|MISS`; on HIT also `X-Cache-Hits` and `X-Cache-Age`. Gated by `OSCILLINK_CACHE_ENABLE`, with `OSCILLINK_CACHE_TTL` and `OSCILLINK_CACHE_CAP`.

- Endpoint rate limits (per-endpoint, best-effort): when enabled per endpoint, responses include `X-EPRL-Limit`, `X-EPRL-Remaining`, and `X-EPRL-Reset` while 429 responses include `Retry-After`.
	- CLI endpoints:
		- `/billing/cli/start`: gated by `OSCILLINK_EPRL_CLI_START_LIMIT` and `OSCILLINK_EPRL_CLI_START_WINDOW` (seconds)
		- `/billing/cli/poll/{code}`: gated by `OSCILLINK_EPRL_CLI_POLL_LIMIT` and `OSCILLINK_EPRL_CLI_POLL_WINDOW` (seconds)
	- Per-IP global limiter headers: `X-IPLimit-*` when `OSCILLINK_IP_RATE_LIMIT>0` and window via `OSCILLINK_IP_RATE_WINDOW`.
	- Global rate limiter headers: `X-RateLimit-*` when enabled via runtime config (`get_rate_limit`).

See README section "Cloud feature flags and headers (beta)" for details.
