# License service ops

This folder contains a minimal example service for JWKS and token renewal. Use it as a reference for your production license system.

## Responsibilities
- Host public JWKS at `/.well-known/jwks.json` with proper `kid` rotation and caching headers (ETag).
- Issue/renew Ed25519-signed license JWTs with claims: `iss`, `aud`, `sub`, `tier`, `limits`, `features`, `nbf`, `exp`, and `kid`.
- Receive aggregated usage batches at `/v1/usage/report` (optional), validate HMAC if configured, and persist counters.

## Rotation and offline grace
- Rotate keys by publishing a new key in JWKS with a new `kid`, then issue new tokens referencing it. Keep the previous key until all old tokens expire.
- The licensed container caches JWKS with TTL (default 3600s) and supports offline grace (default 86400s). Set `OSCILLINK_JWKS_TTL` and `OSCILLINK_JWKS_OFFLINE_GRACE` to tune.

## Renew flow
1) Client (container) approaches expiration.
2) Operator (or automation) fetches a fresh token from `/v1/license/renew` for the same `sub`.
3) Container entrypoint writes the new token to the mounted license file path and restarts gracefully.

## Claims mapping → env
The container maps the token into envs the app enforces:
- `limits.max_nodes` → `OSCILLINK_MAX_NODES`
- `limits.max_dim` → `OSCILLINK_MAX_DIM`
- `limits.qps` and `limits.qps_window` → `OSCILLINK_RATE_LIMIT` and `OSCILLINK_RATE_WINDOW`
- `limits.quota_units` and `limits.quota_window` → `OSCILLINK_KEY_NODE_UNITS_LIMIT`/`OSCILLINK_KEY_NODE_UNITS_WINDOW`
- `limits.monthly_cap|monthly_units` → `OSCILLINK_MONTHLY_CAP`
- `features.*` → `OSCILLINK_FEAT_*`
- `sub` → `OSCILLINK_API_KEYS` (single-key) and, with `tier`, produces `OSCILLINK_KEY_TIERS`

## Security tips
- Serve JWKS and renew endpoints over HTTPS only.
- Use short-lived tokens with leeway to account for clock skew.
- Consider IP allowlists / mTLS for usage report endpoints.
- Monitor failed signature verifications and token claim mismatches.
