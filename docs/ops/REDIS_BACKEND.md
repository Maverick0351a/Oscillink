# Optional Redis Backend

This service can optionally use Redis for distributed state to support horizontal scaling:

- Per-IP rate limiting counters
- Global rate limit counter
- Stripe webhook idempotency (event IDs with TTL)

## Enabling

Set the following environment variables:

- `OSCILLINK_STATE_BACKEND=redis`
- `OSCILLINK_REDIS_URL=redis://localhost:6379/0` (or use `REDIS_URL`)

Optional:

- `OSCILLINK_WEBHOOK_TTL` (seconds, default 604800 = 7 days)

When disabled (`OSCILLINK_STATE_BACKEND` not equal to `redis` or Redis is unreachable), the service falls back to in-memory behavior with identical semantics as before.

## Notes

- The test suite does not require Redis; defaults remain in-memory.
- For production, point `OSCILLINK_REDIS_URL` at a managed Redis and enable the backend. Ensure network and auth are configured.
