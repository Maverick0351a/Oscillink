This document has moved.

New location: [docs/cloud/CLOUD_APP_REFACTOR_PLAN.md](./cloud/CLOUD_APP_REFACTOR_PLAN.md)
# Cloud app modularization plan

## Goals
- Reduce `cloud/app/main.py` size/complexity while preserving behavior and API surface.
- Improve testability and maintainability (routers/services boundaries).
- No breaking changes: keep `cloud.app.main:app` import path and existing routes/headers.

## Current state
- `cloud/app/main.py` ~2400 lines; mixes routing, middleware, quotas, caching, usage logging, Stripe flows, admin.
- Strong test coverage across routes, quotas, webhook idempotency, and admin endpoints.

## Target structure
- cloud/app/
  - main.py (app bootstrap, middleware wiring, include_routers)
  - middleware.py (body size guard, request id, security headers, IP/global rate limiters)
  - services/
    - quotas.py (quota/monthly cap logic)
    - cache.py (bundle TTL LRU cache)
    - usage_log.py (append usage JSONL with HMAC)
    - events.py (webhook persistence helpers: redis/firestore)
    - billing.py (tier map, price map, Stripe helpers)
    - keystore.py (existing)
  - routers/
    - core.py (/v1/settle, /v1/receipt, /v1/bundle, /v1/chain/receipt)
    - jobs.py (/v1/jobs/*)
    - admin.py (/admin/*)
    - billing_webhook.py (/stripe/webhook)

## Staged migration (low risk)
1. Extract pure utilities (usage logging, bundle cache) to `services/usage_log.py` and `services/cache.py`. Wire main to them. (DONE)
2. Extract webhook persistence to `services/events.py` (redis/firestore). Keep in-memory wrapper in main. (NEXT)
3. Extract Stripe webhook to `routers/billing_webhook.py` using `APIRouter()` (no behavior change). Include from main.
4. Extract admin endpoints to `routers/admin.py`.
5. Extract job endpoints to `routers/jobs.py`.
6. Extract core compute endpoints to `routers/core.py`.
7. Move middleware into `middleware.py` and register from main.

At each stage:
- Re-run tests and ensure parity (headers, response models, metrics, idempotency).
- Keep module-level singletons (e.g., webhook events memory) in a shared or main module as needed to avoid circular imports.

## Compatibility & testing
- Keep `cloud.app.main:app` as the entrypoint.
- Preserve all route paths and response schemas.
- Validate: pytest (full suite), ruff, mypy.

## Notes
- The Starlette/python-multipart PendingDeprecation is upstream; leave code unchanged.
- Firestore/Redis remain optional, wrapped in best-effort try/except.
