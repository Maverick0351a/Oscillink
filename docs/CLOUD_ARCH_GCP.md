This document has moved.

New location: [docs/cloud/CLOUD_ARCH_GCP.md](./cloud/CLOUD_ARCH_GCP.md)
# Cloud Architecture (GCP Draft)

Status: Draft (Should) – informs Phase 2 hardening & commercialization.

## Goals
- Keep SDK stateless in cloud path (no anchor persistence)
- Provide metered settlement + optional advanced diffusion gating
- Support subscription tiers (Free, Pro, Enterprise) with Stripe
- Minimize cold start & simplify multi-region deployment

## High-Level Components

| Component | Service | Purpose |
|-----------|---------|---------|
| API Service | Cloud Run (container) | FastAPI app: /v1/settle, /v1/receipt, etc. |
| Metrics | Cloud Run + /metrics scrape (Managed Prom / Ops) | Latency & usage metrics ingestion |
| Auth / Quota Store | Firestore (Native mode) | API keys, per-key quota overrides, subscription metadata |
| Usage Ledger | Firestore collection (batched writes) | Durable usage records (nodes, node_dim_units) |
| Billing & Entitlements | Stripe + Webhooks (Cloud Run) | Plan management, subscription updates -> Firestore sync |
| Secret Management | Secret Manager | HMAC receipt secret, Stripe webhook secret, Stripe API key |
| Async Jobs | In-memory (per instance) -> Future: Cloud Tasks | Background settles for large N |
| Advanced Gating | In-process diffusion precompute | Optionally Pro+ tier gating weights |

## Request Flow
1. Client sends POST /v1/settle with API key header.
2. API instance loads runtime config (cached N seconds) and looks up key in Firestore cache (in-memory LRU) or direct fetch.
3. Subscription tier from key doc decides feature flags (e.g., `allow_diffusion_gates`).
4. Quota check: sum recent usage windows (see Firestore model) or maintain rolling counter doc with transactional increments.
5. Perform lattice settle entirely in memory (embeddings discarded post response).
6. Emit usage record append (batched) + Prometheus counters.
7. Return receipt/bundle + meta (usage, quota, gating stats if present).

## Firestore Caching Strategy
- Hot path must avoid round trips per request for unchanged key metadata.
- Maintain in-process LRU keyed by api_key -> {tier, quota_overrides, updated_at, etag}.
- Background refresh goroutine (Python task) invalidates entries after TTL.
- Fallback: On cache miss fetch doc; if missing -> 401.

## Quota Algorithm (Rolling Window Simplification)
Option A (Simpler, eventually consistent): track cumulative usage + reset timestamp per key doc. If `now >= reset`, reset counters; else enforce.
Option B (More accurate): store usage shards (subcollection `usage_windows`) per hour; sum last active window + current partial. Start with Option A.

## Stripe Integration (Overview)
- Product: "Oscillink Cloud"
- Prices: `cloud_free` ($0), `cloud_pro_monthly`, `cloud_pro_annual`, `cloud_enterprise` (custom)
- Webhook events consumed: `customer.subscription.created|updated|deleted`, `invoice.payment_succeeded`, `customer.subscription.trial_will_end`.
- Webhook handler maps Stripe `price.id` -> internal tier string.
- Firestore key doc updated transactionally: `{ tier: 'pro', entitlements: {...}, quota_limit: <override?>, updated_at }`.

## Deployment Topology
- Single container image (FastAPI + metrics) -> Cloud Run (min instances >=1 to avoid cold latency for Pro).
- Regional deployment (us-central1) + optional second region for failover; DNS weighted.
- Cloud Build trigger on main branch tag -> image build + deploy.

## Environment / Secrets
| Variable | Source | Notes |
|----------|--------|-------|
| OSCILLINK_MAX_NODES | Env | Hard safety cap (Free may lower via code) |
| OSCILLINK_MAX_DIM | Env | Safety cap |
| RECEIPT_HMAC_SECRET | Secret Manager | For receipt signing |
| STRIPE_API_KEY | Secret Manager | Server-side API operations |
| STRIPE_WEBHOOK_SECRET | Secret Manager | Signature verification |
| FIRESTORE_PROJECT | Env | Explicit when multi-project separation |
| ENABLE_DIFFUSION_GATES | Env (bool) | Global kill-switch (default on) |

## Feature Flags / Tier Matrix (Initial)
| Feature | Free | Pro | Enterprise |
|---------|------|-----|------------|
| Base settle & receipt | Yes | Yes | Yes |
| Async jobs | Yes (smaller N) | Yes | Yes (priority) |
| Diffusion gating | No | Yes | Yes |
| Higher N,D limits | Baseline | +50% | Custom |
| Signed usage receipts | No | Optional | Yes |
| Priority queue / faster SLA | No | No | Yes |

## Scaling Considerations
- CPU bound (NumPy) – scale via more Cloud Run instances; autoscale on concurrent request latency.
- Consider compute-optimized machine types for larger N (~5k) workloads.
- For very large async jobs, migrate job execution into Cloud Tasks workers to isolate from latency-sensitive front path.

## Observability
- Prometheus sidecar not needed; Cloud Run exposes /metrics; use Google Managed Prometheus scraping config.
- Structured JSON logs: each request log includes request_id, api_key (hashed), N, D, duration_ms, quota_remaining.
- Alerting: latency p95 > threshold, error rate surge, quota enforcement anomalies, diffusion gate build ms drift.

## Future Enhancements
- Replace in-memory async queue with Cloud Tasks.
- Distributed rate limiting using Redis (Memorystore) if multi-region concurrency demands.
- Signed usage export to GCS daily for audit & billing reconciliation.
- Canary deploy lane to validate performance regressions pre wide rollout.

## Open Questions
- Do we need per-chain prior persistence for long-running multi-step reasoning? (Out of scope Phase 2.)
- Add ephemeral object cache (e.g., 5 min) keyed by state_sig for idempotent replay? Possibly enterprise only.

---
Feedback welcome; adjust before locking Stripe product IDs & Firestore schema.
