# Firestore Usage & Key Model (Draft)

Status: Draft (Should). Iteration expected once Stripe product IDs finalized.

## Collections Overview
| Collection | Purpose |
|------------|---------|
| `api_keys` | API key metadata, tier, quotas, entitlements, status |
| `usage_ledger` | Append-only per-request usage records (optional if using signed JSONL first) |
| `quota_rollups` | (Optional) Aggregated usage per key per window for fast quota checks |
| `webhook_events` | Raw Stripe webhook payloads (idempotency + audit) |

## `api_keys` Document Schema
Document ID: the raw API key (or a hashed form; if hashed, store `prefix` for debugging display).

```
api_keys/{api_key}
{
  tier: 'free' | 'pro' | 'enterprise',
  status: 'active' | 'revoked' | 'suspended',
  created_at: <timestamp>,
  updated_at: <timestamp>,
  last_seen_at: <timestamp>,
  quota_limit_units: <int|null>,   // override (N*D) per window; null = default tier limit
  quota_window_seconds: <int|null>,
  features: {                      // entitlements toggles
    diffusion_gates: <bool>,
    async_jobs: <bool>,
    signed_usage: <bool>,
    priority_queue: <bool>
  },
  stripe: {                        // linkage to billing
    customer_id: <string|null>,
    subscription_id: <string|null>,
    price_id: <string|null>,
    current_period_end: <timestamp|null>
  },
  metadata: {                      // arbitrary internal notes
    note: <string|null>
  }
}
```

### Key Generation
- Random 32+ bytes base62 string; optionally prefix by tier hint (e.g., `ok_free_...`).
- Store SHA256 hash of key as document ID; store first 6 chars of plaintext as `prefix` for support correlation; never log full key.

## Quota Enforcement Strategy
Baseline (Option A):
- Keep running counters in memory per instance: `window_start`, `used_units`.
- At request: if `now - window_start >= window_length` -> reset counters.
- Compare `(used_units + request_units)` to `quota_limit_units`.
- After success, increment counters & (optionally) write periodic rollup doc (not every request to reduce write amplification).

Consistency trade-off: multi-instance race may allow slight overage; acceptable for early stage. Later: transactional increments using a single Firestore doc with a small retry strategy.

## `usage_ledger` Document Schema (Optional)
If durability in Firestore (instead of / in addition to JSONL) is desired:
```
usage_ledger/{auto_id}
{
  ts: <timestamp>,
  api_key_hash: <string>,
  request_id: <string>,
  event: 'settle' | 'receipt' | 'bundle' | 'chain_receipt' | 'job_settle',
  N: <int>,
  D: <int>,
  units: <int>,
  duration_ms: <float>,
  quota_after: { limit: <int>, remaining: <int>, reset: <timestamp> },
  tier: <string>,
  diffusion_gates: <bool>,
  signed: <bool>
}
```
TTL policy: optionally enable automatic purge after 30-60 days (Firestore TTL) if exporting to BigQuery / GCS first.

## `quota_rollups` (Optional)
Design if more accurate windows needed:
```
quota_rollups/{api_key_hash}_{yyyymmddHH}
{
  window_start: <timestamp>,          // hour bucket start
  api_key_hash: <string>,
  units: <int>,                       // cumulative for hour
  last_update: <timestamp>
}
```
Enforcement reads current hour + previous hour (if crossing boundary) to compute rolling 3600s usage. Start with baseline in-memory algorithm before introducing this complexity.

## Webhook Events
`webhook_events/{stripe_event_id}` stores:
```
{
  received_at: <timestamp>,
  type: <string>,
  raw: <object>,
  processed: <bool>,
  processed_at: <timestamp|null>,
  api_key_hash: <string|null>,
  action: 'tier_update' | 'noop' | 'revocation' | 'provision'
}
```
Idempotency: operation logic checks existing doc; if `processed` true skip re-application.

## Tier -> Feature Resolution
Table (canonical) stored in code (versioned) and optionally a `config/features` doc to allow dynamic overrides.

Example internal map:
```
{
  free:      { diffusion_gates: false, async_jobs: true,  signed_usage: false, priority_queue: false },
  pro:       { diffusion_gates: true,  async_jobs: true,  signed_usage: true,  priority_queue: false },
  enterprise:{ diffusion_gates: true,  async_jobs: true,  signed_usage: true,  priority_queue: true }
}
```

## Stripe Sync Flow
1. Webhook arrives -> verify signature via STRIPE_WEBHOOK_SECRET.
2. If event is subscription lifecycle, extract subscription + price id.
3. Map price id -> tier.
4. Lookup existing api_key (by customer metadata or mapping doc) – if none, optionally create pending key record.
5. Update `api_keys/{hash}` doc with tier, features, stripe block, updated_at.
6. Mark webhook event processed.

## Security Considerations
- All lookups use hashed key ID; plaintext never stored.
- Limit Firestore IAM to Cloud Run service account for collections.
- Webhook endpoint isolated path `/stripe/webhook` with strict method + IP logging.
- Rate limit key creation API (if exposed) and store creation IP hash for abuse analysis.

## Open Questions
- Need separate collection for `api_key_rotations`? (Maybe later.)
- Should we sign usage ledger entries at write time? (Enterprise feature: yes.)

---
Feedback encouraged before implementation. Adjust in tandem with `CLOUD_ARCH_GCP.md`.
