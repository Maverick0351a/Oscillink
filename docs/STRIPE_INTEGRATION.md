# Stripe Integration (Draft)

Status: Draft (Should). Align product & price IDs before coding.

## Objectives
- Map paid subscription state to feature entitlements (tier -> features)
- Reflect changes in near real-time (webhook driven)
- Maintain idempotent, auditable updates (webhook event store)

## Products & Prices (Proposed)
| Product | Description | Prices |
|---------|-------------|--------|
| `oscillink_cloud` | Base cloud access | `cloud_free`, `cloud_pro_monthly`, `cloud_pro_annual`, `cloud_enterprise` |

Price metadata should include: `tier=free|pro|enterprise`.

## Customer -> API Key Association
Option A: Store `api_key_hash` in Stripe Customer metadata (preferred).  
Option B: Maintain mapping collection `stripe_customers/{customer_id}` referencing key hash(es).

Provisioning flow: when a Pro subscription is created, if customer lacks key, generate one and email customer using transactional template (out of scope; could store placeholder until manual approval for enterprise).

## Webhook Events Consumed
| Event | Action |
|-------|--------|
| `customer.subscription.created` | Set tier & entitlements |
| `customer.subscription.updated` | Adjust tier (upgrade/downgrade), refresh period end |
| `customer.subscription.deleted` | Revoke or downgrade to `free` |
| `invoice.payment_succeeded` | (Optional) usage-based addons future |
| `customer.subscription.trial_will_end` | Notify (email queue) |

Ignore unrelated events (respond 200 early).

## Webhook Handling Steps
1. Verify signature header using `STRIPE_WEBHOOK_SECRET`.
2. Deserialize event; if already stored & processed -> 200.
3. If event type not in handled set -> store raw, mark processed noop -> 200.
4. Extract subscription object; get `customer`, `items[0].price.id`.
5. Resolve tier from price metadata (fallback map in code).
6. Lookup associated `api_keys/{hash}` via customer metadata.
7. Firestore transaction: update key doc (tier, features, stripe block fields) & mark webhook event processed.
8. Return 200.

Retries: Stripe will retry on non-2xx; ensure idempotency by early exit if `processed` true.

## Entitlements Resolution
Runtime resolution order:
1. Fetch key doc (cache)
2. Merge with static tier map (code) – static defines default features
3. Overlay key doc `features` overrides (per customer customizations)
4. Provide final feature set to request context

## Downgrade / Revocation
- On subscription deletion/expiration: set tier to `free` OR `revoked` if payment delinquent (choice: treat delinquency as revoke or degrade; start with degrade -> free).
- Remove diffusion gating entitlement if not free tier.

## Security
- Validate event `livemode` flag matches environment.
- Webhook endpoint secret rotated via Secret Manager.
- Consider restricting source IPs (not fully reliable) -> rely on signature.
- Sanitize & store only necessary subset of subscription object in key doc.

## Manual Overrides
Allow support/admin to patch `api_keys/{hash}` doc (e.g., temporary quota extension). Add `overrides: { note: str, expires_at: ts }` block recorded in audit log.

## Usage-Based Extensions (Future)
If introducing usage-based billing add-on:
- Track billable units per period in Firestore (separate doc or aggregated field)
- Emit invoice line items through Stripe Billing (requires reporting job)
- Webhook `invoice.created` to reconcile pre-invoice usage snapshot

## Testing Strategy
- Local: use Stripe CLI `stripe listen --forward-to localhost:8000/stripe/webhook`
- Mock: fixture events JSON in tests exercising handler logic (signature bypass in test mode)
- Integration: ephemeral test mode keys & subscriptions creation via automated script

## Failure Scenarios
| Failure | Mitigation |
|---------|------------|
| Webhook outage | Stripe retries; backlog stored once endpoint restored |
| Firestore transient error | Retry transaction w/ exponential backoff (small attempts) |
| Missing customer metadata (no key) | Generate placeholder key doc flagged `pending_provision` |

## Open Questions
- Enterprise contract IDs mapping -> additional price ids or separate product? (Lean: separate price ids.)
- Need email notifications for trial end? (Out of scope initially.)

---
Refine before implementation & coordinate with `FIRESTORE_USAGE_MODEL.md`.
