This document has moved.

New location: [docs/billing/STRIPE_INTEGRATION.md](./billing/STRIPE_INTEGRATION.md)
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

Price metadata should include: `tier=free|beta|enterprise` (Pro exists but is hidden during early beta).

## Customer -> API Key Association
Option A: Store `api_key_hash` in Stripe Customer metadata (preferred).
Option B: Maintain mapping collection `stripe_customers/{customer_id}` referencing key hash(es).

Provisioning flow: during early beta, keys are provisioned manually from the Stripe Dashboard. When Pro is re-enabled, you can auto-provision on subscription creation by looking up the customer and generating a key.

## Webhook Events Consumed
| Event | Action |
|-------|--------|
| `customer.subscription.created` | Set tier & entitlements |
| `customer.subscription.updated` | Adjust tier (upgrade/downgrade), refresh period end |
| `customer.subscription.deleted` | Revoke or downgrade to `free` |
| `invoice.payment_succeeded` | (Optional) usage-based addons future |
| `customer.subscription.trial_will_end` | Notify (email queue) |

Ignore unrelated events (respond 200 early).

## Checkout Success Flow (Hosted by Stripe)

When using Stripe Checkout or Payment Links (no custom website), configure the success URL to point to the API:

- Success URL: `https://<your-domain>/billing/success?session_id={CHECKOUT_SESSION_ID}`

On return, the API will:

1. Retrieve the Checkout Session and associated Subscription via the Stripe API.
2. Resolve the tier from the subscription's price using the price→tier map.
3. Generate an API key if one is not already attached in `subscription.metadata.api_key`.
4. Store/update the key in the configured keystore (in-memory or Firestore) with appropriate status/entitlements.
5. Render a minimal HTML page that reveals the API key and quickstart instructions.

Environment prerequisites:

- `STRIPE_SECRET_KEY` (or `STRIPE_API_KEY`) must be set on the server.
- Optionally configure `OSCILLINK_STRIPE_PRICE_MAP` to map price IDs to tiers. Include `beta` for the beta plan; Pro is optional and disabled in helper scripts unless explicitly allowed.

Notes:

- Enterprise tiers may be marked `pending` (manual activation) depending on catalog config.
- Webhooks still update entitlements idempotently; the success page is user-facing convenience.

## Production mapping (ODIN)

For the ODIN Protocol production environment, set the live price→tier mapping and confirm the configured redirect base:

- Environment variable example (Windows PowerShell syntax shown for value formatting only):
	- `OSCILLINK_STRIPE_PRICE_MAP="price_beta_123:beta;price_live_free:free;price_live_enterprise:enterprise"`  # add pro mapping when enabling Pro
- Stripe price lookup_key values have been set for traceability:
	- `price_cloud_beta_monthly` → Beta
	- `price_cloud_pro_monthly` → Pro (hidden during beta)
	- `price_cloud_enterprise` → Enterprise
- Hosted Checkout success redirect base (Payment Links):
	- `https://api.odinprotocol.com`

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
