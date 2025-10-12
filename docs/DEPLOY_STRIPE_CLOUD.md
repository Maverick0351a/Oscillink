This document has moved.

New location: [docs/cloud/DEPLOY_STRIPE_CLOUD.md](./cloud/DEPLOY_STRIPE_CLOUD.md)
# Deploying Stripe-only Onboarding (Production)

This guide covers deploying the Stripe Payment Links flow with the built-in success page at `/billing/success` so users can buy a plan and immediately get an API key—no custom website needed.

## What you’re deploying

- Hosted checkout via Stripe Payment Links
- After completion: redirect to your API’s success page
  - `https://api.odinprotocol.com/billing/success?session_id={CHECKOUT_SESSION_ID}`
- The success page provisions (or reuses) an API key and displays it to the user
- Webhooks keep subscription → entitlements in sync

## Required environment variables

Set these on your production server/process (live keys):

- `STRIPE_SECRET_KEY` — Your live secret key (e.g., `sk_live_...`)
- `STRIPE_WEBHOOK_SECRET` — Webhook endpoint signing secret (`whsec_...`)
- `OSCILLINK_STRIPE_PRICE_MAP` — Map Stripe price IDs → tiers (include `beta`; Pro is hidden during early beta unless explicitly enabled in scripts)
  - ODIN production mapping:
    - `price_1SGaX7LcPYf7t6osDDmhtyUZ:free`
    - `price_1SGaX7LcPYf7t6osBWkbJgXG:pro`
    - `price_1SGaX8LcPYf7t6os42l7qFbT:enterprise`
- Optional (recommended): `OSCILLINK_ALLOW_UNVERIFIED_STRIPE=0`

See also: docs/billing/STRIPE_INTEGRATION.md → “Production mapping (ODIN)”.

## Stripe configuration

- Prices (live):
  - Pro (hidden during early beta): lookup_key `price_cloud_pro_monthly` → id `price_1SGaX7LcPYf7t6osBWkbJgXG`
  - Enterprise: lookup_key `price_cloud_enterprise` → id `price_1SGaX8LcPYf7t6os42l7qFbT`
  - Prices should include `metadata.tier` for clarity: `pro`, `enterprise`
- Payment Link:
  - After completion action: Redirect to your website
  - Success URL: `https://api.odinprotocol.com/billing/success?session_id={CHECKOUT_SESSION_ID}`
  - Note: A live Payment Link is already published in README under “Get API Key →”.

You can also generate a link via the helper script:

- `python scripts/stripe_create_payment_link.py --tier pro --allow-pro --base-url https://api.odinprotocol.com`  # Pro gated during beta

## Webhook

Create a webhook endpoint in the Stripe Dashboard pointing to your API:

- Endpoint URL: `https://api.odinprotocol.com/stripe/webhook`
- Secret: copy the generated `whsec_...` into `STRIPE_WEBHOOK_SECRET`
- Events (at minimum):
  - `customer.subscription.created`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
  - `invoice.payment_succeeded`
  - Optional: `checkout.session.completed` (if useful for auditing)

The server verifies the signature when `STRIPE_WEBHOOK_SECRET` is set. Unverified mutations are ignored unless `OSCILLINK_ALLOW_UNVERIFIED_STRIPE=1` (not recommended in prod).

## Sanity checks

1. Open the Payment Link and complete a purchase.
2. You should be redirected to `/billing/success` and see your API key.
3. In Stripe, confirm the subscription item’s `price.id` matches your mapped tier.
4. Call the API with your key and confirm it works (e.g., `/v1/receipt`).

## Troubleshooting

- Success page shows an error:
  - Ensure `STRIPE_SECRET_KEY` is set and valid (live key)
  - Ensure the success URL includes `session_id={CHECKOUT_SESSION_ID}`
- Wrong or missing tier:
  - Verify `OSCILLINK_STRIPE_PRICE_MAP` includes the exact `price.id` from the live subscription
  - Check the subscription’s first item price id is the one intended
- Webhook signature errors:
  - Confirm `STRIPE_WEBHOOK_SECRET`
  - Make sure the endpoint URL matches exactly and you’re using the live secret

## Optional: local testing

You can simulate webhooks locally with the Stripe CLI:

- `stripe listen --forward-to localhost:8000/stripe/webhook`

For Windows PowerShell temporary environment variables during local runs:

```powershell
$Env:STRIPE_SECRET_KEY = 'sk_test_xxx'\n$Env:STRIPE_WEBHOOK_SECRET = 'whsec_xxx'
$Env:OSCILLINK_STRIPE_PRICE_MAP = 'price_test_free:free;price_test_beta:beta;price_test_enterprise:enterprise'  # Pro optional
```

---

For architecture and integration details, see `docs/billing/STRIPE_INTEGRATION.md`.
