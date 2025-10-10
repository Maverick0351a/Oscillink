# Deploying to Cloud Run (env example)

Use `.env.cloudrun.example` as a reference for required and optional env vars.

Minimum for automated key provisioning via webhook:
- STRIPE_SECRET_KEY
- STRIPE_WEBHOOK_SECRET
- OSCILLINK_STRIPE_PRICE_MAP (include your live Beta price id: `price_...:beta`)
- OSCILLINK_KEYSTORE_BACKEND=firestore (recommended in prod)
- OSCILLINK_CUSTOMERS_COLLECTION (for portal/cancel)
- OSCILLINK_WEBHOOK_EVENTS_COLLECTION (for idempotency)

Then configure Stripe to send events to `POST https://<your-cloudrun-url>/stripe/webhook`.
