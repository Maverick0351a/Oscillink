# Oscillink Coherent — Scalable Working Memory & Hallucination Suppression

<p align="center">
	<img src="assets/oscillink_hero.svg" alt="Oscillink" width="720" />
	<br/>
	<sub>Attach Oscillink as a post‑retrieval coherence layer to reduce hallucinations and make decisions explainable.</sub>
	<br/>
	<a href="https://pypi.org/project/oscillink/">PyPI</a> · <a href="docs/API.md">SDK API</a> · <a href="#use-the-cloud">Cloud</a> · <a href="https://buy.stripe.com/7sY9AUbcK1if2y6d2g2VG08">Get API Key (Beta)</a> · <a href="OscillinkWhitepaper.tex">Whitepaper</a> · <a href="LICENSE">Apache‑2.0</a>
	<br/>
	<a href="https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml/badge.svg"/></a>
	<a href="https://pypi.org/project/oscillink/"><img alt="PyPI" src="https://img.shields.io/pypi/v/oscillink.svg"/></a>
	<a href="https://pypi.org/project/oscillink/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/oscillink.svg"/></a>
	<a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/Maverick0351a/Oscillink.svg"/></a>
  
</p>

## Oscillink Coherent — Scalable Working Memory & Hallucination Suppression

- Attach to any generative model. Drop in after initial retrieval or candidate generation to produce an explainable, globally coherent working memory state.
- Replace brittle RAG heuristics. Move from ad‑hoc top‑k filters to a physics‑based lattice that minimizes energy and produces signed receipts (ΔH, null points, chain verdicts).
- Scale to lattice‑of‑lattices. The same SPD contract composes hierarchically (see `docs/SCALING.md`)—from a few hundred nodes to layered shard summaries with virtually no architectural rewrite.
- Controllable hallucination suppression. Gate low‑trust sources → observed 42.9% → 0.0% hallucination rate in controlled fact retrieval (Notebook 04) with F1 uplift of ~64%.

## What you get

- Turn disconnected chunks into an explainable working memory, with deterministic math (SPD system)
- Control hallucinations with diffusion gates; observed 42.9% → 0.0% hallucination rate in a controlled task (Notebook 04)
- Signed receipts: energy metrics (ΔH), per‑node components, null points for audits
- Fast: ~10 ms settle on laptop‑scale problems (no training required)

### How it works (brief math)

Oscillink refines embeddings by minimizing a convex energy over a mutual‑kNN lattice and solving one SPD linear system (deterministic, training‑free):

$$
H(U)=\lambda_G\|U-Y\|_F^2+\lambda_C\,\mathrm{tr}(U^\top L_{\mathrm{sym}}U)+\lambda_Q\,\mathrm{tr}((U-\mathbf{1}\psi^\top)^\top B\,(U-\mathbf{1}\psi^\top))+\lambda_P\,\mathrm{tr}(U^\top L_{\mathrm{path}}U)
$$

Stationary system with SPD guarantee (unique solution, fast CG):

$$
M\,U^\star=F,\quad M=\lambda_G I+\lambda_C L_{\mathrm{sym}}+\lambda_Q B+\lambda_P L_{\mathrm{path}},\quad F=\lambda_G Y+\lambda_Q B\,\mathbf{1}\psi^\top
$$

Receipts report the total energy drop $\Delta H_{\text{total}}=H(Y)-H(U^\star)\ge 0$ and surface null‑point outliers for audit.

## Install

```bash
pip install oscillink
```

## 60‑second SDK quickstart

```python
import numpy as np
from oscillink import OscillinkLattice

Y = np.random.randn(120, 128).astype(np.float32)
psi = (Y[:20].mean(0) / (np.linalg.norm(Y[:20].mean(0)) + 1e-12)).astype(np.float32)

lat = OscillinkLattice(Y, kneighbors=6)
lat.set_query(psi)
lat.settle()
print(lat.bundle(k=5))           # Top‑k coherent items
print(lat.receipt()['deltaH_total'])  # Energy drop for audit
```

Want more control? Compute diffusion gates and pass them to `set_query`:

```python
from oscillink import compute_diffusion_gates
gates = compute_diffusion_gates(Y, psi, kneighbors=6, beta=1.0, gamma=0.15)
lat.set_query(psi, gates=gates)
lat.settle()
```

---

## Run the server (operators)

Local (dev):

- Python 3.11; install dev extras and start the API:
	- Install: `pip install -e .[dev]`
	- Start: `uvicorn cloud.app.main:app --port 8000`
- For local development, disable HTTPS redirect: `OSCILLINK_FORCE_HTTPS=0`.
- Optional: set `STRIPE_SECRET_KEY` (and `STRIPE_WEBHOOK_SECRET` if testing webhooks locally via Stripe CLI).

Docker:

- Build with the production Dockerfile: `docker build -t oscillink-cloud -f cloud/Dockerfile .`
- Run: `docker run -p 8000:8080 -e PORT=8080 -e OSCILLINK_FORCE_HTTPS=0 oscillink-cloud`

Cloud Run (prod):

- Use `cloud/Dockerfile`. Our container respects `PORT` and runs Gunicorn+Uvicorn as a non‑root user with a HEALTHCHECK.
- Deploy with the environment variables in the checklist below (Stripe keys, price map, and optional Firestore collections).
- Grant the service account Firestore/Datastore User as noted.

## Use the Cloud

Call a hosted Oscillink with a simple HTTP POST. No infra required.

### Plans

- Free: 5M node·dim/month, community support
- Beta Access ($19/mo): 25M units/month (hard cap), beta phase — limited support, cancel anytime
- Enterprise: Unlimited, SLA, dedicated support (contact us)

### Beta-only Stripe setup (Quickstart)

For the early beta (no public domain required):

1) Subscribe: Use the hosted Stripe link for Beta Access ($19/mo): https://buy.stripe.com/7sY9AUbcK1if2y6d2g2VG08

2) Server env (operators):
- `STRIPE_SECRET_KEY` — required
- `OSCILLINK_STRIPE_PRICE_MAP="price_live_beta_id:beta"` — map your live Beta price ID to `beta`
- Optional for caps/portal: `OSCILLINK_MONTHLY_USAGE_COLLECTION`, `OSCILLINK_CUSTOMERS_COLLECTION`

Windows quick-setup (local dev):

- Run `scripts\setup_billing_local.ps1` to be prompted for your Stripe secret, webhook secret (optional), and Beta price ID mapping. It will set the environment for the current PowerShell session and print a tip to start the server.

3) Provisioning: Keys are provisioned manually during beta. Reply to your Stripe receipt email or email travisjohnson@oscillink.com with the receipt email you used; we’ll send your key within 24 hours.

4) Test: Call `POST /v1/settle` with `X-API-Key` and verify results and headers (see examples below).

### 1) Get an API key

- Pay via the hosted Stripe link (no domain required):
	- Beta Access ($19/mo): https://buy.stripe.com/7sY9AUbcK1if2y6d2g2VG08

- During early beta (no public domain yet):
	- We’ll provision your API key manually after payment. Reply to your Stripe receipt email or email travisjohnson@oscillink.com with the receipt email you used, and we’ll send your key within 24 hours.

Notes for operators:

- Server must have `STRIPE_SECRET_KEY` set. Optional `OSCILLINK_STRIPE_PRICE_MAP` sets price→tier mapping.
- See docs/STRIPE_INTEGRATION.md for full details.
- No redirect flow yet: using Stripe’s hosted confirmation page. Keys are provisioned manually from Stripe Dashboard until a public domain is live.
- To enforce the beta hard cap (25M units/month), configure the monthly cap for the `beta` tier in your runtime settings; exceeding the cap returns 429 with `X-Monthly-*` headers.

Cloud Run + Firestore checklist (early beta):

- Cloud Run service env:
	- `PORT` (Cloud Run injects 8080; our Dockerfile respects `PORT`)
	- `OSCILLINK_FORCE_HTTPS=1` (already default in Dockerfile)
	- `STRIPE_SECRET_KEY` and, if using webhooks, `STRIPE_WEBHOOK_SECRET`
	- `OSCILLINK_KEYSTORE_BACKEND=firestore` to enable Firestore keystore (optional; memory by default)
	- `OSCILLINK_CUSTOMERS_COLLECTION=oscillink_customers` to enable Billing Portal lookups (optional)
	- `OSCILLINK_MONTHLY_USAGE_COLLECTION=oscillink_monthly_usage` to persist monthly caps (optional)
	- `OSCILLINK_WEBHOOK_EVENTS_COLLECTION=oscillink_webhooks` to persist webhook idempotency (optional)
	- Set `OSCILLINK_STRIPE_PRICE_MAP` to include your live price ids → tiers (include `beta`).
- Firestore (in same GCP project):
	- Enable Firestore in Native mode.
	- Service Account used by Cloud Run must have roles: Datastore User (or Firestore User). Minimal perms for collections above.
	- No required indexes for the default code paths (point lookups by document id).
- Webhook endpoint (optional for beta): deploy public URL and configure Stripe to call `POST /stripe/webhook` with the secret; leave disabled while keys are provisioned manually.

### Automate API key provisioning after payment (optional)

You can automate key creation either via a success page redirect or purely by Stripe Webhooks (or both for redundancy):

- Success URL flow (requires public domain):
	- Configure the Payment Link or Checkout Session `success_url` to `https://yourdomain.com/billing/success?session_id={CHECKOUT_SESSION_ID}`.
	- Server verifies the session with Stripe using `STRIPE_SECRET_KEY`, generates an API key, saves `api_key → (customer_id, subscription_id)` in Firestore if `OSCILLINK_CUSTOMERS_COLLECTION` is set, and returns a confirmation page (one‑time display).
	- Idempotency: gate on `session_id` and/or persist a provisioning record (e.g., in `OSCILLINK_WEBHOOK_EVENTS_COLLECTION`).

- Webhook flow (works with or without success redirect):
	- Set `STRIPE_WEBHOOK_SECRET` and point Stripe to `POST /stripe/webhook`.
	- On `checkout.session.completed` (or `customer.subscription.created`), verify signature + timestamp freshness; reject stale events.
	- Ensure idempotency by recording processed `event.id` to `OSCILLINK_WEBHOOK_EVENTS_COLLECTION` (Firestore) before provisioning.
	- Generate an API key into your keystore (`OSCILLINK_KEYSTORE_BACKEND=firestore` recommended) and persist the customers mapping via `OSCILLINK_CUSTOMERS_COLLECTION`.
	- Optional: email the key using your transactional email provider or provide a “retrieve key” admin workflow.

Environment recap for automation:

- `STRIPE_SECRET_KEY` — required to verify sessions and manage subscriptions
- `STRIPE_WEBHOOK_SECRET` — required for secure webhook handling
- `OSCILLINK_CUSTOMERS_COLLECTION` — Firestore mapping: `api_key → {customer_id, subscription_id}`
- `OSCILLINK_WEBHOOK_EVENTS_COLLECTION` — Firestore store for webhook idempotency
- `OSCILLINK_KEYSTORE_BACKEND=firestore` — enable Firestore keystore (optional; memory by default)

### 2) Call the API

Headers:

- `X-API-Key: <your_key>`
- `Content-Type: application/json`

Endpoints (current versioned prefix is captured from settings; default `v1`):

- `POST /v1/settle` — compute settle + optional receipt and bundle
- `POST /v1/receipt` — compute receipt only
- `POST /v1/bundle` — compute bundle only
- `POST /v1/chain/receipt` — chain verdict for a path prior

Minimal curl example:

```bash
curl -X POST https://api.yourdomain.com/v1/settle \
	-H "X-API-Key: $YOUR_API_KEY" \
	-H "Content-Type: application/json" \
	-d '{
		"Y": [[0.1,0.2],[0.3,0.4],[0.5,0.6]],
		"psi": [0.1,0.2],
		"params": {"kneighbors": 2},
		"options": {"bundle_k": 2, "include_receipt": true}
	}'
```

Python client snippet:

```python
import os, httpx

API_BASE = os.environ.get("OSCILLINK_API_BASE", "https://api.yourdomain.com")
API_KEY = os.environ["OSCILLINK_API_KEY"]

payload = {
	"Y": [[0.1,0.2],[0.3,0.4],[0.5,0.6]],
	"psi": [0.1,0.2],
	"params": {"kneighbors": 2},
	"options": {"bundle_k": 2, "include_receipt": True},
}

r = httpx.post(f"{API_BASE}/v1/settle", json=payload, headers={"X-API-Key": API_KEY})
r.raise_for_status()
print(r.json())
```

Response shape (abridged):

- `state_sig: str` — checksum of lattice state for audit
- `bundle: list[dict]` — top‑k results with scores
- `receipt: dict` — energy breakdown (if requested)
- `timings_ms: dict` — perf timings
- `meta: dict` — quota/rate limit headers are returned as `X-Quota-*` (per‑key quotas), plus `X-RateLimit-*` (global) and `X-IPLimit-*` (per‑IP); monthly caps via `X-Monthly-*` when enabled

### Quotas, limits, and headers

- Global and per‑IP rate limits are enforced; exceeding returns 429 with headers indicating remaining and reset
- Per‑key quotas (units consumed = N×D) and monthly caps by tier
	- Beta plan: hard cap at 25M units/month; exceeding returns 429
- Headers you’ll see:
	- Per‑key quota window: `X-Quota-Limit`, `X-Quota-Remaining`, `X-Quota-Reset`
	- Global rate limit: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`
	- Per‑IP rate limit: `X-IPLimit-Limit`, `X-IPLimit-Remaining`, `X-IPLimit-Reset`
	- Monthly caps (if enabled): `X-Monthly-Cap`, `X-Monthly-Used`, `X-Monthly-Remaining`, `X-Monthly-Period`

### Optional: Redis for scale

Horizontal scaling is supported via an optional Redis backend for distributed rate limits and webhook idempotency.

- Set `OSCILLINK_STATE_BACKEND=redis` and `OSCILLINK_REDIS_URL=redis://...`
- Details: docs/REDIS_BACKEND.md

### Beta notice

⚠️ Beta Notice: Cloud API is in beta. Expect occasional downtime, breaking changes with notice, and email‑only support. Hard usage caps enforced. Production use at your own risk.

### Manage or cancel your subscription

Two ways to manage billing once you have an API key:

- Self‑service billing portal (user):
	- Endpoint: `POST /billing/portal`
	- Auth: `X-API-Key: <your_key>`
	- Response: `{ "url": "https://billing.stripe.com/..." }` — open this URL in a browser to manage payment method, invoices, or cancel.
	- Requires server to have `STRIPE_SECRET_KEY` and a Firestore mapping collection set via `OSCILLINK_CUSTOMERS_COLLECTION` (the `/billing/success` flow persists `api_key → (stripe_customer_id, subscription_id)` for portal lookups). Optional `OSCILLINK_PORTAL_RETURN_URL` controls the post‑portal return URL.

	Minimal example:
	```bash
	curl -X POST https://api.yourdomain.com/billing/portal \
		-H "X-API-Key: $YOUR_API_KEY"
	```

- Admin cancel (operator):
	- Endpoint: `POST /admin/billing/cancel/{api_key}?immediate=true|false`
	- Auth: `X-Admin-Secret: <OSCILLINK_ADMIN_SECRET>`
	- Behavior: Cancels the Stripe subscription mapped to `api_key`. If `immediate=true` (or server env `OSCILLINK_STRIPE_CANCEL_IMMEDIATE=1`), the subscription is cancelled immediately; otherwise it cancels at period end. The API key is suspended right away.
	- Requires the same Firestore mapping collection (`OSCILLINK_CUSTOMERS_COLLECTION`) and `STRIPE_SECRET_KEY`.

	Minimal example:
	```bash
	curl -X POST "https://api.yourdomain.com/admin/billing/cancel/$USER_API_KEY?immediate=false" \
		-H "X-Admin-Secret: $OSCILLINK_ADMIN_SECRET"
	```

Server env summary for billing management:

- `STRIPE_SECRET_KEY` — Stripe API key for server‑side operations
- `OSCILLINK_CUSTOMERS_COLLECTION` — Firestore collection name used to persist `api_key → {stripe_customer_id, subscription_id}`
- `OSCILLINK_PORTAL_RETURN_URL` — Optional return URL after the Stripe Billing Portal (default `https://oscillink.com`)
- `OSCILLINK_ADMIN_SECRET` — Required for admin endpoints
- `OSCILLINK_STRIPE_CANCEL_IMMEDIATE` — Optional default for admin cancel behavior (`1/true` for immediate)

---

## Performance (SDK reference)

- Graph build: ~18 ms
- Settle: ~10 ms
- Receipt: ~3 ms

Total: < 40 ms for N≈1200 on a laptop (Python 3.11, NumPy BLAS). Use `scripts/benchmark.py` for your hardware.

Scalability at a glance:

- One matvec is O(Nk); total solve is approximately O(D · cg_iters · N · k)
- Typical CG iterations ≈ 3–4 at tol ~1e‑3 with Jacobi (thanks to SPD)

Hallucination control (controlled study): trap rate reduced 0.33 → 0.00 with F1 uplift (see whitepaper for setup details)

## Docs & examples

- SDK API: `docs/API.md`
- Math overview: `docs/SPEC.md`
- Receipts schema and examples: `docs/RECEIPTS.md`
- Advanced cloud topics: `docs/CLOUD_ARCH_GCP.md`, `docs/CLOUD_ADVANCED_DIFFUSION_ENDPOINT.md`, `docs/FIRESTORE_USAGE_MODEL.md`, `docs/STRIPE_INTEGRATION.md`
- Whitepaper: Oscillink — A Symmetric Positive Definite Lattice for Scalable Working Memory & Hallucination Control (`OscillinkWhitepaper.tex`)
- Examples: `examples/quickstart.py`, `examples/diffusion_gated.py`
- Notebooks: `notebooks/`

## Support & branding

- Contact: travisjohnson@oscillink.com
- Branding: Oscillink is a brand of Odin Protocol Inc. (trademark filing for Oscillink planned). 

## Troubleshooting (Cloud)

- 403 Unauthorized
	- Check the `X-API-Key` header is present and correct
	- If running your own server, ensure `OSCILLINK_API_KEYS` or the keystore contains your key

- 429 Too Many Requests
	- You’ve hit a quota or rate limit; inspect `X-Quota-*`, `X-RateLimit-*`, and `X-IPLimit-*` headers (and `X-Monthly-*` if caps are enabled) for remaining and reset

- Success page didn’t show a key after payment
	- Verify the Stripe payment link redirects to `/billing/success?session_id={CHECKOUT_SESSION_ID}`
	- Ensure the server has `STRIPE_SECRET_KEY` and price→tier mapping configured; see `docs/STRIPE_INTEGRATION.md`

- Redis not used despite being configured
	- Set `OSCILLINK_STATE_BACKEND=redis` and provide `OSCILLINK_REDIS_URL` (or `REDIS_URL`); see `docs/REDIS_BACKEND.md`

## Contributing & License

- Apache‑2.0. See `LICENSE`
- Issues and PRs welcome. See `CONTRIBUTING.md`

---

© 2025 Odin Protocol Inc. (Oscillink brand)

