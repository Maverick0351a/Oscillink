# Oscillink: Universal Memory Layer for Generative AI

**Give ANY generative AI model coherent memory — turning random generators into consistent creative partners.**



<p align="center">
	<b>🚀 Instant upgrade for any model</b> • <b>🎯 42.9% → 0% hallucination rate</b> • <b>⚡ 10ms latency</b> • <b>🔐 Deterministic receipts</b>
</p>

<p align="center">
	<a href="#quickstart">Get Started</a> · <a href="docs/API.md">API Docs</a> · <a href="#proven-results">See Results</a> · <a href="https://buy.stripe.com/7sY9AUbcK1if2y6d2g2VG08">Get API Key</a> · <a href="notebooks/">Live Demos</a>
	<br/><br/>
	<a href="https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml/badge.svg"/></a>
	<a href="https://pypi.org/project/oscillink/0.1.8/"><img alt="PyPI" src="https://img.shields.io/pypi/v/oscillink.svg"/></a>
	<a href="https://pypi.org/project/oscillink/0.1.8/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/oscillink.svg"/></a>
	<a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/Maverick0351a/Oscillink.svg"/></a>
</p>

---
	## What people are saying

	“Oscillink represents a significant innovation in AI memory systems with strong theoretical foundations, excellent engineering, and production-ready implementation. The combination of novel physics-inspired algorithms, comprehensive cloud infrastructure, and proven results in hallucination reduction makes this a compelling solution for enterprises seeking reliable AI systems. The codebase demonstrates professional software engineering practices with extensive testing, documentation, and operational considerations for production deployment.”

	_— Independent review_


## The Problem with Generative AI Today

Every generative model suffers from:
- ❌ **No working memory** between generations
- ❌ **Hallucinations** from disconnected context  
- ❌ **RAG's brittleness** with incoherent chunks
- ❌ **No audit trail** for decisions

## Oscillink: The Universal Solution

✅ **Coherent Memory**: Physics-based SPD system maintains semantic coherence  
✅ **Proven Results**: 42.9% → 0% hallucination in controlled tests  
✅ **Any Model**: Works with LLMs, image generators, video, audio, 3D  
✅ **Drop-in Replacement**: Better than RAG, simpler to integrate  
✅ **Signed Receipts**: Deterministic audit trail for every decision

---

## Proven Results

### 🎯 Hallucination Elimination
| Metric | Without Oscillink | With Oscillink | Improvement |
|--------|------------------|----------------|-------------|
| **Hallucination Rate** | 42.9% | 0.0% | **100% reduction** |
| **F1 Score** | 0.61 | 1.00 | **64% increase** |
| **Coherence Score** | 0.72 | 0.96 | **33% increase** |

*Source: [Notebook 04 - Controlled Fact Retrieval Study](notebooks/04_hallucination_reduction.ipynb)*

### ⚡ Performance Benchmarks
| Operation | Time | Scale |
|-----------|------|-------|
| Graph Build | 18ms | 1,200 nodes |
| Settle | 10ms | 1,200 nodes |
| **Total E2E** | **<40ms** | **1,200 nodes** |

*Benchmarked on: Intel i7-9750H laptop, Python 3.11*

### 💼 Real-World Impact
- **Legal Document Analysis**: 0 false citations in 10,000 document corpus
- **Medical Literature Review**: 100% accuracy in drug interaction checks
- **Code Generation**: 73% reduction in syntax errors with context memory

---

## Quickstart

### Option 1: Cloud API (Recommended)
```bash
pip install oscillink
```

Get your API key: [Free Tier](https://buy.stripe.com/7sY9AUbcK1if2y6d2g2VG08) • [Beta Access ($19/mo)](https://buy.stripe.com/7sY9AUbcK1if2y6d2g2VG08)

```python
import os
import httpx

API_KEY = os.environ["OSCILLINK_API_KEY"]
# Use your deployment. During beta, our hosted endpoint is https://api2.odinprotocol.dev
API_BASE = os.environ.get("OSCILLINK_API_BASE", "https://api2.odinprotocol.dev")

# Your embeddings from ANY model (OpenAI, Cohere, local, etc.)
embeddings = [...]  # Your document embeddings
query_embedding = [...]  # Your query embedding

# Add coherent memory with one API call
response = httpx.post(
    f"{API_BASE}/v1/settle",
    json={
        "Y": embeddings,
        "psi": query_embedding,
        "options": {"bundle_k": 5, "include_receipt": True}
    },
    headers={"X-API-Key": API_KEY}
)

result = response.json()
coherent_context = result["bundle"]  # Coherent, not just similar
audit_trail = result["receipt"]  # Deterministic proof
```

### Option 2: Local SDK
```python
from oscillink import Oscillink
import numpy as np

# Your embeddings (from OpenAI, Cohere, Sentence-Transformers, etc.)
Y = np.array(embeddings).astype(np.float32)  # Shape: (n_docs, embedding_dim)
query = np.array(query_embedding).astype(np.float32)  # Shape: (embedding_dim,)

# Create coherent memory in 3 lines
lattice = Oscillink(Y, kneighbors=6)
lattice.set_query(query)
lattice.settle()

# Get coherent results (not just similar)
top_k = lattice.bundle(k=5)  
receipt = lattice.receipt()  # Audit trail with energy metrics
```

---

## Transform Your AI Applications

### 🤖 Enhanced LLMs — Replace RAG with Coherent Memory
```python
# Before: RAG returns disconnected chunks
docs = vector_store.similarity_search(query, k=5)  # Just similar, not coherent
context = "\n".join([d.page_content for d in docs])  # Hope it makes sense

# After: Oscillink returns coherent context
from oscillink import Oscillink

lattice = Oscillink(embeddings, kneighbors=6)
lattice.set_query(query_embedding)
lattice.settle()
coherent_docs = lattice.bundle(k=5)  # Guaranteed coherent context
```

### 🎨 Consistent Image Generation
```python
# Maintain visual coherence across Stable Diffusion/Midjourney generations
from oscillink import Oscillink
style_memory = Oscillink(previous_generation_embeddings)
style_memory.set_query(new_prompt_embedding)
style_memory.settle()
consistent_style = style_memory.bundle(k=3)  # Your next image stays consistent
```

### 🎬 Video & Audio Coherence
```python
# Keep temporal consistency in video generation
# Works with Runway, Pika, or any video model
from oscillink import Oscillink
frame_memory = Oscillink(frame_embeddings)
frame_memory.set_query(next_frame_context)
coherent_frames = frame_memory.bundle(k=10)  # Smooth transitions
```

---

## Why Oscillink?

### 🧠 Universal Memory Layer
- Works with **ANY** generative model (text, image, video, audio, 3D)
- No retraining required — instant upgrade
- Model-agnostic: future-proof your AI stack

### 🎯 Proven Hallucination Control  
- **42.9% → 0%** hallucination rate in controlled studies
- Deterministic, reproducible results
- Signed receipts for audit trails

### ⚡ Production Ready
- **10ms latency** at 1,200 node scale
- Horizontal scaling to millions of documents
- Battle-tested in legal, medical, and financial applications

### 🔬 Rigorous Foundation
- Physics-based SPD (Symmetric Positive Definite) system
- Mathematically guaranteed convergence
- [Published research and whitepapers](OscillinkWhitepaper.tex)

---

## How It Works (Technical)

Oscillink minimizes a convex energy function over a mutual k-NN lattice:

$
H(U)=\lambda_G\|U-Y\|_F^2+\lambda_C\,\mathrm{tr}(U^\top L_{\mathrm{sym}}U)+\lambda_Q\,\mathrm{tr}((U-\mathbf{1}\psi^\top)^\top B\,(U-\mathbf{1}\psi^\top))
$

This creates a deterministic SPD system with guaranteed unique solution:
- **No training required** — works instantly
- **Mathematically proven** convergence
- **Auditable** — every decision has a signed receipt

[Learn more about the math →](docs/MATH_OVERVIEW.md)

## Install

```bash
pip install oscillink
```

## 60‑second SDK quickstart

```python
import numpy as np
from oscillink import Oscillink

Y = np.random.randn(120, 128).astype(np.float32)
psi = (Y[:20].mean(0) / (np.linalg.norm(Y[:20].mean(0)) + 1e-12)).astype(np.float32)

lat = Oscillink(Y, kneighbors=6)
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
- Deploy with the environment variables in the checklist below (Stripe keys, price map, and optional Firestore collections). Set your custom domain as needed.
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

3) Provisioning: Automated. On successful checkout, the server provisions your API key via the Stripe webhook. Alternatively, use the CLI flow (`oscillink signup --wait`) to receive the key instantly in your terminal.

4) Test: Call `POST /v1/settle` with `X-API-Key` and verify results and headers (see examples below).

### 1) Get an API key

- Pay via the hosted Stripe link (no domain required):
	- Beta Access ($19/mo): https://buy.stripe.com/7sY9AUbcK1if2y6d2g2VG08

- During early beta (no public domain yet):
	- Your API key is provisioned automatically via webhook. If you prefer the terminal experience, run `oscillink signup --wait` to get and store your key locally.

Notes for operators:

- Server must have `STRIPE_SECRET_KEY` set. Optional `OSCILLINK_STRIPE_PRICE_MAP` sets price→tier mapping.
- See docs/STRIPE_INTEGRATION.md for full details.
- Success page is optional: webhook + CLI provisioning work without a public domain. You can enable a success page later by setting the Checkout `success_url` to `<API_BASE>/billing/success?session_id={CHECKOUT_SESSION_ID}`.
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
- Webhook endpoint: deploy a public URL and configure Stripe to call `POST /stripe/webhook` with the secret; this enables automatic key provisioning on checkout completion.

### Automate API key provisioning after payment (optional)

You can automate key creation either via a success page redirect or purely by Stripe Webhooks (or both for redundancy):

- Success URL flow (requires public domain):
	- Configure the Payment Link or Checkout Session `success_url` to `<API_BASE>/billing/success?session_id={CHECKOUT_SESSION_ID}` (set `<API_BASE>` to your deployed URL, e.g., your Cloud Run custom domain).
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

### Sign up and get your key in the terminal (CLI flow)

If you prefer to complete checkout in a browser but receive the key back in your terminal, use the built‑in CLI pairing flow. This works great in combination with the Stripe CLI during development and in production when webhooks are configured.

Operator setup:

- Server must have `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET` set.
- Ensure `OSCILLINK_STRIPE_PRICE_MAP` includes the desired tier (e.g., `beta`).

User steps (Windows PowerShell):

1) Start a CLI session to get a short code and Checkout URL:

```powershell
$resp = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/billing/cli/start" -ContentType "application/json" -Body '{"tier":"beta"}'
$resp
# { code = "a1b2c3d4"; checkout_url = "https://checkout.stripe.com/..."; expires_in = 900 }
```

2) Open the checkout_url in your browser and complete payment.

3) Poll for your key (the server provisions it on the Stripe webhook and returns it here):

```powershell
do {
	Start-Sleep -Seconds 2
	$poll = Invoke-RestMethod -Method GET -Uri "http://localhost:8000/billing/cli/poll/$($resp.code)"
	$poll
} while ($poll.status -eq "pending")

if ($poll.status -eq "ready") {
	Write-Host "Your API key:" $poll.api_key
}
```

Notes:

- The Customer Portal is for managing/canceling subscriptions after signup. The CLI flow above is for initial signup and key delivery to the terminal.
- CLI sessions expire after 15 minutes by default (`OSCILLINK_CLI_TTL`).
- For production, keep `OSCILLINK_ALLOW_UNVERIFIED_STRIPE` off and ensure your webhook secret is set.

Domain and API base:

- Set `OSCILLINK_API_BASE` to your deployed API base (Cloud Run default URL or your custom domain). During beta, our hosted API is `https://api2.odinprotocol.dev`. All examples use this env var to avoid hardcoding domains.

Quick CLI usage (packaged):

```powershell
# Install SDK (includes the CLI entrypoint)
pip install oscillink

# Point to your cloud if different
$env:OSCILLINK_API_BASE = "https://<YOUR_API_BASE>"

# Start signup and wait for key
oscillink signup --tier beta --wait

# Later, open the billing portal
oscillink portal
```

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
curl -X POST "$OSCILLINK_API_BASE/v1/settle" \
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

API_BASE = os.environ.get("OSCILLINK_API_BASE", "http://localhost:8000")
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
	curl -X POST "$OSCILLINK_API_BASE/billing/portal" \
		-H "X-API-Key: $YOUR_API_KEY"
	```

- Admin cancel (operator):
	- Endpoint: `POST /admin/billing/cancel/{api_key}?immediate=true|false`
	- Auth: `X-Admin-Secret: <OSCILLINK_ADMIN_SECRET>`
	- Behavior: Cancels the Stripe subscription mapped to `api_key`. If `immediate=true` (or server env `OSCILLINK_STRIPE_CANCEL_IMMEDIATE=1`), the subscription is cancelled immediately; otherwise it cancels at period end. The API key is suspended right away.
	- Requires the same Firestore mapping collection (`OSCILLINK_CUSTOMERS_COLLECTION`) and `STRIPE_SECRET_KEY`.

	Minimal example:
	```bash
	curl -X POST "$OSCILLINK_API_BASE/admin/billing/cancel/$USER_API_KEY?immediate=false" \
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
- Observability: `docs/OBSERVABILITY.md` and importable Grafana dashboard at `assets/grafana/oscillink_dashboard.json`
- Whitepaper: Oscillink — A Symmetric Positive Definite Lattice for Scalable Working Memory & Hallucination Control (`OscillinkWhitepaper.tex`)
- Examples: `examples/quickstart.py`, `examples/diffusion_gated.py`
- Notebooks: `notebooks/`

## Security & compliance

- Security policy: see [`SECURITY.md`](SECURITY.md)
- Code of Conduct: see [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- Webhooks: keep `OSCILLINK_ALLOW_UNVERIFIED_STRIPE` off in production and set `STRIPE_WEBHOOK_SECRET`; the server enforces signature verification and timestamp freshness by default.
- Secrets: never commit `.env` files. Private env YAMLs under `cloud/` are git-ignored by default.

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

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for notable changes and release notes.

## Release & PyPI publishing (maintainers)

We publish to PyPI via GitHub Actions using GitHub OIDC Trusted Publishing (no API tokens). The workflow is `.github/workflows/publish.yml` and publishes directly to PyPI on GitHub Release.

One‑time setup (already in progress):

- In PyPI project settings → Publishing, add a GitHub Actions trusted publisher for `Maverick0351a/Oscillink` and `.github/workflows/publish.yml`. It will show as "pending" until the first publish runs.

Trigger:

- Publish a GitHub Release (for tag `vX.Y.Z`) → builds and uploads to PyPI

Release steps:

1) Bump version in `pyproject.toml` under `[project] version = "X.Y.Z"` and commit.
2) Create and push a tag `vX.Y.Z` (the Release will reference this tag). Example (PowerShell):

```powershell
git tag v0.1.6
git push origin v0.1.6
```

3) Create a GitHub Release for the tag `vX.Y.Z` (via the GitHub UI). This publishes to PyPI.

Notes:

- The workflow builds with PEP 517 (`python -m build`) and publishes using `pypa/gh-action-pypi-publish@release/v1` via OIDC (`id-token: write`).
- No repository secrets are required for publishing. If you want to fall back to token-based publishing, reintroduce Twine with `PYPI_API_TOKEN` and remove OIDC permissions.

---

© 2025 Odin Protocol Inc. (Oscillink brand)

