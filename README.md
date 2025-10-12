# Oscillink — Self‑Optimizing Memory for Generative Models and Databases

Build coherence into retrieval and generation. Deterministic receipts for every decision. Latency that scales gracefully with corpus size.

<p align="left">
	<a href="https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml/badge.svg"/></a>
	<a href="https://github.com/Maverick0351a/Oscillink/actions/workflows/codeql.yml"><img alt="CodeQL" src="https://github.com/Maverick0351a/Oscillink/actions/workflows/codeql.yml/badge.svg"/></a>
	<a href="https://github.com/Maverick0351a/Oscillink/actions/workflows/pip-audit.yml"><img alt="pip-audit" src="https://github.com/Maverick0351a/Oscillink/actions/workflows/pip-audit.yml/badge.svg"/></a>
	<a href="https://app.codecov.io/gh/Maverick0351a/Oscillink/tree/main"><img alt="Coverage" src="https://codecov.io/gh/Maverick0351a/Oscillink/branch/main/graph/badge.svg"/></a>
	<a href="https://pypi.org/project/oscillink/"><img alt="PyPI" src="https://img.shields.io/pypi/v/oscillink.svg"/></a>
	<a href="https://pypi.org/project/oscillink/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/oscillink.svg"/></a>
	<a href="LICENSE"><img alt="License" src="https://img.shields.io/pypi/l/oscillink.svg"/></a>
	<a href="PATENTS.md"><img alt="Patent pending" src="https://img.shields.io/badge/Patent-pending-blueviolet.svg"/></a>
	<br/>
	<sub>CI: Python 3.10–3.12 × NumPy 1.x/2.x</sub>

</p>

Setup: synthetic “facts + traps” dataset — see the notebook for N, k, trials, seed. Reproducible via `notebooks/04_hallucination_reduction.ipynb`. Traps flagged; gate=0.01, off‑topic damp=0.5.

- ⚡ Latency scales smoothly: with fixed D, k, and CG tol, settle tends to remain stable with denser graphs. Reference E2E < 40 ms at N≈1200 on a laptop CPU.
- 🎯 Hallucination control: in our controlled study[1], Oscillink reduced trap selections while maintaining competitive F1. See the notebook for setup and reproduction.
- 🧾 Receipts: SHA‑256 state signature; optional HMAC‑SHA256. [schema](docs/RECEIPTS.md)
- 🔧 Universal: works with any embeddings, no retraining. [adapters](#adapters--model-compatibility) · [quickstart](examples/quickstart.py)
- 📈 Self‑optimizing: learns parameters over time. [adaptive suite](scripts/bench_adaptive_suite.py)
- 🚀 Production: scales to millions. See scripts under `scripts/` for reproducible benchmarks.

<p>
	<a href="#quickstart">Get Started</a> · <a href="docs/API.md">API Docs</a> · <a href="#proven-results">See Results</a> · <a href="notebooks/">Live Demos</a>
</p>

## Use Oscillink: SDK or Licensed Container

- SDK (pip): Keep everything local. Best for adding coherence directly inside your application process. Start at [Install](#install) and [60‑second SDK quickstart](#60-second-sdk-quickstart).
- Licensed Container (customer‑managed): Run the API entirely inside your VPC or cluster with license‑based entitlements. Jump to [Licensed Container (customer‑managed)](#licensed-container-customer-managed). Operators can then dive into [Operations](docs/OPERATIONS.md) and [Networking](docs/NETWORKING.md).

### Table of contents

- [TL;DR](#tldr)
- [Quickstart](#quickstart)
- [Adapters & model compatibility](#adapters--model-compatibility)
- [Licensed Container (customer-managed)](#licensed-container-customer-managed)
- [Cloud API (beta)](#option-b-cloud-api-beta)
- [Proven Results](#proven-results)
- [Performance](#performance-sdk-reference)
- [How It Works](#how-it-works-technical)
- [Install](#install)
- [Run the server](#run-the-server-operators)
- [Use the Cloud](#use-the-cloud)
- [Docs & examples](#docs--examples)
- [Troubleshooting](#troubleshooting-cloud)
- [Pricing (licensed container)](#pricing-licensed-container)

### SDK at a glance

```python
from oscillink import Oscillink

# Y: (N, D) document embeddings; psi: (D,) query embedding (float32 recommended)
lattice = Oscillink(Y, kneighbors=6)
lattice.set_query(psi)
lattice.settle()
topk = lattice.bundle(k=5)  # coherent results
receipts = lattice.receipt()  # deterministic audit (energy breakdown)
```

### Quick install (60 seconds)

```bash
pip install oscillink
python - <<'PY'
import numpy as np
from oscillink import Oscillink
Y = np.random.randn(80,128).astype('float32')
psi = (Y[:10].mean(0)/ (np.linalg.norm(Y[:10].mean(0))+1e-9)).astype('float32')
lat = Oscillink(Y, kneighbors=6, lamG=1.0, lamC=0.5, lamQ=4.0)
lat.set_query(psi); lat.settle()
print(lat.bundle(k=5)); print(lat.receipt()['deltaH_total'])
PY
```



### Quick checks (Windows PowerShell)

```powershell
# Compare cosine baseline vs Oscillink; writes JSON summary (no plotting)
python scripts/competitor_benchmark.py --input examples\real_benchmark_sample.jsonl --format jsonl --text-col text --id-col id --label-col label --trap-col trap --query-index 0 --k 5 --json --out benchmarks\competitor_sample.json

# Scaling harness; emits JSONL suitable for analysis
python scripts/scale_benchmark.py --N 400 800 1200 --D 64 --k 6 --trials 2 > benchmarks\scale.jsonl
```

With fixed D=64, k=6, tol=1e-3 and Jacobi preconditioning, CG converges in ~3–4 iterations; E2E time trends sublinear in practice with improved connectivity at larger N. Laptop ref: 3.5 GHz i7, Python 3.11, NumPy MKL/Accelerate.

<p align="center"><img alt="Oscillink" src="assets/oscillink_hero.png" width="640"/></p>

<p align="center"><i>Current: v0.1.12 • API v1 • Cloud: beta</i></p>

---

## TL;DR
- Coherent memory for any model (LLMs, image, video, audio, 3D) — no retraining.
- Deterministic receipts for auditability and reproducibility.
- SPD system with guaranteed convergence; typical E2E latency < 40 ms at N≈1200.
- Coherence‑first retrieval that works alongside RAG — see metrics below.

Quick links:
- [SDK Quickstart](#quickstart) · [API + Receipts](docs/API.md) · [Cloud (beta)](#use-the-cloud)


## The Problem with Generative AI Today

Every generative model suffers from:
- **No working memory** between generations
- **Hallucinations** from disconnected context
- **RAG's brittleness** with incoherent chunks
- **No audit trail** for decisions

## Oscillink: The Universal Solution

✅ **Coherent memory**: Physics-based SPD system maintains semantic coherence
✅ **Proven results**: See metrics below (controlled study and benchmarks)
✅ **Any model**: Works with LLMs, image generators, video, audio, 3D
✅ **Coherence‑first retrieval**: Can sit alongside similarity‑only RAG
✅ **Signed receipts**: Deterministic audit trail for every decision

---



## Quickstart

### Option A: Local SDK
```python
from oscillink import Oscillink
import numpy as np

# Your embeddings (from OpenAI, Cohere, Sentence-Transformers, etc.)
Y = np.array(embeddings).astype(np.float32)  # Shape: (n_docs, embedding_dim)
psi = np.array(query_embedding).astype(np.float32)  # Shape: (embedding_dim,)

# Create coherent memory in 3 lines
lattice = Oscillink(Y, kneighbors=6)
lattice.set_query(psi)
lattice.settle()

# Get coherent results (not just similar)
top_k = lattice.bundle(k=5)
receipt = lattice.receipt()  # Audit trail with energy metrics
```

Requirements:
 Python 3.10–3.12; NumPy >= 1.22 and < 3.0 (1.x and 2.x supported)
- Embeddings: shape (N, D), dtype float32 recommended; near unit-norm preferred

Compatibility:
- OS: Windows, macOS, Linux
 - Python: 3.10–3.12
- NumPy: 1.22–2.x (tested in CI)
- CPU only; no GPU required

### Adapters & model compatibility

Oscillink is designed to be universal and light on dependencies:

- Bring your own embeddings: from OpenAI, Cohere, or local models; just supply `Y: (N,D)` and your query `psi: (D,)`.
- Minimal deps: NumPy + small helpers. We avoid heavy, model‑specific stacks by design.
- Adapters: see `oscillink.adapters.*` for simple utilities (e.g., text embedding helpers). You can use any embedding pipeline you prefer.
- Per‑model adjustments: for best results you may tune `kneighbors` and the lattice weights (`lamC`, `lamQ`) to your domain; the CLI `--tune` flag and the adaptive profile suite provide quick, data‑driven defaults.
- Preprocessing: optional `smart_correct` can reduce incidental traps on noisy inputs (code/URL aware).

This flexibility ensures Oscillink works with your existing stack without imposing a large dependency footprint.

### Option B: Cloud API (beta)
Cloud is strictly opt‑in. The SDK never sends data anywhere.
```bash
pip install oscillink
```

Then obtain an API key (see [Use the Cloud](#use-the-cloud)) and call the API:

```python
import os
import httpx

API_KEY = os.environ["OSCILLINK_API_KEY"]
# Use your deployment. During beta, our hosted endpoint is https://api2.odinprotocol.dev
API_BASE = os.environ.get("OSCILLINK_API_BASE", "https://api2.odinprotocol.dev")

# Your embeddings from ANY model (OpenAI, Cohere, local, etc.)
embeddings = [...]  # Your document embeddings
psi = [...]  # Your query embedding

# Add coherent memory with one API call
response = httpx.post(
	f"{API_BASE}/v1/settle",
	json={
		"Y": embeddings,
		"psi": psi,
		"options": {"bundle_k": 5, "include_receipt": True}
	},
	headers={"X-API-Key": API_KEY}
)

result = response.json()
coherent_context = result["bundle"]  # Coherent, not just similar
audit_trail = result["receipt"]  # Deterministic proof
```

## Cloud feature flags and headers (beta)

These toggles are off by default and safe to enable gradually. They only affect the cloud server.

- Adaptive profiles (per-key tuning):
	- Enable overrides: `OSCILLINK_ADAPTIVE_PROFILES=1`
	- Enable learning (bounded EMA + epsilon-greedy): `OSCILLINK_ADAPTIVE_LEARN=1`
	- Tuning knobs: `OSCILLINK_ADAPTIVE_ALPHA` (default `0.2`), `OSCILLINK_ADAPTIVE_EPS` (default `0.1`), `OSCILLINK_ADAPTIVE_MARGIN` (default `0.0`)
	- Responses include `X-Profile-Id` header, and `meta.profile_id` in JSON.

- Bundle cache (in-memory TTL LRU, per API key):
	- Enable: `OSCILLINK_CACHE_ENABLE=1`
	- TTL seconds: `OSCILLINK_CACHE_TTL` (default `300`)
	- Per-key capacity: `OSCILLINK_CACHE_CAP` (default `128`)
	- `/bundle` responses include `X-Cache: HIT|MISS`; on HIT also `X-Cache-Hits` and `X-Cache-Age`.


---

## Proven Results

### 🎯 Reduced hallucinations (developer‑verifiable)
- On the included sample dataset, Oscillink reduced trap selections compared to a cosine baseline while maintaining competitive F1. Reproduce with one command and inspect the JSON output.

Reproduce (Windows PowerShell) — JSON only:

```powershell
python scripts/competitor_benchmark.py --input examples\real_benchmark_sample.jsonl --format jsonl --text-col text --id-col id --label-col label --trap-col trap --query-index 0 --k 5 --json --out benchmarks\competitor_sample.json
python scripts/plot_benchmarks.py --competitor benchmarks\competitor_sample.json --out-dir assets\benchmarks
```

- Hallucination reduction: baseline shows traps present (1=yes), Oscillink default/tuned suppress them on this dataset while maintaining speed. Use your own data via the CLI to validate on your domain.

Aggregate across many queries (mean ± 95% CI):

```powershell
python scripts/competitor_benchmark.py --input examples\real_benchmark_sample.jsonl --format jsonl --text-col text --id-col id --label-col label --trap-col trap --all-queries --k 5 --json --out benchmarks\competitor_multi.json
python scripts/plot_benchmarks.py --competitor benchmarks\competitor_multi.json --out-dir assets\benchmarks
```

<!-- Image plots removed -->

### ⚡ Performance benchmarks and scaling

- End‑to‑end latency typically under 40 ms at N≈1200 on a laptop (Python 3.11), with “light” receipts. Graph build and solve times scale smoothly with N.
- The scaling harness emits JSONL so you can analyze timings on your hardware (no plotting required).

Reproduce on your machine (JSONL only):

```powershell
python scripts/scale_benchmark.py --N 400 800 1200 --D 64 --k 6 --trials 2 > benchmarks\scale.jsonl
python scripts/plot_benchmarks.py --scale benchmarks\scale.jsonl --out-dir assets\benchmarks
```

### 💼 Real‑world impact (representative)
- Coherent retrieval reduces false citations in large corpora (see receipts for audit).
- Autocorrect‑assisted preprocessing reduces incidental traps on noisy inputs.

### Reproduce receipts and proofs

```powershell
pip install -e .[dev]
python scripts\proof_hallucination.py --seed 123 --n 1200 --d 128
```

Example receipt (abridged):

```json
{
	"state_sig": "sha256:9c1d…",
	"energy": {
		"H": 1.234,
		"deltaH_total": -0.456,
		"terms": {"data": 0.321, "graph": 0.789, "query": 0.124}
	},
	"params": {"kneighbors": 6, "lamG": 1.0, "lamC": 0.5, "lamQ": 4.0},
	"timings_ms": {"build": 18.0, "settle": 10.2, "receipt": 3.1}
}
```

---

## How it compares to similarity‑only RAG

- Coherence‑first context selection vs. nearest‑neighbor chunks — see the metrics above (controlled study).
- Complementary: can sit alongside your existing RAG stack (reuse embeddings and vector store).
- Latency reference: ≈10 ms settle, <40 ms E2E at N≈1200 on reference hardware.

Oscillink complements vector similarity with a global coherence objective — keep your embeddings and store; add explainable memory shaping on top.

---

## Transform Your AI Applications

### 🤖 Enhanced LLMs — Coherence‑first beyond similarity‑only RAG
```python
# Before: Similarity‑only RAG may return disconnected chunks
docs = vector_store.similarity_search(query, k=5)  # Just similar, not coherent
context = "\n".join([d.page_content for d in docs])  # Hope it makes sense

# After: Oscillink returns coherent context (see metrics below)
from oscillink import Oscillink

lattice = Oscillink(embeddings, kneighbors=6)
lattice.set_query(query_embedding)
lattice.settle()
coherent_docs = lattice.bundle(k=5)  # Coherent context with deterministic receipts
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

### 🎯 Proven hallucination control
- Demonstrated improvements in a controlled study (see metrics above)
- Deterministic, reproducible results
- Signed receipts for audit trails

### ⚡ Production Ready
- **10ms latency** at 1,200 node scale
- Horizontal scaling to millions of documents
- Pilot deployments in legal, medical, and financial domains

### 🧰 Operators (production-ready knobs)
- Determinism: set `deterministic_k=True` or a `neighbor_seed`, fix CG `tol`, sign receipts.
- Safety: `OSCILLINK_DIFFUSION_GATES_ENABLED=0` as an emergency kill switch for adaptive gates.
- Observability: log `state_sig`, `profile_id`, `ustar_iters`, `ustar_res`, `deltaH_total`, `settle_ms`.

### 🔬 Rigorous Foundation
- Physics-based SPD (Symmetric Positive Definite) system
- Mathematically guaranteed convergence
- [Published research and whitepapers](OscillinkWhitepaper.tex)

---

## How It Works (Technical)

Oscillink minimizes a convex energy function over a mutual k-NN lattice:

$$
H(U)=\lambda_G\|U-Y\|_F^2+\lambda_C\,\mathrm{tr}(U^\top L_{\mathrm{sym}}U)+\lambda_Q\,\mathrm{tr}((U-\mathbf{1}\psi^\top)^\top B\,(U-\mathbf{1}\psi^\top))
$$

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

### Redis-backed CLI sessions (operators)

For higher reliability and horizontal scaling of the CLI pairing flow, enable Redis-backed sessions. When enabled, short-lived CLI pairing records are stored in Redis with key `cli:{code}` and an expiry equal to `OSCILLINK_CLI_TTL` (default 900 seconds). If Redis is unavailable at runtime, the server automatically falls back to in-memory behavior.

Environment variables:

- `OSCILLINK_CLI_SESSIONS_BACKEND=redis` — select Redis backend (default is `memory`)
- `OSCILLINK_STATE_BACKEND=redis` — enable Redis in the app (shared for other features like rate limits)
- `OSCILLINK_REDIS_URL=redis://localhost:6379/0` — Redis connection URL (or use `REDIS_URL`)
- `OSCILLINK_CLI_TTL=900` — TTL in seconds for CLI sessions

Minimal docker-compose for local dev:

```yaml
version: '3.8'
services:
	redis:
		image: redis:7-alpine
		ports:
			- '6379:6379'
	api:
		build:
			context: .
			dockerfile: cloud/Dockerfile
		environment:
			PORT: 8080
			OSCILLINK_FORCE_HTTPS: '0'
			OSCILLINK_STATE_BACKEND: redis
			OSCILLINK_CLI_SESSIONS_BACKEND: redis
			OSCILLINK_REDIS_URL: redis://redis:6379/0
			STRIPE_SECRET_KEY: ${STRIPE_SECRET_KEY:-}
			STRIPE_WEBHOOK_SECRET: ${STRIPE_WEBHOOK_SECRET:-}
			OSCILLINK_STRIPE_PRICE_MAP: ${OSCILLINK_STRIPE_PRICE_MAP:-}
		ports:
			- '8000:8080'
		depends_on:
			- redis
```

Notes:

- In Redis mode, CLI session expiry is handled by Redis key TTL. The server’s `purge_expired` becomes a no-op when Redis is available. If Redis is misconfigured or temporarily unavailable, the server will fall back to in-memory and purge claimed/expired sessions in-process.
- You can keep the per-endpoint rate limits for `/billing/cli/start` and `/billing/cli/poll/{code}` by setting `OSCILLINK_EPRL_CLI_START_LIMIT`/`OSCILLINK_EPRL_CLI_START_WINDOW` and `OSCILLINK_EPRL_CLI_POLL_LIMIT`/`OSCILLINK_EPRL_CLI_POLL_WINDOW`.

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

## Licensed Container (customer-managed)

If you prefer to run Oscillink entirely inside your VPC and only send minimal license/usage heartbeats, use the licensed container. No embeddings or content ever leave your network.

What you get:
- Entitlements and feature gating via a signed JWT license (Ed25519)
- Local enforcement of caps (nodes, dim, monthly units) and QPS
- Optional minimal telemetry: aggregated counters only

Quickstart (Docker):
- Create a folder `deploy\license` and place your license file `oscillink.lic` (JWT) inside.
- Run `docker-compose -f deploy/docker-compose.yml up -d`.
- The API will be available on http://localhost:8000 (health at `/health`).

Windows one‑liner (PowerShell):

```powershell
docker compose -f deploy\docker-compose.yml up -d
```

Linux/macOS one‑liner (bash/zsh):

```bash
docker compose -f deploy/docker-compose.yml up -d
```

Notes:
- Ensure the directories exist before starting (they’re mounted by the compose file):

	```bash
	mkdir -p deploy/license deploy/data
	# Place your license at deploy/license/oscillink.lic
	```

- If your environment uses the legacy Docker Compose standalone binary, replace `docker compose` with `docker-compose`.

Required environment variables (container):
- `OSCILLINK_LICENSE_PATH` — path to the mounted license file (default in our compose: `/run/secrets/oscillink.lic`)
- `OSCILLINK_JWKS_URL` — URL to your license service JWKS (e.g., `https://license.oscillink.com/.well-known/jwks.json`)

Optional (telemetry/usage):
- `OSCILLINK_TELEMETRY` — set to `minimal` for aggregated counters only (default in compose)
- `OSCILLINK_USAGE_LOG` — local JSONL path to append usage counters (e.g., `/data/usage.jsonl`)
- `OSCILLINK_USAGE_FLUSH_URL` — if set, background flusher posts batches to this URL
- `OSCILLINK_LICENSE_ID` — identifier included in usage reports

Optional (JWT verification tuning):
- `OSCILLINK_JWT_ISS` — expected issuer (iss) to enforce
- `OSCILLINK_JWT_AUD` — expected audience (aud) to enforce
- `OSCILLINK_JWT_LEEWAY` — seconds of clock skew allowance (default 300)
- `OSCILLINK_JWKS_TTL` — JWKS cache TTL seconds (default 3600)
- `OSCILLINK_JWKS_OFFLINE_GRACE` — allow cached JWKS for this many seconds if JWKS URL is unreachable (default 86400)

Windows note:
- The compose file mounts `deploy\\license` and `deploy\\data`. Ensure these directories exist on Windows before starting.

Kubernetes (Helm):
- Chart skeleton is under `deploy/helm/oscillink`. Create a Secret with your license and install the chart:
	- Set `image.repository` and `image.tag` as needed.
	- Mount the license secret at `/run/secrets/oscillink.lic`.
 	- Set `OSCILLINK_LICENSE_PATH` and `OSCILLINK_JWKS_URL` via values.

License service (tiny):
- The repo includes a minimal FastAPI sketch at `license_svc/main.py` with endpoints:
	- `/.well-known/jwks.json` for public keys
	- `/v1/license/renew` to renew tokens
	- `/v1/usage/report` to accept aggregated usage batches (HMAC)

Notes:
- At container start, `entrypoint.sh` verifies the JWT against JWKS and exports entitlements to `/run/oscillink_entitlements.json` and an env file `/run/oscillink_entitlements.env` consumed by the app.
- Readiness endpoint: `/license/status` reflects license status (ok/stale/expired/unlicensed). Set `OSCILLINK_LICENSE_REQUIRED=1` to fail readiness (503) when expired/missing. Helm chart wires readinessProbe to this path and livenessProbe to `/health`.
- License→env mapping: the verifier exports limits to envs honored by the app:
	- `OSCILLINK_MAX_NODES`, `OSCILLINK_MAX_DIM` — enforced against request shapes
	- `OSCILLINK_RATE_LIMIT`/`OSCILLINK_RATE_WINDOW` — global QPS with `X-RateLimit-*` headers
	- `OSCILLINK_KEY_NODE_UNITS_LIMIT`/`OSCILLINK_KEY_NODE_UNITS_WINDOW` — per-key quota with `X-Quota-*` headers
	- `OSCILLINK_MONTHLY_CAP` — monthly unit cap override (else tier catalog applies), with `X-Monthly-*` headers
	- `OSCILLINK_API_KEYS` and `OSCILLINK_KEY_TIERS` — enables auth and tier mapping
	- `OSCILLINK_FEAT_*` — feature toggles overlay (e.g., `OSCILLINK_FEAT_DIFFUSION_GATES=1`)
- Set `OSCILLINK_USAGE_LOG` to a file path to capture local usage JSONL; set `OSCILLINK_USAGE_FLUSH_URL` and `OSCILLINK_LICENSE_ID` to enable periodic batch upload (the flusher retries with backoff and idempotency keys).

Operator introspection:
- Set an admin secret and query the effective limits and overlays for debugging:

```powershell
$env:OSCILLINK_ADMIN_SECRET = "<random>"
curl -H "X-Admin-Secret: $env:OSCILLINK_ADMIN_SECRET" http://localhost:8000/admin/introspect
```

The response includes license status, enforced limits (nodes, dim, rate/quota/monthly caps, per-IP and endpoint rate limits), cache settings, feature overlays (`OSCILLINK_FEAT_*`), and API key configuration mode.

### Kubernetes Helm add-ons

The Helm chart includes optional production hardening resources you can enable via `values.yaml`:

- NetworkPolicy: restricts ingress to port 8080 and allows egress to JWKS endpoints. Enable with `networkPolicy.enabled=true`. Adjust `egressCIDR` and `egressPorts` as needed.
- PodDisruptionBudget: keep at least one pod available during voluntary disruptions. Enable with `pdb.enabled=true` and tune `minAvailable`/`maxUnavailable`.
- HorizontalPodAutoscaler: scale based on CPU/memory utilization. Enable with `hpa.enabled=true` and set utilization targets.
- Ingress: optional HTTP(S) exposure. Enable with `ingress.enabled=true`, set `ingress.host`, and configure TLS via `ingress.tls.*`.

For private clusters, also consider restricting `/metrics` via NetworkPolicy or a service mesh policy.

### Operations: metrics and logging knobs

Operators can enable low-noise JSON access logs and protect the metrics endpoint without code changes:

- Metrics protection: set `OSCILLINK_METRICS_PROTECTED=1` to require `X-Admin-Secret` header on `GET /metrics`.
	- Provide the secret via env (e.g., Helm `envFrom` Secret) as `OSCILLINK_ADMIN_SECRET`.
	- When enabled and the header is missing or mismatched, `/metrics` returns 403.
- JSON access logs: set `OSCILLINK_JSON_LOGS=1` to emit structured JSON logs for each request.
	- Control sampling with `OSCILLINK_LOG_SAMPLE` in [0.0, 1.0] (default `1.0`). Example: `OSCILLINK_LOG_SAMPLE=0.1` to log ~10% of requests.
	- Logs include request id, path, latency (ms), status code, and limited metadata (no bodies).

Container image pinning:

- The licensed container uses a pinned base image `python:3.11.9-slim`.
- For additional supply-chain hardening, consider pinning by digest in your image registry or Helm values (e.g., `image.digest: sha256:…`).
- Our CI includes non-blocking SBOM generation, pip-audit, and Trivy scanning; you can make them blocking in your fork by changing the exit code behavior.

Response shape (abridged):

- `state_sig: str` — checksum of lattice state for audit
- `bundle: list[dict]` — top‑k results with scores
- `receipt: dict` — energy breakdown (if requested)
- `timings_ms: dict` — perf timings
- `meta: dict` — quota/rate limit headers are returned as `X-Quota-*` (per‑key quotas), plus `X-RateLimit-*` (global) and `X-IPLimit-*` (per‑IP); monthly caps via `X-Monthly-*` when enabled

### Quotas, limits, scale, and billing

See:
- Quotas, rate limits and headers: `docs/API.md`
- Redis & horizontal scale: `docs/REDIS_BACKEND.md`
- Stripe/billing portal and admin flows: `docs/STRIPE_INTEGRATION.md`

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

## Real dataset CLI (terminal)

Benchmark Oscillink on your own CSV/JSONL datasets from the terminal. The CLI compares a cosine baseline vs. Oscillink default vs. adaptive (with light tuning) and prints a JSON summary plus optional top‑k IDs.

- Script: `scripts/real_benchmark.py`
- Sample data: `examples/real_benchmark_sample.jsonl`

Windows PowerShell examples:

```powershell
# JSONL format with explicit columns
python scripts/real_benchmark.py --input examples\real_benchmark_sample.jsonl --format jsonl --text-col text --id-col id --label-col label --trap-col trap --query-index 0 --k 5 --json

# CSV format (if your file is CSV)
python scripts/real_benchmark.py --input path\to\your.csv --format csv --text-col text --id-col id --label-col label --query-index 0 --k 10 --json

# Show top-k IDs (omit --json for human-readable)
python scripts/real_benchmark.py --input examples\real_benchmark_sample.jsonl --format jsonl --text-col text --id-col id --query-index 1 --k 5
```

Notes:
- If `sentence-transformers` is installed, the script will use it to embed texts; otherwise it falls back to a deterministic hash embedder for quick smoke tests.
- Outputs are printed to stdout and can be redirected to a file (e.g., `> results.json`).
- For larger runs, prefer `--json` to capture metrics deterministically.


- SDK API: `docs/API.md`
- Math overview: `docs/MATH_OVERVIEW.md`
- Receipts schema and examples: `docs/RECEIPTS.md`
- Advanced cloud topics: `docs/CLOUD_ARCH_GCP.md`, `docs/CLOUD_ADVANCED_DIFFUSION_ENDPOINT.md`, `docs/FIRESTORE_USAGE_MODEL.md`, `docs/STRIPE_INTEGRATION.md`
- Observability: `docs/OBSERVABILITY.md` and importable Grafana dashboard at `assets/grafana/oscillink_dashboard.json`
- Image signing: `docs/IMAGE_SIGNING.md`
- Operations runbook: `docs/OPERATIONS.md`
- Networking & egress: `docs/NETWORKING.md`
- Pricing (licensed container): `docs/PRICING.md`
- OpenAPI baseline: `openapi_baseline.json`
- Whitepaper: Oscillink — A Symmetric Positive Definite Lattice for Scalable Working Memory & Hallucination Control (`OscillinkWhitepaper.tex`)
- Examples: `examples/quickstart.py`, `examples/diffusion_gated.py`
- Notebooks: `notebooks/`

<!-- Testimonial section removed for launch until we can include verifiable citations -->

## Security & compliance

- Security policy: see [`SECURITY.md`](SECURITY.md)
- Code of Conduct: see [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- Webhooks: keep `OSCILLINK_ALLOW_UNVERIFIED_STRIPE` off in production and set `STRIPE_WEBHOOK_SECRET`; the server enforces signature verification and timestamp freshness by default.
- Secrets: never commit `.env` files. Private env YAMLs under `cloud/` are git-ignored by default.

## Legal

- Terms of Service: [`docs/TERMS.md`](docs/TERMS.md)
- Privacy Policy: [`docs/PRIVACY.md`](docs/PRIVACY.md)
- Data Processing Addendum (DPA): [`docs/DPA.md`](docs/DPA.md)

### Patent and OSS usage (FAQ)

- Does “patent pending” affect my use?
	- No. Oscillink is open source under Apache‑2.0. You can use, modify, and distribute the software under that license. Apache‑2.0 includes an explicit patent license to practice the Work as contributed. See `LICENSE` for details.
- Why mention a patent at all?
	- Transparency and virtual marking. Our filing is primarily defensive—to deter bad‑faith, closed duplications and protect the project’s ability to go to market—not to restrict good‑faith OSS adoption.
- Can I use Oscillink commercially?
	- Yes, Apache‑2.0 permits commercial use. Please also review our Terms for brand, cloud, and service usage.

## Privacy & data handling

- SDK (local): does not transmit embeddings or content anywhere. All computation is in-process.
- Cloud API: only processes data sent in the request; no training or retention beyond the request lifecycle unless you explicitly enable caching on your deployment.
- Receipts: if requested, receipts include only derived numeric metrics (energy terms, timings) and a checksum of the lattice state, not raw content.
- Operators: to avoid accidental retention in logs, ensure request bodies are excluded from logging. If enabling Redis/Firestore caches, evaluate your data retention policies and regional storage requirements.

## Support & branding

- Contact: travisjohnson@oscillink.com
- General support: contact@oscillink.com · Security: security@oscillink.com
- Branding: Oscillink is a brand of Odin Protocol Inc. (trademark filing for Oscillink planned).

## Production stability and deprecation policy

- API stability: API v1 endpoints are stable. We will not introduce breaking changes to v1 without a deprecation cycle.
- Deprecation window: For any backward-incompatible changes, we commit to a minimum 6-month deprecation window with clear CHANGELOG/Release notes and deprecation warnings.
- Semantic versioning: Minor/patch releases are backward-compatible for the SDK; breaking changes only in major versions.

## Status and uptime (cloud beta)

- Uptime SLO: During beta, we target 99.5% monthly uptime for the hosted API.
- Status page: Coming soon. Until then, we will post incidents and maintenance windows via GitHub Releases/Discussions and respond via contact@.

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
	- Set `OSCILLINK_STATE_BACKEND=redis` and provide `OSCILLINK_REDIS_URL` (or `REDIS_URL`); see `docs/REDIS_BACKEND.md`.
	- For CLI pairing sessions specifically, also set `OSCILLINK_CLI_SESSIONS_BACKEND=redis` (see “Redis-backed CLI sessions” above).

## Error taxonomy (quick reference)

- API 422 Unprocessable Entity
	- Payload shape/type mismatch (e.g., Y not list-of-list float); ensure `Y: (N,D)` and `psi: (D,)`
- API 429 Too Many Requests
	- Rate/quota exceeded; inspect `X-Quota-*`, `X-RateLimit-*`, `X-IPLimit-*` headers for reset timing
- API 403 Unauthorized
	- Missing/invalid `X-API-Key`, or suspended key
- SDK ValueError
	- Non-finite values or mismatched dimensionality; ensure float32, unit-ish norms recommended
- Webhook signature error
	- Verify `STRIPE_WEBHOOK_SECRET`, check server clock, and `OSCILLINK_STRIPE_MAX_AGE`

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

---

[1] Hallucination headline details: see notebook `notebooks/04_hallucination_reduction.ipynb` (dataset card with N, k, trials, seed) and the CLI sample plot at `assets/benchmarks/competitor_single.png`.

---

## Pricing (licensed container)

Simple per-container licensing with clear entitlements and an enterprise cluster option.

- Starter: $49/container · Team: $199/container · Scale Pack: $699 (5 containers) · Enterprise: $2k–$6k/cluster
- Dev (free) for labs/evaluation with caps
- Annual discount: 2 months free; Founding cohort: first 100 buyers lock pricing for 12 months

See full details, market anchors, and entitlements: `docs/PRICING.md`.
