# Oscillink — Self‑Optimizing Coherent Memory for Embedding Workflows

Build coherence into retrieval and generation. Deterministic receipts for every decision. Latency that scales gracefully with corpus size.

<p align="left">
	<a href="https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml/badge.svg?branch=main"/></a>
	<a href="https://github.com/Maverick0351a/Oscillink/actions/workflows/codeql.yml"><img alt="CodeQL" src="https://github.com/Maverick0351a/Oscillink/actions/workflows/codeql.yml/badge.svg?branch=main"/></a>
	<a href="https://github.com/Maverick0351a/Oscillink/actions/workflows/pip-audit.yml"><img alt="pip-audit" src="https://github.com/Maverick0351a/Oscillink/actions/workflows/pip-audit.yml/badge.svg?branch=main"/></a>
	<a href="https://app.codecov.io/gh/Maverick0351a/Oscillink/tree/main"><img alt="Coverage" src="https://codecov.io/gh/Maverick0351a/Oscillink/branch/main/graph/badge.svg"/></a>
	<a href="https://pypi.org/project/oscillink/"><img alt="PyPI" src="https://img.shields.io/pypi/v/oscillink.svg"/></a>
	<a href="https://pypi.org/project/oscillink/"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/oscillink"/></a>
	<a href="https://pypi.org/project/oscillink/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/oscillink.svg"/></a>
	<a href="LICENSE"><img alt="License" src="https://img.shields.io/pypi/l/oscillink.svg"/></a>
	<a href="PATENTS.md"><img alt="Patent pending" src="https://img.shields.io/badge/Patent-pending-blueviolet.svg"/></a>
	<a href="docs/foundations/MATH_OVERVIEW.md"><img alt="Math" src="https://img.shields.io/badge/Math-Math%20Overview-4B8BBE"/></a>
	<br/>
	<sub>CI: Python 3.10–3.12 × NumPy 1.x/2.x</sub>

</p>

<p align="center"><img alt="Oscillink" src="assets/oscillink_hero.png" width="640"/></p>

<p align="center">
  <a href="https://pypi.org/project/oscillink/"><img alt="pip install oscillink" src="https://img.shields.io/badge/pip%20install-oscillink-3776AB?logo=pypi&logoColor=white"/></a>
</p>

<p align="center"><b>A physics‑inspired, model‑free coherence layer that transforms candidate embeddings into an explainable working‑memory state via convex energy minimization. Deterministic receipts for audit. Conjugate‑gradient solve with SPD guarantees.</b><br/>
<code>pip install oscillink</code></p>

Setup: synthetic “facts + traps” dataset — see the notebook for N, k, trials, seed. Reproducible via `notebooks/04_hallucination_reduction.ipynb`. Traps flagged; gate=0.01, off‑topic damp=0.5.

- Latency scales smoothly: with fixed D, k, and CG tol, settle tends to remain stable with denser graphs. Reference E2E < 40 ms at N≈1200 on a laptop CPU.
- Hallucination control: in our controlled study[1], Oscillink reduced trap selections while maintaining competitive F1. See the notebook for setup and reproduction.
- Receipts: SHA‑256 state signature; optional HMAC‑SHA256. [schema](docs/reference/RECEIPTS.md)
- Universal: works with any embeddings, no retraining. [adapters](#adapters--model-compatibility) · [quickstart](examples/quickstart.py)
- Self‑optimizing: learns parameters over time. [adaptive suite](scripts/bench_adaptive_suite.py)
- Production: scales to millions. See scripts under `scripts/` for reproducible benchmarks.

<p>
	<a href="#quickstart">Get Started</a> · <a href="docs/reference/API.md">API Docs</a> · <a href="#proven-results">See Results</a> · <a href="notebooks/">Live Demos</a>
</p>

<p align="center">
	<a href="https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml">CI</a> ·
	<a href="https://pypi.org/project/oscillink/">PyPI</a> ·
	<a href="docs/reference/API.md">API</a> ·
	<a href="docs/foundations/MATH_OVERVIEW.md">Math</a> ·
	<a href="docs/reference/RECEIPTS.md">Receipts</a> ·
	<a href="benchmarks/">Benchmarks</a> ·
	<a href="notebooks/">Notebooks</a>
</p>

## Use Oscillink: SDK or Licensed Container

- SDK (pip): Keep everything local. Best for adding coherence directly inside your application process. Start at [Install](#install) and [60‑second SDK quickstart](#60-second-sdk-quickstart).
- Licensed Container (customer‑managed): Run the API entirely inside your VPC or cluster with license‑based entitlements. Jump to [Licensed Container (customer‑managed)](#licensed-container-customer-managed). Operators can then dive into [Operations](docs/ops/OPERATIONS.md) and [Networking](docs/ops/NETWORKING.md).

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

<p align="center"><i>API v1 • Cloud: beta</i></p>

> Prefer containers? Use the licensed container for easy distribution inside your VPC or laptop. Try `docker compose -f deploy/docker-compose.yml up -d` after placing your license at `deploy/license/oscillink.lic`. See the “Licensed Container (customer-managed)” section below.

---

## TL;DR
- Coherent memory for any model (LLMs, image, video, audio, 3D) — no retraining.
- Deterministic receipts for auditability and reproducibility.
- SPD system with guaranteed convergence; typical E2E latency < 40 ms at N≈1200.
- Coherence‑first retrieval that works alongside RAG — see metrics below.

Quick links:
- [SDK Quickstart](#quickstart) · [API + Receipts](docs/reference/API.md) · [Cloud (beta)](#use-the-cloud)


## Overview

Oscillink builds a transient graph (mutual‑kNN) over input embeddings and settles to a globally coherent state by minimizing a strictly convex energy. The stationary point is unique (SPD), reproducible, and yields:

- A refined representation U* for the candidate set
- An energy receipt ΔH (with term breakdown)
- Null‑point diagnostics (edge‑level incoherence)
- Optional chain priors for path‑constrained reasoning

No training is required; your embeddings are the model.


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

Minimal 60s example:
```bash
pip install oscillink
python - <<'PY'
import numpy as np
from oscillink import Oscillink

# N x D anchors (float32), D-dim query
Y = np.random.randn(80,128).astype('float32')
psi = (Y[:10].mean(0) / (np.linalg.norm(Y[:10].mean(0)) + 1e-9)).astype('float32')

lat = Oscillink(Y, kneighbors=6, lamG=1.0, lamC=0.5, lamQ=4.0)
lat.set_query(psi)
lat.settle()

print("top-5:", [(b["id"], round(b["score"],3)) for b in lat.bundle(k=5)])
print("ΔH:", lat.receipt()["deltaH_total"])
PY
```

System requirements: Python 3.10–3.12, NumPy ≥1.22 (1.x/2.x supported). CPU only.

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
# Oscillink — Self‑Optimizing Coherent Memory for Embedding Workflows

Build coherence into retrieval and generation. Deterministic receipts for every decision. Latency that scales gracefully with corpus size.

<p align="left">
  <a href="https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml/badge.svg?branch=main"/></a>
  <a href="https://github.com/Maverick0351a/Oscillink/actions/workflows/codeql.yml"><img alt="CodeQL" src="https://github.com/Maverick0351a/Oscillink/actions/workflows/codeql.yml/badge.svg?branch=main"/></a>
  <a href="https://github.com/Maverick0351a/Oscillink/actions/workflows/pip-audit.yml"><img alt="pip-audit" src="https://github.com/Maverick0351a/Oscillink/actions/workflows/pip-audit.yml/badge.svg?branch=main"/></a>
  <a href="https://app.codecov.io/gh/Maverick0351a/Oscillink/tree/main"><img alt="Coverage" src="https://codecov.io/gh/Maverick0351a/Oscillink/branch/main/graph/badge.svg"/></a>
  <a href="https://pypi.org/project/oscillink/"><img alt="PyPI" src="https://img.shields.io/pypi/v/oscillink.svg"/></a>
  <a href="https://pypi.org/project/oscillink/"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/oscillink"/></a>
  <a href="https://pypi.org/project/oscillink/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/oscillink.svg"/></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/pypi/l/oscillink.svg"/></a>
  <a href="PATENTS.md"><img alt="Patent pending" src="https://img.shields.io/badge/Patent-pending-blueviolet.svg"/></a>
  <a href="docs/foundations/MATH_OVERVIEW.md"><img alt="Math" src="https://img.shields.io/badge/Math-Math%20Overview-4B8BBE"/></a>
  <br/>
  <sub>CI: Python 3.10–3.12 · NumPy 1.x/2.x</sub>
</p>

<p align="center"><img alt="Oscillink" src="assets/oscillink_hero.png" width="640"/></p>

Abstract. Oscillink is a physics‑inspired, model‑free coherence layer that transforms a set of candidate embeddings into an explainable working‑memory state by minimizing a strictly convex energy on a mutual‑kNN lattice. The system solves a symmetric positive‑definite (SPD) linear system via preconditioned conjugate gradients (CG), returning (i) refined embeddings U*, (ii) a deterministic, signed “receipt” with energy improvement ΔH, and (iii) per‑edge/null‑point diagnostics. No training is required; your embeddings are the model.

Install: `pip install oscillink` · Docs: [API](docs/reference/API.md) · Math: [Overview](docs/foundations/MATH_OVERVIEW.md) · Receipts: [Schema](docs/reference/RECEIPTS.md)

---

## Contents

- Overview
- Quickstart
- Adapters & Compatibility
- Reproducibility
- Performance
- Method (Technical)
- Deployment Options
- Security, Privacy, Legal
- Troubleshooting
- Contributing & License
- Changelog

---

## Overview

Oscillink builds a transient mutual‑kNN graph over N×D anchors `Y` and settles to a unique global optimum of a convex energy. The result is a coherent bundle of items for a query `psi`, along with an auditable “receipt” that explains the decision.

Outputs per run:

- Refined state U* (coherent embeddings)
- Receipt with ΔH (energy drop), term breakdown, CG iterations/residuals, and a SHA‑256 state signature (optional HMAC)
- Diagnostics: null points (incoherent edges), chain/prior scores (optional)

The system is deterministic, explainable, and training‑free.

---

## Quickstart

Minimal, 60 seconds:

```bash
pip install oscillink
python - <<'PY'
import numpy as np
from oscillink import Oscillink

# Anchors (N x D, float32) and a D-dim query
Y = np.random.randn(80,128).astype('float32')
psi = (Y[:10].mean(0) / (np.linalg.norm(Y[:10].mean(0))+1e-9)).astype('float32')

lat = Oscillink(Y, kneighbors=6, lamG=1.0, lamC=0.5, lamQ=4.0)
lat.set_query(psi)
lat.settle()

print("top-5:", [(b.get("id", i), round(b["score"],3)) for i,b in enumerate(lat.bundle(k=5))])
print("ΔH:", lat.receipt()["deltaH_total"])
PY
```

System requirements: Python 3.10–3.12, NumPy ≥ 1.22 (1.x/2.x supported). CPU only.

SDK at a glance:

```python
from oscillink import Oscillink

# Y: (N, D) embeddings; psi: (D,) query (float32 recommended)
lattice = Oscillink(Y, kneighbors=6)
lattice.set_query(psi)
lattice.settle()
bundle  = lattice.bundle(k=5)   # coherent top-k
receipt = lattice.receipt()     # deterministic audit (energy breakdown)
```

---

## Adapters & Compatibility

- Embeddings: Bring your own (OpenAI, Cohere, Sentence‑Transformers, local). Shapes: `Y: (N, D)`, `psi: (D,)`, preferably near unit‑norm float32.
- Dependencies: NumPy + small helpers; no framework lock‑in.
- Adapters: see `oscillink.adapters.*` for simple text embedding utilities.
- Tuning: Practical knobs are `kneighbors`, `lamC`, `lamQ`. CLI `--tune` and the adaptive suite offer data‑driven defaults.
- Preprocessing: optional `smart_correct` can reduce incidental traps in noisy inputs.
- Platforms: Windows, macOS, Linux.

---

## Reproducibility

Claim → how to reproduce (scripts write JSON or plots):

| Claim | How to reproduce |
|---|---|
| Coherence vs cosine baseline on provided sample | `python scripts/competitor_benchmark.py --input examples/real_benchmark_sample.jsonl --format jsonl --text-col text --id-col id --label-col label --trap-col trap --query-index 0 --k 5 --json --out benchmarks/competitor_sample.json` then `python scripts/plot_benchmarks.py --competitor benchmarks/competitor_sample.json --out-dir assets/benchmarks` |
| Scaling timing vs N | `python scripts/scale_benchmark.py --N 400 800 1200 --D 64 --k 6 --trials 2 > benchmarks/scale.jsonl` then `python scripts/plot_benchmarks.py --scale benchmarks/scale.jsonl --out-dir assets/benchmarks` |
| Receipt proof (energy, signature) | `pip install -e .[dev]` then `python scripts/proof_hallucination.py --seed 123 --n 1200 --d 128` |

Notes: The “facts + traps” setup used in the notebook is a controlled study to demonstrate controllability and auditability. Evaluate on your corpus before drawing production conclusions.

---

## Performance

Reference (laptop, Python 3.11, NumPy BLAS, N≈1200, light receipts):

- Graph build: ~18 ms
- Settle: ~10 ms
- Receipt: ~3 ms

Total: under ~40 ms.

As N grows (fixed D, k, tol), CG typically converges in 3–4 iterations with Jacobi preconditioning; end‑to‑end times scale smoothly.

Complexity: one matvec is O(Nk); overall solve ≈ O(D · cg_iters · N · k).

---

## Method (Technical)

Oscillink minimizes a convex energy on a mutual‑kNN lattice:

$$
H(U)=\lambda_G\|U-Y\|_F^2+\lambda_C\,\mathrm{tr}(U^\top L_{\mathrm{sym}}U)+\lambda_Q\,\mathrm{tr}\big((U-\mathbf{1}\psi^\top)^\top\,B\,(U-\mathbf{1}\psi^\top)\big)
$$

with optional path prior term $\lambda_P L_{\text{path}}$. Normal equations yield an SPD system $MU^*=F$ with

$$
M=\lambda_G I+\lambda_C L_{\mathrm{sym}}+\lambda_Q B+\lambda_P L_{\text{path}},\quad F=\lambda_G Y+\lambda_Q B\,\mathbf{1}\psi^\top.
$$

Properties. If $\lambda_G>0$ and $L_{\mathrm{sym}}, B, L_{\text{path}}$ are symmetric PSD, then $M$ is SPD ⇒ unique minimizer $U^*$. We solve with Jacobi‑preconditioned CG. Each run emits a receipt with $\Delta H$, per‑term energy, CG stats, and a SHA‑256 state signature (optional HMAC).

More detail: [Foundations / Math Overview](docs/foundations/MATH_OVERVIEW.md)

---

## Deployment Options

### A. SDK (local)

Keep everything in‑process. See Install and examples under `examples/`.

### B. Licensed container (customer‑managed)

Run the API entirely inside your VPC/cluster. No embeddings or content leave your network; only license/usage heartbeats if enabled.

<details>
<summary>Container quickstart and operations (click to expand)</summary>

#### Quickstart (Docker)

1) Place a license at `deploy/license/oscillink.lic` (Ed25519‑signed JWT).

```bash
docker compose -f deploy/docker-compose.yml up -d
```

API: http://localhost:8000 · health: `/health` · license status: `/license/status`

#### Required env

- `OSCILLINK_LICENSE_PATH` (default `/run/secrets/oscillink.lic`)
- `OSCILLINK_JWKS_URL` (e.g., https://license.oscillink.com/.well-known/jwks.json)

#### Optional

- `OSCILLINK_TELEMETRY=minimal` (aggregated counters only)
- `OSCILLINK_USAGE_LOG=/data/usage.jsonl`
- `OSCILLINK_USAGE_FLUSH_URL` (batch upload)
- JWT verification knobs: `OSCILLINK_JWT_ISS`, `OSCILLINK_JWT_AUD`, `OSCILLINK_JWT_LEEWAY`, `OSCILLINK_JWKS_TTL`, `OSCILLINK_JWKS_OFFLINE_GRACE`

#### Kubernetes (Helm)

- Chart skeleton at `deploy/helm/oscillink`.
- Mount secret to `/run/secrets/oscillink.lic`, set `OSCILLINK_LICENSE_PATH` and `OSCILLINK_JWKS_URL`.
- Optional hardening: NetworkPolicy, PDB, HPA, Ingress.

#### Operator introspection

```bash
# set OSCILLINK_ADMIN_SECRET before running the container
curl -H "X-Admin-Secret: $OSCILLINK_ADMIN_SECRET" http://localhost:8000/admin/introspect
```

Metrics protection: set `OSCILLINK_METRICS_PROTECTED=1` (requires X‑Admin‑Secret on `/metrics`).

</details>

### C. Cloud API (beta)

Opt‑in hosted API for convenience. Obtain an API key and call:

```python
import os, httpx
API_BASE = os.environ.get("OSCILLINK_API_BASE", "https://api2.odinprotocol.dev")
API_KEY  = os.environ["OSCILLINK_API_KEY"]
r = httpx.post(
    f"{API_BASE}/v1/settle",
    json={"Y": [[0.1,0.2]], "psi": [0.1,0.2], "options": {"bundle_k": 1, "include_receipt": True}},
    headers={"X-API-Key": API_KEY},
)
print(r.json())
```

Cloud feature flags, quotas, and Stripe onboarding are documented under `docs/`:

- Cloud architecture & ops: `docs/cloud/CLOUD_ARCH_GCP.md`, `docs/ops/REDIS_BACKEND.md`
- Billing: `docs/billing/STRIPE_INTEGRATION.md`, `docs/billing/PRICING.md`

---

## Security, Privacy, Legal

- Local SDK: does not transmit embeddings/content anywhere.
- Cloud API: processes only request payloads; no training/retention beyond request lifecycle unless explicit caching is enabled.
- Receipts: contain derived numeric metrics and a state checksum; not raw content.
- Webhooks: keep `OSCILLINK_ALLOW_UNVERIFIED_STRIPE=0` in production and set `STRIPE_WEBHOOK_SECRET`.

Policies: [Security](SECURITY.md) · [Privacy](docs/product/PRIVACY.md) · [Terms](docs/product/TERMS.md) · [License](LICENSE) · [Patent notice](PATENTS.md)

Patent & OSS usage (FAQ). Oscillink is open‑source (Apache‑2.0). Apache‑2.0 includes an explicit patent license to practice the contributed Work. Our filing is primarily defensive.

---

## Troubleshooting

- 422 Unprocessable Entity: ensure `Y: (N,D)` and `psi: (D,)` with finite float32 values.
- 403 Unauthorized: missing/invalid `X-API-Key` or suspended key.
- 429 Too Many Requests: rate/quota exceeded; see `X-Quota-*`, `X-RateLimit-*`, `X-IPLimit-*` headers (and `X-Monthly-*` if enabled).
- Webhook signature error: verify `STRIPE_WEBHOOK_SECRET`, server clock, and `OSCILLINK_STRIPE_MAX_AGE`.
- Redis not used: set `OSCILLINK_STATE_BACKEND=redis` and `OSCILLINK_REDIS_URL` (or `REDIS_URL`).

---

## Contributing & License

- License: Apache‑2.0 (see `LICENSE`)
- Contributions welcome (see `CONTRIBUTING.md`)
- Code of Conduct: `CODE_OF_CONDUCT.md`

Support & contacts. General: contact@oscillink.com · Security: security@oscillink.com · Founder: travisjohnson@oscillink.com. Brand: Oscillink is a brand of Odin Protocol Inc.

---

## Changelog

See `CHANGELOG.md` for release notes. Status: API v1 (stable) · Cloud: beta.

---

## Appendix: Datasets and Notebooks

Controlled “facts + traps” dataset and notebook are provided to demonstrate controllability/auditability:

- Notebook: `notebooks/04_hallucination_reduction.ipynb`
- Dataset card (N, k, trials, seed) and CLI plot script: `assets/benchmarks/` and `scripts/plot_benchmarks.py`

---

## Pricing (licensed container)

Simple, per‑container licensing with an enterprise cluster option (see details in `docs/billing/PRICING.md`):

- Starter: $49/container
- Team: $199/container
- Scale Pack: $699 (5 containers)
- Enterprise: $2k–$6k/cluster

Dev (free) for labs/evaluation with caps; annual discount available.

Notes on evaluation design: the hallucination/trap studies are synthetic/controlled and intended to validate auditability and controls, not to claim universal guarantees. For production, validate on your own corpus using the provided harness and receipts.

---

## References

- Math & foundations: `docs/foundations/MATH_OVERVIEW.md`, `docs/foundations/PHYSICS_FOUNDATIONS.md`
- Receipts: `docs/reference/RECEIPTS.md`, `docs/reference/RECEIPT_SCHEMA.md`
- API reference and OpenAPI baseline: `docs/reference/API.md`, `openapi_baseline.json`
- Benchmarks & scripts: `benchmarks/`, `scripts/`, `examples/`, `notebooks/`

Acknowledgments: This repository uses standard sparse linear algebra techniques (mutual‑kNN graphs, Laplacians, SPD solvers) adapted for explainable coherence in embedding workflows.

Contact: For design‑partner inquiries (on‑prem container, regulated deployments): contact@oscillink.com.
