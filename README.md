# Oscillink  Coherent — Scalable Working Memory & Hallucination Suppression

> **Attach to any generative model.** Drop in after initial retrieval or candidate generation to produce an *explainable*, globally coherent working memory state.
>
> **Replace brittle RAG heuristics.** Move from ad‑hoc top‑k filters to a physics‑based lattice that minimizes energy and produces signed receipts (ΔH, null points, chain verdicts).
>
> **Scale to lattice‑of‑lattices.** The same SPD contract composes hierarchically (see `docs/SCALING.md`)—from a few hundred nodes to layered shard summaries with virtually no architectural rewrite.
>
> **Controllable hallucination suppression.** Gate low‑trust sources → observed 42.9% → 0% hallucination rate in controlled fact retrieval (Notebook 04) with improved F1 (+65%).

<p align="center">
<b>60‑Second Try:</b>
</p>

```bash
pip install oscillink-lattice
python - <<'PY'
import numpy as np
from oscillink import OscillinkLattice
Y = np.random.randn(80,128).astype('float32')
psi = (Y[:10].mean(0)/ (np.linalg.norm(Y[:10].mean(0))+1e-9)).astype('float32')
lat = OscillinkLattice(Y, kneighbors=6, lamG=1.0, lamC=0.5, lamQ=4.0)
lat.set_query(psi)
lat.settle()
print(lat.bundle(k=5))
print(lat.receipt()['deltaH_total'])
PY
```

### Who it's for

- Teams building RAG/agent systems that need explainable, deterministic reranking beyond heuristics
- Product owners who care about controllable hallucination suppression and transparent receipts
- Infra/ML engineers who want a fast, model‑free coherence layer with SPD guarantees

### Get started (2 steps)

1) Install: `pip install oscillink-lattice`
2) Run the quick demo: `python examples/quickstart.py` (or the Proof harness below)

**Minimal Hallucination Gating (flag & suppress):**

```python
import numpy as np
from oscillink import OscillinkLattice

corpus = [
  "mars has two moons phobos and deimos",
  "the capital of france is paris",
  "invented fake fact about moon cheese",  # low‑trust / hallucination trap
  "einstein developed general relativity",
  "another spurious claim about ancient laser pyramids"  # low‑trust
]

# simple hash embeddings (adapter will fallback similarly in notebooks)
rng = np.random.default_rng(0)
Y = rng.standard_normal((len(corpus), 96)).astype('float32')
psi = Y[0] / (np.linalg.norm(Y[0]) + 1e-9)

lat = OscillinkLattice(Y, kneighbors=4, lamG=1.0, lamC=0.4, lamQ=2.0)

gates = np.ones(len(corpus), dtype='float32')
for i, text in enumerate(corpus):
  if any(tok in text for tok in ["fake", "spurious"]):
    gates[i] = 0.01  # suppress suspected hallucination sources

lat.set_query(psi, gates=gates)
lat.settle()
bundle = lat.bundle(k=3)
print("Bundle IDs & scores:", [(b['id'], round(b['score'],3)) for b in bundle])
print("deltaH_total:", lat.receipt()['deltaH_total'])
```

**Fast Proof Snapshot** (controlled synthetic & fact tasks):

| Claim | Baseline | Lattice / Gated | Result |
|-------|----------|-----------------|--------|
| Hallucination rate | 42.9% | 0% | −42.9 pp (Notebook 04) |
| F1 (fact task) | 0.33 | 0.55 | +65% relative |
| Coherence energy (ΔH) | 11.18 | 5.54 | −50.5% (Notebook 03) |
| Diffusion gating ΔH (synthetic) | 18,564 | 13,502 | −~27% (benchmark script) |

Full reproduction: notebooks in `notebooks/`, benchmarking scripts in `scripts/`.

### Proof harness (CLI)

For a tiny, reproducible snapshot comparing baseline cosine vs lattice with gating, see the doc and runner:
- Doc: `docs/HALLUCINATION_PROOF.md`
- Runner: `python scripts/proof_hallucination.py --trials 20 --k 3 --seed 0 --json`

Reproducible demo configuration (random embeddings, diffusion gating):

```bash
python scripts/proof_hallucination.py --dataset mars --trials 60 --k 3 --seed 5 --diffusion --json
```

Sample output:

```json
{"trials":60,"k":3,"baseline_hallucination_rate":0.6333,"lattice_hallucination_rate":0.0,"baseline_f1_mean":0.5533,"lattice_f1_mean":0.6344}
```

In words: lattice reduced hallucination rate from ~63% to 0% while improving mean F1 (≈0.55 → ≈0.63) on this controlled setup.

Optional: include a softer metric (average trap share in top‑k) by adding `--trap-share`:

```bash
python scripts/proof_hallucination.py --dataset mars --trials 10 --k 3 --seed 5 --diffusion --trap-share --json
```

**At a Glance (Why Not Just RAG?):**

| Problem (Classic RAG) | Oscillink Approach |
|-----------------------|--------------------|
| Disconnected chunks | SPD energy‑minimized memory state |
| Opaque scoring | Signed receipts (ΔH, null points, chain verdicts) |
| Hallucination leakage | Controllable gating (0% in controlled test¹) |
| One‑off rerank | Continuous coherent settle & bundle diversification |
| Heuristic filtering | Physics‑informed energy shaping |

```
pip install oscillink-lattice  # works with any embeddings
```
Latency: single settle + receipt often ~10–15 ms at modest N on a laptop (see Performance section). Deterministic kNN + CG solve keeps variance low.

<sup>1</sup> 0% hallucination rate achieved in a synthetic, labeled fact retrieval notebook where low‑trust sources were gated; not a blanket production guarantee (see Hallucination Control section).

---

> A graph‑theoretic, SPD‑solved working memory that explains itself. Deep math & scaling details moved to `docs/MATH_OVERVIEW.md` and `docs/SCALING.md` to keep this README focused.

![CI](https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/oscillink-lattice.svg)
![License](https://img.shields.io/github/license/Maverick0351a/Oscillink.svg)
![Python](https://img.shields.io/pypi/pyversions/oscillink-lattice.svg)
![Coverage](https://codecov.io/gh/Maverick0351a/Oscillink/branch/main/graph/badge.svg)

<!-- Additional Health & Adoption Badges -->
![Downloads](https://img.shields.io/pypi/dm/oscillink-lattice.svg)
![Wheel](https://img.shields.io/pypi/wheel/oscillink-lattice.svg)
![Type Hints](https://img.shields.io/badge/type%20hints-PEP561-success)
![Ruff](https://img.shields.io/badge/lint-ruff-informational)
![Last Commit](https://img.shields.io/github/last-commit/Maverick0351a/Oscillink.svg)
![Issues](https://img.shields.io/github/issues/Maverick0351a/Oscillink.svg)
![Stars](https://img.shields.io/github/stars/Maverick0351a/Oscillink.svg)

<!-- If enabling later: Dependabot / Security / License scanning badges can go here. -->

<p align="center">
	<img src="assets/oscillink_hero.svg" alt="Oscillink Lattice – graph-theoretic SPD coherence layer" width="640" />
</p>

**Oscillink Lattice** is a small, fast, *physics‑inspired* coherence layer for generative / embedding workflows – providing structured, explainable short‑term memory without training.
It builds an ephemeral lattice (graph) over candidate vectors and **settles** to the most coherent
state by minimizing a convex energy with a **symmetric positive definite** (SPD) system.

- **Explainable**: exact energy receipts (ΔH) and null‑point diagnostics.
- **Model‑free**: no training — your vectors *are* the model.
- **Safe math**: normalized Laplacian; SPD ensures robust CG convergence.
- **Chain Priors**: encode expected reasoning paths; get **chain receipts** (verdicts, weakest link).

*Designed for:* LLM retrieval & reranking · agent trace consolidation · image/code embedding refinement · explainable short‑term working memory shaping.

*SPD condition:* λ_G > 0 with A, B ⪰ 0 ⇒ M = λ_G I + λ_C L_sym + λ_Q B + λ_P L_path ≻ 0 (CG‑friendly convergence guarantee).

> *Phase‑1 focus:* a pure SDK (no cloud, no data movement). Bring your own embeddings.

---

## 🚀 Key Results (Early Controlled Evaluations)

These headline numbers come from reproducible synthetic & small controlled fact retrieval experiments (see notebooks + scripts). They are **not** generalized production benchmarks yet; they demonstrate *controllability* and *explainability*.

- **0% hallucination rate** (vs 42.9% baseline) in a controlled fact retrieval task (Notebook 04) after gating known misinformation sources.
- **~50% coherence energy reduction** via semantic gating (Notebook 03: ΔH 11.18 → 5.54, −50.5%).
- **+65% F1 accuracy improvement** (0.33 → 0.55) with aggressive gating & structural tuning (lamQ↑, lamC↑) while eliminating hallucinations.
- **Precision uplift** (0.29 → 0.50) by excluding low‑trust nodes through near‑zero gates.
- **Statistical chain verification**: path verdicts + weakest link (z‑scores) for reasoning trajectories.

[Try it yourself →](notebooks/04_hallucination_reduction.ipynb) · [See the benchmarking & notebooks →](notebooks/03_constraint_query.ipynb)

> Nuance: “0% hallucinations” is achievable when false / low‑trust sources are labeled or heuristically flagged for gating. We describe this as **Hallucination Control** rather than a universal guarantee. Claims are reproducible with the provided notebook; broader generalization requires domain‑specific validation.

### Quick Copy (Marketing / Social)
“First physics‑based, SPD lattice to demonstrate controllable hallucination suppression: 42.9% → 0% rate, +65% F1, ~50% coherence energy improvement — all without training.”

---

## Hallucination Control (Controllable Suppression)

Oscillink treats hallucination reduction as an *energy shaping* problem:

1. **Gate assignment**: Assign near‑zero gates (e.g., 0.01) to low‑trust / flagged sources; mild damp (0.5–0.6) to off‑topic items; full weight (1.0) to high‑confidence facts.
2. **Coherence solve**: The SPD system minimizes energy subject to graph structure + query attraction only where permitted.
3. **Bundle selection**: Diversified scoring favors coherent, on‑topic, high‑alignment nodes – traps drop out when their gates suppress query pull.
4. **Receipt auditing**: ΔH + gating stats + null points provide a verifiable trace of what influenced the answer set.

**Controlled Test Outcome (Notebook 04):**

| Metric | Baseline (Cosine Top‑k) | Lattice (Aggressive Gating) | Delta |
|--------|-------------------------|-----------------------------|-------|
| Hallucination Rate | 42.86% | 0.00% | −42.86 pp |
| True Hits | 2 | 3 | +1 |
| Precision | 0.2857 | 0.5000 | +0.2143 |
| Recall | 0.4000 | 0.6000 | +0.2000 |
| F1 | 0.3333 | 0.5455 | +0.2122 (~+65%) |

**Interpretation:** With explicit source gating, Oscillink *eliminated* hallucinations in this synthetic fact task while improving both precision and recall. This demonstrates **controllability** — you can dial out misinformation vectors instead of only hoping a downstream language model refuses them.

> Honesty Note: Real‑world hallucinations are more diverse; this does **not** claim universal elimination. It shows a physics‑based lattice can enforce retrieval hygiene when provided weak supervision over source trust.

### How To Reproduce
```bash
python notebooks/04_hallucination_reduction.ipynb  # (open & run sequentially in Jupyter/Colab)
```
Or adapt the gating pattern:
```python
# gate assignment sketch
base_gate = np.ones(N, dtype=np.float32)
for i, text in enumerate(corpus_lower):
  if any(trigger in text for trigger in trap_signatures):
    base_gate[i] = 0.01  # suppress
  elif off_topic(text):
    base_gate[i] = 0.5   # mild damp
lat.set_query(psi, gates=base_gate)
receipt = lat.receipt()  # inspect receipt['meta']['gates_mean'] etc.
```

### Positioning vs “Just Filter”
Simple filtering removes items but loses auditability & can over‑prune. Oscillink instead **attenuates** influence via gates and *proves* the effect in the receipt (gating stats + ΔH). This preserves optional inclusion if conditions change (e.g., dynamic trust recalibration) while keeping hallucination pressure low.

### Roadmap (Hallucination Control)
- Automated gate inference (diffusion × semantic classifiers × provenance signals)
- Factuality‑weighted energy regularizer
- Multi‑query joint lattice to reduce cross‑turn drift
- Receipt extension: structured hallucination suppression rationale fields

---

## Quickstart
For LLM retrieval / reranking, agent trace consolidation, image or code embedding refinement.

### Installation

From PyPI (recommended):

```bash
pip install oscillink-lattice
```

Cloud + billing (Firestore + Stripe) extras:

```bash
pip install oscillink-lattice[cloud-all]
```

Or install separately:

```bash
pip install oscillink-lattice[cloud]
pip install oscillink-lattice[billing]
```

Editable install for local development / contributions:

```bash
git clone https://github.com/Maverick0351a/Oscillink.git
cd Oscillink
pip install -e .[dev]
```

Then continue with the quickstart below.

```bash
python -m venv .venv
source .venv/bin/activate   # (or .\.venv\Scripts\activate on Windows)
pip install -e .
pytest -q

# optional: run with coverage
pytest -q --cov=oscillink --cov-report=term-missing
```

### Stripe Price Mapping (Cloud Billing)

By default, the cloud webhook maps common test SKUs to tiers:

- `price_cloud_pro_monthly` → `pro`
- `price_cloud_enterprise` → `enterprise`

You can override or extend this via an environment variable:

```bash
# JSON form
set OSCILLINK_STRIPE_PRICE_MAP={"price_123":"pro","price_456":"enterprise"}

# or semicolon form
set OSCILLINK_STRIPE_PRICE_MAP=price_123:pro;price_456:enterprise
```

Notes:
- Env mapping overrides built-ins when keys collide.
- Enterprise tier defaults to `pending` status (requires manual activation) when processed by the webhook.
- For local testing without Stripe library/signature, set `OSCILLINK_ALLOW_UNVERIFIED_STRIPE=1` (never enable in production).
- If `STRIPE_WEBHOOK_SECRET` is set and the `stripe-signature` header is missing, events are only accepted when `OSCILLINK_ALLOW_UNVERIFIED_STRIPE=1`.
  In production, always provide a valid signature header.

See also: Admin diagnostics endpoint `/admin/billing/price-map` to inspect the merged active map and tier catalog.

### Pricing & Tiers (Cloud)

| Tier | Monthly unit cap (node·dim) | Diffusion gating | Activation |
|------|------------------------------|------------------|------------|
| free | 5,000,000                    | No               | Automatic  |
| pro  | 50,000,000                   | Yes              | Automatic  |
| enterprise | Unlimited              | Yes              | Manual (pending until activated) |

Notes:
- “Monthly unit cap” counts N × D per request and sums across the calendar month (UTC).
- Enterprise keys created via Stripe start as pending; an admin must activate the key.
- You can override or extend price→tier mapping via `OSCILLINK_STRIPE_PRICE_MAP`.

### Minimal example

Shapes & dtypes: Y: (N,D) float32, psi: (D,) float32, gates: (N,) float32 in [0,1].

```python
import numpy as np
from oscillink import OscillinkLattice

# synthetic anchors (N x D)
Y = np.random.randn(120, 128).astype(np.float32)
psi = (Y[:20].mean(axis=0) / (np.linalg.norm(Y[:20].mean(axis=0)) + 1e-12)).astype(np.float32)

lat = OscillinkLattice(Y, kneighbors=6, lamG=1.0, lamC=0.5, lamQ=4.0)
lat.set_query(psi=psi)

# add an expected chain prior (indices)
lat.add_chain(chain=[2,5,7,9], lamP=0.2)
lat.settle(dt=1.0, max_iters=12, tol=1e-3)

r = lat.receipt()
cr = lat.chain_receipt(chain=[2,5,7,9])
bundle = lat.bundle(k=6)

print(r["deltaH_total"], cr["verdict"], bundle[:3])
```

See `examples/quickstart.py` for a runnable demo.

---

### Advanced Gates (Screened Diffusion) *(Optional)*

For more adaptive query attraction you can derive gating weights via a screened diffusion process over the anchor graph:

Solve (L_sym + γ I) h = β s, where s is a non‑negative similarity source (cosine alignment with the query). The solution h (normalized to [0,1]) acts as spatially propagated “energy” indicating how strongly each node should align to the query.

```python
from oscillink import OscillinkLattice, compute_diffusion_gates
import numpy as np

Y = np.random.randn(400, 96).astype(np.float32)
psi = np.random.randn(96).astype(np.float32)

# Compute diffusion gates (screened Poisson solve)
gates = compute_diffusion_gates(Y, psi, kneighbors=6, beta=1.0, gamma=0.15)

lat = OscillinkLattice(Y, kneighbors=6)
lat.set_query(psi, gates=gates)
lat.settle()
rec = lat.receipt()
```

Tuning notes:
- Increase `gamma` → stronger screening (more local influence, flatter gates).
- Increase `beta` → amplifies high‑similarity injection regions.
- Keep `gamma > 0` for strict SPD; typical range 0.05–0.3.

This feature is optional; omitting it reverts to uniform gates (original behavior). In a future cloud tier it can power “physics‑informed preprocessing” without changing the core settle contract.

#### Receipt Gating Statistics (Experimental)

When a receipt is produced after setting a query (with or without custom gates) the following meta fields are included (Experimental tier):

| Field | Meaning |
|-------|---------|
| `gates_min` | Minimum gating weight after normalization (expect 0 with diffusion; 1 with uniform) |
| `gates_max` | Maximum gating weight (always 1 after normalization) |
| `gates_mean` | Mean gating weight across nodes (uniform = 1.0, diffusion < 1.0 unless constant source) |
| `gates_uniform` | Boolean convenience flag (`True` if all gates identical within tolerance) |

These help quickly distinguish whether adaptive gating materially reshaped query influence and can be logged for monitoring drift when experimenting with diffusion parameters (`beta`, `gamma`). Fields may evolve (naming or additional statistics) prior to promotion out of Experimental.

---

## What this SDK provides

- `OscillinkLattice`: build lattice, settle, receipts, chain_receipt, bundle.
- **Caching**: stationary solution `U*` cached & reused across diagnostics.
- **Export / Import**: `export_state()` / `OscillinkLattice.from_state()` (JSON) and binary `save_state(..., format='npz')` for reproducibility.
- **Receipt Meta & Version**: `receipt()` now returns `version` + `meta` (cache usage, signature, solve stats, convergence). See `docs/RECEIPTS.md`.
- **Callbacks**: register post‑settle diagnostics hooks via `add_settle_callback(fn)`.
- **Forced Refresh**: `refresh_Ustar()` to recompute stationary solution ignoring cache.
- **Benchmarking**: lightweight timing script in `scripts/bench.py`.
- Graph utilities: mutual‑kNN, row‑cap, normalized Laplacian, path Laplacian.
- Solver: Jacobi‑preconditioned CG (pure NumPy).
- Receipts: ΔH (trace identity), per‑node attribution, null‑points, diagnostics.
- Chain Priors: SPD‑safe path Laplacian; **chain verdict** + **weakest link**. Path prior adds λ_P tr(Uᵀ L_path U) where L_path is the Laplacian of the supplied chain (L_path ⪰ 0 preserves SPD).

**Docs:** see `docs/` for math spec, API, detailed [Receipt Schema](docs/RECEIPTS.md), chain guide, and roadmap.

---

### Feature Snapshot

| Capability | What it Means | Why it Matters |
|------------|---------------|----------------|
| Receipts (ΔH) | Decomposed energy improvement | Auditable, explainable ranking signal |
| Deterministic Signatures | Stable hash over structure & params | Repro + tamper detection |
| Chain Priors | Optional path Laplacian | Steer reasoning / narrative continuity |
| Null-Point Diagnostics | Edge-level anomaly surfacing | Spot incoherent nodes fast |
| Stationary Caching | Reuse U* across diagnostics | Lower latency for multiple queries |
| Binary + JSON Export | Round‑trip state & provenance | Persistence & offline analysis |
| Structured Logging | JSON event hooks | Integrate with observability stacks |
| Performance Scripts | Benchmarks, scaling, perf guard | Prevent silent regressions |
| HMAC Receipt Signing | Integrity sealing | Trust boundary enforcement |


## Performance (Indicative)

On a modern laptop (Python 3.11, NumPy default BLAS), a medium case `N=1200, D=128, k=8` typically reports (single trial):

```
graph_build_ms:   ~18
ustar_solve_ms:   ~40   (CG iters ≈ 25–35)
settle_ms:        ~6–10
receipt_ms:       ~3
```

Numbers vary with BLAS + hardware. Use `scripts/benchmark.py` to profile:

```bash
python scripts/benchmark.py --N 1200 --D 128 --kneighbors 8 --trials 3 --json
```

CI includes a permissive perf guard (`scripts/perf_check.py`). Tighten tolerance once the baseline stabilizes.


## Design principles

SPD guarantee: with A ≥ 0, B ≥ 0 and λ_G > 0 the system matrix M = λ_G I + λ_C L_sym + λ_Q B + λ_P L_path is symmetric positive‑definite (CG‑friendly).

- **Normalized Laplacian**: better conditioning across datasets.
- **SPD system**: \(M = \lambda_G I + \lambda_C L_\mathrm{sym} + \lambda_Q B + \lambda_P L_{path}\).
- **Implicit settle**: \((I+\Delta t M)U^+=U+\Delta t(\lambda_G Y + \lambda_Q B 1\psi^\top)\).
- **Receipts**: exact ΔH via trace identity; normalized residuals for null points.

### Symbol Map

| Code Param | Math | Meaning |
|------------|------|---------|
| lamG | λ_G | Anchor identity pull |
| lamC | λ_C | Graph coherence (L_sym) weight |
| lamQ | λ_Q | Query attraction weight |
| lamP | λ_P | Path (chain) prior weight |
| kneighbors | k | Mutual kNN parameter |

### Export / Import Example (with Provenance Hash)

```python
state = lat.export_state()
print(state['provenance'])  # stable provenance hash for reproducibility lineage
# persist JSON (e.g., json.dump)
lat2 = OscillinkLattice.from_state(state)
lat2.settle()
```

### Cached U* Example

```python
r1 = lat.receipt()        # computes U*
r2 = lat.bundle(k=5)      # reuses cached U*
print(lat.stats["ustar_solves"], lat.stats["ustar_cache_hits"])  # introspect cache
```

### Receipt Meta & Version

Each `receipt()` call now returns a structure:

```python
rec = lat.receipt()
print(rec["version"])   # e.g. "1.0"
print(rec["meta"])      # { 'ustar_cached': bool, 'signature': str, 'ustar_solves': int, ... }
```

`meta.signature` is a stable hash of lattice‑defining parameters (includes adjacency fingerprint & chain metadata); if it changes, the cached `U*` is invalidated automatically.

#### Convergence Fields

Each stationary solve records:

| Field | Meaning |
|-------|---------|
| `ustar_converged` | Residual <= tolerance for last stationary CG solve |
| `ustar_res` | Final residual (max norm across RHS columns) |
| `ustar_iters` | Iterations used |

These live in `receipt()['meta']` for observability & regression detection.

### Forcing a Fresh U*

If you mutate underlying data (or simply want to measure solve time again) you can force recomputation:

```python
lat.refresh_Ustar()   # invalidates cache & recomputes
rec2 = lat.receipt()
```

### Settle Callbacks

Register functions to observe settling progress / instrumentation. Each callback receives `(lattice, diagnostics_dict)` after a successful `settle()` step.

```python
def on_settle(lat, info):
	# e.g., log deltaH or residual norms
	print("ΔH", info.get("deltaH_total"))

lat.add_settle_callback(on_settle)
lat.settle(max_iters=3)
lat.remove_settle_callback(on_settle)
```

### Query & Gating API

You can supply a gating vector (per‑node relevance weights) either alongside the query or separately:

```python
lat.set_query(psi)                # sets psi (optionally gates if provided)
lat.set_gates(np.ones(lat.N))     # explicit gating (validates length)
```

### Chain Export / Import

Exported state preserves original chain ordering (`chain_nodes`) for exact path reconstruction. On import, the path Laplacian is rebuilt deterministically when chain metadata is present.

### Benchmark Script

A small script (`scripts/benchmark.py`) is provided to sanity‑check performance:

```bash
# human-readable summary
python scripts/benchmark.py --N 1500 --D 128 --kneighbors 8

# JSON mode for automation / CI
python scripts/benchmark.py --json --N 800 --D 64 --trials 2 > bench.json
```

It reports neighbor graph build, settle, and stationary solve timings plus ΔH. JSON mode adds per-trial and aggregate stats.

### Stats Introspection

Runtime counters accumulate in `lat.stats`:

```python
print(lat.stats)
# {'ustar_solves': int, 'ustar_cache_hits': int, 'last_signature': '...', ...}
```

### Deterministic Neighbor Construction

Pass `deterministic_k=True` to force a stable full sort (tie‑break by index) when building the mutual‑kNN graph:

```python
lat = OscillinkLattice(Y, kneighbors=6, deterministic_k=True)
```

Alternatively, provide `neighbor_seed` to keep fast partitioning while adding a minuscule jitter for reproducible tie resolution:

```python
lat = OscillinkLattice(Y, kneighbors=6, neighbor_seed=1234)
```

Export/import preserves `kneighbors`, `deterministic_k`, and `neighbor_seed`.
If `kneighbors >= N` the implementation safely clamps to `N-1` to avoid internal partition errors while preserving determinism. Responses surface both `kneighbors_requested` and `kneighbors_effective` in `meta`.

Failure mode transparency: if CG reaches `max_iters` before residual <= tol, `ustar_converged=False` is surfaced (receipt still returned) so you can adjust parameters and retry.

### Lightweight Logging Adapter

Attach any callable `(event: str, payload: dict)` to observe lattice lifecycle events (`init`, `settle`, `ustar_solve`, `ustar_cache_hit`, `receipt`, `add_chain`, `clear_chain`, `refresh_ustar`, `invalidate_cache`).

```python
events = []
lat.set_logger(lambda ev, data: events.append((ev, data)))
lat.settle(max_iters=4)
lat.receipt()
print(events[:3])
```

Detach by `lat.set_logger(None)`.

### Provenance Hash

`export_state()` includes `provenance`, a digest over core arrays (Y, ψ, gating vector coerced to float32), key parameters, and an adjacency fingerprint to enable integrity / reproducibility checks. Changing dtype or node ordering alters the hash.

Receipt `meta` also includes adjacency statistics: `avg_degree`, `edge_density`.

### Why Receipts?

Receipts are structured, reproducible diagnostics that make each lattice invocation **auditable**:
- Deterministic signature (`state_sig`) couples parameters + adjacency fingerprint + chain metadata.
- ΔH decomposition quantifies how much the system *optimized* coherence vs anchors vs query pull.
- Convergence + timing fields (`ustar_iters`, `ustar_res`, `ustar_solve_ms`, `graph_build_ms`) allow regression tracking.
- Optional HMAC signing delivers tamper‑evident integrity for downstream pipelines or caching layers.

See `docs/RECEIPT_SCHEMA.md` for the authoritative field list.

#### Null-Point Capping & Summary (New)

Large lattices can surface many null points (edge-level incoherence diagnostics). To bound payload size you can cap the number of emitted entries by setting the environment variable `OSCILLINK_RECEIPT_NULL_CAP` to an integer > 0. When active:

* Only the top-K highest z-score null points are returned.
* A `null_points_summary` object is added under `receipt['meta']`:

```
{
  "total": <int>,          # total null points detected before capping
  "returned": <int>,       # number actually returned (<= cap)
  "capped": <bool>         # true if truncation occurred
}
```

If unset or `0`, all null points are returned (original behavior). This is especially useful for cloud responses and log hygiene.

#### Release / Tagging Guidance
When cutting a release:
1. Update `CHANGELOG.md` (move Unreleased entries under a new version heading).
2. Bump `version` in `pyproject.toml` & `oscillink/__init__.__version__`.
3. Run the test & benchmark sanity checks (`pytest -q`, `python scripts/benchmark.py --json --N 400 --D 64 --trials 1`).
4. Tag and push: `git tag vX.Y.Z && git push --tags`.
5. (Optional) Publish to PyPI.

### Bundle Ranking Example

`bundle(k)` blends coherence anomaly (z-scored drop) with alignment to the query embedding and applies a simple MMR diversification.

```python
bundle = lat.bundle(k=5)
for item in bundle:
	print(item['id'], item['score'], item['align'])
```

Each entry includes:
- `id`: node index
- `score`: combined ranking score
- `align`: cosine alignment with query embedding


### Receipt Signing (Integrity)

Optionally sign receipts with HMAC‑SHA256 over (`state_sig` || `deltaH_total` || `version`)—rotate secrets to revoke historical trust.

```python
lat.set_receipt_secret("my-shared-secret")
rec = lat.receipt()
print(rec['meta']['signature'])  # { algorithm, payload, signature }
```

Changing lattice state (e.g., adding a chain, updating gates/query) alters the internal `state_sig` and produces a new receipt signature. Omit the secret (or pass `None`) to disable signing.

#### Verifying a Signed Receipt

Use the helper `verify_receipt` (package export) or call `lat.verify_current_receipt` (convenience) to validate integrity:

```python
from oscillink import verify_receipt
rec = lat.receipt()
assert verify_receipt(rec, "my-shared-secret")
assert lat.verify_current_receipt("my-shared-secret")
```

If payload fields or the signature are tampered with the verification returns `False`.

For mixed environments, the extended helper allows enforcing or down‑scoping modes:

```python
from oscillink.core.receipts import verify_receipt_mode
ok, payload = verify_receipt_mode(rec, "my-shared-secret", require_mode=None, minimal_subset=True)
```

- `require_mode='extended'` ensures only extended payloads pass.
- `minimal_subset=True` lets you accept an extended payload while verifying only the minimal subset (compatibility mode), returning the reduced payload if minimal verification succeeds.

#### Signature Scope (Current Minimal Payload)

The HMAC payload currently covers only two fields:

```
{
	"state_sig": <deterministic lattice signature>,
	"deltaH_total": <float>
}
```

Rationale:
1. `state_sig` already commits to Y adjacency fingerprint, query/gates (rounded), λ parameters, chain presence & ordering (length), and neighbor construction parameters. Any structural mutation or parameter drift changes this hash.
2. `deltaH_total` is the principal scalar optimization outcome (coherence improvement) consumers may want to trust. Including it prevents replay of a prior improvement value for the same structural state.

Excluded (for now): iteration counts, residuals, timing metrics, per-node diagnostics. These are useful operationally but can evolve (naming / semantics) and would cause unnecessary signature churn. They may be promoted into the signed payload later under a versioned scheme.

Extension Path:
- Add `version` and a bounded set of convergence stats (`ustar_res`, `ustar_iters`) behind a minor release after documenting stability.
- Introduce a `sig_v` field to allow additive expansion without breaking existing verifiers.

If you need a broader integrity envelope today, verify the receipt JSON externally by recomputing `state_sig` via a fresh lattice reconstruction (using exported state) and comparing numeric fields within tolerance. A roadmap item tracks potential expansion of the signing scope.

##### Extended Signature Mode (New)

You can opt-in to a richer signed payload that includes solver convergence stats and parameter provenance:

```python
lat.set_receipt_secret("shared-secret")
lat.set_signature_mode("extended")  # or "minimal" (default)
rec = lat.receipt()
print(rec['meta']['signature']['payload'])
```

Extended payload shape:

```jsonc
{
  "sig_v": 1,
  "mode": "extended",
  "state_sig": "...",
  "deltaH_total": 12.34,
  "ustar_iters": 17,
  "ustar_res": 0.00042,
  "ustar_converged": true,
  "params": {"lamG":1.0,"lamC":0.5,"lamQ":4.0,"lamP":0.0},
  "graph": {"k":6, "deterministic_k":true, "neighbor_seed":123}
}
```

Minimal mode payload for comparison:

```jsonc
{
  "sig_v": 1,
  "mode": "minimal",
  "state_sig": "...",
  "deltaH_total": 12.34
}
```

`sig_v` (signature schema version) lets future releases add fields while older verifiers can branch on version. Current value: 1.

---
## Positioning vs Vector DBs & Rerankers

Oscillink is a transient coherence/refinement layer, not a store or heavy neural reranker.

- Bring your own candidate embeddings (often from a vector DB retrieval step).
- Apply Oscillink to induce a globally coherent adjustment and receive structured receipts.
- Feed the refined bundle or chain verdict downstream (generation, reasoning, routing).

It complements: (1) vector DBs for scalable recall, (2) cross‑encoders / rerankers for semantic precision. Use Oscillink when you need *explainable* short‑term memory shaping with deterministic math.

---

## Open Core & Cloud (Phase 2 Preview)

The SDK stays Apache‑2.0 and self‑contained. Cloud functionality is strictly opt‑in (no network use unless you run the service). A lightweight cloud layer (FastAPI) ships with:

- Hosted settlement & receipts (stateless per request; embeddings not persisted)
- API key authentication (header `x-api-key`)
- Request correlation header (`x-request-id` echo)
- Prometheus metrics endpoint (`/metrics`)
- Usage metering (nodes + node_dim_units) surfaced in every response
- Global rate limiting (configurable)
- Per‑API key quota (node‑dimension units over a window)
- Async job submission & polling for larger workloads
- Monthly tier caps (node·dim units per calendar month)
- Admin key management (manual enterprise activation / overrides)

Planned (not yet implemented):

- Persistent usage log export (JSONL)
- OpenAPI spec export script & published schema artifact
- Optional distributed quota backend (e.g., Redis) for multi‑replica deployment
- Multi‑tenant usage billing webhooks / signed usage receipts

### Running the Cloud Service Locally

Install with cloud extras:

```bash
pip install -e .[cloud-all]
```

Run with Uvicorn:

```bash
uvicorn cloud.app.main:app --reload --port 8000
```

Health check:

```bash
curl -s http://localhost:8000/health | jq
```

### Docker

Build and run the container (defaults to port 8000):

```bash
docker build -t oscillink-cloud .
docker run --rm -p 8000:8000 oscillink-cloud
```

### Deploy to Cloud Run

Container is Cloud Run‑ready. The service listens on `PORT` (defaults to 8000 locally). Recommended steps:

1) Build & push the image

```bash
# Artifact Registry (recommended)
gcloud builds submit --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/oscillink/oscillink-cloud:latest"
```

2) Deploy to Cloud Run

```bash
gcloud run deploy oscillink-cloud \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/oscillink/oscillink-cloud:latest" \
  --platform=managed --region=${REGION} --allow-unauthenticated \
  --cpu=1 --memory=1Gi --max-instances=10 \
  --set-env-vars=OSCILLINK_KEYSTORE_BACKEND=firestore,OSCILLINK_FIRESTORE_COLLECTION=oscillink_api_keys \
  --set-env-vars=OSCILLINK_MONTHLY_USAGE_COLLECTION=oscillink_monthly_usage,OSCILLINK_WEBHOOK_EVENTS_COLLECTION=oscillink_webhook_events \
  --set-env-vars=STRIPE_WEBHOOK_SECRET=whsec_xxx,OSCILLINK_ADMIN_SECRET=change_me
```

3) Service account & permissions

- Grant the Cloud Run runtime service account Firestore access (Datastore User / Firestore roles)
- Provide Stripe webhook secret (never commit it); configure your Stripe endpoint URL to `/stripe/webhook`

4) Smoke test

```bash
curl -s "https://YOUR_SERVICE_URL/health"
```

### OpenAPI Schema Export

Generate the current OpenAPI spec (writes `openapi.json`):

```bash
python scripts/export_openapi.py --out openapi.json
```

In CI, the workflow exports and uploads this as an artifact (`openapi-schema`). Downstream tooling (SDK generation, diff checks) can retrieve the artifact per build.

You can publish this artifact (e.g., attach to a release) or diff it in CI to detect breaking interface changes.

#### OpenAPI Contract Gating (CI)

Pull requests invoke `scripts/check_openapi_diff_simple.py` to ensure no existing path or HTTP method is removed (additions allowed). The check fails the build on deletions, providing an early guardrail for accidental breaking changes. Future enhancement will fetch the prior main branch artifact instead of a same-build fallback.

### Performance Baseline & Regression Checks (Experimental)

The script `scripts/perf_check.py` compares current run timings against a JSON baseline (`scripts/perf_baseline.json`). In CI it is non-blocking (logs variance); for stricter gating you can fail on regression by removing the fallback `|| echo` segment in the workflow.

To refresh the baseline (after intentional perf improvement):

```bash
python scripts/perf_snapshot.py --out scripts/perf_baseline.json --N 400 --D 64 --kneighbors 6 --trials 3
```

Then commit the updated baseline file.

### Cloud Governance Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OSCILLINK_RATE_LIMIT` / `OSCILLINK_RATE_WINDOW` | Global process-wide request throttle | 0 (disabled) / 60s |
| `OSCILLINK_IP_RATE_LIMIT` / `OSCILLINK_IP_RATE_WINDOW` | Per-IP request limiter (in-memory) | 0 (disabled) / 60s |
| `OSCILLINK_TRUST_XFF` | Trust first `x-forwarded-for` IP (deploy behind trusted proxy) | 0 |
| `OSCILLINK_STRIPE_MAX_AGE` | Max Stripe webhook age (seconds) before rejection | 300 |
| `OSCILLINK_API_KEYS` | Comma list of static API keys (legacy simple auth) | (unset) |
| `OSCILLINK_USAGE_LOG` | JSONL usage event file path | (unset) |
| `OSCILLINK_USAGE_SIGNING_SECRET` | HMAC-sign usage lines for tamper evidence | (unset) |
| `OSCILLINK_ALLOW_UNVERIFIED_STRIPE` | Allow processing Stripe webhooks without a verified signature (testing only; never enable in production) | 0 |
| `OSCILLINK_MONTHLY_USAGE_COLLECTION` | Firestore collection for monthly usage persistence (optional) | (unset) |
| `OSCILLINK_ADMIN_SECRET` | Admin endpoints shared secret, required via `x-admin-secret` header | (unset) |
| `OSCILLINK_RECEIPT_NULL_CAP` | Max number of null-point entries to include in a receipt (0 = no cap) | 0 |

Headers surfaced when active:
- Global rate: `X-RateLimit-*`
- Per-IP rate: `X-IPLimit-*`
- Quota: `X-Quota-*`
- Monthly cap: `X-Monthly-*`

Webhook replay protection: events older than `OSCILLINK_STRIPE_MAX_AGE` (via `stripe-signature` header `t=`) are rejected with `400 webhook timestamp too old` before deeper processing or signature verification.

---

### Firestore backends (production setup)

Firestore is supported for real customers; enable it via environment and provide GCP credentials.

- Prerequisites:
  - Install `google-cloud-firestore`
  - Provide GCP credentials (e.g., set `GOOGLE_APPLICATION_CREDENTIALS`), and ensure the correct project is selected
- Enable Firestore key store:
  - `OSCILLINK_KEYSTORE_BACKEND=firestore`
  - Optional: `OSCILLINK_FIRESTORE_COLLECTION=oscillink_api_keys` (default)
- Persist monthly usage (optional but recommended for multi‑replica):
  - `OSCILLINK_MONTHLY_USAGE_COLLECTION=oscillink_monthly_usage`
  - The service best‑effort hydrates and persists per‑key, per‑period counters
- Persist webhook events (optional):
  - `OSCILLINK_WEBHOOK_EVENTS_COLLECTION=oscillink_webhook_events`
- Stripe verification (recommended):
  - Install `stripe`; set `STRIPE_WEBHOOK_SECRET`

Notes & caveats:
- Monthly caps are enforced best‑effort with optional Firestore persistence; for strict global enforcement under high concurrency, consider transactional updates or a queue (e.g., Cloud Tasks/Pub/Sub) to serialize writes.
- Rolling per‑process quota is in‑memory; monthly counters can be shared via Firestore as shown above.
- Webhook persistence is idempotent (won’t overwrite an existing id); failures are swallowed silently—add logging or retries if you need stronger guarantees.

### Admin diagnostics (Cloud)

Admin-only endpoints (require `x-admin-secret` header with `OSCILLINK_ADMIN_SECRET`):

- `GET /admin/billing/price-map`
  - Returns the active Stripe price→tier mapping and the tier catalog:
    `{ "price_map": {"price_cloud_pro_monthly": "pro", ...}, "tiers": {"pro": {"monthly_unit_cap": 50000000, ...}, ... } }`

- `GET /admin/usage/{api_key}`
  - Returns in-memory quota window state and monthly usage for the API key. Example shape:
    `{ "api_key": "abc123", "quota": {"limit": 1000000, "remaining": 25000, "reset": 1731111111, ...}, "monthly": {"period": "202510", "limit": 50000000, "used": 12345, "remaining": 49987655} }`

Response headers (when active):
- Quota: `X-Quota-Limit`, `X-Quota-Remaining`, `X-Quota-Reset`
- Monthly caps: `X-Monthly-Cap`, `X-Monthly-Used`, `X-Monthly-Remaining`, `X-Monthly-Period`

Response body surface (when requesting settle/receipt/bundle):
- `meta.usage.monthly` block mirrors the monthly details.

## API Stability

Stability tiers (< 1.0.0) communicate upgrade expectations:

| Tier | Contract | Examples |
|------|----------|----------|
| Stable | Field name + basic structure preserved; only additive fields | `state_sig`, `timings_ms.total_settle_ms`, `meta.N`, `meta.D`, `meta.kneighbors_requested`, `meta.kneighbors_effective` |
| Evolving | Additive changes likely; semantics may tighten; removals rare | `meta.usage`, `meta.quota`, usage log JSON line fields, Prometheus bucket layout |
| Experimental | May change or disappear; feedback period | Async job `result.meta` subset, future extra receipt diagnostics |

Policies:
1. Adding new optional fields is non-breaking.
2. Promotion path: Experimental → Evolving → Stable after ≥1 minor version unchanged.
3. Deprecations (if any) appear in `CHANGELOG.md` with removal horizon.

Contributor guidance:
- Default new response/meta fields to Experimental unless clearly foundational.
- Avoid renaming Stable fields; add new + deprecate old instead.
- Update this table + CHANGELOG on promotions or deprecations.

---

## License

Apache‑2.0 for the SDK and receipts schema. (Future hosted billing / pricing automation components may adopt a distinct license; the core SDK remains Apache‑2.0.) See `LICENSE`.

---

## Contributing

Issues & PRs welcome. Please:
- Use the provided **Bug report** / **Feature request** templates.
- Follow the checklist in the PR template.
- Update `CHANGELOG.md` for user-visible changes.

### Developer Tooling
- Install git hooks: `pre-commit install`
- Auto-fix lint: `ruff check . --fix`
- Coverage (XML + terminal): `pytest --cov=oscillink --cov-report=xml --cov-report=term-missing`
- Fast dev cycle helper (optional): `python scripts/benchmark.py --N 400 --D 64 --kneighbors 6 --trials 1`

See `CONTRIBUTING.md` for full guidelines and release process.

## Differentiation Proof (Energy, Chain, Nulls, Gating)

Below is a concise, reproducible showcase of what makes Oscillink different from a plain RAG rerank loop:

1. Exact energy receipt (ΔH) after settling an SPD system.
2. Chain prior verdict + weakest link detection (edge + z‑score) exposing brittle path segments.
3. Null‑point diagnostics (nearly incoherent edges) surfaced directly from the receipt.
4. Optional diffusion gating reducing energy (ΔH) relative to uniform query pull.

### One‑Shot Proof Mode

Run the unified proof (JSON for automation):

```bash
python scripts/benchmark.py --proof --json
```
Example (truncated):

```json
{
  "config": {"D": 64, "N": 1000, "kneighbors": 6},
  "deltaH": 5589.3599,
  "null_points": 998,
  "sample_null": {"edge": [0, 927], "residual": 0.3467, "z": 17.2783},
  "chain_verdict": false,
  "weakest_link": {"edge": [0, 1], "k": 0, "zscore": 31.6069},
  "settle_ms": 2.0392
}
```
Interpretation:
- High null_points count reveals many structurally weak local edges for this random synthetic graph (expected for noise embeddings) — production data typically shows a much smaller coherent defect set.
- Weakest link with large z‑score pinpoints the most tensioned edge under the chain prior, providing a direct audit surface (you can go look at those two embeddings).
- Sub‑3ms settle on N=1000 synthetic points demonstrates low‑latency deterministic convergence.

### Quickstart Receipt & Bundle

```bash
python examples/quickstart.py
```
Excerpt:
```
receipt(deltaH): 1371.5505  nulls: 120
chain verdict: False  weakest: {'edge': [2, 5], 'zscore': 10.9087}
bundle top-3: [ {id:95, score:1.2268, align:0.3066}, ... ]
```
Bundle entries blend alignment and coherence anomaly; you can inspect per-item fields for transparent re-ranking logic.

### Diffusion Gating Comparison

Adaptive (screened diffusion) gates vs uniform:

```bash
python scripts/benchmark_gating_compare.py
```
Sample summary:
```
uniform_deltaH   mean=18563.70
diffusion_deltaH mean=13502.22
```
Lower ΔH under diffusion gating indicates the system found a more globally coherent settled state by spatially modulating query attraction (alignment uplift may be slightly negative in purely random synthetic data; on real semantic corpora diffusion typically preserves or improves top-k alignment while reducing energy).

### Why This Matters vs Classic RAG

| Aspect | Classic RAG / Reranker | Oscillink Lattice |
|--------|------------------------|-------------------|
| Global Objective | None (pairwise / local) | Convex SPD energy minimized deterministically |
| Explainability | Heuristic scores | Structured receipt: ΔH, null points, path verdict |
| Path Reasoning | External / ad-hoc | Native chain Laplacian prior + weakest link |
| Anomaly Surfacing | Manual error cases | Automatic null-point extraction |
| Determinism | Embedding + floating noise | Deterministic kNN + fixed solver tolerances |
| Integrity | Unsealed scores | Optional HMAC-signed receipts |
| Adaptive Gating | Rare / custom | Built-in diffusion gating primitive |

### Reproducing the Proof Locally

```bash
# 1. Energy + chain + nulls (human readable)
python scripts/benchmark.py --proof

# 2. Add bundle stats and diffusion gating
python scripts/benchmark.py --proof --bundle-k 8 --diffusion

# 3. Structured JSON for logging / dashboards
python scripts/benchmark.py --proof --json > proof.json
```

### Next (Planned Enhancements)
- Real text corpus example (semantic null contrast to random synthetic).
- Side-by-side with a cross-encoder reranker showing receipt-aligned uplift.
- Dashboard notebook: track ΔH distributions & null density over time.
- Extended signed payload including convergence fields (opt-in already available as `extended` mode).

## Demonstration Notebooks (Work in Progress)

| Notebook | Purpose | Status |
|----------|---------|--------|
| `01_chain_reasoning.ipynb` | Multi-hop chain prior vs naive cosine; persistence export/import | Scaffolded |
| `02_energy_landscape.ipynb` | ΔH convergence curve; energy vs iteration; (future) diffusion comparison | Scaffolded |
| `03_constraint_query.ipynb` | Support claim X while suppressing Y via gating | Scaffolded |
| `04_hallucination_reduction.ipynb` | Evaluation harness for hallucination reduction (RAG vs bundle) | Stub |

Open these under `notebooks/` to explore emerging features. As they mature, examples will graduate into documented recipes.

### Optional Embedding Extras

Install a real encoder (sentence-transformers) for higher-fidelity text demos:

```bash
pip install "sentence-transformers>=2.2.0"  # optional, not required for core SDK
```

The helper `embed_texts` in `oscillink.adapters.text` automatically falls back to deterministic hash embeddings if the library is missing.

### Diffusion Gating (Adaptive Query Influence)

A screened diffusion preprocessing step can reduce total lattice energy (ΔH) substantially by spatially modulating query attraction strength across the graph.

**Empirical Result (Synthetic Benchmark Example)**
```
uniform_deltaH_mean   ≈ 18,563
diffusion_deltaH_mean ≈ 13,502   (≈27% reduction)
```
(Method: `scripts/benchmark_gating_compare.py`, N=1000 (default in script), k=6, 5 trials; random Gaussian embeddings.)

**Tradeoff:** On pure noise embeddings, alignment uplift can dip slightly (query signal is localized). On semantically structured corpora we expect either neutral or positive top-k semantic alignment while still improving coherence (ΔH). The gating vector is optional.

**Usage:**
```python
from oscillink import OscillinkLattice, compute_diffusion_gates

# Y: (N,D) float32 anchors, psi: (D,) float32 query
lat = OscillinkLattice(Y, kneighbors=6)
# Optional adaptive gates
gates = compute_diffusion_gates(Y, psi, kneighbors=6, beta=1.0, gamma=0.1)
lat.set_query(psi, gates=gates)   # omit gates=... to fall back to uniform
lat.settle()
rec = lat.receipt()  # rec["deltaH_total"] reflects post-gated coherence
```
Parameters:
- `beta` (≥0): scales source injection; higher increases separation in gate strengths.
- `gamma` (>0): screening; larger → more local, flatter gates; small but positive retains SPD.
- `kneighbors`: structural resolution; should match lattice construction for consistency.

**When to Use:**
- Large, heterogeneous candidate pools where uniform pull causes incoherent energy spikes.
- Multi-topic retrieval where you want spatial smoothing around high-alignment clusters.
- As a deterministic, model-free alternative to learned reweighting.

Reproduce the benchmark:
```bash
python scripts/benchmark_gating_compare.py
```
(Adjust `--trials`, `--N`, `--D`, `--kneighbors` to validate on your scale.)
