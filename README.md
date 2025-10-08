# Oscillink Lattice — Short‑Term Coherence SDK (Phase 1)

> A graph‑theoretic, SPD‑solved working memory that explains itself.

![CI](https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/oscillink-lattice.svg)
![License](https://img.shields.io/github/license/Maverick0351a/Oscillink.svg)
![Python](https://img.shields.io/pypi/pyversions/oscillink-lattice.svg)
![Coverage](https://codecov.io/gh/Maverick0351a/Oscillink/branch/main/graph/badge.svg)

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
pip install -e .[cloud]
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

### OpenAPI Schema Export

Generate the current OpenAPI spec (writes `openapi.json`):

```bash
python -m scripts.export_openapi --out openapi.json
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

Headers surfaced when active:
- Global rate: `X-RateLimit-*`
- Per-IP rate: `X-IPLimit-*`
- Quota: `X-Quota-*`
- Monthly cap: `X-Monthly-*`

Webhook replay protection: events older than `OSCILLINK_STRIPE_MAX_AGE` (via `stripe-signature` header `t=`) are rejected with `400 webhook timestamp too old` before deeper processing or signature verification.

---

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

