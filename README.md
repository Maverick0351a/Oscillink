# Oscillink Lattice — Short‑Term Coherence SDK (Phase 1)

![CI](https://github.com/Maverick0351a/Oscillink/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/oscillink-lattice.svg)
![License](https://img.shields.io/github/license/Maverick0351a/Oscillink.svg)
![Python](https://img.shields.io/pypi/pyversions/oscillink-lattice.svg)
![Coverage](https://img.shields.io/badge/coverage-ci--artifact-informational)

**Oscillink Lattice** is a small, fast, *physics‑inspired* memory enhancer for generative models.
It builds an ephemeral lattice (graph) over candidate vectors and **settles** to the most coherent
state by minimizing a convex energy with a **symmetric positive definite** (SPD) system.

- **Explainable**: exact energy receipts (ΔH) and null‑point diagnostics.
- **Model‑free**: no training — your vectors *are* the model.
- **Safe math**: normalized Laplacian; SPD ensures robust CG convergence.
- **Chain Priors**: encode expected reasoning paths; get **chain receipts** (verdicts, weakest link).

> *Phase‑1 focus:* a pure SDK (no cloud, no data movement). Bring your own embeddings.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # (or .\.venv\Scripts\activate on Windows)
pip install -e .
pytest -q

# optional: run with coverage
pytest -q --cov=oscillink --cov-report=term-missing
```

### Minimal example

```python
import numpy as np
from oscillink.core.lattice import OscillinkLattice

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

## What this SDK provides

- `OscillinkLattice`: build lattice, settle, receipts, chain_receipt, bundle.
- **Caching**: stationary solution `U*` cached & reused across diagnostics.
- **Export / Import**: `export_state()` / `OscillinkLattice.from_state()` (JSON) and binary `save_state(..., format='npz')` for reproducibility.
- **Receipt Meta & Version**: `receipt()` now returns `version` + `meta` (cache usage, signature, solve stats, convergence). See `docs/RECEIPT_SCHEMA.md`.
- **Callbacks**: register post‑settle diagnostics hooks via `add_settle_callback(fn)`.
- **Forced Refresh**: `refresh_Ustar()` to recompute stationary solution ignoring cache.
- **Benchmarking**: lightweight timing script in `scripts/bench.py`.
- Graph utilities: mutual‑kNN, row‑cap, normalized Laplacian, path Laplacian.
- Solver: Jacobi‑preconditioned CG (pure NumPy).
- Receipts: ΔH (trace identity), per‑node attribution, null‑points, diagnostics.
- Chain Priors: SPD‑safe path Laplacian; **chain verdict** + **weakest link**.

**Docs:** see `docs/` for math spec, API, receipts schema, chain guide, and roadmap.

---

## Design principles

- **Normalized Laplacian**: better conditioning across datasets.
- **SPD system**: \(M = \lambda_G I + \lambda_C L_\mathrm{sym} + \lambda_Q B + \lambda_P L_{path}\).
- **Implicit settle**: \((I+\Delta t M)U^+=U+\Delta t(\lambda_G Y + \lambda_Q B 1\psi^\top)\).
- **Receipts**: exact ΔH via trace identity; normalized residuals for null points.

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

`export_state()` includes `provenance`, a digest over core arrays (Y, ψ, gating vector), key parameters, and an adjacency fingerprint to enable integrity / reproducibility checks across exports.

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

Optionally sign receipts with HMAC‑SHA256 over `state_sig` + `deltaH_total`.

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

---

## License

Apache‑2.0 for the SDK and receipts schema. See `LICENSE`.

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

