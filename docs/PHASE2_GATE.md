This document has moved.

New location: [docs/ops/PHASE2_GATE.md](./ops/PHASE2_GATE.md)
# Phase 2 Gating Criteria

This document defines the readiness gate for advancing from Phase 1 (foundational lattice + governance primitives) to Phase 2 (expanded functionality, multi-process scaling, richer analytics).

## Objectives
Ensure core semantics are stable and observable before adding higher-level features or distributed deployment patterns.

## Gate Checklist

### 1. Functional Stability
- [ ] All existing endpoints pass test suite (CI green).
- [ ] Receipt meta fields marked Stable have no breaking shape changes for one minor version window.
- [ ] Chain and bundle operations yield deterministic signatures under identical inputs + seeds.

### 2. Performance Baseline
- [ ] Benchmark baseline file (`benchmarks/baseline.json`) populated with real timings (median of >=5 runs, warmed).
- [ ] A simple regression check script (planned) compares current settle median to baseline +/- tolerance (target tolerance: +25%).
- [ ] Any intentional performance regression is documented in CHANGELOG with rationale.

### 3. Governance & Limits
- [ ] Rate limiting and quota behavior documented (README) with single-process scope disclaimers.
- [ ] Quota & rate headers verified in integration tests.
- [ ] Usage logging path & optional HMAC signature validated manually (sample log with signature).

### 4. Observability
- [ ] Prometheus metrics namespace stable (no renames without deprecation entry).
- [ ] Add a metric for job queue depth (optional enhancement) OR explicitly defer with rationale.
- [ ] Baseline metrics dashboard (manual / README snippet) showing: request rate, latency histogram, node throughput.

### 5. API Stability Classification
- [ ] README 'API Stability' section lists Stable vs Evolving vs Experimental fields.
- [ ] Any field transition (e.g., Evolving -> Stable) logged in CHANGELOG.

### 6. Multi-Process / Deployment Notes
- [ ] README clarifies that in-memory rate limit + quota are per-process and require external coordination (redis / gateway) for horizontal scaling.
- [ ] Guidance stub for future external store integration.

### 7. Security & Integrity
- [ ] Document recommended rotation cadence for API keys and how dynamic reload works.
- [ ] Document receipt signature use-case vs usage log signature (different threat surfaces).

### 8. Developer Experience
- [ ] Example quickstart updated to include kneighbors clamping note.
- [ ] CHANGELOG present and maintained for at least two consecutive versions.
- [ ] CONTRIBUTING updated with guidance on adding new meta fields (stability tagging + tests).

### 9. Deferred (Explicit)
Items consciously deferred into Phase 2 for focus:
- Distributed / multi-process coherent quota & rate limiting.
- Persistent job store & resumable jobs.
- Advanced provenance diff tooling.
- Automated performance regression CI check (script placeholder only in Phase 1).

## Exit Criteria
Phase 2 work may begin when all non-deferred checklist items are checked AND at least one maintainer signs off in PR referencing this document.

## Rationale
Locking core semantics early minimizes migration churn and clarifies which surfaces are safe to build upon externally.
