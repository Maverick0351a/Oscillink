This document has moved.

New location: [docs/ops/PROD_READINESS_CHECKLIST.md](./ops/PROD_READINESS_CHECKLIST.md)
# Production Readiness Checklist

> Status legend: [ ] pending · [~] in progress · [x] complete

## High Priority

- [x] Add `state_sig` to receipt meta (consistency with cloud & docs)
- [ ] Preserve adjacency symmetry after row cap (ensure SPD assumptions)
- [ ] Fix deterministic neighbor sort tie-break (index ascending)
- [ ] Reorder monthly cap + quota checks before compute in all endpoints (`/v1/receipt`, `/v1/bundle`, `/v1/chain/receipt`)
- [x] Harden Stripe webhook (ignore tier mutations without verified signature unless explicitly allowed by `OSCILLINK_ALLOW_UNVERIFIED_STRIPE=1`)
- [ ] Fix extras recursion: replace `cloud-all` self reference with explicit union list
- [ ] Guard usage log directory creation (skip `os.makedirs` on empty dir path)
- [ ] Clamp `kneighbors` inside `OscillinkLattice` (`1 <= k <= N-1`)
- [x] Update README env var table (add `OSCILLINK_MONTHLY_USAGE_COLLECTION`, `OSCILLINK_ALLOW_UNVERIFIED_STRIPE`) + signature requirement note
- [x] Add null-point capping env var (`OSCILLINK_RECEIPT_NULL_CAP`) + meta summary

## Medium Priority

- [ ] Tests: state_sig present, adjacency symmetric, deterministic_k reproducibility, kneighbors clamp behavior, webhook guard logic, usage log dir guard
- [ ] README note clarifying receipt signature scope (fields covered are minimal core: `state_sig`, `deltaH_total` unless extended)
- [ ] Optional: extend signed payload (include `version`, `ustar_res`) or document rationale for minimal set
- [ ] Improve provenance diff detail (list differing components)
- [ ] Vectorize coherence drop & per-node components (edge list; reduce O(N^2 D) loops)
- [ ] Cache coherence_drop for reuse between receipt() and bundle()
- [ ] Re-symmetrize after path Laplacian weights if any row scaling applied (double-check invariants)
- [ ] Add Prometheus counters for quota/monthly cap rejections
- [ ] Add retention policy / purge for in-memory webhook events & job store

## Low Priority / Future Enhancements

- [ ] Redis backend for quota + monthly usage + webhook events (multi-replica scaling)
- [ ] Signed usage receipts endpoint + verification helper
- [ ] Pagination & filtering for admin webhook events (Firestore paging)
- [ ] Alert hooks (cap threshold webhooks/email)
- [ ] Iterative solver fallback for diffusion gating (CG) + large-N safeguards
- [ ] Multi-secret receipt signing (key rotation with key id)
- [ ] Strict logging mode (surface callback/logging errors)
- [ ] CORS/security headers for public deployment profile
- [ ] Add CITATION / references section

## Verification Gates

- Lint (ruff) clean
- Tests green (including new invariants)
- SPD sanity: symmetry check + v^T M v > 0 sampled
- Performance baseline unchanged (CG iterations, solve ms)
- Security: webhook mutation only on verified signature path

## Release Blockers (Must be [x] Before Charging Users)

- High Priority list all [x]
- Tests for those changes [x]
- README updated (env + signature scope) [x]
- Webhook guarded [x]
- Quota/monthly enforcement ordering consistent [x]

---

_This file should be updated as each item is addressed; move completed high-priority items to the bottom or mark [x]._
