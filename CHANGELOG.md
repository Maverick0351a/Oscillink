# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres (loosely) to Semantic Versioning while < 1.0 (minor bumps may still be breaking).

## [Unreleased]
### Added
- Planned: per-phase timing metrics, provenance diff helper, scaling benchmark, convergence warnings (Tier 1 roadmap).

## [0.1.1] - 2025-10-06
### Added
- Binary NPZ persistence (`save_state(format="npz")` / `from_npz`).
- Adjacency meta stats (`avg_degree`, `edge_density`) in receipts.
- Structured JSON line logger (`json_line_logger`).
- Performance comparison utilities (`compare_perf`, `perf_snapshot`, `perf_check`).
- Provenance hash embedded in exported state for reproducibility.
- HMAC receipt signing + verification helpers (`verify_receipt`, `verify_current_receipt`).
- Convergence diagnostics for stationary solve (iters, residual, converged flag).
- Chain order persistence & gating API.

### Changed
- Receipt schema expanded (see `docs/RECEIPT_SCHEMA.md`).
- Export/import now retain chain nodes & provenance when available.

### Tests
- Expanded suite to 27 tests covering receipts, provenance, NPZ roundtrip, performance comparison, logger output.

### Security
- Added integrity via optional HMAC-SHA256 signing for receipts.

## [0.1.0] - 2025-10-05
### Added
- Initial alpha release with lattice construction, CG solver, chain priors, receipts (ΔH decomposition), benchmarking script, basic CI.

---

Links:
- Receipt Schema: `docs/RECEIPT_SCHEMA.md`
- Roadmap: `docs/ROADMAP.md`
