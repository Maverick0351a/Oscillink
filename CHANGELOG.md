# [0.1.9] - 2025-10-10
# [0.1.9] - 2025-10-10
## [Unreleased]
## [0.1.13] - 2025-10-11
### Changed
- README polish: moved logo directly under badges for better GitHub rendering.

## [0.1.12] - 2025-10-11
### Added
- Production-ready licensed-container operator suite: metrics protection, JSON access logs, security presets, TLS samples, networking guide, and signing guidance.
### Changed
- Bumped SDK to 0.1.12; Helm appVersion updated and default image tag set to v0.1.12.
### Notes
- To publish to PyPI: create a GitHub Release with tag `v0.1.12` (workflow `.github/workflows/publish.yml`).

### Added
- Helm operator presets and examples: GKE/EKS/AKS values, security-first preset, private/proxy and ingress privacy overlays.
- cert-manager samples (ClusterIssuer/Certificate) and controller-specific variants.
- Licensed-container ops: metrics protection (`OSCILLINK_METRICS_PROTECTED`), JSON access logs (`OSCILLINK_JSON_LOGS`, `OSCILLINK_LOG_SAMPLE`).
- Operator docs: `docs/ops/OPERATIONS.md`, `docs/ops/IMAGE_SIGNING.md`, `docs/ops/NETWORKING.md`.
- Supply chain CI: SBOM (Syft), Trivy scan (non-blocking), and nightly pip-audit; CodeQL retained.

### Changed
- Pinned licensed Docker base image to `python:3.11.9-slim` and documented digest pinning.
- Helm chart bumped to 0.1.1 and README linked new operator docs.
### Changed
- Dropped Python 3.9 support; project now supports Python 3.10–3.12 only.

### Removed
- Experimental OLF prototype and related artifacts under `experiments/` to avoid confusion and coverage noise.
### Changed
- NumPy policy aligned: support 1.22–<3.0 (1.x and 2.x). Updated pyproject and README.
- CI expanded to Python 3.9–3.12 × NumPy 1.x/2.x matrix; retains ruff+mypy and perf checks.

### Added
- Security scans: CodeQL workflow and nightly `pip-audit` job.
- Legal/commercial docs: Terms (docs/product/TERMS.md), Privacy (docs/product/PRIVACY.md), DPA template (docs/billing/DPA.md); linked from README and project URLs.
- README: Production stability/deprecation policy and Cloud beta status/SLO notes.

### Removed
- Unattributed testimonial block for launch; will reintroduce with verifiable citations.

### Maintenance
- Hardened marketing language ("pilot deployments"), ensured all benchmark claims remain reproducible with hardware refs.

### Fixed
- README license badge switched to PyPI-backed shield for reliable rendering.

### Maintenance
- Internal docs/readme polishing; no functional code changes.

# [0.1.5] - 2025-10-08
# [0.1.6] - 2025-10-10
### Changed
- Switch to PyPI Trusted Publishing (GitHub OIDC) — no API tokens required.
- Direct-to-PyPI on GitHub Release; removed TestPyPI step from workflow.
- Organized imports in FastAPI app to satisfy linting.

### Documentation
- README updated with OIDC publish instructions and simplified release flow.

### CI/CD
- Consolidated publish workflow: build sdist/wheel and publish on Release via `pypa/gh-action-pypi-publish`.

# [0.1.5] - 2025-10-08
### Added
- In-memory per-IP rate limiting (`OSCILLINK_IP_RATE_LIMIT`, `OSCILLINK_IP_RATE_WINDOW`, `OSCILLINK_TRUST_XFF`) with response headers (`X-IPLimit-*`).
- Webhook timestamp freshness enforcement for Stripe (`OSCILLINK_STRIPE_MAX_AGE`, default 300s) rejecting stale replay attempts.
- CI OpenAPI diff gating step invoking `scripts/check_openapi_diff_simple.py` on pull requests (path/method removal detection fails build).

### Changed
- Version bumped to 0.1.5.

### Documentation
- README: Added environment variable descriptions for new governance controls and CI contract gating note.

### Security / Governance
- Strengthened abuse protection (dual layer: global + per-IP) and replay defense (timestamp freshness) for webhooks.

# [0.1.4] - 2025-10-07
### Added
- `sig_v` field in signed receipt payloads (minimal & extended) for forward-compatible signature schema evolution.
- Version gating in `verify_receipt_mode` via `required_sig_v`.
- Simple OpenAPI diff script `scripts/check_openapi_diff_simple.py` for path/method removal detection.

### Changed
- Version bumped to 0.1.4.

### Documentation
- README updated with `sig_v` examples and rationale.

# [0.1.3] - 2025-10-07
### Added
- Root-level re-export `verify_receipt_mode` for convenience import.
- CI workflow now uploads OpenAPI schema artifact and runs a perf smoke job.

### Changed
- Version bumped to 0.1.3.

### Documentation
- README updated with CI OpenAPI artifact note and performance baseline guidance.

# Changelog

All notable changes to this project will be documented in this file.

Format loosely follows Keep a Changelog. While < 1.0.0, minor version bumps MAY include breaking changes and will be documented here.

## [0.1.2] - 2025-10-07
### Added
- Extended receipt signature mode (`set_signature_mode('extended')`) signing convergence stats & parameters.
- Helper `verify_receipt_mode` for minimal/extended verification & optional minimal subset verification.
- OpenAPI surface test (`test_openapi_surface.py`).
- Performance smoke test (`test_perf_smoke.py`).
- Neighbor seed & signature mode tests (`test_signature_modes.py`).
- Prometheus gauge `oscillink_job_queue_depth` tracking queued + running async jobs.
- Diffusion gating preprocessor `compute_diffusion_gates` and related examples / benchmarks (`examples/diffusion_gated.py`, `scripts/benchmark_gating_compare.py`).
- Receipt gating statistics meta fields (Experimental): `gates_min`, `gates_max`, `gates_mean`, `gates_uniform`.

### Changed
- Version bumped to 0.1.2.
- Deterministic neighbor ordering & row cap symmetry hardening integrated (carried forward from 0.1.1 work).

### Documentation
- README: Extended Signature Mode section & API Stability section.
- Expanded diffusion gating explanation.

### Tests
- Added invariants, gating stats, signature modes, OpenAPI surface, performance smoke.

### Planned
- Populate `benchmarks/baseline.json` with real median timings for perf regression gating.
- External (multi-process) quota/rate limiter guidance.

## [0.1.1] - 2025-10-07
### Added
- Dynamic per-request reload of API keys, rate limit, and quota env vars (single-process hot reconfig).
- Quota headers (`X-Quota-Limit`, `X-Quota-Remaining`, `X-Quota-Reset`).
- Rate limit headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`).
- `kneighbors_requested` and `kneighbors_effective` meta fields for transparency when clamping occurs.
- Runtime configuration helper module (`cloud/app/runtime_config.py`).
- Optional usage JSONL logging with optional HMAC signature (`OSCILLINK_USAGE_SIGNING_SECRET`).

### Changed
- Clamp kneighbors to `min(requested, N-1)` to avoid argpartition edge cases.
- Refactored `cloud/app/main.py` to remove duplicated environment parsing logic in favor of runtime helpers.

### Fixed
- Duplicate Prometheus metric registration guarded (important for test reloads / dev hot-reload).
- Ensure `state_sig` always present even when receipt omitted.

### Tests
- Added test asserting kneighbors clamp and meta reporting.

## [0.1.0] - 2025-10-01
Initial scaffold release (core lattice construction, settle solver, receipts, chain receipt, bundle API endpoints, basic governance: API key auth, rate limiting, per-key quota, Prometheus metrics, usage logging foundation).
