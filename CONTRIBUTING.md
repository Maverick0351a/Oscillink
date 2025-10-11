# Contributing

Thank you for considering a contribution! This project aims for **clarity, reproducibility, and safety** (SPD math, explicit receipts).

## Ground Rules
- Keep public APIs backward compatible (additive changes favored over breaking ones until v1.0).
- Always accompany new features with tests (functional + at least 1 negative / edge case).
- Maintain solver invariants: SPD construction and deterministic signatures.
- Update `CHANGELOG.md` for user-visible changes.

## Development Setup
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -e .[dev]
pre-commit install  # install git hooks (auto-lint/format/type on commit)

# One-shot checks
ruff check .
mypy oscillink
pytest -q
```

## Issue Tracking
- Use the **Bug report** and **Feature request** templates (auto-applied in GitHub UI).
- Provide minimal reproducible code for bugs; include version + Python info.

## Pull Requests
1. Fork & branch (`feature/<short-description>`).
2. Write/change code + tests.
3. Run lint, type check, tests locally.
4. Update docs / README snippets if behavior is user-facing.
5. Update `CHANGELOG.md` (add under `[Unreleased]`).
6. Open PR; fill out the PR template checklist.

### Fast Checklist
- [ ] Tests pass (`pytest -q`)
- [ ] Coverage reasonable (`pytest --cov=oscillink --cov-report=term-missing`)
	- Full HTML & trend on Codecov: https://codecov.io/gh/Maverick0351a/Oscillink
- [ ] Patch coverage not significantly below project coverage (see Codecov PR comment; soft targets: project ≥85%, patch ≥80%).
- [ ] Lint passes (`ruff check .`)
- [ ] Types clean (`mypy oscillink`)
- [ ] CHANGELOG updated (if user-facing)
- [ ] Docs / README updated

## Performance Considerations
Use `scripts/benchmark.py` (JSON mode) or `scripts/scale_benchmark.py` for quick regressions. Non-trivial slowdowns (>10%) in `ustar_solve_ms` should be justified or optimized.

## Release Process
Maintainers only:
1. Ensure `[Unreleased]` section is curated.
2. Bump version in `pyproject.toml` & `oscillink/__init__.py`.
3. Run full test suite + a representative benchmark.
4. Commit & tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
5. GitHub Actions `Publish to PyPI` uses GitHub OIDC trusted publishing (no token secrets required).

## Security / Integrity
- Receipts can be signed; avoid logging secrets.
- Report vulnerabilities privately (see `SECURITY.md`).

## Style
- Follow Ruff defaults (see `pyproject.toml`).
- Prefer explicitness over clever one-liners in numerical code.

## Questions?
Open a discussion / issue with the appropriate template.

Happy hacking!
