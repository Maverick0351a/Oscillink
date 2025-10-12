This document has moved.

New location: [docs/guides/CHAIN_GUIDE.md](./guides/CHAIN_GUIDE.md)
# Chain Priors and Chain Receipts

**Why**: encode expected reasoning/workflow paths (A→B→C→D) without breaking SPD.

- Build a path Laplacian over the chain; add `λ_P Tr(U^T L_path U)` to the energy.
- Use small λ_P (0.1–0.3) to avoid masking genuine conflicts.
- Optional directionality via staged gate ramps (do not add anti‑symmetric edges).

**Chain Receipt**
- Reports pass/fail, weakest link, per‑edge z‑scores, and chain coherence gain.
- Use z‑threshold ≈ 2.5–3.0 as a default.
