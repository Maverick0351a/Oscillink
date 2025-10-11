# Patents and Virtual Marking

Status: Patent pending

This page provides public notice and virtual marking under 35 U.S.C. § 287.

## Application details

- Jurisdiction: United States
- Type: Utility — Provisional (35 U.S.C. § 111(b))
- Application No.: 63/897,572
- Filing date: 2025‑10‑11
- Title: Oscillink: A Self‑Optimizing Physics‑Based Memory System with Deterministic Audit Trails
- Inventor: Travis Jacob Johnson
- Applicant/Owner: Odin Protocol Inc. (Oscillink)
- Status: Provisional filed; patent pending

Additional U.S. and international applications may be pending. This page will be updated as new filings or issued patents become available.

## Condensed abstract (public disclosure)

Oscillink discloses systems, methods, and apparatus for computing an explainable, coherent working‑memory state over embeddings or latent vectors by minimizing a strictly convex energy on a sparsified graph (a “lattice”). The core contract assembles a symmetric positive definite (SPD) operator

M = λ_G I + λ_C L_sym + λ_Q B + λ_P L_path,

where L_sym is a normalized Laplacian of a mutual‑kNN graph, B is a diagonal gate matrix, and L_path is an optional path prior. The system solves for the unique equilibrium U⋆ using preconditioned conjugate gradient (PCG) with deterministic tolerances and emits deterministic, cryptographically signable receipts including total energy drop (ΔH), per‑node/edge attributions, and “null‑points” (incoherence loci). Parameters {λ_G, λ_C, λ_Q, λ_P} can be adapted via bounded updates using production telemetry (“self‑optimizing”).

Extensions cover: (i) a coherence database/retrieval layer, (ii) software/firmware/hardware kernels (e.g., GPU/FPGA/ASIC implementations), (iii) a quantum memory supervisory controller where the lattice informs pulse schedules under safety bounds, and (iv) phased‑array/field control where beamforming weights are optimized under the same SPD contract subject to constraints. Empirically, the apparatus demonstrates favorable inverse‑like scaling in common regimes, reduced hallucination in retrieval pipelines, and deterministic audit trails suitable for regulated deployment.

## Contact

Licensing and inquiries: travisjohnson@oscillink.com

## Disclaimer

This document is for public notice only and is not legal advice. It does not confer any license, express or implied. Application details may change as prosecution proceeds. Payment identifiers and other sensitive submission artifacts are intentionally omitted from this repository.
