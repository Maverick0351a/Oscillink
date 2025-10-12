# Oscillink Screened Diffusion Whitepaper — Outline (Draft)

## 1. Executive Summary
- Problem: Ad‑hoc weighting of query influence limits interpretability & adaptivity.
- Contribution: Screened diffusion derived gating integrated with convex SPD coherence lattice.
- Outcomes: Locality‑controlled semantic propagation, energy receipts, reproducible signatures.

## 2. Motivation & Positioning
- Gap between heuristic rerankers and opaque deep re‑scorers.
- Need for transparent short‑term coherence shaping.
- Physics analogy: energy injection (query), diffusion (graph), dissipation (screening), coupling (vector field settle).

## 3. Mathematical Foundations
### 3.1 Lattice Vector Field (Existing)
- Stationary system: (λ_G I + λ_C L_sym + λ_Q diag(b) + λ_P L_path) U = λ_G Y + λ_Q diag(b) 1 ψ^T.
- SPD guarantees & convergence properties.
### 3.2 Screened Diffusion Scalar Field
- Equation: (L_sym + γ I) h = β s.
- Source s ≥ 0 from cosine similarity (extensions: kernelized, multi-query superposition).
- Spectral interpretation: h = Φ (Λ + γ I)^{-1} Φ^T s (low‑frequency attenuation vs γ).
### 3.3 Coupling Strategies
- Current implementation: h → gating vector b.
- Future joint minimization over E(U, h) (bi‑convex structure; alternating updates / closed form in h).

## 4. Energy Functional Derivation
- Augmented energy: E(U,h) = 1/2 U^T (λ_G I + λ_C L_sym) U + 1/2 λ_Q Σ_i h_i ||u_i - q||^2 + 1/2 (||∇_G h||^2 + γ||h||^2 - 2β s^T h).
- Euler–Lagrange → screened Poisson for h; linear system for U.
- Conservation & locality rationale (screening vs pure diffusion).

## 5. Algorithm Design
- Step 1: Build graph (mutual kNN, row cap, normalized Laplacian).
- Step 2: Solve screened diffusion (direct / CG depending on N).
- Step 3: Normalize h → [0,1]; inject as gates.
- Step 4: Stationary solve (cached) or single settle step.
- Complexity: O(N k D + N k) graph + O(N^3) direct (small N) or O(iter * (N k + N D)) iterative.

## 6. Implementation Notes
- Deterministic modes: full sort vs stochastic tie jitter.
- Numerical stability: γ > 0 ensures invertibility; fallback uniform gating on failure.
- Memory considerations for large N (sparse Laplacian representation roadmap).

## 7. Empirical Evaluation Plan
- Metrics: ΔH improvement distribution, bundle relevance uplift, alignment variance stabilization.
- Ablations: γ sweep, β sweep, k_diffusion vs k_lattice, deterministic vs seeded.
- Baselines: uniform gates, heuristic sigmoid similarity weights, top‑p mask.

## 8. Interpretability & Receipts
- h as explanatory map: per-node energy influence heatmap.
- Extended receipts: add {h_min, h_max, h_mean, injection_mass}.
- Potential SHAP‑style marginal analysis with/without diffusion.

## 9. Extensions & Research Directions
- Joint alternating minimization (coupled field iteration).
- Temporal diffusion (progressive expansion for streaming coherence).
- Multi-source diffusion (multi-query / chain anchored injection pattern).
- Spectral shaping (band‑pass filters on Λ to emphasize meso‑scale structure).
- Quantum analogy: Hamiltonian framing & amplitude conservation variants.

## 10. Security & Integrity Considerations
- Signature extension: include hash(h) in state_sig variant for stronger reproducibility.
- Adversarial robustness: diffusion smooths sparse manipulations; outline threat model.

## 11. Deployment & Tiering
- SDK: optional helper (`compute_diffusion_gates`).
- Cloud Pro: feature-gated endpoint with metrics & usage line flag.
- Pricing levers: gating compute surcharge, parameter range unlocks.

## 12. Roadmap & Milestones
- v1.1: Preprocessing only (current state).
- v1.2: Receipts include diffusion stats.
- v1.3: Optional sparse Laplacian backend.
- v2.0: Joint U,h optimization + spectral mode introspection API.

## 13. Conclusion
- Adds formal physical analogy & principled adaptivity without sacrificing determinism.
- Sets foundation for future coupled field & spectral products.

## Appendix A: Notation
- Table of symbols (U, Y, ψ, h, L_sym, β, γ, λ_*).

## Appendix B: Pseudocode
- Concise algorithm listing with complexity annotations.

## Appendix C: Failure Modes & Mitigations
- Singular system fallback, extreme similarity concentration, degenerate graph (isolates).
