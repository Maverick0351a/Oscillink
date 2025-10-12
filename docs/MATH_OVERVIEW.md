This document has moved.

New location: [docs/foundations/MATH_OVERVIEW.md](./foundations/MATH_OVERVIEW.md)
# Math Overview (Concise)

This overview orients you to the core mathematical objects used by Oscillink Lattice. For deeper formalism see `docs/foundations/SPEC.md` and API nuances in `docs/reference/API.md`. For physical intuition (energy, diffusion, control), see `docs/foundations/PHYSICS_FOUNDATIONS.md`.

## 1. Objective & Stationary State

Oscillink maintains a matrix U (adjusted anchor representations) minimizing a convex energy:

E(U) = λ_G ||U - Y||_F^2 + λ_C Tr(Uᵀ L_sym U) + λ_Q Tr(Uᵀ B 1 ψᵀ) + λ_P Tr(Uᵀ L_path U)

Where:
- Y ∈ R^{N×D} original anchor embeddings
- L_sym normalized graph Laplacian (mutual kNN)
- B diagonal query attraction mask (from gates in [0,1])
- ψ ∈ R^{D} query vector (unit‑normalized)
- L_path Laplacian of optional chain prior

Taking derivative and setting ∂E/∂U = 0 gives a linear SPD system per column:

(λ_G I + λ_C L_sym + λ_Q B + λ_P L_path) U = λ_G Y + λ_Q B 1 ψᵀ

Denote M = λ_G I + λ_C L_sym + λ_Q B + λ_P L_path. With λ_G > 0 and positive semidefinite L_sym, B, L_path we have M ≻ 0 (symmetric positive‑definite) guaranteeing a unique minimizer and efficient Conjugate Gradient convergence.

## 2. Receipts (Energy Improvement)

ΔH = E(Y) - E(U*)

Instead of reconstructing E(Y) explicitly we exploit a trace identity over the stationary solution to compute ΔH efficiently and deterministically. Receipts expose:
- `deltaH_total` scalar improvement
- Per‑node contributions (attribution)
- Null points: edges with large normalized residuals (structural incoherence surfaces)

## 3. Gating & Diffusion

Gating introduces a spatially varying diagonal term B. Uniform gating (all ones) reduces to classic coherence + query pull. Adaptive gating:
1. Construct a non‑negative source s_i = max(0, cos(y_i, ψ))
2. Solve (L_sym + γ I) h = β s (screened diffusion / Poisson)
3. Normalize h to [0,1] → gates
4. Set B = diag(gates)

This spreads query influence smoothly across graph neighborhoods while suppressing noisy / off‑topic nodes (low diffusion response). Hallucination control is realized by *manual* or heuristic assignment of near‑zero gates to low‑trust nodes.

## 4. Chain Prior

Given an ordered chain C of nodes, build its path Laplacian L_path (positive semidefinite). Adding λ_P Tr(Uᵀ L_path U) penalizes incoherent variation along the expected reasoning trajectory. Chain receipt returns:
- Verdict (bool) if standardized tension below threshold
- Weakest link edge with z‑score

## 5. Determinism & Signatures

A structural signature `state_sig` hashes:
- Adjacency fingerprint (sorted edge list or deterministic build seed)
- λ parameters, kneighbors, gating checksum, chain metadata length
- Query vector rounding & gating vector rounding

This ensures reproducibility & enables optional HMAC signing over minimal or extended payloads.

## 6. Complexity (High Level)

| Component | Cost (approx) |
|-----------|---------------|
| Mutual kNN build | O(k N log N) (partial sorts) |
| Laplacian assembly | O(k N) |
| CG solve (t iters) | O(t k N D) |
| Diffusion gating solve | Similar CG with single RHS |

Empirically t ≈ 20–40 for typical λ settings and moderate condition numbers.

## 7. Null Points & Residuals

Normalized residuals from the stationary linear system highlight edges where the local coherence assumption strains. High z residual edges → candidate anomalies or semantic discontinuities. These become `null_points` in the receipt.

## 8. Energy Reduction Interpretation

Lower ΔH_total under adaptive gating vs uniform indicates the lattice found a more internally coherent adjustment while still representing the query. In practice monitor both ΔH_total and top‑k alignment metrics.

## 9. Extension Hooks (Planned)
- Joint multi‑query block system
- Low‑rank Laplacian updates for streaming
- Preconditioners (Jacobi → ILU / spectral) to reduce CG iterations

## 10. References / Analogies
- Graph Laplacians & diffusion (Chung, 1997)
- Screened Poisson / diffusion kernels for spatial weighting
- Convex quadratic energy minimization under SPD systems

_This concise overview keeps the README lightweight; deep math stays in SPEC & future whitepaper._
