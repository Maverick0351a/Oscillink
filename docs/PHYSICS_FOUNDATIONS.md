This document has moved.

New location: [docs/foundations/PHYSICS_FOUNDATIONS.md](./foundations/PHYSICS_FOUNDATIONS.md)
# Physics Foundations of Oscillink Lattice

This note connects Oscillink’s math to physical intuition. We frame embeddings as a field settling on a graph under conservative forces (coherence), external drive (query), soft constraints (chains), and a controllable coupling (gating). The core consequence is a symmetric positive definite (SPD) energy with a unique stationary solution, enabling deterministic, auditable behavior at production scale.

- Read together with: [Math overview](./MATH_OVERVIEW.md) and the draft [Screened Diffusion outline](./DIFFUSION_WHITEPAPER_OUTLINE.md)
- API usage: [API](./API.md) · [Receipts](./RECEIPTS.md)

## 1) Energy landscape and stationary state

Let $U\in\mathbb{R}^{N\times D}$ be adjusted representations of anchors $Y\in\mathbb{R}^{N\times D}$ on a mutual $k$-NN graph with normalized Laplacian $L_{\mathrm{sym}}$. A query vector $\psi\in\mathbb{R}^D$ pulls the field via a diagonal coupling $B=\operatorname{diag}(b)$ with $b\in[0,1]^N$. An optional chain prior contributes an additional Laplacian $L_{\mathrm{path}}$.

We minimize the convex energy

$$
\begin{aligned}
H(U)
&= \lambda_G\,\|U-Y\|_F^2
 \\ &\quad+ \lambda_C\,\mathrm{tr}\big(U^\top L_{\mathrm{sym}}\,U\big)
 \\ &\quad+ \lambda_Q\,\mathrm{tr}\big((U-\mathbf{1}\,\psi^\top)^\top B\,(U-\mathbf{1}\,\psi^\top)\big)
 \\ &\quad+ \lambda_P\,\mathrm{tr}\big(U^\top L_{\mathrm{path}}\,U\big)
\end{aligned}
$$

The Euler–Lagrange condition $\nabla_U H=0$ yields a linear SPD system per column of $U$:

$$
\underbrace{\big(\lambda_G I + \lambda_C L_{\mathrm{sym}} + \lambda_Q B + \lambda_P L_{\mathrm{path}}\big)}_{M\succ0}\,U
= \lambda_G Y + \lambda_Q B\,\mathbf{1}\,\psi^\top.
$$

With $\lambda_G>0$ and positive semidefinite $L_{\mathrm{sym}},B,L_{\mathrm{path}}$, the operator $M$ is SPD, guaranteeing a unique minimizer $U^*$ and rapid Conjugate Gradient (CG) convergence. $H(U)$ acts as a Lyapunov function for any gradient-based settle scheme.

## 2) Gradient flow and physical interpretation

Continuous-time gradient descent follows

$$
\frac{\mathrm{d}U}{\mathrm{d}t} = -\nabla_U H(U) = -\,M\,U + \lambda_G Y + \lambda_Q B\,\mathbf{1}\,\psi^\top.
$$

The flow exponentially relaxes to the stationary solution $U^*$ of the linear system above. In practice we solve the SPD system directly (CG) rather than simulating dynamics; both viewpoints are equivalent in the limit.

- Data term ($\lambda_G$): springs anchoring $U$ to $Y$ (restoring force)
- Coherence term ($\lambda_C$): graph elasticity penalizing sharp variations over edges
- Query term ($\lambda_Q$): external field aligning $U$ towards $\psi$, gated by $b$
- Chain term ($\lambda_P$): elastic rod penalty along an ordered path (reasoning prior)

## 3) Diffusion gating as screened Poisson

Adaptive gates are obtained by a steady-state diffusion with absorption (screening):

$$
\big(L_{\mathrm{sym}} + \gamma I\big)\,h = \beta\,s,\qquad h\in\mathbb{R}^N,\ s\ge 0.
$$

- Source $s_i=\max\{0,\,\cos(y_i,\psi)\}$ injects query-relevant mass
- $\gamma>0$ sets a diffusion length scale $\ell\sim\gamma^{-1/2}$ (attenuates long-range spread)
- Normalize $h$ to $[0,1]$ and set $b=h$ to form $B=\operatorname{diag}(b)$

This screened diffusion spreads influence over coherent neighborhoods while suppressing off-topic or noisy regions (low $h$). See [Diffusion outline](./DIFFUSION_WHITEPAPER_OUTLINE.md) for spectral interpretations and extensions.

## 4) Control perspective and constraints

Gates $b$ act as a control field modulating the coupling to the query. The current implementation computes $b$ independently, but one can view a joint objective

$$
\mathcal{J}(U,h) = H(U;B=\operatorname{diag}(h)) + \tfrac{1}{2}\,\|\nabla_G h\|^2 + \tfrac{\gamma}{2}\,\|h\|^2 - \beta\,s^\top h
$$

subject to $0\le h\le 1$. Alternating minimization over $(U,h)$ recovers the screened Poisson equation in $h$ and the SPD linear system in $U$.

## 5) Chain prior as elastic energy

Given an ordered chain $C$, its path Laplacian $L_{\mathrm{path}}$ penalizes inconsistent variation along the reasoning trajectory. Physically this is a series of springs along the path; the receipt’s “tension” statistic is proportional to the standardized energy on chain edges. See [Chain guide](./CHAIN_GUIDE.md).

## 6) Receipts as energy accounting

Define the energy drop (improvement)

$$
\Delta H = H(Y) - H(U^*),
$$

computed efficiently via identities at the stationary point. We expose:
- $\Delta H_{\text{total}}$ (scalar)
- Per-node contributions (attribution)
- Null points: high standardized residual edges (locations of strained coherence)

Details in [Receipts](./RECEIPTS.md) and the [Math overview](./MATH_OVERVIEW.md).

## 7) Hallucination reduction as free-energy shaping

Heuristically, aggressive, off-topic pulls inject “free energy” that the lattice must dissipate over the graph, sometimes selecting incoherent items. Screened diffusion gates reduce spurious long-range injection, aligning with a free-energy regularization view. In probabilistic terms, one can frame an additional KL regularizer towards a background prior over neighborhoods; our deterministic gating approximates this behavior without sampling.

## 8) Determinism, signatures, and complexity

Because $M\succ0$ the solution is unique. We hash structural inputs (graph fingerprint, parameters, query/gate summaries) into a `state_sig` and optionally sign receipts. Complexity scales roughly as $\mathcal{O}(D\,t\,N\,k)$ for $t$ CG iterations; the diffusion solve is a single RHS of similar form. See [Math overview](./MATH_OVERVIEW.md) for a tabular summary.

## 9) Parameter-to-physics map (cheat sheet)

- $\lambda_G$ — spring stiffness to anchors (data fidelity)
- $\lambda_C$ — graph elasticity (coherence smoothing)
- $\lambda_Q$ — external field strength (query coupling)
- $\lambda_P$ — chain stiffness (path prior)
- $\beta$ — diffusion source gain; $\gamma$ — screening (attenuation length)
- $k$ — lattice connectivity (degree)

## 10) From theory to code

- Compute gates: `compute_diffusion_gates(Y, psi, kneighbors, beta, gamma)` → $b\in[0,1]^N$
- Set query: `lattice.set_query(psi, gates=b)` → forms $B=\operatorname{diag}(b)$
- Settle: `lattice.settle()` → solves $M\,U=\lambda_GY+\lambda_QB\,\mathbf{1}\,\psi^\top$ via CG
- Inspect: `lattice.receipt()` → $\Delta H$, per-node attribution, null points, timings

References:
- [Math overview](./MATH_OVERVIEW.md)
- [Screened Diffusion outline](./DIFFUSION_WHITEPAPER_OUTLINE.md)
- [API](./API.md) · [Receipts](./RECEIPTS.md) · [Chain guide](./CHAIN_GUIDE.md)
