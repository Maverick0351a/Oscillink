# Roadmap

This roadmap outlines near–term developer tooling and platform work (Phase 2) and forward–looking research and networking directions (Phase 3). Phase 1 milestones (SDK core) are complete and tracked in CHANGELOG.

---

## Phase 2: Developer Tools & Platform

| Item | Package | Status | Notes |
|---|---|---|---|
| Oscillink Debugger/Visualizer | `oscillink-debug` | Planned | WebGL energy surface, solver traces, null-point heat maps |
| Universal Model Adapter | `oscillink-adapters` | Planned | Provider registry (OpenAI/Cohere/Anthropic/HF/local), unit-norm, cache |
| Self-Optimizing Database | `oscillink-db` | Planned | EMA + epsilon-greedy, receipts-driven metrics, Firestore profiles |

### 1. Oscillink Debugger/Visualizer (oscillink-debug)

Capabilities:
- 3D energy landscape visualization
- Real–time settling animation with iteration tracking
- Null–point heat maps showing semantic discontinuities
- Interactive graph structure explorer

Technical approach:
- WebGL rendering of energy function H(U) surface and solver traces

### 2. Universal Model Adapter (oscillink-adapters)

Capabilities:
- Auto–detect embedding providers (OpenAI, Cohere, Anthropic, Hugging Face, local)
- Automatic normalization to unit vectors in L2 space
- Content–hash based caching with TTL

Technical approach:
- Provider registry with dtype/shape validation; pluggable backends

### 3. Self–Optimizing Database (oscillink-db)

Capabilities:
- Persistent parameter evolution using EMA with bounded updates
- Query pattern learning via state–signature clustering
- Distributed consensus using SPD convergence guarantees

Technical approach:
- Firestore–backed profiles with epsilon–greedy exploration; receipts–driven metrics

---

## Phase 3: Network & Intelligence

### 4. Coherent Intranet for AI (oscillink-mesh)

Capabilities:
- Private memory networks between AI agents
- Coherence–preserving message passing
- Consensus via energy minimization across nodes

Technical approach:
- Distributed SPD system where each node maintains local U; global convergence via gradient/energy exchange

### 5. Quantum Memory (oscillink-quantum)

Concepts:
- Map quantum states to vector representations: ρ → Y ∈ ℝ^(N×D)
- Null points become quantum error syndrome detectors
- Real–time decoherence tracking via ΔH monitoring

Mathematical foundation:
- H_quantum = λ_G||ρ − ρ_target||_F² + λ_C·tr(ρᵀ L_sym ρ) + λ_L·tr(L(ρ))

Feasible implementation:
- Simulate with density matrices first; integrate with Qiskit/Cirq for validation

### 6. Federation Protocol (oscillink-federated)

Capabilities:
- Cross–organization memory sharing with privacy preservation
- Homomorphic operations on encrypted embeddings

Technical approach:
- Secure multi–party computation on SPD matrices; encrypted message passing

---

Notes:
- Phase sequencing is indicative; components can be developed in parallel behind feature flags.
- We will preserve the core design goals: universality (bring–your–own embeddings), deterministic receipts, minimal dependency footprint.
