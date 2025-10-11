"""Example: Comparing uniform vs screened diffusion gating.

Run:
    python examples/diffusion_gated.py

Outputs a small table with:
  - deltaH_total
  - average bundle alignment
  - solve timings

This is a *qualitative* illustration; for rigorous benchmarking use
`scripts/benchmark_gating_compare.py` (added separately).
"""

from __future__ import annotations

import time

import numpy as np

from oscillink import OscillinkLattice, compute_diffusion_gates

N, D = 600, 96
rng = np.random.default_rng(1234)
Y = rng.normal(size=(N, D)).astype(np.float32)
psi = rng.normal(size=(D,)).astype(np.float32)
psi /= np.linalg.norm(psi) + 1e-12

# --- Uniform gating (baseline) ---
lat_u = OscillinkLattice(Y, kneighbors=6)
lat_u.set_query(psi)
T0 = time.time()
lat_u.settle()
settle_u_ms = 1000.0 * (time.time() - T0)
rec_u = lat_u.receipt()
bundle_u = lat_u.bundle(k=8)
mean_align_u = sum(b["align"] for b in bundle_u) / len(bundle_u)

# --- Diffusion gating ---
gates = compute_diffusion_gates(Y, psi, kneighbors=6, beta=1.0, gamma=0.15, neighbor_seed=77)
lat_d = OscillinkLattice(Y, kneighbors=6)
lat_d.set_query(psi, gates=gates)
T1 = time.time()
lat_d.settle()
settle_d_ms = 1000.0 * (time.time() - T1)
rec_d = lat_d.receipt()
bundle_d = lat_d.bundle(k=8)
mean_align_d = sum(b["align"] for b in bundle_d) / len(bundle_d)

print("=== Gating Comparison ===")
print(
    f"Uniform   ΔH={rec_u['deltaH_total']:.4f}  mean_align={mean_align_u:.4f}  settle_ms={settle_u_ms:.2f}"
)
print(
    f"Diffusion ΔH={rec_d['deltaH_total']:.4f}  mean_align={mean_align_d:.4f}  settle_ms={settle_d_ms:.2f}"
)
print(
    f"Gate stats: min={float(gates.min()):.3f} max={float(gates.max()):.3f} mean={float(gates.mean()):.3f}"
)
