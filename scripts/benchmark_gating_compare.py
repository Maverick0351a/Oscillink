#!/usr/bin/env python
"""Compare uniform vs diffusion gating performance & basic metrics.

Usage:
  python scripts/benchmark_gating_compare.py --N 1200 --D 128 --kneighbors 8 --trials 3 --gamma 0.15 --beta 1.0 --json

Outputs summary of:
  - mean settle ms
  - mean receipt ms
  - mean ΔH
  - mean bundle alignment (top-k)

JSON includes per-trial detail for both modes.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from oscillink import OscillinkLattice, compute_diffusion_gates


def run_once(N: int, D: int, kneighbors: int, beta: float, gamma: float, bundle_k: int, seed: int):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(N, D)).astype(np.float32)
    psi = rng.normal(size=(D,)).astype(np.float32)

    # Uniform
    lat_u = OscillinkLattice(Y, kneighbors=kneighbors)
    lat_u.set_query(psi)
    t0 = time.time()
    lat_u.settle()
    settle_u_ms = 1000.0 * (time.time() - t0)
    t1 = time.time()
    rec_u = lat_u.receipt()
    receipt_u_ms = 1000.0 * (time.time() - t1)
    bundle_u = lat_u.bundle(k=bundle_k)
    mean_align_u = float(np.mean([b["align"] for b in bundle_u])) if bundle_u else 0.0

    # Diffusion
    t2 = time.time()
    gates = compute_diffusion_gates(
        Y,
        psi,
        kneighbors=kneighbors,
        beta=beta,
        gamma=gamma,
        neighbor_seed=seed,
    )
    gate_ms = 1000.0 * (time.time() - t2)
    lat_d = OscillinkLattice(Y, kneighbors=kneighbors)
    lat_d.set_query(psi, gates=gates)
    t3 = time.time()
    lat_d.settle()
    settle_d_ms = 1000.0 * (time.time() - t3)
    t4 = time.time()
    rec_d = lat_d.receipt()
    receipt_d_ms = 1000.0 * (time.time() - t4)
    bundle_d = lat_d.bundle(k=bundle_k)
    mean_align_d = float(np.mean([b["align"] for b in bundle_d])) if bundle_d else 0.0

    return {
        "uniform": {
            "settle_ms": settle_u_ms,
            "receipt_ms": receipt_u_ms,
            "deltaH_total": rec_u["deltaH_total"],
            "mean_align": mean_align_u,
        },
        "diffusion": {
            "gate_build_ms": gate_ms,
            "settle_ms": settle_d_ms,
            "receipt_ms": receipt_d_ms,
            "deltaH_total": rec_d["deltaH_total"],
            "mean_align": mean_align_d,
            "gate_min": float(gates.min()),
            "gate_max": float(gates.max()),
            "gate_mean": float(gates.mean()),
        },
    }


def summarize(trials):
    def resolve_path(d, path):
        cur = d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"Path component '{part}' missing while resolving '{path}'")
            cur = cur[part]
        return cur

    def agg(path):
        vals = [resolve_path(t, path) for t in trials]
        return float(np.mean(vals)), float(np.std(vals) if len(vals) > 1 else 0.0)

    metrics = [
        ("uniform.settle_ms", "uniform_settle_ms"),
        ("uniform.receipt_ms", "uniform_receipt_ms"),
        ("uniform.deltaH_total", "uniform_deltaH"),
        ("uniform.mean_align", "uniform_mean_align"),
        ("diffusion.gate_build_ms", "diff_gate_ms"),
        ("diffusion.settle_ms", "diff_settle_ms"),
        ("diffusion.receipt_ms", "diff_receipt_ms"),
        ("diffusion.deltaH_total", "diff_deltaH"),
        ("diffusion.mean_align", "diff_mean_align"),
    ]
    out = {}
    for path, key in metrics:
        mean, std = agg(path)
        out[key] = {"mean": mean, "std": std}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=800)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--kneighbors", type=int, default=8)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.15)
    ap.add_argument("--bundle_k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    trials = []
    for t in range(args.trials):
        trials.append(
            run_once(
                args.N, args.D, args.kneighbors, args.beta, args.gamma, args.bundle_k, args.seed + t
            )
        )

    summary = summarize(trials)

    if args.json:
        print(
            json.dumps(
                {"config": vars(args), "summary": summary, "trials": trials},
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    else:
        print("=== Gating Benchmark Summary ===")
        for k, v in summary.items():
            print(f"{k:22s} mean={v['mean']:.3f} ms/std={v['std']:.3f}")
        uplift = summary["diff_mean_align"]["mean"] - summary["uniform_mean_align"]["mean"]
        print(f"Alignment uplift (diffusion - uniform): {uplift:.4f}")


if __name__ == "__main__":
    main()
