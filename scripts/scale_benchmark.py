#!/usr/bin/env python
"""Scaling benchmark: vary N, D, k and emit JSON lines.

Usage:
  python scripts/scale_benchmark.py --N 500 1000 2000 --D 64 128 --k 4 8 --trials 2 --seed 0 > scale.jsonl

Each line: {"N":..., "D":..., "k":..., "trial": t, "graph_build_ms":..., "ustar_solve_ms":..., "last_settle_ms":..., "deltaH":..., "ustar_iters":..., "ustar_res": ...}

Intended for quick local scaling curves and CI spot checks (keep sizes modest there).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from oscillink import OscillinkLattice


def run_case(N: int, D: int, k: int, trials: int, seed: int):
    rs = np.random.RandomState(seed)
    Y = rs.randn(N, D).astype(np.float32)
    psi = rs.randn(D).astype(np.float32)
    lat = OscillinkLattice(Y, kneighbors=k, deterministic_k=True)
    # Use light receipt detail to avoid O(N^2) diagnostics (null points, per-node components)
    # which can cause large memory usage at higher N during scaling benchmarks.
    lat.set_receipt_detail("light")
    lat.set_query(psi / (np.linalg.norm(psi) + 1e-12))
    lat.add_chain(list(range(min(4, N)))) if N >= 4 else None
    # warm settle
    lat.settle(max_iters=6, tol=1e-3)
    for t in range(trials):
        # force fresh stationary solve timing
        lat.refresh_Ustar(tol=1e-4, max_iters=64)
        rec = lat.receipt()
        out = {
            "N": N,
            "D": D,
            "k": k,
            "trial": t,
            "graph_build_ms": rec["meta"].get("graph_build_ms"),
            "ustar_solve_ms": rec["meta"].get("ustar_solve_ms"),
            "last_settle_ms": rec["meta"].get("last_settle_ms"),
            "deltaH": rec["deltaH_total"],
            "ustar_iters": rec["meta"].get("ustar_iters"),
            "ustar_res": rec["meta"].get("ustar_res"),
            "ustar_converged": rec["meta"].get("ustar_converged"),
        }
        print(json.dumps(out, separators=(",", ":")))
        sys.stdout.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", nargs="+", type=int, required=True, help="List of N sizes")
    p.add_argument("--D", nargs="+", type=int, required=True, help="List of D sizes")
    p.add_argument("--k", nargs="+", type=int, required=True, help="List of kneighbor values")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for N in args.N:
        for D in args.D:
            for k in args.k:
                run_case(N, D, k, args.trials, seed=args.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
