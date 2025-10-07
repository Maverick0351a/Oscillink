#!/usr/bin/env python
"""Lightweight performance benchmark for OscillinkLattice.

Example:
  python scripts/benchmark.py --N 2000 --D 128 --kneighbors 8 --trials 3
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import time

import numpy as np

from oscillink import OscillinkLattice


def run_once(N: int, D: int, kneighbors: int, lamG: float, lamC: float, lamQ: float, lamP: float, chain_len: int, seed: int):
    rs = np.random.RandomState(seed)
    Y = rs.randn(N, D).astype(np.float32)
    psi = (Y[: min(32, N)].mean(axis=0)).astype(np.float32)
    psi /= (np.linalg.norm(psi) + 1e-12)

    t0 = time.time()
    lat = OscillinkLattice(Y, kneighbors=kneighbors, lamG=lamG, lamC=lamC, lamQ=lamQ, deterministic_k=True)
    build_ms = 1000 * (time.time() - t0)

    lat.set_query(psi)
    chain = list(range(0, min(chain_len, N))) if chain_len >= 2 else None
    if chain and lamP > 0:
        lat.add_chain(chain, lamP=lamP)

    t1 = time.time()
    lat.settle(max_iters=12, tol=1e-3)
    settle_ms = 1000 * (time.time() - t1)

    t2 = time.time()
    rec = lat.receipt()
    receipt_ms = 1000 * (time.time() - t2)

    return {
        "build_ms": build_ms,
        "settle_ms": settle_ms,
        "receipt_ms": receipt_ms,
        "deltaH": rec["deltaH_total"],
        "ustar_iters": rec["meta"].get("ustar_iters"),
        "ustar_res": rec["meta"].get("ustar_res"),
        "ustar_converged": rec["meta"].get("ustar_converged"),
        "N": N,
        "D": D,
        "kneighbors": kneighbors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--kneighbors", type=int, default=6)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--lamG", type=float, default=1.0)
    ap.add_argument("--lamC", type=float, default=0.5)
    ap.add_argument("--lamQ", type=float, default=4.0)
    ap.add_argument("--lamP", type=float, default=0.2)
    ap.add_argument("--chain-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="Emit JSON with per-trial and aggregate stats")
    args = ap.parse_args()

    rows = []
    for t in range(args.trials):
        rows.append(run_once(args.N, args.D, args.kneighbors, args.lamG, args.lamC, args.lamQ, args.lamP, args.chain_len, args.seed + t))

    def agg(key):
        vals = [r[key] for r in rows]
        return f"{stats.mean(vals):.2f}±{(stats.pstdev(vals) if len(vals)>1 else 0):.2f}" if vals else "-"

    if args.json:
        aggregates = {k: {
            "mean": float(stats.mean([r[k] for r in rows])),
            "stdev": float(stats.pstdev([r[k] for r in rows])) if len(rows) > 1 else 0.0
        } for k in ["build_ms","settle_ms","receipt_ms","deltaH","ustar_iters","ustar_res"]}
        payload = {
            "config": {"N": args.N, "D": args.D, "kneighbors": args.kneighbors, "trials": args.trials, "lamG": args.lamG, "lamC": args.lamC, "lamQ": args.lamQ, "lamP": args.lamP, "chain_len": args.chain_len},
            "trials": rows,
            "aggregates": aggregates,
            "converged_all": all(r["ustar_converged"] for r in rows),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Oscillink Benchmark (trials={args.trials})")
        print(f"N={args.N} D={args.D} k={args.kneighbors} lamG={args.lamG} lamC={args.lamC} lamQ={args.lamQ} lamP={args.lamP}")
        print(f"build_ms   : {agg('build_ms')}")
        print(f"settle_ms  : {agg('settle_ms')}")
        print(f"receipt_ms : {agg('receipt_ms')}")
        print(f"deltaH     : {agg('deltaH')}")
        print(f"ustar_iters: {agg('ustar_iters')}  res={agg('ustar_res')}  conv={rows[0]['ustar_converged']}")

if __name__ == "__main__":
    main()
