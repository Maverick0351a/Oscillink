#!/usr/bin/env python
"""Simple micro-benchmark for OscillinkLattice.
Run: python scripts/bench.py --N 2000 --D 256 --k 8 --iters 12
"""

import argparse
import json
import statistics
import time

import numpy as np

from oscillink.core.lattice import OscillinkLattice


def run_once(N: int, D: int, k: int, iters: int, seed: int):
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((N, D), dtype=np.float32)
    psi = Y[: max(1, N // 12)].mean(axis=0)
    psi /= np.linalg.norm(psi) + 1e-12
    t0 = time.time()
    lat = OscillinkLattice(Y, kneighbors=k)
    build_t = (time.time() - t0) * 1000
    lat.set_query(psi.astype(np.float32))
    t1 = time.time()
    lat.settle(max_iters=iters)
    settle_t = (time.time() - t1) * 1000
    r = lat.receipt()
    return {
        "build_ms": build_t,
        "settle_ms": settle_t,
        "cg_iters": r["cg_iters"],
        "deltaH": r["deltaH_total"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    rows = [run_once(args.N, args.D, args.k, args.iters, seed=i) for i in range(args.runs)]
    agg = {
        "N": args.N,
        "D": args.D,
        "k": args.k,
        "iters": args.iters,
        "runs": args.runs,
        "build_ms_mean": statistics.mean(r["build_ms"] for r in rows),
        "settle_ms_mean": statistics.mean(r["settle_ms"] for r in rows),
        "cg_iters_mean": statistics.mean(r["cg_iters"] for r in rows),
        "deltaH_mean": statistics.mean(r["deltaH"] for r in rows),
    }
    print(json.dumps({"runs": rows, "summary": agg}, indent=2))


if __name__ == "__main__":
    main()
