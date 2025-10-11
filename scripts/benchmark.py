#!/usr/bin/env python
"""Unified benchmark & differentiation proof for OscillinkLattice.

Modes:
    * Aggregate performance (default, trials>1) -- existing benchmark output
    * Proof mode (``--proof``) single run includes:
            - ΔH (energy) from receipt
            - Null point count (coherence gaps)
            - Chain verdict & weakest link (edge + z-score)
            - Optional bundle alignment mean (top-k bundle)
            - Optional diffusion gating comparison (``--diffusion``)

Examples:
    python scripts/benchmark.py --N 2000 --D 128 --kneighbors 8 --trials 3
    python scripts/benchmark.py --proof --N 500 --D 96 --bundle-k 8 --diffusion
    python scripts/benchmark.py --proof --json
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time
import tracemalloc

import numpy as np

from oscillink import OscillinkLattice, compute_diffusion_gates


def run_once(
    N: int,
    D: int,
    kneighbors: int,
    lamG: float,
    lamC: float,
    lamQ: float,
    lamP: float,
    chain_len: int,
    seed: int,
    memprof: bool = False,
    receipt_detail: str = "full",
):
    rs = np.random.RandomState(seed)
    Y = rs.randn(N, D).astype(np.float32)
    psi = (Y[: min(32, N)].mean(axis=0)).astype(np.float32)
    psi /= np.linalg.norm(psi) + 1e-12

    t0 = time.time()
    lat = OscillinkLattice(
        Y, kneighbors=kneighbors, lamG=lamG, lamC=lamC, lamQ=lamQ, deterministic_k=True
    )
    build_ms = 1000 * (time.time() - t0)

    lat.set_query(psi)
    set_receipt = getattr(lat, "set_receipt_detail", None)
    if callable(set_receipt):
        set_receipt(receipt_detail)
    chain = list(range(0, min(chain_len, N))) if chain_len >= 2 else None
    if chain and lamP > 0:
        lat.add_chain(chain, lamP=lamP)

    t1 = time.time()
    lat.settle(max_iters=12, tol=1e-3)
    settle_ms = 1000 * (time.time() - t1)

    t2 = time.time()
    rec = lat.receipt()
    receipt_ms = 1000 * (time.time() - t2)
    peak_mb = None
    if memprof:
        # Measure peak memory after main ops (requires start/stop around run in caller)
        current, peak = tracemalloc.get_traced_memory()
        peak_mb = float(peak) / 1024.0 / 1024.0

    chain_receipt = None
    weakest = None
    verdict = None
    if chain:
        chain_receipt = lat.chain_receipt(chain)
        weakest = chain_receipt.get("weakest_link")
        verdict = chain_receipt.get("verdict")

    return {
        "build_ms": build_ms,
        "settle_ms": settle_ms,
        "receipt_ms": receipt_ms,
        "deltaH": rec["deltaH_total"],
        "null_points": len(rec.get("null_points", [])),
        "sample_null": (rec.get("null_points") or [None])[0],
        "chain_verdict": verdict,
        "weakest_link": weakest,
        "ustar_iters": rec["meta"].get("ustar_iters"),
        "ustar_res": rec["meta"].get("ustar_res"),
        "ustar_converged": rec["meta"].get("ustar_converged"),
        "N": N,
        "D": D,
        "kneighbors": kneighbors,
        **({"peak_mb": peak_mb} if peak_mb is not None else {}),
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
    ap.add_argument("--json", action="store_true", help="Emit JSON with per-trial / proof stats")
    ap.add_argument(
        "--receipt-mode",
        dest="receipt_mode",
        type=str,
        default="full",
        choices=["full", "light"],
        help="Receipt detail level for timing",
    )
    ap.add_argument(
        "--memprof",
        action="store_true",
        help="Capture peak memory via tracemalloc and include in output",
    )
    ap.add_argument(
        "--proof", action="store_true", help="Run single-run proof output (energy, chain, nulls)"
    )
    ap.add_argument(
        "--bundle-k",
        type=int,
        default=0,
        help="When >0 in proof mode, also compute bundle and mean alignment",
    )
    ap.add_argument(
        "--diffusion", action="store_true", help="Compare diffusion gating (proof mode only)"
    )
    args = ap.parse_args()

    if args.proof:
        if args.memprof:
            tracemalloc.start()
        rows = [
            run_once(
                args.N,
                args.D,
                args.kneighbors,
                args.lamG,
                args.lamC,
                args.lamQ,
                args.lamP,
                args.chain_len,
                args.seed,
                memprof=args.memprof,
                receipt_detail=args.receipt_mode,
            )
        ]
        proof_row = rows[0]
        bundle_align_mean = None
        bundle = None
        if args.bundle_k > 0:
            # Reconstruct lattice quickly for bundle (reuse RNG seed for reproducibility)
            rs = np.random.RandomState(args.seed + 42)
            Yb = rs.randn(args.N, args.D).astype(np.float32)
            psi_b = (Yb[: min(32, args.N)].mean(axis=0)).astype(np.float32)
            psi_b /= np.linalg.norm(psi_b) + 1e-12
            lat_b = OscillinkLattice(
                Yb,
                kneighbors=args.kneighbors,
                lamG=args.lamG,
                lamC=args.lamC,
                lamQ=args.lamQ,
                deterministic_k=True,
            )
            lat_b.set_query(psi_b)
            chain_b = list(range(0, min(args.chain_len, args.N))) if args.chain_len >= 2 else None
            if chain_b and args.lamP > 0:
                lat_b.add_chain(chain_b, lamP=args.lamP)
            lat_b.settle(max_iters=12, tol=1e-3)
            bundle = lat_b.bundle(k=args.bundle_k)
            if bundle:
                bundle_align_mean = float(np.mean([b["align"] for b in bundle]))

        diffusion_block = None
        if args.diffusion:
            rsd = np.random.RandomState(args.seed + 99)
            Yd = rsd.randn(args.N, args.D).astype(np.float32)
            psi_d = (Yd[: min(32, args.N)].mean(axis=0)).astype(np.float32)
            psi_d /= np.linalg.norm(psi_d) + 1e-12
            gates = compute_diffusion_gates(
                Yd.astype(np.float32),
                psi_d.astype(np.float32),
                kneighbors=args.kneighbors,
                deterministic_k=True,
            )
            lat_u = OscillinkLattice(
                Yd,
                kneighbors=args.kneighbors,
                lamG=args.lamG,
                lamC=args.lamC,
                lamQ=args.lamQ,
                deterministic_k=True,
            )
            lat_u.set_query(psi_d)
            lat_u.settle(max_iters=12, tol=1e-3)
            rec_u = lat_u.receipt()
            lat_d = OscillinkLattice(
                Yd,
                kneighbors=args.kneighbors,
                lamG=args.lamG,
                lamC=args.lamC,
                lamQ=args.lamQ,
                deterministic_k=True,
            )
            lat_d.set_query(psi_d, gates=gates)
            lat_d.settle(max_iters=12, tol=1e-3)
            rec_d = lat_d.receipt()
            diffusion_block = {
                "uniform_deltaH": rec_u["deltaH_total"],
                "diffusion_deltaH": rec_d["deltaH_total"],
                "gate_min": float(gates.min()),
                "gate_max": float(gates.max()),
                "gate_mean": float(gates.mean()),
            }
        proof_payload = {
            "config": {"N": args.N, "D": args.D, "kneighbors": args.kneighbors},
            "deltaH": proof_row["deltaH"],
            "null_points": proof_row["null_points"],
            "sample_null": proof_row["sample_null"],
            "chain_verdict": proof_row["chain_verdict"],
            "weakest_link": proof_row["weakest_link"],
            "settle_ms": proof_row["settle_ms"],
            "bundle_mean_align": bundle_align_mean,
            "diffusion": diffusion_block,
        }
        if args.json:
            print(json.dumps(proof_payload, separators=(",", ":"), sort_keys=True))
            return
        print("=== Oscillink Proof ===")
        print(f"N={args.N} D={args.D} k={args.kneighbors}")
        print(
            f"ΔH={proof_row['deltaH']:.3f}  null_points={proof_row['null_points']} sample_null={proof_row['sample_null']}"
        )
        print("chain verdict:", proof_row["chain_verdict"], "weakest:", proof_row["weakest_link"])
        if bundle_align_mean is not None:
            print(f"bundle mean align: {bundle_align_mean:.4f}")
        print(f"settle_ms={proof_row['settle_ms']:.2f}")
        if args.memprof and rows and rows[0].get("peak_mb") is not None:
            print(f"peak_memory={rows[0]['peak_mb']:.1f} MB")
        if diffusion_block:
            print(
                f"diffusion ΔH={diffusion_block['diffusion_deltaH']:.3f} vs uniform ΔH={diffusion_block['uniform_deltaH']:.3f}"
            )
            print(
                "gates min={gate_min:.3f} max={gate_max:.3f} mean={gate_mean:.3f}".format(
                    **diffusion_block
                )
            )
        print("(Use --json for structured output)")
        return

    rows = []
    if args.memprof:
        tracemalloc.start()
    for t in range(args.trials):
        rows.append(
            run_once(
                args.N,
                args.D,
                args.kneighbors,
                args.lamG,
                args.lamC,
                args.lamQ,
                args.lamP,
                args.chain_len,
                args.seed + t,
                memprof=args.memprof,
                receipt_detail=args.receipt_mode,
            )
        )

    def agg(key):
        vals = [r[key] for r in rows]
        return (
            f"{stats.mean(vals):.2f}±{(stats.pstdev(vals) if len(vals) > 1 else 0):.2f}"
            if vals
            else "-"
        )

    if args.json:
        aggregates = {
            k: {
                "mean": float(stats.mean([r[k] for r in rows])),
                "stdev": float(stats.pstdev([r[k] for r in rows])) if len(rows) > 1 else 0.0,
            }
            for k in ["build_ms", "settle_ms", "receipt_ms", "deltaH", "ustar_iters", "ustar_res"]
        }
        if args.memprof and any("peak_mb" in r for r in rows):
            pm = [r.get("peak_mb") for r in rows if r.get("peak_mb") is not None]
            if pm:
                aggregates["peak_mb"] = {
                    "mean": float(stats.mean(pm)),
                    "stdev": float(stats.pstdev(pm)) if len(pm) > 1 else 0.0,
                }
        payload = {
            "config": {
                "N": args.N,
                "D": args.D,
                "kneighbors": args.kneighbors,
                "trials": args.trials,
                "lamG": args.lamG,
                "lamC": args.lamC,
                "lamQ": args.lamQ,
                "lamP": args.lamP,
                "chain_len": args.chain_len,
                "receipt_mode": args.receipt_mode,
            },
            "trials": rows,
            "aggregates": aggregates,
            "converged_all": all(r["ustar_converged"] for r in rows),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Oscillink Benchmark (trials={args.trials})")
        print(
            f"N={args.N} D={args.D} k={args.kneighbors} lamG={args.lamG} lamC={args.lamC} lamQ={args.lamQ} lamP={args.lamP}"
        )
        print(f"build_ms   : {agg('build_ms')}")
        print(f"settle_ms  : {agg('settle_ms')}")
        print(f"receipt_ms : {agg('receipt_ms')}")
        print(f"deltaH     : {agg('deltaH')}")
        print(
            f"ustar_iters: {agg('ustar_iters')}  res={agg('ustar_res')}  conv={rows[0]['ustar_converged']}"
        )
        # Show chain / null info from first trial for quick insight
        if rows[0]["chain_verdict"] is not None:
            print(f"chain verdict={rows[0]['chain_verdict']} weakest={rows[0]['weakest_link']}")
            print(f"null_points={rows[0]['null_points']} sample_null={rows[0]['sample_null']}")


if __name__ == "__main__":
    main()
