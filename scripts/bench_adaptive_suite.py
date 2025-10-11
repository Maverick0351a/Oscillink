#!/usr/bin/env python3
"""
Convenience wrapper to run the adaptive benchmark suite across datasets and modes.

Produces a consolidated JSON with results for:
- mars (random embeddings)
- mars (semantic embeddings)
- paris (random embeddings)
- paris (semantic embeddings)

Usage:
  python scripts/bench_adaptive_suite.py --trials 30 --k 3 --json
  python scripts/bench_adaptive_suite.py --trials 30 --k 5 --json --semantic-only

Notes:
- Keeps runtime reasonable by using the internal tune/test split of benchmark_adaptive.py
- You can adjust --tune-split per invocation; defaults align with the underlying script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Case:
    dataset: str
    semantic: bool
    k: int


def run_case(case: Case, trials: int, tune_split: float) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/benchmark_adaptive.py",
        "--dataset",
        case.dataset,
        "--trials",
        str(trials),
        "--k",
        str(case.k),
        "--tune-split",
        str(tune_split),
        "--json",
    ]
    if case.semantic:
        cmd.append("--semantic")
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out.strip())


def main():
    ap = argparse.ArgumentParser(description="Run adaptive benchmark suite and consolidate results")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tune-split", type=float, default=0.5)
    ap.add_argument("--semantic-only", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cases: List[Case] = []
    cases.append(Case(dataset="mars", semantic=False, k=args.k))
    cases.append(Case(dataset="mars", semantic=True, k=args.k))
    if not args.semantic_only:
        cases.append(Case(dataset="paris", semantic=False, k=args.k))
        cases.append(Case(dataset="paris", semantic=True, k=args.k))

    results: Dict[str, Any] = {
        "trials": args.trials,
        "k": args.k,
        "tune_split": args.tune_split,
        "cases": [],
    }
    for c in cases:
        res = run_case(c, args.trials, args.tune_split)
        results["cases"].append({"dataset": c.dataset, "semantic": c.semantic, "result": res})

    if args.json:
        print(json.dumps(results, separators=(",", ":")))
    else:
        print("=== Adaptive Benchmark Suite ===")
        print(f"trials={args.trials} k={args.k} tune_split={args.tune_split}")
        for entry in results["cases"]:
            r = entry["result"]
            name = f"{entry['dataset']}{'-sem' if entry['semantic'] else ''}"
            print(f"\n[{name}] adaptive_params={r.get('adaptive_params')}")
            print(
                "F1:",
                r.get("baseline_f1_mean"),
                r.get("default_f1_mean"),
                r.get("adaptive_f1_mean"),
            )
            print(
                "Hall:",
                r.get("baseline_hall_rate"),
                r.get("default_hall_rate"),
                r.get("adaptive_hall_rate"),
            )
            print(
                "Latency(ms):",
                r.get("baseline_time_ms_mean"),
                r.get("default_time_ms_mean"),
                r.get("adaptive_time_ms_mean"),
            )


if __name__ == "__main__":
    main()
