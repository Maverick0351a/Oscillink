#!/usr/bin/env python
"""Compare current benchmark metrics against a stored baseline with tolerance.

Usage:
  python scripts/perf_check.py --baseline scripts/perf_baseline.json --N 400 --D 64 --kneighbors 6 --trials 2

Exits non-zero if mean times regress by more than allowed tolerance percentage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_TOLERANCE_PCT = 35.0  # generous to avoid CI flakiness


def run_benchmark_json(args) -> dict:
    cmd = [
        sys.executable,
        "scripts/benchmark.py",
        "--json",
        "--N",
        str(args.N),
        "--D",
        str(args.D),
        "--kneighbors",
        str(args.kneighbors),
        "--trials",
        str(args.trials),
    ]
    if args.chain_len is not None:
        cmd.extend(["--chain-len", str(args.chain_len)])
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--kneighbors", type=int, default=6)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--chain-len", type=int, default=8)
    ap.add_argument("--tolerance-pct", type=float, default=DEFAULT_TOLERANCE_PCT)
    args = ap.parse_args()

    baseline = json.loads(args.baseline.read_text())
    current = run_benchmark_json(args)

    metrics = ["build_ms", "settle_ms", "receipt_ms"]
    failures = []
    for m in metrics:
        base_mean = baseline["aggregates"][m]["mean"]
        curr_mean = current["aggregates"][m]["mean"]
        if base_mean <= 0:
            continue
        pct = 100.0 * (curr_mean - base_mean) / base_mean
        if pct > args.tolerance_pct:
            failures.append(
                f"Metric {m} regressed {pct:.1f}% (baseline={base_mean:.2f}, current={curr_mean:.2f})"
            )
    result = {
        "baseline": baseline["aggregates"],
        "current": current["aggregates"],
        "tolerance_pct": args.tolerance_pct,
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if failures:
        print("Performance regression detected", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
