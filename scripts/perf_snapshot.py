#!/usr/bin/env python
"""Capture a performance snapshot JSON using benchmark --json output."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--kneighbors", type=int, default=6)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--chain-len", type=int, default=8)
    args = ap.parse_args()
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
        "--chain-len",
        str(args.chain_len),
    ]
    out = subprocess.check_output(cmd, text=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Wrote snapshot to {args.out}")


if __name__ == "__main__":
    main()
