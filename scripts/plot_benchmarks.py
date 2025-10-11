#!/usr/bin/env python3
"""
Plot benchmark results with labeled graphs.

Generates PNG charts under assets/benchmarks/ to include in README:
- competitor_single.png: latency & F1/hallucination bars for a single competitor_benchmark JSON
- scale_timing.png: scaling curves (graph build, solve, settle) from scale_benchmark JSONL

Usage (PowerShell):
  python scripts/plot_benchmarks.py --competitor c:\path\comp.json --out-dir assets\benchmarks
  python scripts/plot_benchmarks.py --scale c:\path\scale.jsonl --out-dir assets\benchmarks

Notes:
- Requires matplotlib; install dev extras: `pip install -e .[dev]`
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def _ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_competitor_single(path: str, out_dir: str) -> str:
    with open(path, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    # Latency bars
    labels = [
        "Cosine",
        "Oscillink (default)",
        "Oscillink (tuned)",
    ]
    times = [
        data.get("cosine_time_ms"),
        data.get("oscillink_default_time_ms"),
        data.get("oscillink_tuned_time_ms"),
    ]
    # F1 bars (skip None)
    f1s = [data.get("cosine_f1"), data.get("oscillink_default_f1"), data.get("oscillink_tuned_f1")]
    # Hallucination flags
    halls = [
        data.get("cosine_hallucination"),
        data.get("oscillink_default_hallucination"),
        data.get("oscillink_tuned_hallucination"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Latency
    ax = axes[0]
    vals = [t if isinstance(t, (int, float)) else float("nan") for t in times]
    ax.bar(labels, vals, color=["#888", "#2b8cbe", "#0868ac"]) 
    ax.set_title("Latency (ms)")
    ax.set_ylabel("ms")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    # F1
    ax = axes[1]
    vals_f1 = [f if isinstance(f, (int, float)) else float("nan") for f in f1s]
    ax.bar(labels, vals_f1, color=["#888", "#74c476", "#31a354"]) 
    ax.set_ylim(0, 1)
    ax.set_title("F1 (higher is better)")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    # Hallucination
    ax = axes[2]
    vals_h = [1 if h is True else 0 if h is False else float("nan") for h in halls]
    ax.bar(labels, vals_h, color=["#fb6a4a", "#9ecae1", "#6baed6"]) 
    ax.set_ylim(0, 1)
    ax.set_title("Hallucination present (1=yes)")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle("Competitor vs Oscillink (N={} k={})".format(data.get("N"), data.get("k")))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _ensure_out_dir(out_dir)
    out_path = os.path.join(out_dir, "competitor_single.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _read_jsonl_with_fallback(path: str) -> List[Dict[str, Any]]:
    # PowerShell redirects may produce UTF-16 files; try utf-8 then utf-16 then ignore
    encodings = ["utf-8", "utf-16"]
    text: List[str] = []
    for enc in encodings:
        try:
            with open(path, encoding=enc) as f:
                text = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    if not text:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.readlines()
    rows: List[Dict[str, Any]] = []
    for line in text:
        s = line.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except Exception:
            continue
    return rows


def plot_scale(path: str, out_dir: str) -> str:
    lines: List[Dict[str, Any]] = _read_jsonl_with_fallback(path)
    # Group by N (take mean of metrics per N)
    byN: Dict[int, Dict[str, float]] = {}
    counts: Dict[int, int] = {}
    for row in lines:
        N = int(row.get("N", 0))
        counts[N] = counts.get(N, 0) + 1
        agg = byN.setdefault(N, {"graph_build_ms": 0.0, "ustar_solve_ms": 0.0, "last_settle_ms": 0.0})
        for key in agg:
            val = row.get(key)
            if isinstance(val, (int, float)):
                agg[key] += float(val)
    for N, agg in byN.items():
        c = float(counts[N])
        for key in list(agg.keys()):
            agg[key] = agg[key] / c if c > 0 else float("nan")

    Ns = sorted(byN.keys())
    gb = [byN[n]["graph_build_ms"] for n in Ns]
    us = [byN[n]["ustar_solve_ms"] for n in Ns]
    ls = [byN[n]["last_settle_ms"] for n in Ns]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(Ns, gb, label="Graph build")
    ax.plot(Ns, us, label="Solve (U*)")
    ax.plot(Ns, ls, label="Settle")
    ax.set_xlabel("N (documents)")
    ax.set_ylabel("ms (mean)")
    ax.set_title("Scaling curves — lower is better")
    ax.legend()
    fig.tight_layout()
    _ensure_out_dir(out_dir)
    out_path = os.path.join(out_dir, "scale_timing.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot benchmark results to PNGs in assets/benchmarks/")
    ap.add_argument("--competitor", default=None, help="Path to competitor_benchmark JSON result")
    ap.add_argument("--scale", default=None, help="Path to scale_benchmark JSONL results")
    ap.add_argument("--out-dir", default=os.path.join("assets", "benchmarks"))
    args = ap.parse_args()

    if not args.competitor and not args.scale:
        ap.error("Provide at least --competitor or --scale")

    if args.competitor:
        path = plot_competitor_single(args.competitor, args.out_dir)
        print(f"Wrote {path}")
    if args.scale:
        path = plot_scale(args.scale, args.out_dir)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
