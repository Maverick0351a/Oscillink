#!/usr/bin/env python3
"""
Plot benchmark results with labeled graphs.

Generates PNG charts under assets/benchmarks/ to include in README:
- competitor_single.png: latency & F1/hallucination bars for a single competitor_benchmark JSON
- competitor_multi.png: aggregated means with 95% CI error bars when competitor JSON contains aggregate
- scale_timing.png: scaling curves (graph build, solve, settle) from scale_benchmark JSONL
- http_latency.png: HTTP latency CDF/histogram from http_benchmark JSONL

Usage (PowerShell):
    python scripts/plot_benchmarks.py --competitor C:\\path\\comp.json --out-dir assets\\benchmarks
    python scripts/plot_benchmarks.py --scale C:\\path\\scale.jsonl --out-dir assets\\benchmarks
    python scripts/plot_benchmarks.py --http C:\\path\\http.jsonl --out-dir assets\\benchmarks

Notes:
- Requires matplotlib; install dev extras: `pip install -e .[dev]`
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


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
    # Prefer trap share if available; fallback to hallucination boolean
    trap_shares = [
        data.get("cosine_trap_share"),
        data.get("oscillink_default_trap_share"),
        data.get("oscillink_tuned_trap_share"),
    ]
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
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    # F1
    ax = axes[1]
    vals_f1 = [f if isinstance(f, (int, float)) else float("nan") for f in f1s]
    ax.bar(labels, vals_f1, color=["#888", "#74c476", "#31a354"])
    ax.set_ylim(0, 1)
    ax.set_title("F1 (higher is better)")
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    # Traps / hallucination panel
    ax = axes[2]
    if any(isinstance(t, (int, float)) for t in trap_shares):
        vals_ts = [
            (t * 100.0 if isinstance(t, (int, float)) else float("nan")) for t in trap_shares
        ]
        ax.bar(labels, vals_ts, color=["#fb6a4a", "#9ecae1", "#6baed6"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of top-k that are traps")
        ax.set_title("Trap share (lower is better)")
    else:
        vals_h = [1 if h is True else 0 if h is False else float("nan") for h in halls]
        ax.bar(labels, vals_h, color=["#fb6a4a", "#9ecae1", "#6baed6"])
        ax.set_ylim(0, 1)
        ax.set_title("Hallucination present (1=yes)")
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle("Competitor vs Oscillink (N={} k={})".format(data.get("N"), data.get("k")))
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    _ensure_out_dir(out_dir)
    out_path = os.path.join(out_dir, "competitor_single.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_competitor_multi(path: str, out_dir: str) -> str:
    with open(path, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    agg: Dict[str, Any] = data.get("aggregate", {})
    n_queries = len(data.get("per_query", []))
    N = data.get("N")
    k = data.get("k")

    labels = [
        "Cosine",
        "Oscillink (default)",
        "Oscillink (tuned)",
    ]

    def _mean_ci(block: Dict[str, Any], key: str) -> Tuple[float, float]:
        sub = block.get(key) if isinstance(block, dict) else None
        if isinstance(sub, dict):
            mean = sub.get("mean")
            ci = sub.get("ci95")
            m = float(mean) if isinstance(mean, (int, float)) else float("nan")
            c = float(ci) if isinstance(ci, (int, float)) else 0.0
            return m, c
        return float("nan"), 0.0

    cos = agg.get("cosine", {})
    dfl = agg.get("oscillink_default", {})
    tun = agg.get("oscillink_tuned", {})

    # Latency
    lat_means = []
    lat_errs = []
    for blk in (cos, dfl, tun):
        m, e = _mean_ci(blk, "time_ms")
        lat_means.append(m)
        lat_errs.append(e)

    # F1
    f1_means = []
    f1_errs = []
    for blk in (cos, dfl, tun):
        m, e = _mean_ci(blk, "f1")
        f1_means.append(m)
        f1_errs.append(e)

    # Trap share -> percent
    ts_means = []
    ts_errs = []
    for blk in (cos, dfl, tun):
        m, e = _mean_ci(blk, "trap_share")
        m = m * 100.0 if np.isfinite(m) else m
        e = e * 100.0
        ts_means.append(m)
        ts_errs.append(e)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    # Latency
    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x, lat_means, yerr=lat_errs, capsize=4, color=["#888", "#2b8cbe", "#0868ac"])
    ax.set_title("Latency (ms) — mean ± 95% CI")
    ax.set_ylabel("ms")
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=20, ha="right")

    # F1
    ax = axes[1]
    ax.bar(x, f1_means, yerr=f1_errs, capsize=4, color=["#888", "#74c476", "#31a354"])
    ax.set_ylim(0, 1)
    ax.set_title("F1 (higher is better) — mean ± 95% CI")
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=20, ha="right")

    # Trap share
    ax = axes[2]
    ax.bar(x, ts_means, yerr=ts_errs, capsize=4, color=["#fb6a4a", "#9ecae1", "#6baed6"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of top-k that are traps")
    ax.set_title("Trap share (lower is better) — mean ± 95% CI")
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=20, ha="right")

    fig.suptitle(f"Aggregated competitor vs Oscillink (Q={n_queries}, N={N}, k={k})")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    _ensure_out_dir(out_dir)
    out_path = os.path.join(out_dir, "competitor_multi.png")
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
        agg = byN.setdefault(
            N, {"graph_build_ms": 0.0, "ustar_solve_ms": 0.0, "last_settle_ms": 0.0}
        )
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


def _ecdf(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr.sort()
    n = arr.size
    if n == 0:
        return np.array([]), np.array([])
    y = np.arange(1, n + 1) / n
    return arr, y


def plot_http_latency(path: str, out_dir: str) -> str:
    rows: List[Dict[str, Any]] = _read_jsonl_with_fallback(path)
    lat = [float(r.get("latency_ms", float("nan"))) for r in rows if r.get("ok")]
    xs, ys = _ecdf(lat)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # CDF
    ax = axes[0]
    ax.plot(xs, ys, label="CDF")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Fraction ≤ x")
    ax.set_title("HTTP latency CDF (successful requests)")
    # Histogram
    ax = axes[1]
    ax.hist(lat, bins=30, color="#9ecae1", edgecolor="#3182bd")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("HTTP latency histogram")
    fig.tight_layout()
    _ensure_out_dir(out_dir)
    out_path = os.path.join(out_dir, "http_latency.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot benchmark results to PNGs in assets/benchmarks/")
    ap.add_argument("--competitor", default=None, help="Path to competitor_benchmark JSON result")
    ap.add_argument("--scale", default=None, help="Path to scale_benchmark JSONL results")
    ap.add_argument("--out-dir", default=os.path.join("assets", "benchmarks"))
    ap.add_argument("--http", default=None, help="Path to http_benchmark JSONL results")
    args = ap.parse_args()

    if not args.competitor and not args.scale and not args.http:
        ap.error("Provide at least --competitor or --scale or --http")

    if args.competitor:
        # Auto-detect if the competitor result is a multi-query aggregate
        try:
            with open(args.competitor, encoding="utf-8") as f:
                comp_data = json.load(f)
        except Exception:
            comp_data = {}
        if isinstance(comp_data, dict) and "aggregate" in comp_data:
            path = plot_competitor_multi(args.competitor, args.out_dir)
        else:
            path = plot_competitor_single(args.competitor, args.out_dir)
        print(f"Wrote {path}")
    if args.scale:
        path = plot_scale(args.scale, args.out_dir)
        print(f"Wrote {path}")
    if args.http:
        path = plot_http_latency(args.http, args.out_dir)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
