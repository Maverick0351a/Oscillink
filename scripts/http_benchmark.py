#!/usr/bin/env python3
"""
Simple HTTP load benchmark for Oscillink Cloud endpoints.

Sends concurrent POST requests to /v1/settle (or any URL) and records per-request latency.
Writes JSONL with one row per request plus a summary JSON alongside.

Usage (PowerShell):
    python scripts/http_benchmark.py --url https://<rev-url>/v1/settle --api-key testkey ^
                 --requests 200 --concurrency 20 --out benchmarks\\http_canary.jsonl

Notes:
  - By default uses a tiny valid Oscillink payload; provide --payload-file to override.
  - Install dev extras first: `pip install -e .[dev]`
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import httpx


def _default_payload() -> Dict[str, Any]:
    return {
        "Y": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        "psi": [0.1, 0.2],
        "params": {"kneighbors": 2},
        "options": {"bundle_k": 2, "include_receipt": True},
    }


def _load_payload(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return _default_payload()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Result:
    t_start: float
    latency_ms: float
    status_code: int
    ok: bool


async def _worker(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    results: List[Result],
    iters: int,
) -> None:
    for _ in range(iters):
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json=payload, headers=headers, timeout=30)
            ok = r.status_code == 200
            status = r.status_code
        except Exception:
            ok = False
            status = 0
        t1 = time.perf_counter()
        results.append(Result(t_start=t0, latency_ms=(t1 - t0) * 1000.0, status_code=status, ok=ok))


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _write_outputs(path_jsonl: str, results: List[Result]) -> str:
    os.makedirs(os.path.dirname(path_jsonl) or ".", exist_ok=True)
    with open(path_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    # Summary next to JSONL
    lat = [r.latency_ms for r in results]
    ok = sum(1 for r in results if r.ok)
    total = len(results)
    summary = {
        "count": total,
        "ok": ok,
        "error": total - ok,
        "p50_ms": _percentile(lat, 0.50),
        "p90_ms": _percentile(lat, 0.90),
        "p95_ms": _percentile(lat, 0.95),
        "p99_ms": _percentile(lat, 0.99),
        "min_ms": min(lat) if lat else float("nan"),
        "max_ms": max(lat) if lat else float("nan"),
    }
    with open(os.path.splitext(path_jsonl)[0] + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return os.path.splitext(path_jsonl)[0] + "_summary.json"


async def _run(
    url: str,
    api_key: str,
    requests: int,
    concurrency: int,
    payload_file: Optional[str],
    out: str,
    host_header: Optional[str] = None,
) -> None:
    payload = _load_payload(payload_file)
    headers: Dict[str, str] = {"X-API-Key": api_key, "Content-Type": "application/json"}
    if host_header:
        # Allow overriding the Host header (useful for Cloud Run tagged revisions with TrustedHost middleware)
        headers["Host"] = host_header
    results: List[Result] = []
    # Prefer HTTP/2 if available; otherwise fall back to HTTP/1.1
    use_http2 = True
    try:
        import h2  # type: ignore  # noqa: F401
    except Exception:
        use_http2 = False
    async with httpx.AsyncClient(http2=use_http2, verify=True) as client:
        per_worker = requests // concurrency
        rem = requests % concurrency
        tasks = []
        for i in range(concurrency):
            iters = per_worker + (1 if i < rem else 0)
            if iters <= 0:
                continue
            tasks.append(
                asyncio.create_task(_worker(client, url, headers, payload, results, iters))
            )
        t0 = time.perf_counter()
        await asyncio.gather(*tasks)
        t1 = time.perf_counter()
        duration_s = t1 - t0
    summary_path = _write_outputs(out, results)
    lat = [r.latency_ms for r in results]
    print(
        json.dumps(
            {
                "url": url,
                "requests": requests,
                "concurrency": concurrency,
                "duration_s": duration_s,
                "throughput_rps": requests / duration_s if duration_s > 0 else None,
                "p50_ms": _percentile(lat, 0.50),
                "p90_ms": _percentile(lat, 0.90),
                "p95_ms": _percentile(lat, 0.95),
                "p99_ms": _percentile(lat, 0.99),
                "ok": sum(1 for r in results if r.ok),
                "errors": sum(1 for r in results if not r.ok),
                "out_jsonl": out,
                "summary": summary_path,
            },
            indent=2,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="HTTP load benchmark for Oscillink endpoints")
    ap.add_argument("--url", required=True, help="Target URL, e.g. https://.../v1/settle")
    ap.add_argument("--api-key", required=True, help="API key for X-API-Key header")
    ap.add_argument("--requests", type=int, default=200, help="Total number of requests")
    ap.add_argument("--concurrency", type=int, default=20, help="Concurrent workers")
    ap.add_argument("--payload-file", default=None, help="Optional JSON file with request payload")
    ap.add_argument(
        "--out", default=os.path.join("benchmarks", "http_canary.jsonl"), help="Output JSONL path"
    )
    ap.add_argument(
        "--host-header", default=None, help="Optional value for Host header (for Cloud Run tags)"
    )
    args = ap.parse_args()

    asyncio.run(
        _run(
            args.url,
            args.api_key,
            args.requests,
            args.concurrency,
            args.payload_file,
            args.out,
            args.host_header,
        )
    )


if __name__ == "__main__":
    main()
