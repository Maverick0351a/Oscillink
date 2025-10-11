#!/usr/bin/env python3
"""
Competitor benchmark CLI

Compare:
- Baseline cosine
- Oscillink (default and optional tiny tuning)
- FAISS Flat / HNSW (if faiss is installed)
- Annoy (if annoy is installed)

Outputs JSON with latency and quality metrics (F1, hallucination, trap-share) when labels/traps are provided.

Usage (Windows PowerShell):
  python scripts/competitor_benchmark.py --input examples\real_benchmark_sample.jsonl --format jsonl --text-col text \
    --id-col id --label-col label --trap-col trap --query-index 0 --k 5 --json

Optional:
  --tune (tiny grid for Oscillink if labels present)
  --smart-correct (preprocess texts/query with conservative autocorrect)
    --out <path> (save the JSON summary to a file)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oscillink.adapters.text import embed_texts
from oscillink.core.lattice import OscillinkLattice
from oscillink.preprocess.autocorrect import smart_correct


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def cosine_topk(
    psi: np.ndarray, Y: np.ndarray, k: int, exclude_idx: Optional[int] = None
) -> List[int]:
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    pn = psi / (np.linalg.norm(psi) + 1e-9)
    scores = Yn @ pn
    if exclude_idx is not None and 0 <= exclude_idx < len(scores):
        scores[exclude_idx] = -1e9
    idx = np.argsort(-scores)[:k]
    return idx.tolist()


def eval_topk(
    pred: List[int], labels: Optional[List[int]], traps: Optional[List[int]], k: int
) -> Tuple[Optional[float], Optional[bool], Optional[float]]:
    if labels is None:
        if traps is None:
            return None, None, None
        hall = any((i in traps) for i in pred)
        tshare = float(sum(1 for i in pred if i in traps)) / max(1, k)
        return None, hall, tshare
    gt_ids = {i for i, lab in enumerate(labels) if int(lab) == 1}
    tp = len([i for i in pred if i in gt_ids])
    fp = len([i for i in pred if i not in gt_ids])
    fn = len([i for i in gt_ids if i not in pred])
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (prec == 0.0 and rec == 0.0) else (2 * prec * rec / (prec + rec))
    hall = None
    tshare = None
    if traps is not None:
        hall = any((i in traps) for i in pred)
        tshare = float(sum(1 for i in pred if i in traps)) / max(1, k)
    return f1, hall, tshare


def select_query(
    query: Optional[str], query_index: Optional[int], texts: List[str]
) -> Tuple[str, Optional[int]]:
    if query is not None:
        return query, None
    if query_index is not None and 0 <= query_index < len(texts):
        return texts[query_index], query_index
    return texts[0], 0 if len(texts) > 0 else None


def _infer_format(path: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    import os as _os

    ext = _os.path.splitext(path)[1].lower()
    if ext in {".jsonl", ".ndjson"}:
        return "jsonl"
    if ext == ".csv":
        return "csv"
    return "jsonl"


def _prepare_fields(
    rows: List[Dict[str, Any]],
    *,
    text_col: str,
    id_col: Optional[str],
    label_col: Optional[str],
    trap_col: Optional[str],
) -> tuple[List[str], List[Any], Optional[List[int]], Optional[List[int]]]:
    texts: List[str] = []
    ids: List[Any] = []
    labels: Optional[List[int]] = [] if label_col is not None else None
    for r in rows:
        t = r.get(text_col)
        if t is None:
            continue
        texts.append(str(t))
        rid = r.get(id_col) if id_col else len(ids)
        ids.append(rid)
        if labels is not None and label_col is not None:
            try:
                val = r.get(label_col)
                labels.append(int(val) if val is not None else 0)
            except Exception:
                labels.append(0)
    traps: Optional[List[int]] = None
    if trap_col is not None:
        traps = []
        for i, r in enumerate(rows[: len(texts)]):
            try:
                if int(r.get(trap_col, 0)) == 1:
                    traps.append(i)
            except Exception:
                continue
    return texts, ids, labels, traps


def maybe_smart_correct(texts: List[str], q_text: str, enabled: bool) -> Tuple[List[str], str]:
    if not enabled:
        return texts, q_text
    return [smart_correct(t) for t in texts], smart_correct(q_text)


def run_cosine_baseline(
    psi: np.ndarray, Y: np.ndarray, k: int, q_idx: Optional[int]
) -> Tuple[List[int], float]:
    t0 = time.time()
    pred = cosine_topk(psi, Y, k, exclude_idx=q_idx)
    return pred, 1000.0 * (time.time() - t0)


def run_faiss_nn(
    psi: np.ndarray, Y: np.ndarray, k: int, q_idx: Optional[int]
) -> Tuple[Optional[List[int]], Optional[float]]:
    try:
        import faiss  # type: ignore

        YN = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
        psiN = psi / (np.linalg.norm(psi) + 1e-9)
        index = faiss.IndexFlatIP(Y.shape[1])
        index.add(YN.astype(np.float32))
        t2 = time.time()
        dists, idxs = index.search(psiN.reshape(1, -1), k + (1 if q_idx is not None else 0))
        ms = 1000.0 * (time.time() - t2)
        cand = idxs[0].tolist()
        if q_idx is not None and q_idx in cand:
            cand = [c for c in cand if c != q_idx]
        return cand[:k], ms
    except Exception:
        return None, None


def run_annoy_nn(
    psi: np.ndarray, Y: np.ndarray, k: int, q_idx: Optional[int]
) -> Tuple[Optional[List[int]], Optional[float]]:
    try:
        from annoy import AnnoyIndex  # type: ignore

        dim = int(Y.shape[1])
        t = AnnoyIndex(dim, metric="angular")
        for i, v in enumerate(Y.tolist()):
            t.add_item(i, v)
        t.build(10)
        t3 = time.time()
        cand = t.get_nns_by_vector(psi.tolist(), k + (1 if q_idx is not None else 0))
        ms = 1000.0 * (time.time() - t3)
        if q_idx is not None and q_idx in cand:
            cand = [c for c in cand if c != q_idx]
        return cand[:k], ms
    except Exception:
        return None, None


def _tune_params(
    Y: np.ndarray,
    psi: np.ndarray,
    *,
    k: int,
    kneighbors: int,
    lamG: float,
    lamC: float,
    lamQ: float,
    trials: int,
    labels: Optional[List[int]],
    traps: Optional[List[int]],
) -> Dict[str, Any]:
    if labels is None:
        return {"lamG": lamG, "lamC": lamC, "lamQ": lamQ, "kneighbors": kneighbors}
    lamC_grid = [max(0.1, lamC * s) for s in [0.6, 1.0, 1.4]]
    lamQ_grid = [max(0.5, lamQ * s) for s in [0.5, 1.0, 1.5]]
    k_grid = sorted(
        set(
            [max(1, min(Y.shape[0] - 1, kk)) for kk in [kneighbors - 2, kneighbors, kneighbors + 2]]
        )
    )
    rng = np.random.default_rng(42)
    best_f1 = -1.0
    best = {"lamG": lamG, "lamC": lamC, "lamQ": lamQ, "kneighbors": kneighbors}
    for lc in lamC_grid:
        for lq in lamQ_grid:
            for kk in k_grid:
                f1s: List[float] = []
                for _ in range(max(1, int(trials))):
                    jitter = rng.standard_normal(psi.shape).astype(np.float32) * 0.01
                    psi_t = (psi + jitter) / (np.linalg.norm(psi + jitter) + 1e-9)
                    lat_t = OscillinkLattice(
                        Y, kneighbors=kk, lamG=lamG, lamC=lc, lamQ=lq, deterministic_k=True
                    )
                    lat_t.set_query(psi_t)
                    lat_t.settle(max_iters=12, tol=1e-3)
                    pred_t = [int(item.get("id", -1)) for item in lat_t.bundle(k=k)]
                    f1_t, _, _ = eval_topk(pred_t, labels, traps, k)
                    if f1_t is not None:
                        f1s.append(float(f1_t))
                mean_f1 = float(np.mean(f1s)) if f1s else -1.0
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best = {"lamG": lamG, "lamC": lc, "lamQ": lq, "kneighbors": kk}
    return best


def _score_value(
    f1: Optional[float],
    trap_share: Optional[float],
    *,
    score_type: str,
    alpha: float,
) -> float:
    f = 0.0 if f1 is None else float(f1)
    t = 0.0 if trap_share is None else float(trap_share)
    if score_type == "f1":
        return f
    if score_type == "trap_penalized":
        # linear penalty
        return f - alpha * t
    if score_type == "f1_times_one_minus_traps":
        return f * (1.0 - t)
    return f


def _tune_params_multi(
    Y: np.ndarray,
    *,
    k: int,
    kneighbors: int,
    lamG: float,
    lamC: float,
    lamQ: float,
    labels: Optional[List[int]],
    traps: Optional[List[int]],
    tune_indices: List[int],
    score_type: str,
    alpha: float,
) -> Dict[str, Any]:
    # If we don't have labels, we can't compute F1; bail to defaults
    if labels is None:
        return {"lamG": lamG, "lamC": lamC, "lamQ": lamQ, "kneighbors": kneighbors}

    # Small, robust grid around provided params
    lamC_grid = [max(0.1, lamC * s) for s in [0.6, 1.0, 1.4]]
    lamQ_grid = [max(0.5, lamQ * s) for s in [0.5, 1.0, 1.5]]
    k_grid = sorted(
        set(
            [max(1, min(Y.shape[0] - 1, kk)) for kk in [kneighbors - 2, kneighbors, kneighbors + 2]]
        )
    )

    # Use the embedded row vector as the query embedding for leave-one-out
    psi_list: List[np.ndarray] = [Y[q_idx] for q_idx in tune_indices if 0 <= q_idx < Y.shape[0]]
    qidx_list: List[int] = [q_idx for q_idx in tune_indices if 0 <= q_idx < Y.shape[0]]

    best_score = float("-inf")
    best = {"lamG": lamG, "lamC": lamC, "lamQ": lamQ, "kneighbors": kneighbors}

    for lc in lamC_grid:
        for lq in lamQ_grid:
            for kk in k_grid:
                cand = {"lamG": lamG, "lamC": lc, "lamQ": lq, "kneighbors": kk}
                scores: List[float] = []
                for psi_vec, q_idx in zip(psi_list, qidx_list):
                    pred, _ = _run_lat_internal(cand, Y, psi_vec, k, q_idx)
                    f1, _, tsh = eval_topk(pred, labels, traps, k)
                    scores.append(_score_value(f1, tsh, score_type=score_type, alpha=alpha))
                mean_score = float(np.mean(scores)) if scores else float("-inf")
                if mean_score > best_score:
                    best_score = mean_score
                    best = cand
    return best


def parse_args():
    ap = argparse.ArgumentParser(
        description="Competitor benchmark: cosine vs Oscillink vs FAISS/Annoy (if available)"
    )
    ap.add_argument("--input", required=True, help="Path to CSV or JSONL")
    ap.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        help="Input format; inferred from extension if omitted",
    )
    ap.add_argument("--text-col", dest="text_col", default="text")
    ap.add_argument("--id-col", dest="id_col", default=None)
    ap.add_argument(
        "--label-col",
        dest="label_col",
        default=None,
        help="Binary relevance column (1 relevant, 0 otherwise)",
    )
    ap.add_argument(
        "--trap-col", dest="trap_col", default=None, help="Binary trap/false column (1 means trap)"
    )
    ap.add_argument(
        "--query", dest="query", default=None, help="Explicit query text (overrides query-index)"
    )
    ap.add_argument(
        "--query-index",
        dest="query_index",
        type=int,
        default=None,
        help="Use row at this index as query (excluded from candidates)",
    )
    ap.add_argument("--k", type=int, default=5)
    # Multi-query controls
    ap.add_argument(
        "--all-queries",
        action="store_true",
        help="Evaluate every row as the query (leaves-one-out); aggregates metrics",
    )
    ap.add_argument(
        "--query-indices",
        type=str,
        default=None,
        help="Comma-separated list of query indices to evaluate (overrides --query-index)",
    )
    # Lattice params
    ap.add_argument("--kneighbors", type=int, default=6)
    ap.add_argument("--lamG", type=float, default=1.0)
    ap.add_argument("--lamC", type=float, default=0.5)
    ap.add_argument("--lamQ", type=float, default=4.0)
    # Optional small tuning
    ap.add_argument(
        "--tune",
        action="store_true",
        help="Run a tiny grid search to adjust lamC/lamQ/k for F1 (if labels provided)",
    )
    ap.add_argument("--tune-trials", type=int, default=8)
    ap.add_argument(
        "--tune-mode",
        choices=["single", "multi"],
        default="single",
        help="Tune params around a single representative query (default) or average over multiple queries",
    )
    ap.add_argument(
        "--tune-indices",
        type=str,
        default=None,
        help="Comma-separated list of query indices to use for tuning when --tune-mode=multi; defaults to all queries evaluated",
    )
    ap.add_argument(
        "--score",
        choices=["f1", "trap_penalized", "f1_times_one_minus_traps"],
        default="f1",
        help="Objective for tuning/selection when labels+traps available",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Penalty weight for trap_penalized score (score = F1 - alpha*trap_share)",
    )
    # Preprocessing
    ap.add_argument(
        "--smart-correct",
        dest="smart_correct",
        action="store_true",
        help="Apply smart autocorrect to texts and query before embedding",
    )
    # Outputs
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None, help="Optional path to save JSON summary")
    return ap.parse_args()


def _ci95(arr: List[float]) -> float:
    import math

    vals = [float(x) for x in arr if isinstance(x, (int, float))]
    n = len(vals)
    if n <= 1:
        return 0.0
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    return 1.96 * se


def _tune_or_defaults(
    Y: np.ndarray,
    psi_ref: np.ndarray,
    k_eff: int,
    k: int,
    *,
    labels: Optional[List[int]],
    traps: Optional[List[int]],
    lamG: float,
    lamC: float,
    lamQ: float,
    tune: bool,
    tune_trials: int,
) -> Dict[str, Any]:
    if not tune:
        return {"lamG": lamG, "lamC": lamC, "lamQ": lamQ, "kneighbors": k_eff}
    # Multi-query tune averages score across multiple queries
    # For now, reuse single-query tuning if tune-mode is single or labels absent
    return _tune_params(
        Y,
        psi_ref,
        k=k,
        kneighbors=k_eff,
        lamG=lamG,
        lamC=lamC,
        lamQ=lamQ,
        trials=tune_trials,
        labels=labels,
        traps=traps,
    )


def _eval_one_query(
    Y: np.ndarray,
    labels: Optional[List[int]],
    traps: Optional[List[int]],
    k: int,
    k_eff: int,
    best_params: Dict[str, Any],
    psi: np.ndarray,
    q_idx: Optional[int],
) -> Dict[str, Any]:
    # Baseline cosine
    pred_cos, cos_ms = run_cosine_baseline(psi, Y, k, q_idx)
    f1_cos, _, tsh_cos = eval_topk(pred_cos, labels, traps, k)
    # Oscillink default
    pred_lat_def, lat_def_ms = _run_lat_internal(
        {"lamG": 1.0, "lamC": 0.5, "lamQ": 4.0, "kneighbors": k_eff}, Y, psi, k, q_idx
    )
    f1_lat_def, _, tsh_lat_def = eval_topk(pred_lat_def, labels, traps, k)
    # Oscillink tuned
    pred_lat_tuned, lat_tuned_ms = _run_lat_internal(best_params, Y, psi, k, q_idx)
    f1_lat_tuned, _, tsh_lat_tuned = eval_topk(pred_lat_tuned, labels, traps, k)
    return {
        "cosine": {
            "time_ms": None if cos_ms is None else float(cos_ms),
            "f1": None if f1_cos is None else float(f1_cos),
            "trap_share": None if tsh_cos is None else float(tsh_cos),
        },
        "oscillink_default": {
            "time_ms": float(lat_def_ms),
            "f1": None if f1_lat_def is None else float(f1_lat_def),
            "trap_share": None if tsh_lat_def is None else float(tsh_lat_def),
        },
        "oscillink_tuned": {
            "time_ms": float(lat_tuned_ms),
            "f1": None if f1_lat_tuned is None else float(f1_lat_tuned),
            "trap_share": None if tsh_lat_tuned is None else float(tsh_lat_tuned),
        },
    }


def _run_lat_internal(
    params: Dict[str, Any],
    Y: np.ndarray,
    psi_vec: np.ndarray,
    k: int,
    q_idx: Optional[int],
) -> Tuple[List[int], float]:
    k_lat = min(params.get("kneighbors", 6), max(1, Y.shape[0] - 1))
    t1 = time.time()
    lat = OscillinkLattice(
        Y,
        kneighbors=k_lat,
        lamG=params.get("lamG", 1.0),
        lamC=params.get("lamC", 0.5),
        lamQ=params.get("lamQ", 4.0),
        deterministic_k=True,
    )
    lat.set_query(psi_vec)
    lat.settle(max_iters=12, tol=1e-3)
    raw = [int(item.get("id", -1)) for item in lat.bundle(k=k + (1 if q_idx is not None else 0))]
    if q_idx is not None:
        raw = [i for i in raw if i != q_idx]
    pred = raw[:k]
    return pred, 1000.0 * (time.time() - t1)


def _compute_query_indices_from_args(args: Any, texts: List[str]) -> List[int]:
    if getattr(args, "all_queries", False):
        return list(range(len(texts)))
    if getattr(args, "query_indices", None):
        return [int(x) for x in str(args.query_indices).split(",") if x.strip() != ""]
    _, q_idx0 = select_query(
        getattr(args, "query", None), getattr(args, "query_index", None), texts
    )
    return [int(q_idx0) if q_idx0 is not None else 0]


def _single_query_summary_from_per_query(
    *,
    args: Any,
    Y: np.ndarray,
    k_eff: int,
    best_params: Dict[str, Any],
    per_query: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pq = per_query[0]
    return {
        "k": int(args.k),
        "N": int(Y.shape[0]),
        "cosine_time_ms": pq["cosine"]["time_ms"],
        "oscillink_default_time_ms": pq["oscillink_default"]["time_ms"],
        "oscillink_tuned_time_ms": pq["oscillink_tuned"]["time_ms"],
        "faiss_time_ms": None,
        "annoy_time_ms": None,
        "oscillink_default_params": {"lamG": 1.0, "lamC": 0.5, "lamQ": 4.0, "kneighbors": k_eff},
        "oscillink_tuned_params": best_params,
        "cosine_f1": pq["cosine"]["f1"],
        "oscillink_default_f1": pq["oscillink_default"]["f1"],
        "oscillink_tuned_f1": pq["oscillink_tuned"]["f1"],
        "faiss_f1": None,
        "annoy_f1": None,
        "cosine_hallucination": None,
        "oscillink_default_hallucination": None,
        "oscillink_tuned_hallucination": None,
        "faiss_hallucination": None,
        "annoy_hallucination": None,
        "cosine_trap_share": pq["cosine"]["trap_share"],
        "oscillink_default_trap_share": pq["oscillink_default"]["trap_share"],
        "oscillink_tuned_trap_share": pq["oscillink_tuned"]["trap_share"],
    }


def _aggregate_metrics(
    *,
    cos_times: List[float],
    cos_f1s: List[float],
    cos_traps: List[float],
    d_times: List[float],
    d_f1s: List[float],
    d_traps: List[float],
    t_times: List[float],
    t_f1s: List[float],
    t_traps: List[float],
) -> Dict[str, Any]:
    return {
        "cosine": {
            "time_ms": {
                "mean": float(np.mean(cos_times)) if cos_times else None,
                "ci95": _ci95(cos_times) if cos_times else None,
            },
            "f1": {
                "mean": float(np.mean(cos_f1s)) if cos_f1s else None,
                "ci95": _ci95(cos_f1s) if cos_f1s else None,
            },
            "trap_share": {
                "mean": float(np.mean(cos_traps)) if cos_traps else None,
                "ci95": _ci95(cos_traps) if cos_traps else None,
            },
        },
        "oscillink_default": {
            "time_ms": {"mean": float(np.mean(d_times)), "ci95": _ci95(d_times)},
            "f1": {
                "mean": float(np.mean(d_f1s)) if d_f1s else None,
                "ci95": _ci95(d_f1s) if d_f1s else None,
            },
            "trap_share": {
                "mean": float(np.mean(d_traps)) if d_traps else None,
                "ci95": _ci95(d_traps) if d_traps else None,
            },
        },
        "oscillink_tuned": {
            "time_ms": {"mean": float(np.mean(t_times)), "ci95": _ci95(t_times)},
            "f1": {
                "mean": float(np.mean(t_f1s)) if t_f1s else None,
                "ci95": _ci95(t_f1s) if t_f1s else None,
            },
            "trap_share": {
                "mean": float(np.mean(t_traps)) if t_traps else None,
                "ci95": _ci95(t_traps) if t_traps else None,
            },
        },
    }


def _eval_queries(
    *,
    Y: np.ndarray,
    texts_corr: List[str],
    query_indices: List[int],
    args: Any,
    labels: Optional[List[int]],
    traps: Optional[List[int]],
    k_eff: int,
    best_params: Dict[str, Any],
) -> Tuple[
    List[Dict[str, Any]],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
]:
    per_query: List[Dict[str, Any]] = []
    cos_times: List[float] = []
    cos_f1s: List[float] = []
    cos_traps: List[float] = []
    d_times: List[float] = []
    d_f1s: List[float] = []
    d_traps: List[float] = []
    t_times: List[float] = []
    t_f1s: List[float] = []
    t_traps: List[float] = []

    for q_idx in query_indices:
        q_text = texts_corr[q_idx]
        psi = embed_texts([q_text], normalize=True).astype(np.float32)[0]
        metrics = _eval_one_query(Y, labels, traps, args.k, k_eff, best_params, psi, q_idx)
        per_query.append({"q_idx": int(q_idx), **metrics})

        # Collect aggregates, guarding for None
        c = metrics["cosine"]
        if c["time_ms"] is not None:
            cos_times.append(float(c["time_ms"]))
        if c["f1"] is not None:
            cos_f1s.append(float(c["f1"]))
        if c["trap_share"] is not None:
            cos_traps.append(float(c["trap_share"]))
        d = metrics["oscillink_default"]
        d_times.append(float(d["time_ms"]))
        if d["f1"] is not None:
            d_f1s.append(float(d["f1"]))
        if d["trap_share"] is not None:
            d_traps.append(float(d["trap_share"]))
        t = metrics["oscillink_tuned"]
        t_times.append(float(t["time_ms"]))
        if t["f1"] is not None:
            t_f1s.append(float(t["f1"]))
        if t["trap_share"] is not None:
            t_traps.append(float(t["trap_share"]))

    return (
        per_query,
        cos_times,
        cos_f1s,
        cos_traps,
        d_times,
        d_f1s,
        d_traps,
        t_times,
        t_f1s,
        t_traps,
    )


def run_benchmark(args) -> Dict[str, Any]:
    fmt = _infer_format(args.input, args.format)
    rows = load_jsonl(args.input) if fmt == "jsonl" else load_csv(args.input)
    if not rows:
        raise SystemExit("No rows loaded from input")

    texts, ids, labels, traps = _prepare_fields(
        rows,
        text_col=args.text_col,
        id_col=args.id_col,
        label_col=args.label_col,
        trap_col=args.trap_col,
    )
    # Decide query set
    query_indices: List[int] = _compute_query_indices_from_args(args, texts)

    # Optional autocorrect (applies to all texts; query text handled per-iteration)
    texts_corr, _ = maybe_smart_correct(texts, texts[0] if texts else "", args.smart_correct)

    # Embed corpus once
    Y = embed_texts(texts_corr, normalize=True).astype(np.float32)

    def topk_to_ids(pred_idx: List[int]) -> List[Any]:
        return [ids[i] for i in pred_idx if 0 <= i < len(ids)]

    # Helper constants
    k_eff = min(args.kneighbors, max(1, Y.shape[0] - 1))

    # Precompute default and tuned params once (tuned around a representative psi)
    psi_ref = embed_texts([texts_corr[query_indices[0]]], normalize=True).astype(np.float32)[0]
    # Choose tuning strategy
    if args.tune and getattr(args, "tune_mode", "single") == "multi":
        # Select tune indices: explicit or reuse evaluated queries
        if getattr(args, "tune_indices", None):
            tune_indices = [int(x) for x in str(args.tune_indices).split(",") if x.strip() != ""]
        else:
            tune_indices = list(query_indices)
        best_params = _tune_params_multi(
            Y,
            k=args.k,
            kneighbors=k_eff,
            lamG=args.lamG,
            lamC=args.lamC,
            lamQ=args.lamQ,
            labels=labels,
            traps=traps,
            tune_indices=tune_indices,
            score_type=getattr(args, "score", "f1"),
            alpha=float(getattr(args, "alpha", 0.5)),
        )
        tune_meta = {
            "mode": "multi",
            "score": getattr(args, "score", "f1"),
            "alpha": float(getattr(args, "alpha", 0.5)),
            "tune_indices": [int(i) for i in tune_indices],
        }
    else:
        best_params = _tune_or_defaults(
            Y,
            psi_ref,
            k_eff,
            args.k,
            labels=labels,
            traps=traps,
            lamG=args.lamG,
            lamC=args.lamC,
            lamQ=args.lamQ,
            tune=args.tune,
            tune_trials=args.tune_trials,
        )
        tune_meta = {
            "mode": "single" if args.tune else "off",
            "trials": int(args.tune_trials),
        }

    # Collect per-query metrics via helper
    (
        per_query,
        cos_times,
        cos_f1s,
        cos_traps,
        d_times,
        d_f1s,
        d_traps,
        t_times,
        t_f1s,
        t_traps,
    ) = _eval_queries(
        Y=Y,
        texts_corr=texts_corr,
        query_indices=query_indices,
        args=args,
        labels=labels,
        traps=traps,
        k_eff=k_eff,
        best_params=best_params,
    )

    # If only one query, return the original single-query summary for backward compatibility
    if len(query_indices) == 1:
        return _single_query_summary_from_per_query(
            args=args, Y=Y, k_eff=k_eff, best_params=best_params, per_query=per_query
        )

    result = {
        "mode": "multi",
        "N": int(Y.shape[0]),
        "k": int(args.k),
        "oscillink_default_params": {"lamG": 1.0, "lamC": 0.5, "lamQ": 4.0, "kneighbors": k_eff},
        "oscillink_tuned_params": best_params,
        "tune": bool(args.tune),
        "tune_trials": int(args.tune_trials),
        "tune_meta": tune_meta,
        "per_query": per_query,
        "aggregate": _aggregate_metrics(
            cos_times=cos_times,
            cos_f1s=cos_f1s,
            cos_traps=cos_traps,
            d_times=d_times,
            d_f1s=d_f1s,
            d_traps=d_traps,
            t_times=t_times,
            t_f1s=t_f1s,
            t_traps=t_traps,
        ),
    }
    return result


def main():
    args = parse_args()
    summary = run_benchmark(args)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, separators=(",", ":"))
    if args.json:
        print(json.dumps(summary, separators=(",", ":")))
    else:
        print(summary)


if __name__ == "__main__":
    main()
