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


def cosine_topk(psi: np.ndarray, Y: np.ndarray, k: int, exclude_idx: Optional[int] = None) -> List[int]:
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    pn = psi / (np.linalg.norm(psi) + 1e-9)
    scores = (Yn @ pn)
    if exclude_idx is not None and 0 <= exclude_idx < len(scores):
        scores[exclude_idx] = -1e9
    idx = np.argsort(-scores)[:k]
    return idx.tolist()


def eval_topk(pred: List[int], labels: Optional[List[int]], traps: Optional[List[int]], k: int) -> Tuple[Optional[float], Optional[bool], Optional[float]]:
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


def select_query(query: Optional[str], query_index: Optional[int], texts: List[str]) -> Tuple[str, Optional[int]]:
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


def run_cosine_baseline(psi: np.ndarray, Y: np.ndarray, k: int, q_idx: Optional[int]) -> Tuple[List[int], float]:
    t0 = time.time()
    pred = cosine_topk(psi, Y, k, exclude_idx=q_idx)
    return pred, 1000.0 * (time.time() - t0)


def run_faiss_nn(psi: np.ndarray, Y: np.ndarray, k: int, q_idx: Optional[int]) -> Tuple[Optional[List[int]], Optional[float]]:
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


def run_annoy_nn(psi: np.ndarray, Y: np.ndarray, k: int, q_idx: Optional[int]) -> Tuple[Optional[List[int]], Optional[float]]:
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
    k_grid = sorted(set([max(1, min(Y.shape[0]-1, kk)) for kk in [kneighbors-2, kneighbors, kneighbors+2]]))
    rng = np.random.default_rng(42)
    best_f1 = -1.0
    best = {"lamG": lamG, "lamC": lamC, "lamQ": lamQ, "kneighbors": kneighbors}
    for lc in lamC_grid:
        for lq in lamQ_grid:
            for kk in k_grid:
                f1s: List[float] = []
                for _ in range(max(1, int(trials))):
                    jitter = (rng.standard_normal(psi.shape).astype(np.float32) * 0.01)
                    psi_t = (psi + jitter) / (np.linalg.norm(psi + jitter) + 1e-9)
                    lat_t = OscillinkLattice(Y, kneighbors=kk, lamG=lamG, lamC=lc, lamQ=lq, deterministic_k=True)
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


def parse_args():
    ap = argparse.ArgumentParser(description="Competitor benchmark: cosine vs Oscillink vs FAISS/Annoy (if available)")
    ap.add_argument("--input", required=True, help="Path to CSV or JSONL")
    ap.add_argument("--format", choices=["csv", "jsonl"], help="Input format; inferred from extension if omitted")
    ap.add_argument("--text-col", dest="text_col", default="text")
    ap.add_argument("--id-col", dest="id_col", default=None)
    ap.add_argument("--label-col", dest="label_col", default=None, help="Binary relevance column (1 relevant, 0 otherwise)")
    ap.add_argument("--trap-col", dest="trap_col", default=None, help="Binary trap/false column (1 means trap)")
    ap.add_argument("--query", dest="query", default=None, help="Explicit query text (overrides query-index)")
    ap.add_argument("--query-index", dest="query_index", type=int, default=None, help="Use row at this index as query (excluded from candidates)")
    ap.add_argument("--k", type=int, default=5)
    # Lattice params
    ap.add_argument("--kneighbors", type=int, default=6)
    ap.add_argument("--lamG", type=float, default=1.0)
    ap.add_argument("--lamC", type=float, default=0.5)
    ap.add_argument("--lamQ", type=float, default=4.0)
    # Optional small tuning
    ap.add_argument("--tune", action="store_true", help="Run a tiny grid search to adjust lamC/lamQ/k for F1 (if labels provided)")
    ap.add_argument("--tune-trials", type=int, default=8)
    # Preprocessing
    ap.add_argument("--smart-correct", dest="smart_correct", action="store_true", help="Apply smart autocorrect to texts and query before embedding")
    # Outputs
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None, help="Optional path to save JSON summary")
    return ap.parse_args()


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
    q_text, q_idx = select_query(args.query, args.query_index, texts)
    texts, q_text = maybe_smart_correct(texts, q_text, args.smart_correct)

    # Embed
    Y = embed_texts(texts, normalize=True).astype(np.float32)
    psi = embed_texts([q_text], normalize=True).astype(np.float32)[0]

    def topk_to_ids(pred_idx: List[int]) -> List[Any]:
        return [ids[i] for i in pred_idx if 0 <= i < len(ids)]

    # Baseline cosine
    pred_cos, cos_ms = run_cosine_baseline(psi, Y, args.k, q_idx)
    f1_cos, hall_cos, tsh_cos = eval_topk(pred_cos, labels, traps, args.k)

    # Oscillink default & tuned
    k_eff = min(args.kneighbors, max(1, Y.shape[0] - 1))

    def run_lat(params: Dict[str, Any]) -> Tuple[List[int], float]:
        k_lat = min(params.get("kneighbors", k_eff), max(1, Y.shape[0]-1))
        t1 = time.time()
        lat = OscillinkLattice(Y, kneighbors=k_lat, lamG=params.get("lamG", 1.0), lamC=params.get("lamC", 0.5), lamQ=params.get("lamQ", 4.0), deterministic_k=True)
        lat.set_query(psi)
        lat.settle(max_iters=12, tol=1e-3)
        pred = [int(item.get("id", -1)) for item in lat.bundle(k=args.k)]
        return pred, 1000.0 * (time.time() - t1)

    pred_lat_def, lat_def_ms = run_lat({"lamG": 1.0, "lamC": 0.5, "lamQ": 4.0, "kneighbors": k_eff})
    f1_lat_def, hall_lat_def, tsh_lat_def = eval_topk(pred_lat_def, labels, traps, args.k)

    if args.tune:
        best_params = _tune_params(Y, psi, k=args.k, kneighbors=k_eff, lamG=args.lamG, lamC=args.lamC, lamQ=args.lamQ, trials=args.tune_trials, labels=labels, traps=traps)
    else:
        best_params = {"lamG": args.lamG, "lamC": args.lamC, "lamQ": args.lamQ, "kneighbors": k_eff}
    pred_lat_tuned, lat_tuned_ms = run_lat(best_params)
    f1_lat_tuned, hall_lat_tuned, tsh_lat_tuned = eval_topk(pred_lat_tuned, labels, traps, args.k)

    # FAISS (optional)
    pred_faiss, faiss_ms = run_faiss_nn(psi, Y, args.k, q_idx)

    # Annoy (optional)
    pred_annoy, annoy_ms = run_annoy_nn(psi, Y, args.k, q_idx)

    summary: Dict[str, Any] = {
        "k": int(args.k),
        "N": int(Y.shape[0]),
        "cosine_time_ms": None if cos_ms is None else float(cos_ms),
        "oscillink_default_time_ms": float(lat_def_ms),
        "oscillink_tuned_time_ms": float(lat_tuned_ms),
        "faiss_time_ms": None if faiss_ms is None else float(faiss_ms),
        "annoy_time_ms": None if annoy_ms is None else float(annoy_ms),
        "oscillink_default_params": {"lamG": 1.0, "lamC": 0.5, "lamQ": 4.0, "kneighbors": k_eff},
        "oscillink_tuned_params": best_params,
        # Quality
        "cosine_f1": None if f1_cos is None else float(f1_cos),
        "oscillink_default_f1": None if f1_lat_def is None else float(f1_lat_def),
        "oscillink_tuned_f1": None if f1_lat_tuned is None else float(f1_lat_tuned),
        "faiss_f1": None if (pred_faiss is None or labels is None) else float(eval_topk(pred_faiss, labels, traps, args.k)[0] or 0.0),
        "annoy_f1": None if (pred_annoy is None or labels is None) else float(eval_topk(pred_annoy, labels, traps, args.k)[0] or 0.0),
        "cosine_hallucination": None if hall_cos is None else bool(hall_cos),
        "oscillink_default_hallucination": None if hall_lat_def is None else bool(hall_lat_def),
        "oscillink_tuned_hallucination": None if hall_lat_tuned is None else bool(hall_lat_tuned),
        "faiss_hallucination": None if (pred_faiss is None or traps is None) else bool(eval_topk(pred_faiss, labels, traps, args.k)[1]),
        "annoy_hallucination": None if (pred_annoy is None or traps is None) else bool(eval_topk(pred_annoy, labels, traps, args.k)[1]),
    }

    # Include top-k IDs for transparency
    summary.update({
        "cosine_topk": topk_to_ids(pred_cos),
        "oscillink_default_topk": topk_to_ids(pred_lat_def),
        "oscillink_tuned_topk": topk_to_ids(pred_lat_tuned),
        "faiss_topk": None if pred_faiss is None else topk_to_ids(pred_faiss),
        "annoy_topk": None if pred_annoy is None else topk_to_ids(pred_annoy),
    })

    return summary


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
