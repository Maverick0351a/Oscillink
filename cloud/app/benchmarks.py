from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from oscillink.adapters.text import embed_texts
from oscillink.core.lattice import OscillinkLattice
from oscillink.preprocess.autocorrect import smart_correct

from .config import get_settings

# Note: no direct imports from keystore/features here to avoid unused warnings; we resolve via .main helpers


router = APIRouter()
_API_VERSION = get_settings().api_version


class CompetitorBenchPayload(BaseModel):
    texts: List[str]
    labels: Optional[List[int]] = None
    traps: Optional[List[int]] = None
    ids: Optional[List[Any]] = None
    query: Optional[str] = None
    query_index: Optional[int] = None
    k: int = 5
    kneighbors: int = 6
    lamG: float = 1.0
    lamC: float = 0.5
    lamQ: float = 4.0
    tune: bool = False
    tune_trials: int = 8
    smart_correct: bool = False


def _cosine_topk(psi: np.ndarray, Y: np.ndarray, k: int, exclude_idx: Optional[int]) -> List[int]:
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    pn = psi / (np.linalg.norm(psi) + 1e-9)
    scores = Yn @ pn
    if exclude_idx is not None and 0 <= exclude_idx < len(scores):
        scores[exclude_idx] = -1e9
    idx = np.argsort(-scores)[:k]
    return idx.tolist()


def _eval_topk(
    pred: List[int], labels: Optional[List[int]], traps: Optional[List[int]], k: int
) -> Tuple[Optional[float], Optional[bool]]:
    if labels is None:
        if traps is None:
            return None, None
        hall = any((i in traps) for i in pred)
        return None, hall
    gt_ids = {i for i, lab in enumerate(labels) if int(lab) == 1}
    tp = len([i for i in pred if i in gt_ids])
    fp = len([i for i in pred if i not in gt_ids])
    fn = len([i for i in gt_ids if i not in pred])
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (prec == 0.0 and rec == 0.0) else (2 * prec * rec / (prec + rec))
    hall = None
    if traps is not None:
        hall = any((i in traps) for i in pred)
    return f1, hall


def _run_faiss(
    psi: np.ndarray, Y: np.ndarray, k: int, q_idx: Optional[int]
) -> Tuple[Optional[List[int]], Optional[float]]:
    try:
        import faiss  # type: ignore

        Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
        pn = psi / (np.linalg.norm(psi) + 1e-9)
        index = faiss.IndexFlatIP(Y.shape[1])
        index.add(Yn.astype(np.float32))
        t0 = time.time()
        _d, idxs = index.search(pn.reshape(1, -1), k + (1 if q_idx is not None else 0))
        ms = 1000.0 * (time.time() - t0)
        cand = idxs[0].tolist()
        if q_idx is not None and q_idx in cand:
            cand = [c for c in cand if c != q_idx]
        return cand[:k], ms
    except Exception:
        return None, None


def _run_annoy(
    psi: np.ndarray, Y: np.ndarray, k: int, q_idx: Optional[int]
) -> Tuple[Optional[List[int]], Optional[float]]:
    try:
        from annoy import AnnoyIndex  # type: ignore

        dim = int(Y.shape[1])
        t = AnnoyIndex(dim, metric="angular")
        for i, v in enumerate(Y.tolist()):
            t.add_item(i, v)
        t.build(10)
        t0 = time.time()
        cand = t.get_nns_by_vector(psi.tolist(), k + (1 if q_idx is not None else 0))
        ms = 1000.0 * (time.time() - t0)
        if q_idx is not None and q_idx in cand:
            cand = [c for c in cand if c != q_idx]
        return cand[:k], ms
    except Exception:
        return None, None


def feature_context(x_api_key: str | None = Depends(lambda x_api_key=None: x_api_key)):
    # Reuse existing feature resolution from main via keystore metadata
    from . import main as _main  # type: ignore

    ks = _main.get_keystore()
    meta = ks.get(x_api_key) if x_api_key else None
    feats = _main.resolve_features(meta)
    return {"api_key": x_api_key, "features": feats}


def _select_query(
    texts: List[str], query: Optional[str], query_index: Optional[int]
) -> Tuple[str, Optional[int]]:
    if query is not None:
        return query, None
    if query_index is not None and 0 <= query_index < len(texts):
        return texts[query_index], query_index
    return texts[0], 0 if texts else None


def _run_oscillink(
    Y: np.ndarray, psi: np.ndarray, k: int, k_eff: int, params: Dict[str, Any]
) -> Tuple[List[int], float]:
    k_lat = min(int(params.get("kneighbors", k_eff)), max(1, int(Y.shape[0]) - 1))
    t1 = time.time()
    lat = OscillinkLattice(
        Y,
        kneighbors=k_lat,
        lamG=float(params.get("lamG", 1.0)),
        lamC=float(params.get("lamC", 0.5)),
        lamQ=float(params.get("lamQ", 4.0)),
        deterministic_k=True,
    )
    lat.set_query(psi)
    lat.settle(max_iters=12, tol=1e-3)
    pred = [int(item.get("id", -1)) for item in lat.bundle(k=k)]
    return pred, 1000.0 * (time.time() - t1)


def _tune_params(
    Y: np.ndarray,
    psi: np.ndarray,
    k: int,
    k_eff: int,
    base: Dict[str, Any],
    labels: Optional[List[int]],
    traps: Optional[List[int]],
    trials: int,
) -> Dict[str, Any]:
    if labels is None:
        return {**base, "kneighbors": k_eff}
    lamC_grid = [max(0.1, float(base.get("lamC", 0.5)) * s) for s in [0.6, 1.0, 1.4]]
    lamQ_grid = [max(0.5, float(base.get("lamQ", 4.0)) * s) for s in [0.5, 1.0, 1.5]]
    k_grid = sorted(
        set([max(1, min(int(Y.shape[0]) - 1, kk)) for kk in [k_eff - 2, k_eff, k_eff + 2]])
    )
    best = {
        "lamG": float(base.get("lamG", 1.0)),
        "lamC": float(base.get("lamC", 0.5)),
        "lamQ": float(base.get("lamQ", 4.0)),
        "kneighbors": k_eff,
    }
    best_f1 = -1.0
    rng = np.random.default_rng(42)
    for lc in lamC_grid:
        for lq in lamQ_grid:
            for kk in k_grid:
                f1s: List[float] = []
                for _ in range(max(1, int(trials))):
                    jitter = rng.standard_normal(psi.shape).astype(np.float32) * 0.01
                    psi_t = (psi + jitter) / (np.linalg.norm(psi + jitter) + 1e-9)
                    lat_t = OscillinkLattice(
                        Y,
                        kneighbors=kk,
                        lamG=float(base.get("lamG", 1.0)),
                        lamC=lc,
                        lamQ=lq,
                        deterministic_k=True,
                    )
                    lat_t.set_query(psi_t)
                    lat_t.settle(max_iters=12, tol=1e-3)
                    pred_t = [int(item.get("id", -1)) for item in lat_t.bundle(k=k)]
                    f1_t, _ = _eval_topk(pred_t, labels, traps, k)
                    if f1_t is not None:
                        f1s.append(float(f1_t))
                mean_f1 = float(np.mean(f1s)) if f1s else -1.0
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best = {
                        "lamG": float(base.get("lamG", 1.0)),
                        "lamC": lc,
                        "lamQ": lq,
                        "kneighbors": kk,
                    }
    return best


@router.post(f"/{_API_VERSION}/bench/competitor")
def competitor_benchmark(
    payload: CompetitorBenchPayload,
    request: Request,
    response: Response,
    ctx=Depends(feature_context),
):
    # Resolve fields
    texts = payload.texts or []
    if not texts:
        raise HTTPException(status_code=400, detail="texts must be non-empty")
    labels = payload.labels
    traps = payload.traps
    ids = payload.ids or list(range(len(texts)))
    if payload.smart_correct:
        texts = [smart_correct(t) for t in texts]
    # Select query
    q_text, q_idx = _select_query(texts, payload.query, payload.query_index)

    # Embed
    Y = embed_texts(texts, normalize=True).astype(np.float32)
    psi = embed_texts([q_text], normalize=True).astype(np.float32)[0]
    N, D = int(Y.shape[0]), int(Y.shape[1])
    if N <= 1:
        raise HTTPException(status_code=400, detail="need at least 2 texts to benchmark")

    # Quota/Monthly usage enforcement using main helpers (late import to avoid cycles)
    from . import main as _main  # type: ignore

    x_api_key = ctx.get("api_key")
    units = N * D
    monthly_ctx = _main._check_monthly_cap(x_api_key, units)
    remaining, limit, reset_at = _main._check_and_consume_quota(x_api_key, units)

    # Baseline cosine
    t0 = time.time()
    pred_cos = _cosine_topk(psi, Y, payload.k, exclude_idx=q_idx)
    cosine_ms = 1000.0 * (time.time() - t0)
    f1_cos, hall_cos = _eval_topk(pred_cos, labels, traps, payload.k)

    # Oscillink default
    k_eff = min(payload.kneighbors, max(1, N - 1))
    pred_lat_def, lat_def_ms = _run_oscillink(
        Y, psi, payload.k, k_eff, {"lamG": 1.0, "lamC": 0.5, "lamQ": 4.0, "kneighbors": k_eff}
    )
    f1_lat_def, hall_lat_def = _eval_topk(pred_lat_def, labels, traps, payload.k)

    # Optional tiny tuning (if labels provided)
    base_params = {"lamG": payload.lamG, "lamC": payload.lamC, "lamQ": payload.lamQ}
    best_params = (
        _tune_params(Y, psi, payload.k, k_eff, base_params, labels, traps, payload.tune_trials)
        if (payload.tune and labels is not None)
        else {**base_params, "kneighbors": k_eff}
    )

    pred_lat_tuned, lat_tuned_ms = _run_oscillink(Y, psi, payload.k, k_eff, best_params)
    f1_lat_tuned, hall_lat_tuned = _eval_topk(pred_lat_tuned, labels, traps, payload.k)

    # Optional FAISS/Annoy
    pred_faiss, faiss_ms = _run_faiss(psi, Y, payload.k, q_idx)
    pred_annoy, annoy_ms = _run_annoy(psi, Y, payload.k, q_idx)

    def _ids_from_idx(idx_list: Optional[List[int]]) -> Optional[List[Any]]:
        if idx_list is None:
            return None
        return [ids[i] for i in idx_list if 0 <= i < len(ids)]

    summary: Dict[str, Any] = {
        "k": int(payload.k),
        "N": N,
        "cosine_time_ms": float(cosine_ms),
        "oscillink_default_time_ms": float(lat_def_ms),
        "oscillink_tuned_time_ms": float(lat_tuned_ms),
        "faiss_time_ms": None if faiss_ms is None else float(faiss_ms),
        "annoy_time_ms": None if annoy_ms is None else float(annoy_ms),
        "oscillink_default_params": {"lamG": 1.0, "lamC": 0.5, "lamQ": 4.0, "kneighbors": k_eff},
        "oscillink_tuned_params": best_params,
        "cosine_f1": None if f1_cos is None else float(f1_cos),
        "oscillink_default_f1": None if f1_lat_def is None else float(f1_lat_def),
        "oscillink_tuned_f1": None if f1_lat_tuned is None else float(f1_lat_tuned),
        "faiss_f1": None
        if (pred_faiss is None or labels is None)
        else float((_eval_topk(pred_faiss, labels, traps, payload.k)[0]) or 0.0),
        "annoy_f1": None
        if (pred_annoy is None or labels is None)
        else float((_eval_topk(pred_annoy, labels, traps, payload.k)[0]) or 0.0),
        "cosine_hallucination": None if hall_cos is None else bool(hall_cos),
        "oscillink_default_hallucination": None if hall_lat_def is None else bool(hall_lat_def),
        "oscillink_tuned_hallucination": None if hall_lat_tuned is None else bool(hall_lat_tuned),
        "faiss_hallucination": None
        if (pred_faiss is None or traps is None)
        else bool(_eval_topk(pred_faiss, labels, traps, payload.k)[1]),
        "annoy_hallucination": None
        if (pred_annoy is None or traps is None)
        else bool(_eval_topk(pred_annoy, labels, traps, payload.k)[1]),
        "cosine_topk": _ids_from_idx(pred_cos),
        "oscillink_default_topk": _ids_from_idx(pred_lat_def),
        "oscillink_tuned_topk": _ids_from_idx(pred_lat_tuned),
        "faiss_topk": _ids_from_idx(pred_faiss),
        "annoy_topk": _ids_from_idx(pred_annoy),
    }

    # Metrics and headers
    try:
        from . import main as _main  # type: ignore

        _main.USAGE_NODES.inc(N)
        _main.USAGE_NODE_DIM_UNITS.inc(N * D)
        headers = _main._quota_headers(remaining, limit, reset_at)
        for k, v in headers.items():
            response.headers.setdefault(k, v)
        # Monthly headers (informational)
        if monthly_ctx:
            response.headers.setdefault("X-Monthly-Cap", str(monthly_ctx["limit"]))
            response.headers.setdefault("X-Monthly-Used", str(monthly_ctx["used"]))
            response.headers.setdefault("X-Monthly-Remaining", str(monthly_ctx["remaining"]))
            response.headers.setdefault("X-Monthly-Period", str(monthly_ctx["period"]))
        # Usage log
        _main._append_usage(
            {
                "ts": time.time(),
                "event": "competitor_benchmark",
                "api_key": x_api_key,
                "N": N,
                "D": D,
                "units": units,
                "duration_ms": float(summary.get("oscillink_tuned_time_ms", 0.0)),
                "quota": None
                if limit == 0
                else {"limit": limit, "remaining": remaining, "reset": int(reset_at)},
                "monthly": None
                if not monthly_ctx
                else {
                    "limit": monthly_ctx["limit"],
                    "used": monthly_ctx["used"],
                    "remaining": monthly_ctx["remaining"],
                    "period": monthly_ctx["period"],
                },
            }
        )
    except Exception:
        pass

    return summary
