#!/usr/bin/env python3
"""
Benchmark: baseline cosine vs Oscillink (default) vs Oscillink (adaptive tuning)

- Datasets: small synthetic text sets (mars/paris) with ground truth and traps
- Metrics: F1, hallucination rate (any trap in top-k), optional trap-share
- Latency: baseline cosine and lattice settle+bundle wall time (ms)
- Adaptive: small grid search tuned on first split of trials, evaluated on remaining

Usage examples:
  python scripts/benchmark_adaptive.py --dataset mars --trials 20 --k 3 --json
  python scripts/benchmark_adaptive.py --dataset paris --trials 30 --k 5 --semantic --json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import List, Set, Tuple

import numpy as np

from oscillink.adapters.text import embed_texts
from oscillink.core.lattice import OscillinkLattice


@dataclass
class TrialResult:
    f1: float
    hallucination: bool
    trap_share: float
    time_ms: float


@dataclass
class Summary:
    trials: int
    k: int
    dataset: str
    semantic: bool
    # Quality
    baseline_f1_mean: float
    default_f1_mean: float
    adaptive_f1_mean: float
    baseline_hall_rate: float
    default_hall_rate: float
    adaptive_hall_rate: float
    # Latency
    baseline_time_ms_mean: float
    default_time_ms_mean: float
    adaptive_time_ms_mean: float
    # Params selected
    adaptive_params: dict
    # Optional
    baseline_trap_share_mean: float | None = None
    default_trap_share_mean: float | None = None
    adaptive_trap_share_mean: float | None = None

    def to_json(self) -> str:
        d = asdict(self)
        # Drop optional trap share if None
        if d.get("baseline_trap_share_mean") is None:
            d.pop("baseline_trap_share_mean", None)
            d.pop("default_trap_share_mean", None)
            d.pop("adaptive_trap_share_mean", None)
        return json.dumps(d, separators=(",", ":"))


def f1_score(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cosine_topk(psi: np.ndarray, Y: np.ndarray, k: int) -> List[int]:
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    pn = psi / (np.linalg.norm(psi) + 1e-9)
    scores = Yn @ pn
    idx = np.argsort(-scores)[:k]
    return idx.tolist()


def eval_topk(
    pred: List[int], gt_ids: Set[int], trap_ids: Set[int], k: int
) -> tuple[float, bool, float]:
    tp = len([i for i in pred if i in gt_ids])
    fp = len([i for i in pred if i not in gt_ids])
    fn = len([i for i in gt_ids if i not in pred])
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f1_score(prec, rec)
    hall = any(i in trap_ids for i in pred)
    tshare = len([i for i in pred if i in trap_ids]) / max(1, k)
    return f1, hall, tshare


def bundle_topk(lat: OscillinkLattice, k: int) -> List[int]:
    b = lat.bundle(k=k)
    return [int(item.get("id", -1)) for item in b]


def _dataset_mars() -> Tuple[List[str], Set[int], Set[int]]:
    corpus = [
        "mars has two moons phobos and deimos",  # pos
        "the capital of france is paris",  # pos (off-topic)
        "fake fact about moon cheese",  # trap
        "einstein developed general relativity",  # filler
        "spurious claim about ancient laser pyramids",  # trap
        "mars atmosphere mostly CO2",  # pos
        "random note about oceans",
        "spurious rumor about cheese aliens",  # trap-like text but we only count ones with keywords
    ]
    gt_ids = {0, 5}
    trap_ids = {i for i, t in enumerate(corpus) if ("fake" in t) or ("spurious" in t)}
    return corpus, gt_ids, trap_ids


def _dataset_paris() -> Tuple[List[str], Set[int], Set[int]]:
    corpus = [
        "Paris is the capital of France.",
        "The Eiffel Tower is located in Paris.",
        "Berlin is the capital of France.",
        "The Louvre houses famous artworks including the Mona Lisa.",
        "Tokyo is the capital of Japan.",
        "France borders Brazil across the Mediterranean Sea.",
        "The Seine river flows through Paris.",
        "The Colosseum is in Rome.",
        "Paris uses the Yen as its primary currency.",
        "Notre-Dame Cathedral is a landmark in Paris.",
    ]
    gt_ids = {0, 1, 3, 6, 9}
    trap_ids = {2, 5, 8}
    return corpus, gt_ids, trap_ids


def generate_embeddings(
    corpus: List[str], semantic: bool, rng: np.random.Generator, d: int = 96
) -> tuple[np.ndarray, np.ndarray]:
    if semantic:
        Y = embed_texts(corpus, normalize=True).astype(np.float32)
        psi = (Y[0] / (np.linalg.norm(Y[0]) + 1e-9)).astype(np.float32)
        return Y, psi
    Y = rng.standard_normal((len(corpus), d)).astype(np.float32)
    psi = Y[0] / (np.linalg.norm(Y[0]) + 1e-9)
    return Y, psi


def eval_baseline(
    Y: np.ndarray, psi: np.ndarray, k: int, gt_ids: Set[int], trap_ids: Set[int]
) -> TrialResult:
    t0 = time.time()
    pred = cosine_topk(psi, Y, k)
    t_ms = 1000.0 * (time.time() - t0)
    f1, hall, tshare = eval_topk(pred, gt_ids, trap_ids, k)
    return TrialResult(f1=f1, hallucination=hall, trap_share=tshare, time_ms=t_ms)


def eval_lattice(
    Y: np.ndarray,
    psi: np.ndarray,
    k: int,
    gt_ids: Set[int],
    trap_ids: Set[int],
    *,
    kneighbors: int,
    lamG: float,
    lamC: float,
    lamQ: float,
) -> TrialResult:
    N, _ = Y.shape
    k_eff = min(kneighbors, max(1, N - 1))
    t0 = time.time()
    lat = OscillinkLattice(
        Y, kneighbors=k_eff, lamG=lamG, lamC=lamC, lamQ=lamQ, deterministic_k=True
    )
    lat.set_query(psi)
    lat.settle(max_iters=12, tol=1e-3)
    pred = bundle_topk(lat, k)
    t_ms = 1000.0 * (time.time() - t0)
    f1, hall, tshare = eval_topk(pred, gt_ids, trap_ids, k)
    return TrialResult(f1=f1, hallucination=hall, trap_share=tshare, time_ms=t_ms)


def tune_params(corpus: List[str], semantic: bool, trials: int, seed: int, k: int) -> dict:
    """Small grid search across a few plausible params. Returns best params by mean F1."""
    # Grids kept intentionally tiny to avoid overscope/time
    lamG_grid = [1.0]
    lamC_grid = [0.3, 0.5, 0.7]
    lamQ_grid = [2.0, 4.0, 6.0]
    kneigh_grid = [4, 6, 8]
    rng = np.random.default_rng(seed)

    # Ground truth/traps depend only on dataset (corpus)
    # Compute once
    gt_ids = set()
    trap_ids = set()
    if "paris" in corpus[0].lower() or any("Paris" in t for t in corpus):
        _, gt_ids, trap_ids = _dataset_paris()
    else:
        _, gt_ids, trap_ids = _dataset_mars()

    # Tuning trials
    means = []
    for lamG in lamG_grid:
        for lamC in lamC_grid:
            for lamQ in lamQ_grid:
                for kngh in kneigh_grid:
                    f1s: List[float] = []
                    # Average over small sample of trials for robustness
                    for _ in range(trials):
                        local_rng = np.random.default_rng(
                            rng.integers(0, 2**32 - 1, dtype=np.uint64).item()
                        )
                        Y, psi = generate_embeddings(corpus, semantic, local_rng)
                        res = eval_lattice(
                            Y,
                            psi,
                            k,
                            gt_ids,
                            trap_ids,
                            kneighbors=kngh,
                            lamG=lamG,
                            lamC=lamC,
                            lamQ=lamQ,
                        )
                        f1s.append(res.f1)
                    means.append(((lamG, lamC, lamQ, kngh), float(np.mean(f1s) if f1s else 0.0)))
    means.sort(key=lambda x: x[1], reverse=True)
    best = means[0][0] if means else (1.0, 0.5, 4.0, 6)
    return {"lamG": best[0], "lamC": best[1], "lamQ": best[2], "kneighbors": int(best[3])}


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark baseline vs Oscillink default vs Oscillink adaptive"
    )
    ap.add_argument("--dataset", type=str, choices=["mars", "paris"], default="mars")
    ap.add_argument("--semantic", action="store_true", help="Use semantic/text embeddings")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--report-trap-share", action="store_true")
    ap.add_argument(
        "--tune-split",
        type=float,
        default=0.5,
        help="Fraction of trials for adaptive tuning (rest for test)",
    )
    # Optional: run with fixed params (skip tuning)
    ap.add_argument(
        "--use-fixed-params",
        action="store_true",
        help="Use provided lamG/lamC/lamQ/kneighbors for adaptive; skip tuning",
    )
    ap.add_argument("--lamG", type=float, default=None)
    ap.add_argument("--lamC", type=float, default=None)
    ap.add_argument("--lamQ", type=float, default=None)
    ap.add_argument("--kneighbors", type=int, default=None)
    args = ap.parse_args()

    # Load dataset
    if args.dataset == "paris":
        corpus, gt_ids, trap_ids = _dataset_paris()
    else:
        corpus, gt_ids, trap_ids = _dataset_mars()

    # Determine adaptive params
    if args.use_fixed_params:
        # Validate fixed params are provided
        lamG = args.lamG if args.lamG is not None else 1.0
        lamC = args.lamC if args.lamC is not None else 0.5
        lamQ = args.lamQ if args.lamQ is not None else 4.0
        kneigh = int(args.kneighbors) if args.kneighbors is not None else 6
        best_params = {"lamG": lamG, "lamC": lamC, "lamQ": lamQ, "kneighbors": kneigh}
        tune_trials = 0
        test_trials = max(1, args.trials)
    else:
        # Split trials
        tune_trials = max(1, int(args.trials * args.tune_split))
        test_trials = max(1, args.trials - tune_trials)
        # Tune adaptive params on tune split
        best_params = tune_params(
            corpus, args.semantic, trials=tune_trials, seed=args.seed, k=args.k
        )

    # Fixed default params
    default_params = {"lamG": 1.0, "lamC": 0.5, "lamQ": 4.0, "kneighbors": 6}

    # Run test split
    rng = np.random.default_rng(args.seed + 1337)
    base_f1s: List[float] = []
    def_f1s: List[float] = []
    ada_f1s: List[float] = []
    base_halls: List[int] = []
    def_halls: List[int] = []
    ada_halls: List[int] = []
    base_times: List[float] = []
    def_times: List[float] = []
    ada_times: List[float] = []
    base_tshares: List[float] = []
    def_tshares: List[float] = []
    ada_tshares: List[float] = []

    for _ in range(test_trials):
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1, dtype=np.uint64).item())
        Y, psi = generate_embeddings(corpus, args.semantic, local_rng)

        # baseline
        r_base = eval_baseline(Y, psi, args.k, gt_ids, trap_ids)
        base_f1s.append(r_base.f1)
        base_halls.append(1 if r_base.hallucination else 0)
        base_times.append(r_base.time_ms)
        base_tshares.append(r_base.trap_share)

        # default lattice
        r_def = eval_lattice(Y, psi, args.k, gt_ids, trap_ids, **default_params)
        def_f1s.append(r_def.f1)
        def_halls.append(1 if r_def.hallucination else 0)
        def_times.append(r_def.time_ms)
        def_tshares.append(r_def.trap_share)

        # adaptive lattice
        r_ada = eval_lattice(Y, psi, args.k, gt_ids, trap_ids, **best_params)
        ada_f1s.append(r_ada.f1)
        ada_halls.append(1 if r_ada.hallucination else 0)
        ada_times.append(r_ada.time_ms)
        ada_tshares.append(r_ada.trap_share)

    summary = Summary(
        trials=test_trials,
        k=args.k,
        dataset=args.dataset,
        semantic=bool(args.semantic),
        baseline_f1_mean=float(np.mean(base_f1s) if base_f1s else 0.0),
        default_f1_mean=float(np.mean(def_f1s) if def_f1s else 0.0),
        adaptive_f1_mean=float(np.mean(ada_f1s) if ada_f1s else 0.0),
        baseline_hall_rate=float(np.mean(base_halls) if base_halls else 0.0),
        default_hall_rate=float(np.mean(def_halls) if def_halls else 0.0),
        adaptive_hall_rate=float(np.mean(ada_halls) if ada_halls else 0.0),
        baseline_time_ms_mean=float(np.mean(base_times) if base_times else 0.0),
        default_time_ms_mean=float(np.mean(def_times) if def_times else 0.0),
        adaptive_time_ms_mean=float(np.mean(ada_times) if ada_times else 0.0),
        adaptive_params=best_params,
        baseline_trap_share_mean=(float(np.mean(base_tshares)) if args.report_trap_share else None),
        default_trap_share_mean=(float(np.mean(def_tshares)) if args.report_trap_share else None),
        adaptive_trap_share_mean=(float(np.mean(ada_tshares)) if args.report_trap_share else None),
    )

    if args.json:
        print(summary.to_json())
    else:
        print("Trials:", summary.trials)
        print("Dataset:", summary.dataset, "Semantic:", summary.semantic)
        print("k:", summary.k)
        print("Adaptive params:", summary.adaptive_params)
        print(
            f"F1 (mean): baseline={summary.baseline_f1_mean:.3f} default={summary.default_f1_mean:.3f} adaptive={summary.adaptive_f1_mean:.3f}"
        )
        print(
            f"Hallucination rate: baseline={summary.baseline_hall_rate:.3f} default={summary.default_hall_rate:.3f} adaptive={summary.adaptive_hall_rate:.3f}"
        )
        print(
            f"Latency ms (mean): baseline={summary.baseline_time_ms_mean:.1f} default={summary.default_time_ms_mean:.1f} adaptive={summary.adaptive_time_ms_mean:.1f}"
        )
        if args.report_trap_share:
            print(
                f"Trap-share (mean): baseline={(summary.baseline_trap_share_mean or 0.0):.3f} default={(summary.default_trap_share_mean or 0.0):.3f} adaptive={(summary.adaptive_trap_share_mean or 0.0):.3f}"
            )


if __name__ == "__main__":
    main()
