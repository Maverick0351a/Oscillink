#!/usr/bin/env python3
"""
Micro-bench: density (row_cap_val) and warm-start vs cold-start on synthetic sets.

Reports F1, hallucination rate, CG iters, settle time across tiny scenarios.

Usage examples:
  python scripts/bench_micro_dynamics.py --dataset mars --trials 20 --k 3 --json
  python scripts/bench_micro_dynamics.py --dataset paris --trials 20 --k 3 --json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from oscillink.adapters.text import embed_texts
from oscillink.core.lattice import OscillinkLattice


@dataclass
class RunSummary:
    dataset: str
    k: int
    trials: int
    density: Dict[str, float]
    warmstart: Dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def _dataset_mars() -> Tuple[List[str], Set[int], Set[int]]:
    corpus = [
        "mars has two moons phobos and deimos",
        "the capital of france is paris",
        "fake fact about moon cheese",
        "einstein developed general relativity",
        "spurious claim about ancient laser pyramids",
        "mars atmosphere mostly CO2",
        "random note about oceans",
        "spurious rumor about cheese aliens",
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
    corpus: List[str], d: int = 96, scale: int = 1, noise: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (Y, psi, base_n). If scale>1, replicate base corpus with small noise.

    base_n is the original corpus length used to map indices back for eval.
    """
    Y_base = embed_texts(corpus, normalize=True).astype(np.float32)
    base_n = len(corpus)
    if scale <= 1:
        psi = (Y_base[0] / (np.linalg.norm(Y_base[0]) + 1e-9)).astype(np.float32)
        return Y_base, psi, base_n
    blocks = []
    rng = np.random.default_rng(12345)
    for _s in range(scale):
        jitter = rng.standard_normal(Y_base.shape).astype(np.float32) * float(noise)
        Yb = Y_base + jitter
        Yn = Yb / (np.linalg.norm(Yb, axis=1, keepdims=True) + 1e-9)
        blocks.append(Yn.astype(np.float32))
    Y = np.concatenate(blocks, axis=0)
    psi = (blocks[0][0] / (np.linalg.norm(blocks[0][0]) + 1e-9)).astype(np.float32)
    return Y, psi, base_n


def cosine_topk(psi: np.ndarray, Y: np.ndarray, k: int) -> List[int]:
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    pn = psi / (np.linalg.norm(psi) + 1e-9)
    scores = Yn @ pn
    idx = np.argsort(-scores)[:k]
    return idx.tolist()


def f1_score(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def eval_topk(
    pred: List[int], gt_ids: Set[int], trap_ids: Set[int], k: int, base_n: Optional[int] = None
) -> Tuple[float, float, float]:
    # Map predictions back to base indices if the dataset was scaled (replicated)
    pred_mapped = [int(i % base_n) for i in pred] if (base_n is not None and base_n > 0) else pred
    tp = len([i for i in pred if i in gt_ids])
    # Recompute against mapped indices for correctness when scaled
    tp = len([i for i in pred_mapped if i in gt_ids])
    fp = len([i for i in pred_mapped if i not in gt_ids])
    fn = len([i for i in gt_ids if i not in pred_mapped])
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f1_score(prec, rec)
    hall = 1.0 if any(i in trap_ids for i in pred_mapped) else 0.0
    return f1, hall, float(tp)


def run_density(
    Y: np.ndarray,
    psi: np.ndarray,
    k: int,
    base_n: Optional[int] = None,
    with_dynamics: bool = False,
) -> Dict[str, float]:
    # Evaluate three caps: 0.5 (sparse), 1.0 (default), 2.0 (dense)
    cfgs = [(0.5, "cap0.5"), (1.0, "cap1.0"), (2.0, "cap2.0")]
    out: Dict[str, float] = {}
    for cap, name in cfgs:
        t0 = time.time()
        lat = OscillinkLattice(
            Y, kneighbors=6, row_cap_val=cap, lamG=1.0, lamC=0.5, lamQ=4.0, deterministic_k=True
        )
        # rebuild effective adjacency with new cap by reconstructing instance (lightweight for small Y)
        # For now, we use private flow: row_cap_val is only in __init__, so re-instantiation is the safe path.
        # If a public rebuild_graph() exists later, switch to that.
        lat.set_query(psi)
        s = lat.settle(max_iters=12, tol=1e-3)
        b = lat.bundle(k=k)
        pred = [int(e.get("id", -1)) for e in b]
        f1, hall, _ = eval_topk(pred, gt_ids, trap_ids, k, base_n=base_n)
        out[f"{name}_f1"] = f1
        out[f"{name}_hall"] = hall
        out[f"{name}_iters"] = float(s.get("iters", 0))
        out[f"{name}_t_ms"] = float(s.get("t_ms", 0.0))
        out[f"{name}_latency_ms"] = (time.time() - t0) * 1000.0
        if with_dynamics:
            rec = lat.receipt()
            dyn = rec.get("meta", {}).get("dynamics")
            if dyn:
                out[f"{name}_temp"] = float(dyn.get("temperature", 0.0))
                out[f"{name}_visc"] = float(dyn.get("viscosity_step", 0.0))
    return out


def run_warmstart(
    Y: np.ndarray,
    psi: np.ndarray,
    k: int,
    base_n: Optional[int] = None,
    inertia: float = 0.0,
    with_dynamics: bool = False,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    lat = OscillinkLattice(Y, kneighbors=6, lamG=1.0, lamC=0.5, lamQ=4.0, deterministic_k=True)
    lat.set_query(psi)
    # Cold start: simulate by resetting U to Y between settles
    t0 = time.time()
    s1 = lat.settle(max_iters=12, tol=1e-3)
    b1 = lat.bundle(k=k)
    pred1 = [int(e.get("id", -1)) for e in b1]
    f11, h11, _ = eval_topk(pred1, gt_ids, trap_ids, k, base_n=base_n)

    # "Similar" query: tiny perturbation toward psi (keep warm state)
    psi2 = psi * 0.95 + 0.05 * Y[0]
    psi2 = psi2 / (np.linalg.norm(psi2) + 1e-12)
    lat.set_query(psi2)
    # Inertia approximation: blend state toward previous U before next settle
    if inertia > 0.0:
        lat.U = ((1.0 - float(inertia)) * lat.Y + float(inertia) * lat.U).astype(np.float32)
    s2 = lat.settle(max_iters=12, tol=1e-3)
    b2 = lat.bundle(k=k)
    pred2 = [int(e.get("id", -1)) for e in b2]
    f12, h12, _ = eval_topk(pred2, gt_ids, trap_ids, k, base_n=base_n)
    out["warm_iters1"] = float(s1.get("iters", 0))
    out["warm_iters2"] = float(s2.get("iters", 0))
    out["warm_t1_ms"] = float(s1.get("t_ms", 0.0))
    out["warm_t2_ms"] = float(s2.get("t_ms", 0.0))
    out["warm_f1_1"] = f11
    out["warm_f1_2"] = f12
    out["warm_hall_1"] = h11
    out["warm_hall_2"] = h12
    out["warm_total_latency_ms"] = (time.time() - t0) * 1000.0
    if with_dynamics:
        rec = lat.receipt()
        dyn = rec.get("meta", {}).get("dynamics")
        if dyn:
            out["warm_temp2"] = float(dyn.get("temperature", 0.0))
            out["warm_visc2"] = float(dyn.get("viscosity_step", 0.0))
    return out


def _run_once(corpus: List[str], args) -> Tuple[Dict[str, float], Dict[str, float]]:
    Y, psi, base_n = generate_embeddings(corpus, scale=args.scale, noise=args.noise)
    d = run_density(Y, psi, args.k, base_n=base_n, with_dynamics=args.with_dynamics)
    w = run_warmstart(
        Y,
        psi,
        args.k,
        base_n=base_n,
        inertia=max(0.0, min(1.0, args.inertia)),
        with_dynamics=args.with_dynamics,
    )
    return d, w


def _parse_args():
    ap = argparse.ArgumentParser(description="Micro-bench density and warm-start effects")
    ap.add_argument("--dataset", type=str, choices=["mars", "paris"], default="mars")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--scale", type=int, default=1, help="Scale factor for dataset size (replicate with noise)"
    )
    ap.add_argument("--noise", type=float, default=0.01, help="Embedding noise for scaled replicas")
    ap.add_argument(
        "--inertia", type=float, default=0.0, help="Warm-start inertia blend factor in [0,1]"
    )
    ap.add_argument(
        "--with-dynamics", action="store_true", help="Capture dynamics (requires env flag set)"
    )
    return ap.parse_args()


def _print_human(summary: RunSummary, with_dynamics: bool) -> None:
    print("=== Micro Dynamics Bench ===")
    print(f"dataset={summary.dataset} k={summary.k} trials={summary.trials}")
    print("-- density --")
    base_density = summary.density
    for name in ("cap0.5", "cap1.0", "cap2.0"):
        print(
            f"{name}: f1={base_density[name + '_f1']:.3f} hall={base_density[name + '_hall']:.3f} "
            f"iters={base_density[name + '_iters']:.1f} t_ms={base_density[name + '_t_ms']:.2f} "
            f"lat_ms={base_density[name + '_latency_ms']:.2f}"
        )
        if with_dynamics:
            print(
                f"     temp={base_density.get(name + '_temp', 0.0):.4f} visc={base_density.get(name + '_visc', 0.0):.4f}"
            )
    print("-- warm-start --")
    base_warm = summary.warmstart
    print(
        f"iters1={base_warm['warm_iters1']:.1f} iters2={base_warm['warm_iters2']:.1f} "
        f"t1={base_warm['warm_t1_ms']:.2f} t2={base_warm['warm_t2_ms']:.2f} "
        f"f1_1={base_warm['warm_f1_1']:.3f} f1_2={base_warm['warm_f1_2']:.3f} "
        f"hall1={base_warm['warm_hall_1']:.3f} hall2={base_warm['warm_hall_2']:.3f} "
        f"total_lat_ms={base_warm['warm_total_latency_ms']:.2f}"
    )
    if with_dynamics:
        print(
            f"     warm_temp2={base_warm.get('warm_temp2', 0.0):.4f} warm_visc2={base_warm.get('warm_visc2', 0.0):.4f}"
        )


def main():
    args = _parse_args()

    if args.dataset == "paris":
        corpus, gt, traps = _dataset_paris()
    else:
        corpus, gt, traps = _dataset_mars()

    global gt_ids, trap_ids
    gt_ids, trap_ids = gt, traps

    base_density: Dict[str, float] = {
        "cap0.5_f1": 0.0,
        "cap1.0_f1": 0.0,
        "cap2.0_f1": 0.0,
        "cap0.5_hall": 0.0,
        "cap1.0_hall": 0.0,
        "cap2.0_hall": 0.0,
        "cap0.5_iters": 0.0,
        "cap1.0_iters": 0.0,
        "cap2.0_iters": 0.0,
        "cap0.5_t_ms": 0.0,
        "cap1.0_t_ms": 0.0,
        "cap2.0_t_ms": 0.0,
        "cap0.5_latency_ms": 0.0,
        "cap1.0_latency_ms": 0.0,
        "cap2.0_latency_ms": 0.0,
    }
    base_warm: Dict[str, float] = {
        "warm_iters1": 0.0,
        "warm_iters2": 0.0,
        "warm_t1_ms": 0.0,
        "warm_t2_ms": 0.0,
        "warm_f1_1": 0.0,
        "warm_f1_2": 0.0,
        "warm_hall_1": 0.0,
        "warm_hall_2": 0.0,
        "warm_total_latency_ms": 0.0,
    }

    for _ in range(args.trials):
        d, w = _run_once(corpus, args)
        for k2, v in d.items():
            base_density[k2] += v
        for k2, v in w.items():
            base_warm[k2] += v

    # average
    for k2 in base_density:
        base_density[k2] /= max(1, args.trials)
    for k2 in base_warm:
        base_warm[k2] /= max(1, args.trials)

    summary = RunSummary(
        dataset=args.dataset,
        k=args.k,
        trials=args.trials,
        density=base_density,
        warmstart=base_warm,
    )
    if args.json:
        print(summary.to_json())
    else:
        _print_human(summary, args.with_dynamics)


if __name__ == "__main__":
    main()
