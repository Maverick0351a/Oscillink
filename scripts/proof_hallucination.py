#!/usr/bin/env python3
"""
Proof harness: quantify hallucination suppression (baseline vs gated lattice).

- Builds a small synthetic labeled corpus with a couple of known traps ("fake", "spurious").
- Computes a cosine top-k baseline and a gated Oscillink lattice bundle.
- Reports hallucination rate (presence of any trap in top-k) and F1 over a tiny ground truth set.

Usage:
  python scripts/proof_hallucination.py --trials 20 --k 3 --seed 0 --json

Outputs JSON summary by default when --json is passed; otherwise prints a human summary.

NOTE: This is an illustrative, controlled demo (matches the README narrative). It is
not a general benchmark; for real datasets, adapt the labeling and gating accordingly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import List, Set, Tuple

import numpy as np

from oscillink import OscillinkLattice, compute_diffusion_gates
from oscillink.adapters.text import embed_texts


@dataclass
class TrialMetrics:
    hallucination: bool
    precision: float
    recall: float
    f1: float
    trap_share: float


@dataclass
class Summary:
    trials: int
    k: int
    baseline_hallucination_rate: float
    lattice_hallucination_rate: float
    baseline_f1_mean: float
    lattice_f1_mean: float
    baseline_trap_share_mean: float | None = None
    lattice_trap_share_mean: float | None = None

    def to_json(self) -> str:
        d = asdict(self)
        # Drop optional fields if not computed
        if d.get("baseline_trap_share_mean") is None:
            d.pop("baseline_trap_share_mean", None)
            d.pop("lattice_trap_share_mean", None)
        return json.dumps(d, separators=(",", ":"))


def f1(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cosine_topk(psi: np.ndarray, Y: np.ndarray, k: int) -> List[int]:
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    pn = psi / (np.linalg.norm(psi) + 1e-9)
    scores = Yn @ pn
    idx = np.argsort(-scores)[:k]
    return idx.tolist()


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
    # Gold facts relevant to the Paris query
    gt_ids = {0, 1, 3, 6, 9}
    # Traps (explicit false claims)
    trap_ids = {2, 5, 8}
    return corpus, gt_ids, trap_ids


def trial_once(
    rng: np.random.Generator,
    k: int,
    *,
    dataset: str,
    lamG: float,
    lamC: float,
    lamQ: float,
    kneighbors: int | None,
    trap_gate: float,
    offtopic_gate: float,
    allow_threshold: float,
    semantic: bool,
    use_diffusion: bool,
    diff_beta: float,
    diff_gamma: float,
    verbose: bool,
    strict_exclude: bool,
) -> tuple[TrialMetrics, TrialMetrics]:
    # Select dataset
    if dataset == "paris":
        corpus, gt_ids, trap_ids = _dataset_paris()
    else:
        corpus, gt_ids, trap_ids = _dataset_mars()
    N, D = len(corpus), 96

    # Embeddings
    if semantic:
        # Use hash/semantic adapter for some structure (deterministic by text content)
        Y = embed_texts(corpus, normalize=True).astype(np.float32)
        psi = (Y[0] / (np.linalg.norm(Y[0]) + 1e-9)).astype(np.float32)
    else:
        # Random embeddings (stable per trial via rng)
        Y = rng.standard_normal((N, D)).astype(np.float32)
        psi = Y[0] / (np.linalg.norm(Y[0]) + 1e-9)

    # Baseline: cosine top-k
    def eval_topk(pred: list[int]) -> TrialMetrics:
        tp = len([i for i in pred if i in gt_ids])
        fp = len([i for i in pred if i not in gt_ids])
        fn = len([i for i in gt_ids if i not in pred])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tshare = len([i for i in pred if i in trap_ids]) / max(1, k)
        return TrialMetrics(
            hallucination=any(i in trap_ids for i in pred),
            precision=prec,
            recall=rec,
            f1=f1(prec, rec),
            trap_share=tshare,
        )

    pred_base = cosine_topk(psi, Y, k)
    if verbose:
        print("[baseline top-k]", pred_base)
    base_metrics = eval_topk(pred_base)

    # Lattice with gates: heavily downweight traps
    effective_k = (
        min(6, max(1, N - 1)) if kneighbors is None else min(max(1, kneighbors), max(1, N - 1))
    )
    lat = OscillinkLattice(Y, kneighbors=effective_k, lamG=lamG, lamC=lamC, lamQ=lamQ)
    gates = np.ones(N, dtype=np.float32)
    for i in range(N):
        if i in trap_ids:
            gates[i] = min(gates[i], trap_gate)
        # mild damp for off-topic (toy example): heuristics can be tuned
        if i not in gt_ids and i not in trap_ids:
            gates[i] = min(gates[i], offtopic_gate)

    # Optional diffusion gating multiplier
    if use_diffusion:
        diff = compute_diffusion_gates(
            Y,
            psi,
            kneighbors=effective_k,
            beta=diff_beta,
            gamma=diff_gamma,
            deterministic_k=False,
        )
        gates = (gates * diff).astype(np.float32)

    # Strict exclude removes traps entirely pre-lattice (demo-mode switch)
    exclude_mask = np.ones(N, dtype=bool)
    if strict_exclude:
        for i in trap_ids:
            exclude_mask[i] = False
    # Optional allow list thresholding to mimic notebook filtering
    allowed = np.array([g > allow_threshold for g in gates]) & exclude_mask

    def lattice_topk() -> list[int]:
        if allowed.any() and allowed.sum() < N:
            idx_map = np.nonzero(allowed)[0]
            Y_sub = Y[idx_map]
            gates_sub = gates[idx_map]
            # If only a single node remains, return it directly to avoid invalid kneighbors
            if len(idx_map) == 1:
                return [int(idx_map[0])]
            # Subgraph neighbor count: clamp to at least 1 and at most len(idx_map)-1
            effective_k_sub = min(effective_k, max(1, len(idx_map) - 1))
            lat_sub = OscillinkLattice(
                Y_sub, kneighbors=effective_k_sub, lamG=lamG, lamC=lamC, lamQ=lamQ
            )
            lat_sub.set_query(psi, gates=gates_sub.astype(np.float32))
            lat_sub.settle()
            pred_local = bundle_topk(lat_sub, k)
            return [int(idx_map[j]) for j in pred_local]
        lat.set_query(psi, gates=gates)
        lat.settle()
        return bundle_topk(lat, k)

    pred_lat = lattice_topk()
    if verbose:
        print("[lattice top-k]", pred_lat)

    lat_metrics = eval_topk(pred_lat)

    return base_metrics, lat_metrics


def run(
    trials: int,
    k: int,
    seed: int,
    *,
    dataset: str,
    lamG: float,
    lamC: float,
    lamQ: float,
    kneighbors: int | None,
    trap_gate: float,
    offtopic_gate: float,
    allow_threshold: float,
    semantic: bool,
    use_diffusion: bool,
    diff_beta: float,
    diff_gamma: float,
    verbose: bool,
    strict_exclude: bool,
    report_trap_share: bool,
) -> Summary:
    rng = np.random.default_rng(seed)
    base_hall = 0
    latt_hall = 0
    base_f1s: List[float] = []
    latt_f1s: List[float] = []
    base_tshares: List[float] = []
    latt_tshares: List[float] = []

    for _ in range(trials):
        # reseed per trial to vary embeddings
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1, dtype=np.uint64).item())
        b, latm = trial_once(
            local_rng,
            k,
            dataset=dataset,
            lamG=lamG,
            lamC=lamC,
            lamQ=lamQ,
            kneighbors=kneighbors,
            trap_gate=trap_gate,
            offtopic_gate=offtopic_gate,
            allow_threshold=allow_threshold,
            semantic=semantic,
            use_diffusion=use_diffusion,
            diff_beta=diff_beta,
            diff_gamma=diff_gamma,
            verbose=verbose,
            strict_exclude=strict_exclude,
        )
        base_hall += 1 if b.hallucination else 0
        latt_hall += 1 if latm.hallucination else 0
        base_f1s.append(b.f1)
        latt_f1s.append(latm.f1)
        if report_trap_share:
            base_tshares.append(b.trap_share)
            latt_tshares.append(latm.trap_share)

    return Summary(
        trials=trials,
        k=k,
        baseline_hallucination_rate=base_hall / trials,
        lattice_hallucination_rate=latt_hall / trials,
        baseline_f1_mean=float(np.mean(base_f1s)) if base_f1s else 0.0,
        lattice_f1_mean=float(np.mean(latt_f1s)) if latt_f1s else 0.0,
        baseline_trap_share_mean=(float(np.mean(base_tshares)) if base_tshares else None),
        lattice_trap_share_mean=(float(np.mean(latt_tshares)) if latt_tshares else None),
    )


def main():
    ap = argparse.ArgumentParser(description="Hallucination suppression proof harness")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    # Data
    ap.add_argument("--dataset", type=str, choices=["mars", "paris"], default="mars")
    # Lattice params
    ap.add_argument("--lamG", type=float, default=1.0)
    ap.add_argument("--lamC", type=float, default=0.5)
    ap.add_argument("--lamQ", type=float, default=2.0)
    ap.add_argument("--kneighbors", type=int, default=None)
    # Gating controls
    ap.add_argument("--trap-gate", dest="trap_gate", type=float, default=0.01)
    ap.add_argument("--offtopic-gate", dest="offtopic_gate", type=float, default=0.6)
    ap.add_argument("--allow-threshold", dest="allow_threshold", type=float, default=0.1)
    # Embedding mode
    ap.add_argument(
        "--semantic", action="store_true", help="Use text adapter embeddings instead of random"
    )
    # Diffusion gating
    ap.add_argument("--diffusion", action="store_true", help="Enable diffusion-based gate shaping")
    ap.add_argument("--diff-beta", dest="diff_beta", type=float, default=1.5)
    ap.add_argument("--diff-gamma", dest="diff_gamma", type=float, default=0.1)
    # Verbose & strict-exclude
    ap.add_argument("--verbose", action="store_true", help="Print baseline and lattice selections")
    ap.add_argument(
        "--strict-exclude",
        action="store_true",
        help="Remove known trap ids pre-lattice (demo mode)",
    )
    # Optional softer metric
    ap.add_argument(
        "--trap-share",
        dest="trap_share",
        action="store_true",
        help="Include softer metric: average share of traps in top-k",
    )
    args = ap.parse_args()

    summary = run(
        args.trials,
        args.k,
        args.seed,
        dataset=args.dataset,
        lamG=args.lamG,
        lamC=args.lamC,
        lamQ=args.lamQ,
        kneighbors=args.kneighbors,
        trap_gate=args.trap_gate,
        offtopic_gate=args.offtopic_gate,
        allow_threshold=args.allow_threshold,
        semantic=args.semantic,
        use_diffusion=args.diffusion,
        diff_beta=args.diff_beta,
        diff_gamma=args.diff_gamma,
        verbose=args.verbose,
        strict_exclude=args.strict_exclude,
        report_trap_share=args.trap_share,
    )

    if args.json:
        print(summary.to_json())
    else:
        print("Trials:", summary.trials)
        print("k:", summary.k)
        print("Baseline hallucination rate:", round(summary.baseline_hallucination_rate, 4))
        print("Lattice hallucination rate:", round(summary.lattice_hallucination_rate, 4))
        print("Baseline F1 mean:", round(summary.baseline_f1_mean, 4))
        print("Lattice F1 mean:", round(summary.lattice_f1_mean, 4))
        if args.trap_share:
            print("Baseline trap-share mean:", round(summary.baseline_trap_share_mean or 0.0, 4))
            print("Lattice trap-share mean:", round(summary.lattice_trap_share_mean or 0.0, 4))


if __name__ == "__main__":
    main()
