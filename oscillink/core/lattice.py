from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from .graph import (
    build_path_laplacian,
    mmr_diversify,
    mutual_knn_adj,
    normalized_laplacian,
    row_sum_cap,
)
from .receipts import deltaH_trace, null_points, per_node_components, verify_receipt
from .solver import cg_solve


class OscillinkLattice:
    """Short‑term coherence container (OLC) with chain priors and receipts.

    Phase‑1 enhancements:
      - Parameter validation (SPD guarantees maintained)
      - U* caching with signature invalidation
      - Export / import of state for reproducibility
      - Basic stats tracking (# of U* solves, cache hits)
    """

    def __init__(
        self,
        Y: np.ndarray,
        kneighbors: int = 6,
        row_cap_val: float = 1.0,
        lamG: float = 1.0,
        lamC: float = 0.5,
        lamQ: float = 4.0,
        deterministic_k: bool = False,
        neighbor_seed: Optional[int] = None,
    ):
        # --- parameter validation ---
        if not isinstance(Y, np.ndarray) or Y.ndim != 2:
            raise ValueError("Y must be a 2D numpy array")
        if kneighbors < 1:
            raise ValueError("kneighbors must be >= 1")
        if lamG <= 0:
            raise ValueError("lamG must be > 0 for SPD")
        for name, val in {"lamC": lamC, "lamQ": lamQ}.items():
            if val < 0:
                raise ValueError(f"{name} must be >= 0")
        self.Y = Y.astype(np.float32).copy()
        self.U = self.Y.copy()
        self.N, self.D = self.Y.shape

        # neighbor build config
        # Clamp kneighbors defensively to avoid errors when user passes k >= N
        k_eff = min(kneighbors, max(1, self.N - 1))
        self._kneighbors = k_eff
        self._deterministic_k = bool(deterministic_k)
        self._neighbor_seed = neighbor_seed
        self._row_cap_val = float(row_cap_val)
        t_build0 = time.time()
        A = mutual_knn_adj(
            self.Y,
            k=k_eff,
            deterministic=self._deterministic_k,
            seed=self._neighbor_seed,
        )
        self.A = row_sum_cap(A, self._row_cap_val)
        self.L_sym, self.sqrt_deg = normalized_laplacian(self.A)
        self._graph_build_ms = 1000.0 * (time.time() - t_build0)

        self.B_diag = np.ones(self.N, dtype=np.float32)
        self.psi = np.zeros(self.D, dtype=np.float32)

        self.lamG, self.lamC, self.lamQ = lamG, lamC, lamQ
        self.L_path: Optional[np.ndarray] = None
        self.A_path: Optional[np.ndarray] = None
        self.lamP = 0.0
        self.last: Dict[str, Any] = {"iters": 0, "res": None, "t_ms": None}
        # original chain node ordering (None until add_chain called)
        self._chain_nodes = None  # type: Optional[list[int]]

        # cache for U*
        self._Ustar_cache: Optional[np.ndarray] = None
        self._Ustar_sig: Optional[str] = None
        self.stats: Dict[str, int] = {"ustar_solves": 0, "ustar_cache_hits": 0}
        self._settle_callbacks: list = []  # user-provided callbacks
        self._logger = None
        self._receipt_secret: Optional[bytes] = None
        # signature mode: 'minimal' (default) or 'extended'
        self._signature_mode: str = "minimal"
        # receipt detail mode: 'full' (default) computes all diagnostics; 'light' skips heavy per-node/nulls
        self._receipt_detail: str = "full"
        # last dynamics (optional): populated when OSCILLINK_RECEIPT_DYNAMICS is enabled and settle() is called
        self._last_dynamics: Optional[Dict[str, Any]] = None
        self._log(
            "init",
            {
                "N": self.N,
                "D": self.D,
                "kneighbors_requested": kneighbors,
                "kneighbors_effective": k_eff,
                "deterministic_k": self._deterministic_k,
                "neighbor_seed": self._neighbor_seed,
            },
        )

    # --- Public API ---

    def set_query(self, psi: np.ndarray, gates: Optional[np.ndarray] = None) -> None:
        self.psi = psi.astype(np.float32).copy()
        if gates is not None:
            if gates.shape[0] != self.N:
                raise ValueError("gates length mismatch N")
            self.B_diag = gates.astype(np.float32).copy()
        self._invalidate_cache()

    def set_gates(self, gates: np.ndarray) -> None:
        """Set gating vector (B_diag) separately from query for clarity."""
        if gates.shape[0] != self.N:
            raise ValueError("gates length mismatch N")
        self.B_diag = gates.astype(np.float32).copy()
        self._invalidate_cache()

    def add_chain(
        self,
        chain: List[int],
        lamP: float = 0.2,
        weights: Optional[List[float]] = None,
    ) -> None:
        if lamP < 0:
            raise ValueError("lamP must be >= 0")
        if any((c < 0 or c >= self.N) for c in chain):
            raise ValueError("chain indices out of bounds")
        if len(chain) < 2:
            raise ValueError("chain must contain at least two indices")
        if weights is not None and len(weights) != len(chain) - 1:
            raise ValueError("weights length must equal len(chain)-1")
        Lp, Ap = build_path_laplacian(self.N, chain, weights)
        self.L_path = Lp
        self.A_path = Ap
        self.lamP = float(lamP)
        self._chain_nodes = list(map(int, chain))
        self._invalidate_cache()
        self._log("add_chain", {"length": len(chain), "lamP": lamP})

    def clear_chain(self) -> None:
        self.L_path = None
        self.A_path = None
        self.lamP = 0.0
        self._chain_nodes = None
        self._invalidate_cache()
        self._log("clear_chain", {})

    def settle(
        self,
        dt: float = 1.0,
        max_iters: int = 12,
        tol: float = 1e-3,
        precond: str = "jacobi",
        *,
        warm_start: bool = True,
        inertia: float = 0.0,
    ) -> Dict[str, Any]:
        """Implicit Euler step: (I + dt M) U^+ = U + dt (lamG Y + lamQ B 1 psi^T)."""
        U_prev = self.U.copy()
        RHS = self.lamG * self.Y + self.lamQ * (self.B_diag[:, None] * self.psi[None, :])

        def A_mul(X: np.ndarray) -> np.ndarray:
            out = X
            out = out + dt * (
                self.lamG * X
                + self.lamC * (self.L_sym @ X)
                + self.lamQ * (self.B_diag[:, None] * X)
            )
            if self.L_path is not None and self.lamP > 0.0:
                out = out + dt * (self.lamP * (self.L_path @ X))
            return out

        b = self.U + dt * RHS
        M_diag = None
        if precond == "jacobi":
            diag_base = (
                self.lamG
                + self.lamQ * self.B_diag
                + (self.lamP if self.L_path is not None else 0.0)
            )
            M_diag = 1.0 + dt * diag_base

        t0 = time.time()
        # Choose starting point: cold start, warm start, or inertial blend
        x0 = self._choose_start_x0(warm_start=warm_start, inertia=inertia)

        U_plus, iters, res = cg_solve(
            A_mul,
            b,
            x0=x0,
            M_diag=M_diag,
            tol=tol,
            max_iters=max_iters,
        )
        self.U = U_plus.astype(np.float32)
        self.last = {"iters": int(iters), "res": float(res), "t_ms": 1000.0 * (time.time() - t0)}
        self._log("settle", self.last)
        if res > tol * 10:  # loose heuristic: significantly above desired tolerance
            self._log(
                "settle_convergence_warn", {"res": float(res), "tol": tol, "iters": int(iters)}
            )
        # Optional dynamics metrics (gated by env flag)
        try:
            import os as _os

            dyn_flag = _os.getenv("OSCILLINK_RECEIPT_DYNAMICS", "0").strip().lower()
            if dyn_flag in {"1", "true", "yes"}:
                self._last_dynamics = self._compute_dynamics(U_prev, self.U, int(iters))
        except Exception:
            # Never break settle path for optional diagnostics
            self._last_dynamics = None
        # callbacks
        if self._settle_callbacks:
            for cb in list(self._settle_callbacks):  # copy to allow modification inside cb
                try:
                    cb(self, self.last)
                except Exception:
                    pass  # swallow to avoid breaking main flow
        return self.last

    def solve_Ustar(
        self,
        tol: float = 1e-4,
        max_iters: int = 64,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Compute stationary U* (optionally using cached value)."""
        sig = self._signature()
        if use_cache and self._Ustar_cache is not None and self._Ustar_sig == sig:
            self.stats["ustar_cache_hits"] += 1
            self._log("ustar_cache_hit", {"signature": sig})
            return self._Ustar_cache

        RHS = self.lamG * self.Y + self.lamQ * (self.B_diag[:, None] * self.psi[None, :])

        def M_mul(X: np.ndarray) -> np.ndarray:
            out = (
                self.lamG * X
                + self.lamC * (self.L_sym @ X)
                + self.lamQ * (self.B_diag[:, None] * X)
            )
            if self.L_path is not None and self.lamP > 0.0:
                out = out + self.lamP * (self.L_path @ X)
            return out

        M_diag = (
            self.lamG + self.lamQ * self.B_diag + (self.lamP if self.L_path is not None else 0.0)
        )
        t_solve0 = time.time()
        Ustar, iters_used, res = cg_solve(
            M_mul, RHS, x0=self.Y, M_diag=M_diag, tol=tol, max_iters=max_iters
        )
        solve_ms = 1000.0 * (time.time() - t_solve0)
        Ustar = Ustar.astype(np.float32)
        converged = bool(res <= tol)
        # store last stationary solve stats separately
        self.last_ustar = {"iters": int(iters_used), "res": float(res), "converged": converged}
        self.last_ustar["solve_ms"] = solve_ms
        if use_cache:
            self._Ustar_cache = Ustar
            self._Ustar_sig = sig
        self.stats["ustar_solves"] += 1
        self._log(
            "ustar_solve",
            {
                "signature": sig,
                "tol": tol,
                "max_iters": max_iters,
                "iters": int(iters_used),
                "res": float(res),
                "converged": converged,
                "solve_ms": solve_ms,
            },
        )
        if not converged:
            self._log(
                "ustar_convergence_warn", {"res": float(res), "tol": tol, "iters": int(iters_used)}
            )
        return Ustar

    def refresh_Ustar(self, tol: float = 1e-4, max_iters: int = 64) -> np.ndarray:
        """Force recomputation of U* ignoring any cached value."""
        self._invalidate_cache()
        self._log("refresh_ustar", {})
        return self.solve_Ustar(tol=tol, max_iters=max_iters, use_cache=True)

    def receipt(self) -> Dict[str, Any]:
        from .. import __version__ as pkg_version

        Ustar = self.solve_Ustar()
        # Always compute core energy delta; conditionally compute heavier diagnostics based on detail mode
        dH = deltaH_trace(
            self.U,
            Ustar,
            self.lamG,
            self.lamC,
            self.L_sym,
            self.lamQ,
            self.B_diag,
            self.lamP,
            self.L_path,
        )
        if self._receipt_detail == "light":
            coh_drop = np.zeros(self.N, dtype=np.float32)
            anchor_pen = np.zeros(self.N, dtype=np.float32)
            query_term = np.zeros(self.N, dtype=np.float32)
            nulls_full = []
        else:
            coh_drop, anchor_pen, query_term = per_node_components(
                self.Y,
                Ustar,
                self.A,
                self.L_sym,
                self.sqrt_deg,
                self.lamG,
                self.lamC,
                self.lamQ,
                self.B_diag,
                self.psi,
            )
            nulls_full = null_points(Ustar, self.A, self.sqrt_deg, self.lamC, z_th=3.0)
        # Null-point capping (observability control)
        import os as _os

        cap_raw = _os.getenv("OSCILLINK_RECEIPT_NULL_CAP", "0").strip()
        try:
            cap_val = int(cap_raw)
        except ValueError:
            cap_val = 0
        if cap_val > 0 and len(nulls_full) > cap_val:
            # Simple strategy: keep highest z values
            nulls_sorted = sorted(nulls_full, key=lambda e: e.get("z", 0.0), reverse=True)
            nulls = nulls_sorted[:cap_val]
            null_meta = {
                "total_null_points": len(nulls_full),
                "returned_null_points": cap_val,
                "null_cap_applied": True,
            }
        else:
            nulls = nulls_full
            null_meta = {
                "total_null_points": len(nulls_full),
                "returned_null_points": len(nulls_full),
                "null_cap_applied": False,
            }
        meta = {
            "ustar_cached": bool(
                self._Ustar_cache is not None and self._Ustar_sig == self._signature()
            ),
            "ustar_solves": int(self.stats["ustar_solves"]),
            "ustar_cache_hits": int(self.stats["ustar_cache_hits"]),
            "ustar_converged": bool(getattr(self, "last_ustar", {}).get("converged", True)),
            "ustar_res": float(getattr(self, "last_ustar", {}).get("res", 0.0)),
            "ustar_iters": int(getattr(self, "last_ustar", {}).get("iters", 0)),
            "ustar_solve_ms": float(getattr(self, "last_ustar", {}).get("solve_ms", 0.0)),
            "graph_build_ms": float(getattr(self, "_graph_build_ms", 0.0)),
            # Some tests call receipt() before any settle(); guard None -> 0.0
            "last_settle_ms": float(self.last.get("t_ms") or 0.0),
            # adjacency stats
            "avg_degree": float(np.sum(self.A > 0) / max(self.N, 1)),
            "edge_density": float(np.sum(self.A > 0) / max(self.N * (self.N - 1), 1)),
            # gating stats (Experimental): summarize B_diag distribution
            "gates_min": float(np.min(self.B_diag)),
            "gates_max": float(np.max(self.B_diag)),
            "gates_mean": float(np.mean(self.B_diag)),
            "gates_uniform": bool(np.allclose(self.B_diag, self.B_diag[0])),
            # deterministic lattice signature (added for parity with cloud endpoints & docs)
            "state_sig": self._signature(),
            "receipt_detail": self._receipt_detail,
            # null point summary
            "null_points_summary": null_meta,
        }
        # signing (optional)
        signature_block = None
        if self._receipt_secret is not None:
            if self._signature_mode == "extended":
                payload = {
                    "sig_v": 1,
                    "mode": "extended",
                    "state_sig": self._signature(),
                    "deltaH_total": float(dH),
                    "ustar_iters": int(
                        self.last_ustar.get("iters", 0) if hasattr(self, "last_ustar") else 0
                    ),
                    "ustar_res": float(
                        self.last_ustar.get("res", 0.0) if hasattr(self, "last_ustar") else 0.0
                    ),
                    "ustar_converged": bool(
                        self.last_ustar.get("converged", True)
                        if hasattr(self, "last_ustar")
                        else True
                    ),
                    "params": {
                        "lamG": self.lamG,
                        "lamC": self.lamC,
                        "lamQ": self.lamQ,
                        "lamP": self.lamP,
                    },
                    "graph": {
                        "k": self._kneighbors,
                        "deterministic_k": self._deterministic_k,
                        "neighbor_seed": self._neighbor_seed,
                    },
                }
            else:  # minimal
                payload = {
                    "sig_v": 1,
                    "mode": "minimal",
                    "state_sig": self._signature(),
                    "deltaH_total": float(dH),
                }
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            sig_hex = hmac.new(self._receipt_secret, raw, hashlib.sha256).hexdigest()
            signature_block = {"algorithm": "HMAC-SHA256", "payload": payload, "signature": sig_hex}
            meta["signature"] = signature_block
        out = {
            "version": str(pkg_version),
            "deltaH_total": float(dH),
            "coh_drop_sum": float(np.sum(coh_drop)),
            "anchor_pen_sum": float(np.sum(anchor_pen)),
            "query_term_sum": float(np.sum(query_term)),
            # Some tests invoke receipt() before any settle(); protect against None placeholders.
            "cg_iters": int(self.last.get("iters") or 0),
            "residual": float(self.last.get("res") or 0.0),
            "t_ms": float(self.last.get("t_ms") or 0.0),
            "null_points": nulls,
            "meta": meta,
        }
        # Optionally include last dynamics snapshot under meta when enabled
        try:
            import os as _os

            dyn_flag = _os.getenv("OSCILLINK_RECEIPT_DYNAMICS", "0").strip().lower()
            if (
                dyn_flag in {"1", "true", "yes"}
                and getattr(self, "_last_dynamics", None) is not None
            ):
                meta["dynamics"] = self._last_dynamics
        except Exception:
            pass
        self._log(
            "receipt", {"deltaH_total": out["deltaH_total"], "ustar_cached": meta["ustar_cached"]}
        )
        return out

    def verify_current_receipt(self, secret: bytes | str) -> bool:
        """Convenience validation of the most recent receipt with provided secret.

        Re-computes a fresh receipt (uses cached U* if valid) and applies HMAC verification.
        Returns False if signature block absent or invalid.
        """
        rec = self.receipt()
        return verify_receipt(rec, secret)

    def chain_receipt(self, chain: List[int], z_th: float = 2.5) -> Dict[str, Any]:
        Ustar = self.solve_Ustar()
        di = self.sqrt_deg + 1e-12
        Un = Ustar / di[:, None]
        diffs = Un[:, None, :] - Un[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)

        # structural residuals
        R_s = self.lamC * self.A * d2.astype(np.float32)
        mu_s = R_s.mean(axis=1, keepdims=True)
        sig_s = R_s.std(axis=1, keepdims=True) + 1e-12

        # path residuals
        if self.A_path is None:
            _, Apath = build_path_laplacian(self.N, chain, None)
            A_p = Apath
        else:
            A_p = self.A_path
        R_p = max(self.lamC, 1e-6) * A_p * d2.astype(np.float32)
        mu_p = R_p.mean(axis=1, keepdims=True)
        sig_p = R_p.std(axis=1, keepdims=True) + 1e-12

        edges: List[Dict[str, Any]] = []
        worst = (-1, -1.0, (-1, -1))
        gain = 0.0
        for k in range(len(chain) - 1):
            i, j = int(chain[k]), int(chain[k + 1])
            z_struct = float((R_s[i, j] - mu_s[i, 0]) / sig_s[i, 0])
            z_path = float((R_p[i, j] - mu_p[i, 0]) / sig_p[i, 0])
            rs, rp = float(R_s[i, j]), float(R_p[i, j])
            edges.append(
                {
                    "k": int(k),
                    "edge": [int(i), int(j)],
                    "z_struct": float(z_struct),
                    "z_path": float(z_path),
                    "r_struct": float(rs),
                    "r_path": float(rp),
                }
            )
            if max(z_struct, z_path) > worst[1]:
                worst = (k, max(z_struct, z_path), (i, j))

            # chain coherence gain vs anchors
            ydiff = (self.Y[i] / di[i]) - (self.Y[j] / di[j])
            udiff = Un[i] - Un[j]
            gain += (
                0.5
                * self.lamC
                * max(self.A[i, j], 0.0)
                * (float(ydiff @ ydiff) - float(udiff @ udiff))
            )

        # Ensure numeric typing for mypy: cast to float before comparison
        verdict = all(max(float(e["z_struct"]), float(e["z_path"])) <= float(z_th) for e in edges)
        return {
            "verdict": bool(verdict),
            "weakest_link": {
                "k": int(worst[0]),
                "edge": [int(worst[2][0]), int(worst[2][1])],
                "zscore": float(worst[1]),
            },
            "coherence_gain": float(gain),
            "edges": edges,
        }

    def bundle(self, k: int = 8, alpha: float = 0.5) -> list[dict]:
        """Return top-k diversified bundle of nodes.

        Scoring:
          score = alpha * z(coherence_drop) + (1 - alpha) * cosine(U*_i, psi)

        - coherence_drop: per-node Δ in pairwise energy vs anchors (higher -> more informative anomaly)
        - alignment: semantic alignment of stationary embedding with query
        - z() normalizes coherence_drop to mean 0 / std 1 for comparability
        - MMR diversification (lambda_div=0.5) applied over original Y vectors

        Parameters
        ---------
        k : int
            Number of items to return (after diversification)
        alpha : float in [0,1]
            Trade-off weight (1.0 = pure coherence anomaly, 0.0 = pure alignment)

        Returns
        -------
        list[dict]
            Each element: { 'id': int, 'score': float, 'align': float }

        Notes
        -----
        Uses cached U* if valid; otherwise triggers a stationary solve.
        """
        Ustar = self.solve_Ustar()
        # alignment
        u_norm = np.linalg.norm(Ustar, axis=1, keepdims=True) + 1e-12
        psi_n = self.psi / (np.linalg.norm(self.psi) + 1e-12)
        align = (Ustar / u_norm) @ psi_n
        coh = self._coherence_drop(Ustar)
        # z-score
        mu, sigma = float(np.mean(coh)), float(np.std(coh) + 1e-12)
        z = (coh - mu) / sigma if sigma > 0 else np.zeros_like(coh)
        score = alpha * z + (1 - alpha) * align.squeeze()
        order = mmr_diversify(self.Y, score, k=k, lambda_div=0.5)
        return [{"id": int(i), "score": float(score[i]), "align": float(align[i])} for i in order]

    # --- Callback registration ---
    def add_settle_callback(self, fn) -> None:
        """Register a callback fn(lattice, stats_dict) executed after each settle()."""
        self._settle_callbacks.append(fn)

    def remove_settle_callback(self, fn) -> None:
        try:
            self._settle_callbacks.remove(fn)
        except ValueError:
            pass

    # --- Export / Import helpers ---
    def export_state(
        self, include_graph: bool = True, include_chain: bool = True
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict capturing lattice state for reproducibility."""
        from .. import __version__ as pkg_version  # local import to avoid cycle at top-level

        # provenance hash: stable digest of key numerical state (Y, psi, B_diag, params, adjacency subset)
        # Use same adjacency sampling strategy as signature to avoid huge blobs; include shape + params.
        nz = np.argwhere(self.A > 0)[:2048]
        h = hashlib.sha256()
        h.update(self.Y.tobytes())
        h.update(self.psi.tobytes())
        h.update(self.B_diag.tobytes())
        h.update(np.array([self.lamG, self.lamC, self.lamQ, self.lamP], dtype=np.float64).tobytes())
        h.update(nz.tobytes())
        provenance = h.hexdigest()
        state: Dict[str, Any] = {
            "version": str(pkg_version),
            "shape": [int(self.N), int(self.D)],
            "params": {"lamG": self.lamG, "lamC": self.lamC, "lamQ": self.lamQ, "lamP": self.lamP},
            "Y": self.Y.tolist(),
            "psi": self.psi.tolist(),
            "B_diag": self.B_diag.tolist(),
            "kneighbors": int(self._kneighbors),
            "deterministic_k": bool(self._deterministic_k),
            "neighbor_seed": self._neighbor_seed,
            "provenance": provenance,
        }
        if include_graph:
            state["A"] = self.A.tolist()
        if include_chain and self.L_path is not None:
            # chain reconstruction requires indices present in path adjacency; store edges
            edges = []
            nz = np.argwhere(
                (self.A_path if self.A_path is not None else np.zeros_like(self.A)) > 0
            )
            for i, j in nz:
                if i < j:
                    edges.append([int(i), int(j)])
            state["chain_edges"] = edges
            if self._chain_nodes is not None:
                state["chain_nodes"] = list(self._chain_nodes)
        return state

    def save_state(
        self,
        path: str,
        format: str = "json",
        include_graph: bool = True,
        include_chain: bool = True,
    ) -> None:
        """Persist lattice state to disk.

        format:
          - 'json' (default) writes UTF-8 JSON
          - 'npz' writes a compressed NumPy archive (faster / smaller for large arrays)
        """
        fmt = format.lower()
        state = self.export_state(include_graph=include_graph, include_chain=include_chain)
        if fmt == "json":
            # json already imported at module top; avoid re-import here so we don't create a
            # function-local binding that would shadow the global and break the npz branch.
            import io  # noqa: F401 (reserved for potential streaming / future incremental writes)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, sort_keys=True)
        elif fmt == "npz":
            # separate heavy numeric arrays to avoid JSON overhead
            arrays: dict[str, np.ndarray] = {
                "Y": self.Y,
                "psi": self.psi,
                "B_diag": self.B_diag,
            }
            if include_graph:
                arrays["A"] = self.A
            if include_chain and self._chain_nodes is not None:
                arrays["chain_nodes"] = np.array(self._chain_nodes, dtype=np.int32)
            # store lightweight metadata JSON as a string field
            meta = state.copy()
            # remove arrays duplicated in archive to reduce duplication
            for k in ["Y", "psi", "B_diag", "A", "chain_nodes"]:
                meta.pop(k, None)
            meta_json = json.dumps(meta, sort_keys=True)
            # Build a merged kwargs dict to avoid mypy confusion about positional args
            archive: dict[str, np.ndarray] = {"__meta__": np.array(meta_json)}
            archive.update(arrays)
            np.savez_compressed(path, **archive)
        else:
            raise ValueError("format must be 'json' or 'npz'")

    @classmethod
    def from_npz(cls, path: str) -> OscillinkLattice:
        """Load lattice from a compressed npz produced by save_state(format='npz')."""
        with np.load(path, allow_pickle=False) as data:
            meta_json = str(data["__meta__"])  # stored as 0-d array
            meta = json.loads(meta_json)
            Y = data["Y"].astype(np.float32)
            state = {
                **meta,
                "Y": Y.tolist(),
                "psi": data["psi"].astype(np.float32).tolist(),
                "B_diag": data["B_diag"].astype(np.float32).tolist(),
            }
            if "A" in data.files:
                state["A"] = data["A"].astype(np.float32).tolist()
            if "chain_nodes" in data.files:
                state["chain_nodes"] = data["chain_nodes"].astype(int).tolist()
        return cls.from_state(state)

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> OscillinkLattice:
        Y = np.array(state["Y"], dtype=np.float32)
        params = state.get("params", {})
        lat = cls(
            Y,
            kneighbors=state.get("kneighbors", 6),
            lamG=params.get("lamG", 1.0),
            lamC=params.get("lamC", 0.5),
            lamQ=params.get("lamQ", 4.0),
            deterministic_k=state.get("deterministic_k", False),
            neighbor_seed=state.get("neighbor_seed"),
        )
        psi = np.array(state.get("psi", np.zeros(Y.shape[1], dtype=np.float32)), dtype=np.float32)
        B = np.array(state.get("B_diag", np.ones(Y.shape[0], dtype=np.float32)), dtype=np.float32)
        lat.set_query(psi, gates=B)
        # Restore adjacency if provided (overrides rebuild differences / randomness in ties)
        if "A" in state:
            A = np.array(state["A"], dtype=np.float32)
            if A.shape == (lat.N, lat.N):
                lat.A = A
                lat.L_sym, lat.sqrt_deg = normalized_laplacian(lat.A)
        lamP = params.get("lamP", 0.0)
        if lamP > 0:
            if "chain_nodes" in state:
                lat.add_chain(list(map(int, state["chain_nodes"])), lamP=lamP)
            elif "chain_edges" in state:
                edges = state["chain_edges"]
                if edges:
                    flat = sorted({i for e in edges for i in e})
                    lat.add_chain(flat, lamP=lamP)
        # Optionally store provenance from input for downstream comparison
        if "provenance" in state:
            lat._imported_provenance = state["provenance"]  # type: ignore[attr-defined]
        return lat

    # --- Internal helpers ---
    def _signature(self) -> str:
        # adjacency fingerprint (subset of nonzero indices for stability & perf)
        nz = np.argwhere(self.A > 0)[:2048]
        adj_sig = hashlib.sha256(nz.tobytes()).hexdigest()
        data = {
            "psi": np.round(self.psi, 6).tolist(),
            "B": np.round(self.B_diag, 6).tolist(),
            "lam": [self.lamG, self.lamC, self.lamQ, self.lamP],
            "chain_present": self.L_path is not None,
            "chain_len": len(self._chain_nodes) if self._chain_nodes else 0,
            "k": self._kneighbors,
            "detk": self._deterministic_k,
            "adj": adj_sig,
        }
        raw = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _invalidate_cache(self) -> None:
        self._Ustar_cache = None
        self._Ustar_sig = None
        self._log("invalidate_cache", {})

    def _choose_start_x0(self, *, warm_start: bool, inertia: float) -> np.ndarray:
        """Select CG initial guess x0 based on warm_start/inertia flags."""
        if not warm_start:
            return self.Y
        w = float(max(0.0, min(1.0, inertia)))
        if w <= 0.0:
            return self.U
        return ((1.0 - w) * self.Y + w * self.U).astype(np.float32)

    def rebuild_graph(
        self,
        *,
        row_cap_val: Optional[float] = None,
        kneighbors: Optional[int] = None,
        deterministic_k: Optional[bool] = None,
        neighbor_seed: Optional[int] = None,
    ) -> None:
        """Rebuild adjacency and Laplacian with optional new parameters.

        Parameters are optional; unspecified values reuse current settings. Invalidates U* cache and updates timing.
        """
        # Update config
        if row_cap_val is not None:
            self._row_cap_val = float(row_cap_val)
        if kneighbors is not None:
            self._kneighbors = min(int(kneighbors), max(1, self.N - 1))
        if deterministic_k is not None:
            self._deterministic_k = bool(deterministic_k)
        if neighbor_seed is not None:
            self._neighbor_seed = neighbor_seed
        # Rebuild
        t_build0 = time.time()
        A = mutual_knn_adj(
            self.Y,
            k=self._kneighbors,
            deterministic=self._deterministic_k,
            seed=self._neighbor_seed,
        )
        self.A = row_sum_cap(A, self._row_cap_val)
        self.L_sym, self.sqrt_deg = normalized_laplacian(self.A)
        self._graph_build_ms = 1000.0 * (time.time() - t_build0)
        self._invalidate_cache()
        self._log(
            "rebuild_graph",
            {
                "k": int(self._kneighbors),
                "row_cap_val": float(self._row_cap_val),
                "deterministic_k": self._deterministic_k,
                "neighbor_seed": self._neighbor_seed,
            },
        )

    def _coherence_drop(self, Ustar: np.ndarray) -> np.ndarray:
        """Compute per-node coherence drop term reused across receipt diagnostics and ranking."""
        Yn = self.Y / (self.sqrt_deg[:, None] + 1e-12)
        Un = Ustar / (self.sqrt_deg[:, None] + 1e-12)
        coh = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            # iterate only over neighbors with non-zero weight
            nz = np.nonzero(self.A[i])[0]
            if nz.size == 0:
                continue
            yi = Yn[i]
            ui = Un[i]
            for j in nz:
                w = float(self.A[i, j])
                if w <= 0.0:
                    continue
                ydiff = yi - Yn[j]
                udiff = ui - Un[j]
                coh[i] += 0.5 * self.lamC * w * (float(ydiff @ ydiff) - float(udiff @ udiff))
        return coh

    # --- Dynamics metrics (optional) ---
    def _compute_dynamics(
        self, U_prev: np.ndarray, U_next: np.ndarray, iters: int
    ) -> Dict[str, Any]:
        """Compute a single-step dynamics snapshot.

        Metrics:
          - temperature: mean(||ΔU||^2) across nodes
          - viscosity_step: iters / ΔH_step (ΔH between U_prev and U_next via deltaH_trace)
          - flow: per-edge energy drop in structural term, with top edges
          - coherence_radius: max graph hop distance reached by activated nodes (|ΔU_i| above threshold)
        """
        # Movement / temperature
        dU = (U_next - U_prev).astype(np.float32)
        move2 = np.sum(dU * dU, axis=1)  # per-node squared movement
        temperature = float(np.mean(move2))

        # Step energy drop (approximate) and viscosity
        dH_step = float(
            deltaH_trace(
                U_prev,
                U_next,
                self.lamG,
                self.lamC,
                self.L_sym,
                self.lamQ,
                self.B_diag,
                self.lamP,
                self.L_path,
            )
        )
        viscosity_step = float(iters) / (abs(dH_step) + 1e-12)

        # Coherence flow on edges: drop in structural pairwise energy
        di = self.sqrt_deg + 1e-12
        Up = U_prev / di[:, None]
        Un = U_next / di[:, None]
        nz = np.argwhere(self.A > 0)
        flows: List[Dict[str, Any]] = []
        flow_total = 0.0
        # Limit top edges kept to avoid huge payloads
        TOP_K = 16
        for i, j in nz:
            i_i, j_i = int(i), int(j)
            w = float(self.A[i_i, j_i])
            if w <= 0.0 or i_i == j_i:
                continue
            ydiff_prev = Up[i_i] - Up[j_i]
            ydiff_next = Un[i_i] - Un[j_i]
            e_prev = 0.5 * self.lamC * w * float(ydiff_prev @ ydiff_prev)
            e_next = 0.5 * self.lamC * w * float(ydiff_next @ ydiff_next)
            f = max(0.0, e_prev - e_next)
            if f > 0.0:
                flow_total += f
                flows.append({"edge": [i_i, j_i], "flow": float(f)})
        # Top flows by magnitude
        if flows:
            flows.sort(key=lambda e: e["flow"], reverse=True)
            flows = flows[:TOP_K]

        # Coherence radius via BFS from highly affected nodes
        inf = np.sqrt(move2 + 1e-12)
        if inf.size == 0 or float(np.max(inf)) <= 1e-9:
            radius = 0
        else:
            thr = 0.1 * float(np.max(inf))
            seeds = [int(i) for i in np.where(inf >= thr)[0].tolist()]
            radius = self._bfs_radius(seeds)

        return {
            "temperature": temperature,
            "step_deltaH": dH_step,
            "viscosity_step": viscosity_step,
            "flow_total": float(flow_total),
            "top_flows": flows,
            "radius": int(radius),
            # quick movement stats
            "move2_mean": float(np.mean(move2) if move2.size else 0.0),
            "move2_max": float(np.max(move2) if move2.size else 0.0),
        }

    def _bfs_radius(self, seeds: List[int]) -> int:
        if not seeds:
            return 0
        N = self.N
        visited = np.full(N, False)
        dist = np.full(N, -1, dtype=int)
        q: deque[int] = deque()
        for s in seeds:
            if 0 <= s < N and not visited[s]:
                visited[s] = True
                dist[s] = 0
                q.append(s)
        # build adjacency lists for speed
        neighbors = [np.where(self.A[i] > 0)[0].astype(int).tolist() for i in range(N)]
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if not visited[v]:
                    visited[v] = True
                    dist[v] = dist[u] + 1
                    q.append(v)
        # radius among reached nodes
        return int(np.max(dist)) if np.any(dist >= 0) else 0

    # --- Logging API ---
    def set_logger(self, logger_callable) -> None:
        """Attach a logger callable(event:str, payload:dict). Pass None to detach."""
        self._logger = logger_callable

    def _log(self, event: str, payload: dict) -> None:
        if self._logger is not None:
            try:
                self._logger(event, payload)
            except Exception:
                pass

    # --- Receipt signing API ---
    def set_receipt_secret(self, secret: bytes | str | None) -> None:
        """Configure HMAC-SHA256 signing secret for receipts. Pass None to disable signing."""
        if secret is None:
            self._receipt_secret = None
        else:
            if isinstance(secret, str):
                secret = secret.encode("utf-8")
            self._receipt_secret = secret

    def set_signature_mode(self, mode: str) -> None:
        """Configure signature payload size.

        mode:
          - 'minimal' (default): state_sig + deltaH_total
          - 'extended': adds convergence stats, params and graph build params.
        Any other value raises ValueError.
        """
        m = mode.lower().strip()
        if m not in {"minimal", "extended"}:
            raise ValueError("mode must be 'minimal' or 'extended'")
        self._signature_mode = m

    def set_receipt_detail(self, mode: str) -> None:
        """Configure receipt detail level.

        mode:
          - 'full' (default): includes per-node diagnostics (coh_drop, anchor_pen, query_term) and null points
          - 'light': skips heavy per-node and null point computations (fast path)
        Any other value raises ValueError.
        """
        m = mode.lower().strip()
        if m not in {"full", "light"}:
            raise ValueError("mode must be 'full' or 'light'")
        self._receipt_detail = m

    # --- Representation ---
    def __repr__(self) -> str:  # pragma: no cover (formatting deterministic & simple)
        parts = [
            f"N={self.N}",
            f"D={self.D}",
            f"k={self._kneighbors}",
            f"lamG={self.lamG}",
            f"lamC={self.lamC}",
            f"lamQ={self.lamQ}",
        ]
        if self.lamP > 0 and self._chain_nodes is not None:
            parts.append(f"chain_len={len(self._chain_nodes)}")
            parts.append(f"lamP={self.lamP}")
        if self._Ustar_cache is not None:
            parts.append("U*cached")
        return "OscillinkLattice(" + ", ".join(parts) + ")"


def json_line_logger(stream=None):
    """Return a logger callable that writes compact JSON Lines events.

    Usage:
        lat.set_logger(json_line_logger())
    """
    import json as _json
    import sys

    if stream is None:
        stream = sys.stderr

    def _log(ev: str, payload: dict):  # pragma: no cover (straightforward serialization)
        try:
            obj = {"event": ev, **payload}
            stream.write(_json.dumps(obj, separators=(",", ":")) + "\n")
        except Exception:
            pass

    return _log
