from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple

"""Lightweight adaptive profile system (cloud-only).

Reads per-key learned parameter profiles from Firestore when configured and
returns an active profile id plus a dict of param overrides. Optionally, when
learning is enabled, proposes exploratory overrides and records observations
with bounded EMA updates (safe and opt-in).

Environment:
  - OSCILLINK_LEARNERS_COLLECTION: Firestore collection name. If unset, loader is disabled.
  - OSCILLINK_ADAPTIVE_PROFILES: '1'/'true' to enable applying overrides.
    - OSCILLINK_ADAPTIVE_LEARN: '1'/'true' to enable write-path learning.
    - OSCILLINK_ADAPTIVE_ALPHA: EMA factor for param updates (default 0.2)
    - OSCILLINK_ADAPTIVE_EPS: epsilon for exploration (default 0.1)
    - OSCILLINK_ADAPTIVE_MARGIN: minimum score improvement to update EMA (default 0.0)

Firestore document shape (suggested):
  doc id: api_key
  fields:
    active_profile_id: str (e.g., 'default' or 'p-2025-10-10')
    learned_params: { lamG: float, lamC: float, lamQ: float, kneighbors: int }
    updated_at: float
        ema_score: float (optional)
        last_observation: { ts: float, metrics: {...}, overrides: {...} }
"""


def _enabled() -> bool:
    return os.getenv("OSCILLINK_ADAPTIVE_PROFILES", "0").lower() in {"1", "true", "on", "yes"}


def _collection() -> str:
    return os.getenv("OSCILLINK_LEARNERS_COLLECTION", "").strip()


def _learn_enabled() -> bool:
    return os.getenv("OSCILLINK_ADAPTIVE_LEARN", "0").lower() in {"1", "true", "on", "yes"}


def _alpha() -> float:
    try:
        return max(0.0, min(1.0, float(os.getenv("OSCILLINK_ADAPTIVE_ALPHA", "0.2"))))
    except Exception:
        return 0.2


def _eps() -> float:
    try:
        return max(0.0, min(1.0, float(os.getenv("OSCILLINK_ADAPTIVE_EPS", "0.1"))))
    except Exception:
        return 0.1


def _margin() -> float:
    try:
        return float(os.getenv("OSCILLINK_ADAPTIVE_MARGIN", "0.0"))
    except Exception:
        return 0.0


# Guardrails (promotion cadence / rate limits)
def _min_obs() -> int:
    try:
        return max(1, int(os.getenv("OSCILLINK_ADAPTIVE_MIN_OBS", "500")))
    except Exception:
        return 500


def _min_promote_sec() -> float:
    try:
        return max(0.0, float(os.getenv("OSCILLINK_ADAPTIVE_MIN_PROMOTE_SEC", "900")))
    except Exception:
        return 900.0


def _eps_floor() -> float:
    try:
        return max(0.0, min(1.0, float(os.getenv("OSCILLINK_ADAPTIVE_EPS_FLOOR", "0.02"))))
    except Exception:
        return 0.02


def _heavy_min_obs() -> int:
    try:
        return max(1, int(os.getenv("OSCILLINK_ADAPTIVE_HEAVY_PROMOTE_MIN_OBS", "5000")))
    except Exception:
        return 5000


# Conservative bounds to prevent destabilization
_BOUNDS = {
    "lamG": (0.5, 2.0),
    "lamC": (0.1, 2.5),
    "lamQ": (1.0, 8.0),
    # kneighbors is clamped later to [1, N-1] in lattice builder; still add a soft bound
    "kneighbors": (1, 64),
}


def _clip(name: str, val: Any) -> Any:
    if name not in _BOUNDS:
        return val
    lo, hi = _BOUNDS[name]
    try:
        if isinstance(val, (int, float)):
            if name == "kneighbors":
                return int(max(lo, min(hi, int(val))))
            return float(max(lo, min(hi, float(val))))
    except Exception:
        return None
    return val


def get_active_profile(api_key: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Return (profile_id, overrides) for the api_key.

    If disabled or not found, returns ("baseline", {}). All values are clipped
    to safe ranges. Never raises; returns empty overrides on any error.
    """
    if not _enabled() or not _collection() or not api_key:
        return "baseline", {}
    try:  # pragma: no cover - external dependency
        from google.cloud import firestore  # type: ignore

        client = firestore.Client()
        snap = client.collection(_collection()).document(api_key).get()
        if not snap.exists:
            return "baseline", {}
        data = snap.to_dict() or {}
        prof_id = str(data.get("active_profile_id") or "default")
        lp = data.get("learned_params") or {}
        out: Dict[str, Any] = {}
        for k in ("lamG", "lamC", "lamQ", "kneighbors"):
            if k in lp:
                cv = _clip(k, lp[k])
                if cv is not None:
                    out[k] = cv
        return prof_id, out
    except Exception:
        return "baseline", {}


def propose_overrides(
    api_key: Optional[str],
    base: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return (profile_id, overrides) possibly with exploration applied.

    - If profiles disabled or no collection/key, return ("baseline", {})
    - If learning disabled, behaves like get_active_profile()
    - If learning enabled, with probability eps proposes exploratory params
      around current learned/base values (clipped to safe bounds).
    """
    prof_id, overrides = get_active_profile(api_key)
    if not _learn_enabled() or not _collection() or not api_key:
        return prof_id, overrides
    # exploration around the working set (learned -> base -> defaults)
    working = {
        "lamG": overrides.get("lamG", (base or {}).get("lamG", 1.0)),
        "lamC": overrides.get("lamC", (base or {}).get("lamC", 0.5)),
        "lamQ": overrides.get("lamQ", (base or {}).get("lamQ", 4.0)),
        "kneighbors": overrides.get("kneighbors", (base or {}).get("kneighbors", 6)),
    }
    # Maintain an exploration floor to keep learning alive
    eps_eff = max(_eps(), _eps_floor())
    if random.random() < eps_eff:
        # multiplicative jitter for lam*, additive small integer jitter for k
        def jmul(x: float, s: float = 0.15) -> float:
            try:
                return float(x) * float(max(0.5, min(2.0, 1.0 + random.uniform(-s, s))))
            except Exception:
                return x

        def jaddk(k: int) -> int:
            try:
                return int(k) + random.choice([-1, 0, 1])
            except Exception:
                return k

        exp = {
            "lamG": _clip("lamG", jmul(working["lamG"])),
            "lamC": _clip("lamC", jmul(working["lamC"])),
            "lamQ": _clip("lamQ", jmul(working["lamQ"])),
            "kneighbors": _clip("kneighbors", jaddk(int(working["kneighbors"]))),
        }
        return "explore", exp  # exploration tag; caller may surface as needed
    return prof_id, overrides


def record_observation(
    api_key: Optional[str],
    profile_id: str,
    overrides: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    """Record an observation and optionally update learned params via bounded EMA.

    metrics expected keys (best-effort):
      - duration_ms: float
      - iters: int
      - residual: float
      - tol: float

    Score heuristic: higher is better -> score = -(duration_ms) - 100 * residual
    Update rule: if score > (ema_score + margin), then learned_params <- EMA toward overrides
    All operations are best-effort and gated via env; failures are swallowed.
    """
    if not _enabled():
        return
    if not _learn_enabled():
        return
    if not _collection() or not api_key:
        return
    try:  # pragma: no cover - external dependency
        _record_observation_impl(api_key, profile_id, overrides, metrics)
    except Exception:
        # swallow to avoid impacting request flow
        return


def _record_observation_impl(
    api_key: str,
    profile_id: str,
    overrides: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    # local imports to avoid hard dependency when disabled
    import time as _time

    from google.cloud import firestore  # type: ignore

    def _score(m: Dict[str, Any]) -> float:
        duration = float(m.get("duration_ms", 0.0))
        residual = float(m.get("residual", 0.0))
        return -duration - 100.0 * residual

    def _ema(old: float, new: float) -> float:
        return (1.0 - _alpha()) * old + _alpha() * new

    client = firestore.Client()
    doc_ref = client.collection(_collection()).document(api_key)
    snap = doc_ref.get()
    data = snap.to_dict() if snap.exists else {}
    if data is None:
        data = {}
    lp = (data or {}).get("learned_params") or {}
    ema_score = float((data or {}).get("ema_score", float("-inf")))
    obs_since_promo = int((data or {}).get("obs_since_promo", 0))
    last_promoted_at = float((data or {}).get("last_promoted_at", 0.0))
    current_active = (
        data.get("active_profile_id") if isinstance(data, dict) else None
    ) or profile_id

    # compute score
    score = _score(metrics)

    # init ema if not present
    ema_score = score if ema_score == float("-inf") else _ema(ema_score, score)

    # conditional EMA update toward overrides
    improved = score > (ema_score + _margin())
    # Determine if this is a heavy change (graph rebuild affecting latency/quality strongly)
    heavy_keys = ("kneighbors", "row_cap_val")
    heavy_change = any((k in overrides) and (overrides.get(k) != lp.get(k)) for k in heavy_keys)
    min_obs_needed = _heavy_min_obs() if heavy_change else _min_obs()
    min_sec_needed = _min_promote_sec()

    # Decide promotion eligibility
    now_ts = _time.time()
    eligible = (
        improved
        and ((obs_since_promo + 1) >= min_obs_needed)
        and ((now_ts - last_promoted_at) >= min_sec_needed)
    )

    new_lp = dict(lp)
    did_promote = False
    if eligible and overrides:
        for k in ("lamG", "lamC", "lamQ", "kneighbors"):
            if k in overrides:
                cur = new_lp.get(k, overrides[k])
                try:
                    val = _ema(float(cur), float(overrides[k]))
                except Exception:
                    val = overrides[k]
                new_lp[k] = _clip(k, val)
        did_promote = True

    # Update counters
    new_obs_since = 0 if did_promote else (obs_since_promo + 1)

    payload = {
        "active_profile_id": current_active,
        "learned_params": new_lp if did_promote else lp,
        "ema_score": ema_score,
        "updated_at": now_ts,
        "last_promoted_at": (now_ts if did_promote else last_promoted_at),
        "obs_since_promo": int(new_obs_since),
        "last_observation": {
            "ts": now_ts,
            "profile_id": profile_id,
            "overrides": overrides,
            "metrics": metrics,
            "score": score,
            "improved": bool(improved),
            "promotion": {
                "did_promote": bool(did_promote),
                "heavy_change": bool(heavy_change),
                "min_obs_needed": int(min_obs_needed),
                "min_sec_needed": float(min_sec_needed),
            },
        },
    }
    # Merge to avoid clobbering unrelated fields
    doc_ref.set(payload, merge=True)
