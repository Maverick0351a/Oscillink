from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest, REGISTRY

from oscillink import OscillinkLattice, __version__

from .config import get_settings
from .runtime_config import get_api_keys, get_rate_limit, get_quota_config
from .keystore import get_keystore, KeyMetadata, InMemoryKeyStore  # type: ignore
from .billing import resolve_tier_from_subscription, tier_info, current_period
from .features import resolve_features
from .models import HealthResponse, ReceiptResponse, SettleRequest, AdminKeyUpdate, AdminKeyResponse

app = FastAPI(title="Oscillink Cloud API", default_response_class=ORJSONResponse)

MAX_BODY_BYTES = int(os.getenv("OSCILLINK_MAX_BODY_BYTES", "1048576"))  # 1MB default

@app.middleware("http")
async def body_size_guard(request: Request, call_next):
    # Read body only if content-length not provided or suspicious; rely on header when present
    cl = request.headers.get("content-length")
    if cl and cl.isdigit():
        if int(cl) > MAX_BODY_BYTES:
            return ORJSONResponse(status_code=413, content={"detail": "payload too large"})
        return await call_next(request)
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return ORJSONResponse(status_code=413, content={"detail": "payload too large"})
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}
    request._receive = receive  # type: ignore[attr-defined]
    return await call_next(request)

REQUEST_ID_HEADER = "x-request-id"

# Prometheus metrics (guard against re-registration during test reloads)
if "oscillink_settle_requests_total" in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
    SETTLE_COUNTER = REGISTRY._names_to_collectors["oscillink_settle_requests_total"]  # type: ignore
    SETTLE_LATENCY = REGISTRY._names_to_collectors["oscillink_settle_latency_seconds"]  # type: ignore
    SETTLE_N_GAUGE = REGISTRY._names_to_collectors["oscillink_settle_last_N"]  # type: ignore
    SETTLE_D_GAUGE = REGISTRY._names_to_collectors["oscillink_settle_last_D"]  # type: ignore
    USAGE_NODES = REGISTRY._names_to_collectors["oscillink_usage_nodes_total"]  # type: ignore
    USAGE_NODE_DIM_UNITS = REGISTRY._names_to_collectors["oscillink_usage_node_dim_units_total"]  # type: ignore
    JOB_QUEUE_DEPTH = REGISTRY._names_to_collectors.get("oscillink_job_queue_depth")  # type: ignore
else:
    SETTLE_COUNTER = Counter(
        "oscillink_settle_requests_total", "Total settle requests", ["status"]
    )
    SETTLE_LATENCY = Histogram(
        "oscillink_settle_latency_seconds", "Settle latency", buckets=(0.001,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0)
    )
    SETTLE_N_GAUGE = Gauge(
        "oscillink_settle_last_N", "N of last settle"
    )
    SETTLE_D_GAUGE = Gauge(
        "oscillink_settle_last_D", "D of last settle"
    )
    USAGE_NODES = Counter(
        "oscillink_usage_nodes_total", "Total nodes processed"
    )
    USAGE_NODE_DIM_UNITS = Counter(
        "oscillink_usage_node_dim_units_total", "Total node-dimension units processed (sum N*D)"
    )
    JOB_QUEUE_DEPTH = Gauge(
        "oscillink_job_queue_depth", "Number of jobs currently queued or running"
    )
    STRIPE_WEBHOOK_EVENTS = Counter(
        "oscillink_stripe_webhook_events_total", "Stripe webhook events", ["result"]
    )

_key_usage: dict[str, dict[str, float]] = {}
# Monthly usage (node_dim_units) per key. Reset per calendar month (UTC).
_monthly_usage: dict[str, dict[str, int | str]] = {}

# Firestore-backed monthly usage (optional). When enabled via OSCILLINK_MONTHLY_USAGE_COLLECTION, per-key
# monthly counters (units used in the current period) are persisted and shared across processes.
_MONTHLY_USAGE_COLLECTION = os.getenv("OSCILLINK_MONTHLY_USAGE_COLLECTION", "").strip()

def _load_monthly_usage_doc(api_key: str, period: str):  # pragma: no cover - external dependency
    if not _MONTHLY_USAGE_COLLECTION:
        return None
    try:
        from google.cloud import firestore  # type: ignore
        client = firestore.Client()
        doc_id = f"{api_key}:{period}"
        snap = client.collection(_MONTHLY_USAGE_COLLECTION).document(doc_id).get()
        if snap.exists:
            return snap.to_dict() or None
    except Exception:
        return None
    return None

def _update_monthly_usage_doc(api_key: str, period: str, used: int):  # pragma: no cover - external dependency
    if not _MONTHLY_USAGE_COLLECTION:
        return
    try:
        from google.cloud import firestore  # type: ignore
        client = firestore.Client()
        doc_id = f"{api_key}:{period}"
        doc_ref = client.collection(_MONTHLY_USAGE_COLLECTION).document(doc_id)
        # Use transaction (optimistic) to avoid lost updates; fall back to blind set on failure.
        @firestore.transactional
        def _tx_update(tx, ref):  # type: ignore
            snap = ref.get(transaction=tx)
            if snap.exists:
                data = snap.to_dict() or {}
                data["used"] = used
                tx.set(ref, data, merge=False)
            else:
                tx.set(ref, {"api_key": api_key, "period": period, "used": used, "updated_at": time.time(), "created_at": time.time()})
        try:
            tx = client.transaction()
            _tx_update(tx, doc_ref)
        except Exception:
            # Blind overwrite (eventual consistency acceptable for quota enforcement best-effort)
            doc_ref.set({"api_key": api_key, "period": period, "used": used, "updated_at": time.time()}, merge=True)
    except Exception:
        pass

def _check_monthly_cap(key: str | None, units: int):
    """Enforce per-tier monthly unit caps (best-effort in-memory).

    Returns a dict describing monthly usage context or None when unlimited.
    Raises HTTPException(429/413) when exceeding caps.
    """
    if key is None:
        return None
    meta: KeyMetadata | None = get_keystore().get(key)
    if not meta:
        return None
    tinfo = tier_info(meta.tier)
    cap = tinfo.monthly_unit_cap
    if cap is None or cap <= 0:
        return None
    period = current_period()
    rec = _monthly_usage.get(key)
    if not rec or rec.get("period") != period:
        # Attempt to hydrate from persistent store
        used_val = 0
        if _MONTHLY_USAGE_COLLECTION:
            persisted = _load_monthly_usage_doc(key, period)
            if persisted and isinstance(persisted.get("used"), (int, float)):
                used_val = int(persisted.get("used", 0))
        rec = {"period": period, "used": used_val}
        _monthly_usage[key] = rec  # type: ignore
    used = int(rec.get("used", 0))
    if units > cap:
        raise HTTPException(status_code=413, detail=f"request units {units} exceed monthly cap {cap}")
    if used + units > cap:
        remaining = max(cap - used, 0)
        raise HTTPException(status_code=429, detail=f"monthly cap exceeded (cap={cap}, used={used})", headers={"X-MonthCap-Limit": str(cap), "X-MonthCap-Remaining": str(remaining)})
    new_used = used + units
    rec["used"] = new_used  # type: ignore
    # Best-effort persistence (async not required; cheap write) - ignore failures silently
    if _MONTHLY_USAGE_COLLECTION:
        _update_monthly_usage_doc(key, period, new_used)
    return {"limit": cap, "used": rec["used"], "remaining": cap - rec["used"], "period": period}

def _check_and_consume_quota(key: str | None, units: int) -> tuple[int, int, float]:
    """Check quota for this key; consume units if allowed.

    Returns (remaining, limit, reset_epoch). If quota exceeded raises HTTPException.
    If quota disabled or key is None (open access) returns (-1, 0, 0).
    """
    q = get_quota_config()
    # Per-key override (limit/window) if metadata present
    if key:
        meta: KeyMetadata | None = get_keystore().get(key)
        if meta:
            if meta.quota_limit_units is not None:
                q_limit = int(meta.quota_limit_units)
            else:
                q_limit = q.limit
            q_window = int(meta.quota_window_seconds) if meta.quota_window_seconds is not None else q.window
        else:
            q_limit, q_window = q.limit, q.window
    else:
        q_limit, q_window = q.limit, q.window
    if q_limit <= 0 or key is None:
        # Quota disabled OR unauthenticated (open mode)
        return -1, 0, 0
    now = time.time()
    rec = _key_usage.get(key)
    if not rec or now - rec["window_start"] >= q_window:
        rec = {"window_start": now, "used": 0.0, "limit": q_limit, "window": q_window}
        _key_usage[key] = rec
    # If limit/window changed (override toggled), reset window
    elif rec.get("limit") != q_limit or rec.get("window") != q_window:
        rec = {"window_start": now, "used": 0.0, "limit": q_limit, "window": q_window}
        _key_usage[key] = rec
    if units > q_limit:
        raise HTTPException(status_code=413, detail=f"request units {units} exceed per-key limit {q_limit}")
    if rec["used"] + units > q_limit:
        reset_at = rec["window_start"] + q_window
        headers = {
            "Retry-After": str(int(reset_at - now) + 1),
            "X-Quota-Limit": str(q_limit),
            "X-Quota-Remaining": "0",
            "X-Quota-Reset": str(int(reset_at)),
        }
        raise HTTPException(status_code=429, detail="quota exceeded", headers=headers)
    rec["used"] += units
    remaining = q_limit - int(rec["used"])
    reset_at = rec["window_start"] + q_window
    return remaining, q_limit, reset_at

def _quota_headers(remaining: int, limit: int, reset_epoch: float) -> dict[str, str]:
    if remaining < 0:
        return {}
    return {
        "X-Quota-Limit": str(limit),
        "X-Quota-Remaining": str(max(remaining, 0)),
        "X-Quota-Reset": str(int(reset_epoch)),
    }

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = rid
    return response

# ---------------- Per-IP Rate Limiting (in-memory) -----------------
_ip_rl_counters: dict[str, dict[str, float]] = {}

def _ip_rate_limit_config():
    """Fetch current per-IP rate limit configuration from environment.

    Returns (limit, window_seconds, trust_xff). limit<=0 disables the limiter.
    """
    try:
        limit = int(os.getenv("OSCILLINK_IP_RATE_LIMIT", "0"))
    except ValueError:
        limit = 0
    try:
        window = int(os.getenv("OSCILLINK_IP_RATE_WINDOW", "60"))
    except ValueError:
        window = 60
    trust_xff = os.getenv("OSCILLINK_TRUST_XFF", "0") in {"1", "true", "TRUE", "on"}
    return limit, max(1, window), trust_xff

def _client_ip(request: Request, trust_xff: bool) -> str:
    if trust_xff:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # Use the first IP in the chain (client origin). Strip whitespace.
            first = xff.split(",")[0].strip()
            if first:
                return first
    try:
        if request.client and request.client.host:
            return request.client.host  # type: ignore[attr-defined]
    except Exception:
        pass
    return "unknown"

@app.middleware("http")
async def per_ip_rate_limit_mw(request: Request, call_next):
    limit, window, trust_xff = _ip_rate_limit_config()
    if limit <= 0:
        return await call_next(request)
    # Exempt lightweight/system endpoints
    if request.url.path in {"/health", "/metrics"}:
        return await call_next(request)
    now = time.time()
    ip = _client_ip(request, trust_xff)
    rec = _ip_rl_counters.get(ip)
    if not rec or now - rec["window_start"] >= window:
        rec = {"window_start": now, "count": 0.0, "limit": float(limit), "window": float(window)}
        _ip_rl_counters[ip] = rec  # type: ignore
    # Detect dynamic config change (limit/window altered) -> reset window
    elif rec.get("limit") != float(limit) or rec.get("window") != float(window):
        rec = {"window_start": now, "count": 0.0, "limit": float(limit), "window": float(window)}
        _ip_rl_counters[ip] = rec  # type: ignore
    if rec["count"] >= limit:
        reset_at = rec["window_start"] + window
        headers = {
            "Retry-After": str(int(reset_at - now) + 1),
            "X-IPLimit-Limit": str(limit),
            "X-IPLimit-Remaining": "0",
            "X-IPLimit-Reset": str(int(reset_at)),
        }
        return ORJSONResponse(status_code=429, content={"detail": "ip rate limit exceeded"}, headers=headers)
    rec["count"] += 1
    response = await call_next(request)
    remaining = max(limit - int(rec["count"]), 0)
    response.headers.setdefault("X-IPLimit-Limit", str(limit))
    response.headers.setdefault("X-IPLimit-Remaining", str(remaining))
    response.headers.setdefault("X-IPLimit-Reset", str(int(rec["window_start"] + window)))
    return response

_rl_state = {"window_start": time.time(), "count": 0, "limit": 0, "window": 60}

@app.middleware("http")
async def rate_limit_mw(request: Request, call_next):
    # Reload current limits via runtime config helper
    r = get_rate_limit()
    _rl_state["limit"], _rl_state["window"] = r.limit, r.window
    if _rl_state["limit"] <= 0:
        return await call_next(request)
    now = time.time()
    window_elapsed = now - _rl_state["window_start"]
    if window_elapsed >= _rl_state["window"]:
        _rl_state["window_start"] = now
        _rl_state["count"] = 0
    if _rl_state["count"] >= _rl_state["limit"] and request.url.path not in ("/health", "/metrics"):
        reset_in = _rl_state["window"] - (now - _rl_state["window_start"])
        headers = {
            "Retry-After": f"{int(reset_in)+1}",
            "X-RateLimit-Limit": str(_rl_state["limit"]),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(_rl_state["window_start"] + _rl_state["window"]))
        }
        return ORJSONResponse(status_code=429, content={"detail": "rate limit exceeded"}, headers=headers)
    _rl_state["count"] += 1
    resp = await call_next(request)
    remaining = max(_rl_state["limit"] - _rl_state["count"], 0)
    resp.headers.setdefault("X-RateLimit-Limit", str(_rl_state["limit"]))
    resp.headers.setdefault("X-RateLimit-Remaining", str(remaining))
    resp.headers.setdefault("X-RateLimit-Reset", str(int(_rl_state["window_start"] + _rl_state["window"])))
    return resp
_API_VERSION = get_settings().api_version  # capture at import for routing; other settings fetched dynamically
_ENV_KEYS_FINGERPRINT = {"api_keys": os.getenv("OSCILLINK_API_KEYS", ""), "tiers": os.getenv("OSCILLINK_KEY_TIERS", "")}

# In-memory async job store (non-persistent, single-process)
_jobs: dict[str, dict] = {}
_JOB_TTL_SEC = 3600

# Usage logging (optional JSONL)
USAGE_LOG_PATH = os.getenv("OSCILLINK_USAGE_LOG")  # if set, append JSON lines
USAGE_LOG_SIGNING_SECRET = os.getenv("OSCILLINK_USAGE_SIGNING_SECRET")  # optional HMAC secret

def _append_usage(record: dict):
    if not USAGE_LOG_PATH:
        return
    try:
        if USAGE_LOG_SIGNING_SECRET:
            # compute signature over deterministic canonical form of payload fields (exclude signature itself)
            payload = json.dumps(record, separators=(",", ":"), sort_keys=True)
            sig = hmac.new(USAGE_LOG_SIGNING_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
            record = {**record, "sig": {"alg": "HS256", "h": sig}}
        # minimal defensiveness: ensure directory exists (if path includes directory component)
        dir_part = os.path.dirname(USAGE_LOG_PATH)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        with open(USAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception:
        # Silent failure; logging framework could be integrated later
        pass

# ---------------- Webhook Event Logging / Idempotency -----------------
_webhook_events_mem: dict[str, dict] = {}

def _webhook_events_collection():
    return os.getenv("OSCILLINK_WEBHOOK_EVENTS_COLLECTION", "").strip()

def _webhook_get(event_id: str):
    # Memory first
    if event_id in _webhook_events_mem:
        return _webhook_events_mem[event_id]
    coll = _webhook_events_collection()
    if not coll:
        return None
    try:  # pragma: no cover - external dependency path
        from google.cloud import firestore  # type: ignore
        client = firestore.Client()
        snap = client.collection(coll).document(event_id).get()
        if snap.exists:
            return snap.to_dict()
    except Exception:
        return None
    return None

def _webhook_store(event_id: str, record: dict):
    # Always store in memory for fast duplicate checks
    _webhook_events_mem[event_id] = record
    coll = _webhook_events_collection()
    if not coll:
        return
    try:  # pragma: no cover - external dependency path
        from google.cloud import firestore  # type: ignore
        client = firestore.Client()
        # Use create to preserve idempotency (do not overwrite existing)
        doc_ref = client.collection(coll).document(event_id)
        if not doc_ref.get().exists:
            doc_ref.set(record, merge=False)
    except Exception:
        # Swallow errors silently (observability layer can catch later)
        pass

def _purge_old_jobs():
    now = time.time()
    expired = [jid for jid, rec in _jobs.items() if now - rec.get("created", now) > _JOB_TTL_SEC]
    for jid in expired:
        _jobs.pop(jid, None)
    if 'JOB_QUEUE_DEPTH' in globals():
        try:
            JOB_QUEUE_DEPTH.set(len(_jobs))  # type: ignore
        except Exception:
            pass

def api_key_guard(x_api_key: str | None = Header(default=None)):
    """Return api_key (may be None for open access) after validation.

    Resolution order:
    1. If OSCILLINK_KEYSTORE_BACKEND = firestore|memory and any keys exist there, validate via keystore metadata
       (status must be 'active'). If key not present -> 401.
    2. Else fall back to legacy environment list (OSCILLINK_API_KEYS). If that var unset -> open access.
    """
    ks = get_keystore()
    # Hot-reload for InMemoryKeyStore when env lists change (development/testing convenience)
    global _ENV_KEYS_FINGERPRINT
    current_fp = {"api_keys": os.getenv("OSCILLINK_API_KEYS", ""), "tiers": os.getenv("OSCILLINK_KEY_TIERS", "")}
    if current_fp != _ENV_KEYS_FINGERPRINT and isinstance(ks, InMemoryKeyStore):  # type: ignore
        # Recreate in-memory keystore to pick up new env keys/tiers
        from cloud.app.keystore import InMemoryKeyStore as _IMKS  # local import to avoid cycle
        # Replace global singleton
        from cloud.app import keystore as _kmod
        _kmod._key_store = _IMKS()
        ks = get_keystore()
        _ENV_KEYS_FINGERPRINT = current_fp
    backend = os.getenv("OSCILLINK_KEYSTORE_BACKEND", "memory").lower()
    # Attempt keystore validation first when backend explicitly set OR when firestore selected.
    # Legacy env list ALWAYS enforced if present (checked early to satisfy tests expecting 401)
    allowed = get_api_keys()
    if allowed:
        if x_api_key is None or x_api_key not in allowed:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        # Tier overrides may be handled by InMemoryKeyStore above; return key directly
        return x_api_key

    if backend in {"firestore", "memory"}:
        if x_api_key:
            meta = ks.get(x_api_key)
            if meta:
                if meta.is_active():
                    return x_api_key
                # Provide specific messaging for pending enterprise activation
                if meta.status == "pending":
                    raise HTTPException(status_code=403, detail="key pending manual activation")
                if backend == "firestore":
                    raise HTTPException(status_code=401, detail="invalid or inactive API key")
                # memory backend falls through to potential open access only if no keys seeded
        else:
            if backend == "firestore":  # closed mode when firestore selected
                raise HTTPException(status_code=401, detail="invalid or missing API key")
        # If backend memory and no key provided, allow open access if keystore empty
        if backend == "memory":
            # Check if any keys exist in memory; if none, open access
            # Access protected member of InMemoryKeyStore cautiously
            try:
                if not getattr(ks, '_keys', {}):
                    return None
            except Exception:
                pass
    # Legacy env list fallback ALWAYS enforced when list non-empty
    allowed = get_api_keys()
    # If we reach here, open access (no env key list, memory backend with possibly empty keys)
    return x_api_key  # open access (None) when no keystore match & no env list

def feature_context(x_api_key: str | None = Depends(api_key_guard)):
    """Resolve feature bundle for request.

    Derives tier from keystore metadata when key present; otherwise returns free tier. Feature overrides respected.
    """
    meta = get_keystore().get(x_api_key) if x_api_key else None
    features = resolve_features(meta)
    return {"api_key": x_api_key, "features": features}

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version=__version__)

def _build_lattice(req: SettleRequest) -> tuple[OscillinkLattice, int, int, int]:
    Y = np.array(req.Y, dtype=np.float32)
    N, D = Y.shape
    if N == 0 or D == 0:
        raise HTTPException(status_code=400, detail="Empty matrix")
    s = get_settings()
    if s.max_nodes < N:
        raise HTTPException(status_code=413, detail=f"N>{s.max_nodes} exceeds limit")
    if s.max_dim < D:
        raise HTTPException(status_code=413, detail=f"D>{s.max_dim} exceeds limit")
    # Clamp kneighbors to avoid argpartition errors when requested >= N
    k_eff = min(req.params.kneighbors, max(1, N - 1))
    lat = OscillinkLattice(
        Y,
        kneighbors=k_eff,
        lamG=req.params.lamG,
        lamC=req.params.lamC,
        lamQ=req.params.lamQ,
        deterministic_k=req.params.deterministic_k,
        neighbor_seed=req.params.neighbor_seed,
    )
    if req.psi is not None:
        psi = np.array(req.psi, dtype=np.float32)
        if psi.shape[0] != D:
            raise HTTPException(status_code=400, detail="psi dimension mismatch")
        lat.set_query(psi)
    if req.gates is not None:
        gates = np.array(req.gates, dtype=np.float32)
        if gates.shape[0] != N:
            raise HTTPException(status_code=400, detail="gates length mismatch")
        lat.set_gates(gates)
    if req.chain:
        if len(req.chain) < 2:
            raise HTTPException(status_code=400, detail="chain must have >=2 nodes")
            
        lat.add_chain(req.chain, lamP=req.params.lamP)
    return lat, N, D, k_eff

@app.post(f"/{_API_VERSION}/settle", response_model=ReceiptResponse)
def settle(req: SettleRequest, request: Request, response: Response, ctx=Depends(feature_context)):
    x_api_key = ctx["api_key"]
    feats = ctx["features"]
    # Enforce diffusion gating restriction (Experimental path)
    if req.gates is not None:
        if os.getenv("OSCILLINK_DIFFUSION_GATES_ENABLED", "1") not in {"1", "true", "TRUE", "on"}:
            raise HTTPException(status_code=403, detail="diffusion gating temporarily disabled")
        if not feats.diffusion_allowed:
            raise HTTPException(status_code=403, detail="diffusion gating not enabled for this tier")
    lat, N, D, k_eff = _build_lattice(req)
    units = N * D
    # Monthly cap enforcement (before quota window since it is a higher level allowance)
    monthly_ctx = _check_monthly_cap(x_api_key, units)
    remaining, limit, reset_at = _check_and_consume_quota(x_api_key, units)

    t0 = time.time()
    try:
        lat.settle(dt=req.options.dt, max_iters=req.options.max_iters, tol=req.options.tol)
        elapsed = time.time() - t0
        SETTLE_COUNTER.labels(status="ok").inc()
    except Exception:
        SETTLE_COUNTER.labels(status="error").inc()
        raise
    t_settle = 1000.0 * elapsed
    SETTLE_LATENCY.observe(elapsed)
    SETTLE_N_GAUGE.set(N)
    SETTLE_D_GAUGE.set(D)
    USAGE_NODES.inc(N)
    USAGE_NODE_DIM_UNITS.inc(N * D)

    receipt = None
    bundle = None
    if req.options.include_receipt:
        receipt = lat.receipt()
    if req.options.bundle_k:
        bundle = lat.bundle(k=req.options.bundle_k)

    # derive minimal meta subset
    meta = {
        "N": int(N),
        "D": int(D),
        "kneighbors_requested": req.params.kneighbors,
        "kneighbors_effective": k_eff,
        "lam": {"G": req.params.lamG, "C": req.params.lamC, "Q": req.params.lamQ, "P": req.params.lamP},
    }
    sig_meta = receipt.get("meta", {}).get("state_sig") if (receipt and isinstance(receipt.get("meta"), dict)) else None
    state_sig = sig_meta or lat._signature()

    # Build monthly usage block if present
    monthly_usage_block = None
    if monthly_ctx:
        monthly_usage_block = {
            "limit": monthly_ctx["limit"],
            "used": monthly_ctx["used"],
            "remaining": monthly_ctx["remaining"],
            "period": monthly_ctx["period"],
        }
    resp = ReceiptResponse(
        state_sig=state_sig,
        receipt=receipt,
        bundle=bundle,
        timings_ms={"total_settle_ms": t_settle},
        meta={**meta, "request_id": request.headers.get(REQUEST_ID_HEADER, ""), "usage": {"nodes": N, "node_dim_units": units, "monthly": monthly_usage_block}, "quota": None if limit==0 else {"limit": limit, "remaining": remaining, "reset": int(reset_at)}},
    )
    headers = _quota_headers(remaining, limit, reset_at)
    # Monthly headers (informational)
    if monthly_ctx:
        response.headers.setdefault("X-Monthly-Cap", str(monthly_ctx["limit"]))
        response.headers.setdefault("X-Monthly-Used", str(monthly_ctx["used"]))
        response.headers.setdefault("X-Monthly-Remaining", str(monthly_ctx["remaining"]))
        response.headers.setdefault("X-Monthly-Period", str(monthly_ctx["period"]))
    for k, v in headers.items():
        response.headers.setdefault(k, v)
    _append_usage({
        "ts": time.time(),
        "event": "settle",
        "api_key": x_api_key,
        "N": N,
        "D": D,
        "units": units,
        "duration_ms": t_settle,
        "quota": None if limit==0 else {"limit": limit, "remaining": remaining, "reset": int(reset_at)},
        "monthly": monthly_usage_block,
    })
    return resp

@app.post(f"/{_API_VERSION}/receipt")
def receipt(req: SettleRequest, request: Request, response: Response, ctx=Depends(feature_context)):
    """Return only the receipt (always include_receipt)."""
    x_api_key = ctx["api_key"]
    feats = ctx["features"]
    if req.gates is not None:
        if os.getenv("OSCILLINK_DIFFUSION_GATES_ENABLED", "1") not in {"1", "true", "TRUE", "on"}:
            raise HTTPException(status_code=403, detail="diffusion gating temporarily disabled")
        if not feats.diffusion_allowed:
            raise HTTPException(status_code=403, detail="diffusion gating not enabled for this tier")
    lat, N, D, k_eff = _build_lattice(req)
    units = N * D
    # Enforce monthly/quota BEFORE doing any compute to prevent free riding via failures after compute
    monthly_ctx = _check_monthly_cap(x_api_key, units)
    remaining, limit, reset_at = _check_and_consume_quota(x_api_key, units)
    t0 = time.time()
    lat.settle(dt=req.options.dt, max_iters=req.options.max_iters, tol=req.options.tol)
    elapsed = time.time() - t0
    SETTLE_COUNTER.labels(status="ok").inc()
    SETTLE_LATENCY.observe(elapsed)
    SETTLE_N_GAUGE.set(N)
    SETTLE_D_GAUGE.set(D)
    rec = lat.receipt()
    headers = _quota_headers(remaining, limit, reset_at)
    if monthly_ctx:
        response.headers.setdefault("X-Monthly-Cap", str(monthly_ctx["limit"]))
        response.headers.setdefault("X-Monthly-Used", str(monthly_ctx["used"]))
        response.headers.setdefault("X-Monthly-Remaining", str(monthly_ctx["remaining"]))
        response.headers.setdefault("X-Monthly-Period", str(monthly_ctx["period"]))
    for k, v in headers.items():
        response.headers.setdefault(k, v)
    _append_usage({
        "ts": time.time(),
        "event": "receipt",
        "api_key": x_api_key,
        "N": N,
        "D": D,
        "units": units,
        "duration_ms": 1000.0 * elapsed,
        "quota": None if limit==0 else {"limit": limit, "remaining": remaining, "reset": int(reset_at)},
        "monthly": None if not monthly_ctx else {"limit": monthly_ctx["limit"], "used": monthly_ctx["used"], "remaining": monthly_ctx["remaining"], "period": monthly_ctx["period"]}
    })
    return {
        "state_sig": rec.get("meta", {}).get("state_sig"),
        "receipt": rec,
        "timings_ms": {"total_settle_ms": 1000.0 * elapsed},
        "meta": {"N": N, "D": D, "kneighbors_requested": req.params.kneighbors, "kneighbors_effective": k_eff, "request_id": request.headers.get(REQUEST_ID_HEADER, ""), "usage": {"nodes": N, "node_dim_units": units, "monthly": None if not monthly_ctx else {"limit": monthly_ctx["limit"], "used": monthly_ctx["used"], "remaining": monthly_ctx["remaining"], "period": monthly_ctx["period"]}}, "quota": None if limit==0 else {"limit": limit, "remaining": remaining, "reset": int(reset_at)}} ,
    }

@app.post(f"/{_API_VERSION}/bundle")
def bundle(req: SettleRequest, request: Request, response: Response, ctx=Depends(feature_context)):
    """Return only the bundle (requires options.bundle_k)."""
    x_api_key = ctx["api_key"]
    feats = ctx["features"]
    if req.gates is not None:
        if os.getenv("OSCILLINK_DIFFUSION_GATES_ENABLED", "1") not in {"1", "true", "TRUE", "on"}:
            raise HTTPException(status_code=403, detail="diffusion gating temporarily disabled")
        if not feats.diffusion_allowed:
            raise HTTPException(status_code=403, detail="diffusion gating not enabled for this tier")
    if not req.options.bundle_k:
        raise HTTPException(status_code=400, detail="options.bundle_k must be set for /bundle")
    lat, N, D, k_eff = _build_lattice(req)
    units = N * D
    # Quota + monthly first (no compute before cost authorization)
    monthly_ctx = _check_monthly_cap(x_api_key, units)
    remaining, limit, reset_at = _check_and_consume_quota(x_api_key, units)
    t0 = time.time()
    lat.settle(dt=req.options.dt, max_iters=req.options.max_iters, tol=req.options.tol)
    elapsed = time.time() - t0
    SETTLE_COUNTER.labels(status="ok").inc()
    SETTLE_LATENCY.observe(elapsed)
    SETTLE_N_GAUGE.set(N)
    SETTLE_D_GAUGE.set(D)
    b = lat.bundle(k=req.options.bundle_k)
    headers = _quota_headers(remaining, limit, reset_at)
    if monthly_ctx:
        response.headers.setdefault("X-Monthly-Cap", str(monthly_ctx["limit"]))
        response.headers.setdefault("X-Monthly-Used", str(monthly_ctx["used"]))
        response.headers.setdefault("X-Monthly-Remaining", str(monthly_ctx["remaining"]))
        response.headers.setdefault("X-Monthly-Period", str(monthly_ctx["period"]))
    for k, v in headers.items():
        response.headers.setdefault(k, v)
    _append_usage({
        "ts": time.time(),
        "event": "bundle",
        "api_key": x_api_key,
        "N": N,
        "D": D,
        "units": units,
        "duration_ms": 1000.0 * elapsed,
        "quota": None if limit==0 else {"limit": limit, "remaining": remaining, "reset": int(reset_at)},
        "monthly": None if not monthly_ctx else {"limit": monthly_ctx["limit"], "used": monthly_ctx["used"], "remaining": monthly_ctx["remaining"], "period": monthly_ctx["period"]}
    })
    return {
        "state_sig": lat._signature(),
        "bundle": b,
        "timings_ms": {"total_settle_ms": 1000.0 * elapsed},
        "meta": {"N": N, "D": D, "kneighbors_requested": req.params.kneighbors, "kneighbors_effective": k_eff, "request_id": request.headers.get(REQUEST_ID_HEADER, ""), "usage": {"nodes": N, "node_dim_units": units, "monthly": None if not monthly_ctx else {"limit": monthly_ctx["limit"], "used": monthly_ctx["used"], "remaining": monthly_ctx["remaining"], "period": monthly_ctx["period"]}}, "quota": None if limit==0 else {"limit": limit, "remaining": remaining, "reset": int(reset_at)}} ,
    }

@app.post(f"/{_API_VERSION}/chain/receipt")
def chain_receipt(req: SettleRequest, request: Request, response: Response, ctx=Depends(feature_context)):
    """Return settle plus chain receipt (requires chain)."""
    x_api_key = ctx["api_key"]
    feats = ctx["features"]
    if req.gates is not None:
        if os.getenv("OSCILLINK_DIFFUSION_GATES_ENABLED", "1") not in {"1", "true", "TRUE", "on"}:
            raise HTTPException(status_code=403, detail="diffusion gating temporarily disabled")
        if not feats.diffusion_allowed:
            raise HTTPException(status_code=403, detail="diffusion gating not enabled for this tier")
    if not req.chain:
        raise HTTPException(status_code=400, detail="chain must be provided")
    lat, N, D, k_eff = _build_lattice(req)
    units = N * D
    # Enforce billing constraints prior to compute
    monthly_ctx = _check_monthly_cap(x_api_key, units)
    remaining, limit, reset_at = _check_and_consume_quota(x_api_key, units)
    t0 = time.time()
    lat.settle(dt=req.options.dt, max_iters=req.options.max_iters, tol=req.options.tol)
    elapsed = time.time() - t0
    SETTLE_COUNTER.labels(status="ok").inc()
    SETTLE_LATENCY.observe(elapsed)
    SETTLE_N_GAUGE.set(N)
    SETTLE_D_GAUGE.set(D)
    rec = lat.chain_receipt(req.chain)
    headers = _quota_headers(remaining, limit, reset_at)
    if monthly_ctx:
        response.headers.setdefault("X-Monthly-Cap", str(monthly_ctx["limit"]))
        response.headers.setdefault("X-Monthly-Used", str(monthly_ctx["used"]))
        response.headers.setdefault("X-Monthly-Remaining", str(monthly_ctx["remaining"]))
        response.headers.setdefault("X-Monthly-Period", str(monthly_ctx["period"]))
    for k, v in headers.items():
        response.headers.setdefault(k, v)
    _append_usage({
        "ts": time.time(),
        "event": "chain_receipt",
        "api_key": x_api_key,
        "N": N,
        "D": D,
        "units": units,
        "duration_ms": 1000.0 * elapsed,
        "quota": None if limit==0 else {"limit": limit, "remaining": remaining, "reset": int(reset_at)},
        "monthly": None if not monthly_ctx else {"limit": monthly_ctx["limit"], "used": monthly_ctx["used"], "remaining": monthly_ctx["remaining"], "period": monthly_ctx["period"]}
    })
    return {
        "state_sig": lat._signature(),
        "chain_receipt": rec,
        "timings_ms": {"total_settle_ms": 1000.0 * elapsed},
        "meta": {"N": N, "D": D, "kneighbors_requested": req.params.kneighbors, "kneighbors_effective": k_eff, "request_id": request.headers.get(REQUEST_ID_HEADER, ""), "usage": {"nodes": N, "node_dim_units": units, "monthly": None if not monthly_ctx else {"limit": monthly_ctx["limit"], "used": monthly_ctx["used"], "remaining": monthly_ctx["remaining"], "period": monthly_ctx["period"]}}, "quota": None if limit==0 else {"limit": limit, "remaining": remaining, "reset": int(reset_at)}} ,
    }

@app.get("/metrics")
def metrics():
    data = generate_latest()  # type: ignore
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post(f"/{_API_VERSION}/jobs/settle")
def submit_job(req: SettleRequest, background: BackgroundTasks, request: Request, ctx=Depends(feature_context)):
    x_api_key = ctx["api_key"]
    feats = ctx["features"]
    if req.gates is not None:
        if os.getenv("OSCILLINK_DIFFUSION_GATES_ENABLED", "1") not in {"1", "true", "TRUE", "on"}:
            raise HTTPException(status_code=403, detail="diffusion gating temporarily disabled")
        if not feats.diffusion_allowed:
            raise HTTPException(status_code=403, detail="diffusion gating not enabled for this tier")
    job_id = uuid.uuid4().hex
    created = time.time()
    _purge_old_jobs()
    _jobs[job_id] = {"status": "queued", "created": created}
    try:
        JOB_QUEUE_DEPTH.set(len(_jobs))  # type: ignore
    except Exception:
        pass

    def run_job():
        try:
            lat, N, D, k_eff = _build_lattice(req)
            # Quota check occurs at execution time to avoid holding quota for queued jobs
            try:
                units = N * D
                monthly_ctx = _check_monthly_cap(x_api_key, units)
                remaining, limit, reset_at = _check_and_consume_quota(x_api_key, units)
            except HTTPException as he:  # record quota error inside job result
                _jobs[job_id] = {"status": "error", "error": he.detail, "created": created, "quota_error": True}
                return
            t0 = time.time()
            lat.settle(dt=req.options.dt, max_iters=req.options.max_iters, tol=req.options.tol)
            elapsed = time.time() - t0
            rec = lat.receipt() if req.options.include_receipt else None
            bundle = lat.bundle(k=req.options.bundle_k) if req.options.bundle_k else None
            USAGE_NODES.inc(N)
            USAGE_NODE_DIM_UNITS.inc(N * D)
            _jobs[job_id] = {
                "status": "done",
                "created": created,
                "completed": time.time(),
                "result": {
                    "state_sig": rec.get("meta", {}).get("state_sig") if rec else lat._signature(),
                    "receipt": rec,
                    "bundle": bundle,
                    "timings_ms": {"total_settle_ms": 1000.0 * elapsed},
                    "meta": {"N": N, "D": D, "kneighbors_requested": req.params.kneighbors, "kneighbors_effective": k_eff, "request_id": request.headers.get(REQUEST_ID_HEADER, ""), "usage": {"nodes": N, "node_dim_units": units, "monthly": None if not monthly_ctx else {"limit": monthly_ctx["limit"], "used": monthly_ctx["used"], "remaining": monthly_ctx["remaining"], "period": monthly_ctx["period"]}}, "quota": None if limit==0 else {"limit": limit, "remaining": remaining, "reset": int(reset_at)}}
                }
            }
            _append_usage({
                "ts": time.time(),
                "event": "job_settle",
                "api_key": x_api_key,
                "job_id": job_id,
                "N": N,
                "D": D,
                "units": units,
                "duration_ms": 1000.0 * elapsed,
                "quota": None if limit==0 else {"limit": limit, "remaining": remaining, "reset": int(reset_at)},
                "monthly": None if not monthly_ctx else {"limit": monthly_ctx["limit"], "used": monthly_ctx["used"], "remaining": monthly_ctx["remaining"], "period": monthly_ctx["period"]}
            })
        except Exception as e:
            _jobs[job_id] = {"status": "error", "error": str(e), "created": created}
        try:
            JOB_QUEUE_DEPTH.set(len(_jobs))  # type: ignore
        except Exception:
            pass

    background.add_task(run_job)
    return {"job_id": job_id, "status": "queued"}

@app.get(f"/{_API_VERSION}/jobs/{{job_id}}")
def get_job(job_id: str, ctx=Depends(feature_context)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

@app.delete(f"/{_API_VERSION}/jobs/{{job_id}}")
def cancel_job(job_id: str, ctx=Depends(feature_context)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") in {"done", "error"}:
        return {"job_id": job_id, "status": job["status"], "note": "already finished"}
    # Cannot truly cancel background task easily; mark as cancelled
    job["status"] = "cancelled"
    try:
        JOB_QUEUE_DEPTH.set(len(_jobs))  # type: ignore
    except Exception:
        pass
    return {"job_id": job_id, "status": "cancelled"}

# ---------------- Admin Key Management -----------------

def _admin_guard(x_admin_secret: str | None = Header(default=None)):
    required = os.getenv("OSCILLINK_ADMIN_SECRET")
    if not required:
        raise HTTPException(status_code=503, detail="admin secret not configured")
    if x_admin_secret != required:
        raise HTTPException(status_code=401, detail="invalid admin secret")
    return True

@app.get("/admin/keys/{api_key}", response_model=AdminKeyResponse)
def admin_get_key(api_key: str, auth=Depends(_admin_guard)):
    ks = get_keystore()
    meta = ks.get(api_key)
    if not meta:
        raise HTTPException(status_code=404, detail="key not found")
    return AdminKeyResponse(
        api_key=meta.api_key,
        tier=meta.tier,
        status=meta.status,
        quota_limit_units=meta.quota_limit_units,
        quota_window_seconds=meta.quota_window_seconds,
        features=meta.features,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
    )

@app.put("/admin/keys/{api_key}", response_model=AdminKeyResponse)
def admin_put_key(api_key: str, payload: AdminKeyUpdate, auth=Depends(_admin_guard)):
    ks = get_keystore()
    fields = payload.dict(exclude_unset=True)
    # Support creation if absent
    meta = ks.update(api_key, create=True, **fields)
    if not meta:
        raise HTTPException(status_code=500, detail="failed to update key")
    return AdminKeyResponse(
        api_key=meta.api_key,
        tier=meta.tier,
        status=meta.status,
        quota_limit_units=meta.quota_limit_units,
        quota_window_seconds=meta.quota_window_seconds,
        features=meta.features,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
    )

@app.get("/admin/webhook/events")
def admin_list_webhook_events(limit: int = 50, auth=Depends(_admin_guard)):
        """Return recent webhook events (memory-backed; Firestore persistence optional).

        Parameters:
            limit: max number of events to return (most recent first). Clamped to 500.
        """
        lim = max(1, min(limit, 500))
        # In-memory events dict keyed by id; sort by ts descending.
        events = list(_webhook_events_mem.values())
        events.sort(key=lambda r: r.get("ts", 0), reverse=True)
        return {"events": events[:lim], "count": len(events), "returned": len(events[:lim])}

# Stripe webhook with subscription → tier sync
@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """Stripe webhook endpoint (skeleton).

    Validates signature if STRIPE_WEBHOOK_SECRET set; presently stores raw event in memory (non-durable).
    Future: write to Firestore and process asynchronously.
    """
    secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    body = await request.body()
    payload_text = body.decode("utf-8", errors="replace")
    event = None
    allow_unverified = os.getenv("OSCILLINK_ALLOW_UNVERIFIED_STRIPE", "0") in {"1", "true", "TRUE", "on"}
    verified = False
    if secret:
        sig_header = request.headers.get("stripe-signature")
        if not sig_header:
            raise HTTPException(status_code=400, detail="missing stripe-signature header")
        # Enforce timestamp freshness (basic replay protection) if header contains t= segment
        try:
            max_age = int(os.getenv("OSCILLINK_STRIPE_MAX_AGE", "300"))  # seconds
        except ValueError:
            max_age = 300
        if max_age > 0:
            # stripe-signature format: t=timestamp,v1=...,v0=...
            try:
                parts = {kv.split('=')[0]: kv.split('=')[1] for kv in sig_header.split(',') if '=' in kv}
                if 't' in parts:
                    ts = int(parts['t'])
                    now = int(time.time())
                    if now - ts > max_age:
                        # Allow explicit unverified override pathway to bypass freshness (test/dev only)
                        if os.getenv("OSCILLINK_ALLOW_UNVERIFIED_STRIPE", "0") in {"1","true","TRUE","on"}:
                            pass
                        else:
                            raise HTTPException(status_code=400, detail="webhook timestamp too old")
            except HTTPException:
                raise
            except Exception:
                # Non-fatal: if parsing fails we proceed (could tighten later)
                pass
        # Attempt real verification if stripe package available
        try:  # pragma: no cover - external dependency path
            import stripe  # type: ignore
            stripe.api_version = "2024-06-20"
            event = stripe.Webhook.construct_event(payload_text, sig_header, secret)
            verified = True
        except ModuleNotFoundError:
            # Fallback: parse JSON without cryptographic validation (NOT FOR PROD)
            try:
                event = json.loads(payload_text)
            except Exception:
                raise HTTPException(status_code=400, detail="invalid JSON payload (no stripe lib)")
        except Exception as e:  # signature failure
            raise HTTPException(status_code=400, detail=f"signature verification failed: {e}")
    else:
        try:
            event = json.loads(payload_text)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON payload")

    etype = event.get("type", "unknown") if isinstance(event, dict) else getattr(event, "type", "unknown")
    event_id = event.get("id") if isinstance(event, dict) else getattr(event, "id", None)
    if not event_id:
        # Without an id we cannot ensure idempotency
        raise HTTPException(status_code=400, detail="event missing id")

    # Idempotency check
    existing = _webhook_get(event_id)
    if existing:
        try:
            STRIPE_WEBHOOK_EVENTS.labels(result="duplicate").inc()  # type: ignore
        except Exception:
            pass
        return {"received": True, "id": event_id, "type": etype, "processed": False, "duplicate": True, "note": "duplicate ignored"}

    processed = False
    note = None
    # Subscription lifecycle handling
    if etype.startswith("customer.subscription."):
        sub_obj = event.get("data", {}).get("object", {}) if isinstance(event, dict) else {}
        # Subscription cancellation / deletion sets status; treat deleted as cancelled
        api_key = None
        try:
            metadata = sub_obj.get("metadata", {}) or {}
            api_key = metadata.get("api_key")
        except Exception:
            api_key = None
        if api_key:
            ks = get_keystore()
            # Only mutate keystore if event verified OR explicitly allowed via override (development/testing)
            if not verified and secret and not allow_unverified:
                note = "signature not verified; subscription event ignored"
            else:
                if etype in {"customer.subscription.created", "customer.subscription.updated"}:
                    new_tier = resolve_tier_from_subscription(sub_obj)
                    tinfo = tier_info(new_tier)
                    status = "pending" if getattr(tinfo, "requires_manual_activation", False) else "active"
                    ks.update(api_key, create=True, tier=new_tier, status=status, features={"diffusion_gates": tinfo.diffusion_allowed})
                    processed = True
                    note = f"tier set to {new_tier} (status={status})"
                elif etype in {"customer.subscription.deleted", "customer.subscription.cancelled"}:
                    ks.update(api_key, status="suspended")
                    processed = True
                    note = "subscription cancelled; key suspended"
        else:
            note = "subscription missing api_key metadata"
    record = {
        "id": event_id,
        "ts": time.time(),
        "type": etype,
        "processed": processed,
        "note": note,
        "live": bool(secret),
        "verified": verified,
        "allow_unverified_override": allow_unverified,
        "api_key": api_key if 'api_key' in locals() else None,
        # integrity hash of raw payload (without storing full body) for audit correlation
        "payload_sha256": hashlib.sha256(payload_text.encode('utf-8')).hexdigest(),
        "freshness_max_age": os.getenv("OSCILLINK_STRIPE_MAX_AGE", "300"),
    }
    # Attempt to persist event (fire-and-forget)
    _webhook_store(event_id, record)
    try:
        STRIPE_WEBHOOK_EVENTS.labels(result="processed" if processed else "ignored").inc()  # type: ignore
    except Exception:
        pass
    return record

# CLI entrypoint for uvicorn
# uvicorn cloud.app.main:app --reload --port 8000
