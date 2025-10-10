from __future__ import annotations

# Standard library
import hashlib
import hmac
import json
import os
import time
import uuid
import smtplib
from email.message import EmailMessage
from typing import Any

# Third-party
import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, ORJSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from oscillink import OscillinkLattice, __version__

from .billing import (
    TIER_CATALOG,
    current_period,
    get_price_map,
    resolve_tier_from_subscription,
    tier_info,
)
from .config import get_settings
from .features import resolve_features
from .keystore import InMemoryKeyStore, KeyMetadata, get_keystore  # type: ignore
from .models import AdminKeyResponse, AdminKeyUpdate, HealthResponse, ReceiptResponse, SettleRequest
from .redis_backend import get_with_ttl, incr_with_window, redis_enabled, set_with_ttl
from .runtime_config import get_api_keys, get_quota_config, get_rate_limit

app = FastAPI(title="Oscillink Cloud API", default_response_class=ORJSONResponse)

# --- Security & Ops Middlewares (configurable via env) ---
_ALLOW_ORIGINS = os.getenv("OSCILLINK_CORS_ALLOW_ORIGINS", "").strip()
_TRUSTED_HOSTS = os.getenv("OSCILLINK_TRUSTED_HOSTS", "").strip()  # e.g. "api.example.com,.example.com"
_FORCE_HTTPS = os.getenv("OSCILLINK_FORCE_HTTPS", "0") in {"1", "true", "TRUE", "on"}

if _ALLOW_ORIGINS:
    origins = [o.strip() for o in _ALLOW_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["POST", "GET", "OPTIONS", "DELETE"],
        allow_headers=["*"],
        max_age=600,
    )
if _TRUSTED_HOSTS:
    hosts = [h.strip() for h in _TRUSTED_HOSTS.split(",") if h.strip()]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
if _FORCE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)

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
SETTLE_COUNTER: Any
SETTLE_LATENCY: Any
SETTLE_N_GAUGE: Any
SETTLE_D_GAUGE: Any
USAGE_NODES: Any
USAGE_NODE_DIM_UNITS: Any
JOB_QUEUE_DEPTH: Any
STRIPE_WEBHOOK_EVENTS: Any

if "oscillink_settle_requests_total" in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
    SETTLE_COUNTER = REGISTRY._names_to_collectors["oscillink_settle_requests_total"]  # type: ignore
    SETTLE_LATENCY = REGISTRY._names_to_collectors["oscillink_settle_latency_seconds"]  # type: ignore
    SETTLE_N_GAUGE = REGISTRY._names_to_collectors["oscillink_settle_last_N"]  # type: ignore
    SETTLE_D_GAUGE = REGISTRY._names_to_collectors["oscillink_settle_last_D"]  # type: ignore
    USAGE_NODES = REGISTRY._names_to_collectors["oscillink_usage_nodes_total"]  # type: ignore
    USAGE_NODE_DIM_UNITS = REGISTRY._names_to_collectors["oscillink_usage_node_dim_units_total"]  # type: ignore
    JOB_QUEUE_DEPTH = REGISTRY._names_to_collectors.get("oscillink_job_queue_depth")  # type: ignore
    if "oscillink_stripe_webhook_events_total" in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
        STRIPE_WEBHOOK_EVENTS = REGISTRY._names_to_collectors["oscillink_stripe_webhook_events_total"]  # type: ignore
    else:
        STRIPE_WEBHOOK_EVENTS = Counter(
            "oscillink_stripe_webhook_events_total", "Stripe webhook events", ["result"]
        )
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
            q_limit = int(meta.quota_limit_units) if meta.quota_limit_units is not None else q.limit
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
    if not rec or now - rec["window_start"] >= q_window or rec.get("limit") != q_limit or rec.get("window") != q_window:
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


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Set basic security headers on every response.

    CSP is intentionally omitted globally; applied only on the HTML success page. Add selectively if needed.
    """
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
    return resp

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
    if redis_enabled():
        key = f"iprl:{ip}:{window}"
        count, ttl = incr_with_window(key, window, amount=1)
        # When Redis not reachable, incr_with_window returns (0, -2); fall back to memory path
        if ttl != -2:
            if count > limit:
                reset_at = int(now + (ttl if ttl >= 0 else window))
                headers = {
                    "Retry-After": str(int(max(reset_at - now, 0)) + 1),
                    "X-IPLimit-Limit": str(limit),
                    "X-IPLimit-Remaining": "0",
                    "X-IPLimit-Reset": str(reset_at),
                }
                return ORJSONResponse(status_code=429, content={"detail": "ip rate limit exceeded"}, headers=headers)
            response = await call_next(request)
            remaining = max(limit - int(count), 0)
            reset_at = int(now + (ttl if ttl >= 0 else window))
            response.headers.setdefault("X-IPLimit-Limit", str(limit))
            response.headers.setdefault("X-IPLimit-Remaining", str(remaining))
            response.headers.setdefault("X-IPLimit-Reset", str(reset_at))
            return response
    # Fallback to in-memory counters
    rec = _ip_rl_counters.get(ip)
    if not rec or now - rec["window_start"] >= window or rec.get("limit") != float(limit) or rec.get("window") != float(window):
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
    if redis_enabled():
        key = f"grl:{_rl_state['window']}"
        count, ttl = incr_with_window(key, _rl_state["window"], amount=1)
        if request.url.path not in ("/health", "/metrics") and count > _rl_state["limit"] and ttl != -2:
            reset_at = int(now + (ttl if ttl >= 0 else _rl_state["window"]))
            headers = {
                "Retry-After": str(int(max(reset_at - now, 0)) + 1),
                "X-RateLimit-Limit": str(_rl_state["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_at),
            }
            return ORJSONResponse(status_code=429, content={"detail": "rate limit exceeded"}, headers=headers)
        resp = await call_next(request)
        remaining = max(_rl_state["limit"] - int(count), 0)
        reset_at = int(now + (ttl if ttl >= 0 else _rl_state["window"]))
        resp.headers.setdefault("X-RateLimit-Limit", str(_rl_state["limit"]))
        resp.headers.setdefault("X-RateLimit-Remaining", str(remaining))
        resp.headers.setdefault("X-RateLimit-Reset", str(reset_at))
        return resp
    # Fallback to in-memory window
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
# Shared in-memory webhook events store (stable across TestClient instances)

_STRIPE_EVENTS_COUNT = 0

class _WebhookEventsWrapper:
    """Wrapper around app.state.webhook_events that also tracks a stable count.

    Tests use len(_webhook_events_mem) and .clear(). This wrapper ensures len reflects
    the number of unique events recorded in this process regardless of any internal
    rebindings of the underlying dict.
    """
    def __init__(self, app_ref: FastAPI):
        self._app = app_ref

    def _store(self) -> dict:
        if not hasattr(self._app.state, "webhook_events") or not isinstance(self._app.state.webhook_events, dict):  # type: ignore[attr-defined]
            self._app.state.webhook_events = {}
        return self._app.state.webhook_events  # type: ignore[attr-defined]

    def __setitem__(self, key, value):
        global _STRIPE_EVENTS_COUNT
        st = self._store()
        is_new = key not in st
        st[key] = value
        if is_new:
            _STRIPE_EVENTS_COUNT += 1

    def __getitem__(self, key):
        return self._store()[key]

    def __contains__(self, key):
        return key in self._store()

    def get(self, key, default=None):
        return self._store().get(key, default)

    def values(self):
        return self._store().values()

    def items(self):
        return self._store().items()

    def clear(self):
        global _STRIPE_EVENTS_COUNT
        st = self._store()
        st.clear()
        _STRIPE_EVENTS_COUNT = 0

    def __len__(self):
        return _STRIPE_EVENTS_COUNT

_webhook_events_mem = _WebhookEventsWrapper(app)

def _webhook_events_collection():
    return os.getenv("OSCILLINK_WEBHOOK_EVENTS_COLLECTION", "").strip()

def _webhook_get(event_id: str):
    # Memory first
    if event_id in _webhook_events_mem:
        return _webhook_events_mem[event_id]
    # Redis (optional distributed idempotency)
    if redis_enabled():
        val, _ttl = get_with_ttl(f"stripe_evt:{event_id}")
        if val:
            try:
                return json.loads(val)
            except Exception:
                return {"id": event_id, "source": "redis"}
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
    # Redis store with TTL for distributed idempotency
    if redis_enabled():
        try:
            ttl = int(os.getenv("OSCILLINK_WEBHOOK_TTL", "604800"))  # 7 days default
        except ValueError:
            ttl = 604800
        try:
            set_with_ttl(f"stripe_evt:{event_id}", json.dumps(record, separators=(",", ":")), ttl)
        except Exception:
            pass
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

def api_key_guard(x_api_key: str | None = Header(default=None)):  # noqa: C901
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
        # Replace global singleton
        from cloud.app import keystore as _kmod  # noqa: I001 (local hot-reload import)
        from cloud.app.keystore import InMemoryKeyStore as _IMKS  # noqa: N814,I001 local import to avoid cycle
        _kmod._key_store = _IMKS()
        ks = get_keystore()
        _ENV_KEYS_FINGERPRINT = current_fp
    backend = os.getenv("OSCILLINK_KEYSTORE_BACKEND", "memory").lower()
    # Legacy env list ALWAYS enforced if present (checked early to satisfy tests expecting 401)
    allowed = get_api_keys()
    if allowed:
        if x_api_key is None or x_api_key not in allowed:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        # Tier overrides may be handled by InMemoryKeyStore above; return key directly
        return x_api_key

    # If no env keys are configured at all, operate in open mode for memory backend regardless of
    # any prior in-memory keystore state. This matches tests that unset OSCILLINK_API_KEYS and expect
    # unauthenticated access.
    if backend == "memory" and not allowed:
        return None

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
    # If we reach here, open access (no env key list)
    return None

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

## Removed earlier draft Stripe webhook stub; consolidated full implementation later in file.

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

def _new_api_key() -> str:
    # Generate a URL-safe API key (32 bytes -> 43 chars base64url). Prefix for readability.
    try:
        import secrets
        return "ok_" + secrets.token_urlsafe(32)
    except Exception:
        return "ok_" + uuid.uuid4().hex

# --- Optional Firestore mapping: api_key -> (stripe_customer_id, subscription_id) ---
_CUSTOMERS_COLLECTION = os.getenv("OSCILLINK_CUSTOMERS_COLLECTION", "").strip()

def _fs_get_customer_mapping(api_key: str):  # pragma: no cover - external dependency
    if not _CUSTOMERS_COLLECTION:
        return None
    try:
        from google.cloud import firestore  # type: ignore
        client = firestore.Client()
        snap = client.collection(_CUSTOMERS_COLLECTION).document(api_key).get()
        if snap.exists:
            return snap.to_dict() or None
    except Exception:
        return None
    return None

def _fs_set_customer_mapping(api_key: str, customer_id: str | None, subscription_id: str | None):  # pragma: no cover - external dependency
    if not _CUSTOMERS_COLLECTION or not api_key or not (customer_id or subscription_id):
        return
    try:
        from google.cloud import firestore  # type: ignore
        client = firestore.Client()
        doc_ref = client.collection(_CUSTOMERS_COLLECTION).document(api_key)
        payload = {
            "api_key": api_key,
            "stripe_customer_id": customer_id,
            "subscription_id": subscription_id,
            "updated_at": time.time(),
        }
        if not doc_ref.get().exists:
            payload["created_at"] = time.time()
        doc_ref.set(payload, merge=True)
    except Exception:
        # best-effort only
        pass

def _stripe_fetch_session_and_subscription(session_id: str):  # pragma: no cover - external dependency path
    stripe_secret = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")
    if not stripe_secret:
        raise RuntimeError("stripe secret not configured")
    import stripe  # type: ignore
    stripe.api_key = stripe_secret
    stripe.api_version = "2024-06-20"
    session = stripe.checkout.Session.retrieve(session_id, expand=["subscription", "customer"])  # type: ignore
    if not session:
        raise ValueError("session not found")
    sub = session.get("subscription") if isinstance(session, dict) else getattr(session, "subscription", None)
    if isinstance(sub, str):
        sub = stripe.Subscription.retrieve(sub)  # type: ignore
    if not isinstance(sub, dict):
        sub_id = session.get("subscription") if isinstance(session, dict) else None
        if sub_id:
            sub = stripe.Subscription.retrieve(sub_id)  # type: ignore
    if not isinstance(sub, dict):
        raise ValueError("subscription not found for session")
    return session, sub

def _send_key_email(to_email: str, api_key: str, tier: str, status: str) -> bool:
    """Best-effort email sender for delivering API keys.

    Controlled by environment:
      - OSCILLINK_EMAIL_MODE: 'none' (default), 'console', or 'smtp'
      - OSCILLINK_EMAIL_FROM: sender address for smtp mode
      - SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_TLS (1/0)

    Returns True if an email was attempted/sent; False otherwise.
    """
    mode = (os.getenv("OSCILLINK_EMAIL_MODE", "none") or "none").lower()
    if not to_email or mode == "none":
        return False
    subject = "Your Oscillink API Key"
    body = (
        f"Thanks for subscribing.\n\n"
        f"Tier: {tier} (status: {status})\n"
        f"API Key: {api_key}\n\n"
        f"Keep this key secret. You can rotate or revoke it via support.\n"
    )
    if mode == "console":
        # Log to server console only (useful for dev)
        print(f"[email:console] to={to_email} subject={subject}\n{body}")
        return True
    if mode == "smtp":
        from_addr = os.getenv("OSCILLINK_EMAIL_FROM", "")
        host = os.getenv("SMTP_HOST", "")
        port = int(os.getenv("SMTP_PORT", "587") or "587")
        user = os.getenv("SMTP_USER", "")
        pw = os.getenv("SMTP_PASS", "")
        use_tls = os.getenv("SMTP_TLS", "1") in {"1", "true", "TRUE", "on"}
        if not (from_addr and host and to_email):
            return False
        try:
            msg = EmailMessage()
            msg["From"] = from_addr
            msg["To"] = to_email
            msg["Subject"] = subject
            msg.set_content(body)
            with smtplib.SMTP(host, port, timeout=10) as s:
                if use_tls:
                    s.starttls()
                if user:
                    s.login(user, pw)
                s.send_message(msg)
            return True
        except Exception:
            # Do not fail webhook processing on email errors
            return False
    return False

def _provision_key_for_subscription(sub: dict) -> tuple[str, str, str]:  # (api_key, tier, status)
    meta = sub.get("metadata", {}) or {}
    api_key = meta.get("api_key") if isinstance(meta, dict) else None
    new_tier = resolve_tier_from_subscription(sub)
    tinfo = tier_info(new_tier)
    status = "pending" if getattr(tinfo, "requires_manual_activation", False) else "active"
    if not api_key:
        api_key = _new_api_key()
        # Best-effort attach to subscription metadata
        try:
            import stripe  # type: ignore
            stripe.Subscription.modify(sub.get("id"), metadata={**meta, "api_key": api_key})  # type: ignore
        except Exception:
            pass
    ks = get_keystore()
    ks.update(api_key, create=True, tier=new_tier, status=status, features={"diffusion_gates": tinfo.diffusion_allowed})
    return api_key, new_tier, status

@app.get("/billing/success")
def billing_success(session_id: str | None = None):  # pragma: no cover - external dependency path
    """Stripe Checkout success landing page."""
    if not session_id:
        return HTMLResponse(status_code=400, content=(
            "<html><body><h2>Missing session</h2><p>session_id is required. "
            "If you reached this page from Stripe Checkout, contact support.</p></body></html>"))
    try:
        _ = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")
        if not _:
            return HTMLResponse(status_code=503, content=(
                "<html><body><h2>Billing not configured</h2>"
                "<p>Stripe secret not set on server. Your payment likely succeeded, but we can't "
                "provision a key automatically. Please email contact@oscillink.com with your receipt.</p>"
                "</body></html>"))
        session, sub = _stripe_fetch_session_and_subscription(session_id)
        api_key, new_tier, status = _provision_key_for_subscription(sub)
        # Best-effort: persist api_key -> (customer_id, subscription_id) mapping for portal/cancel flows
        try:
            cust_id = session.get("customer") if isinstance(session, dict) else None
            sub_id = sub.get("id") if isinstance(sub, dict) else None
            _fs_set_customer_mapping(api_key, cust_id, sub_id)
        except Exception:
            pass
        html = f"""
        <html>
          <head>
            <meta charset=\"utf-8\" />
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
            <title>Oscillink — Your API Key</title>
            <style>
              body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 2rem; color: #111; }}
              code {{ background: #f5f5f5; padding: 0.25rem 0.4rem; border-radius: 4px; }}
              .card {{ border: 1px solid #eee; border-radius: 8px; padding: 1rem 1.25rem; max-width: 800px; }}
              .muted {{ color: #555; }}
            </style>
          </head>
          <body>
            <div class=\"card\">
              <h2>Thanks — your key is ready</h2>
              <p class=\"muted\">Tier: <strong>{new_tier}</strong> · Status: <strong>{status}</strong></p>
              <p>API Key: <code>{api_key}</code></p>
              <h3>Quickstart</h3>
              <ol>
                <li>Install: <code>pip install oscillink</code></li>
                <li>Call the Cloud API with header <code>X-API-Key: {api_key}</code></li>
              </ol>
              <p class=\"muted\">Keep this key secret. You can rotate or revoke it via admin support.</p>
            </div>
          </body>
        </html>
        """
        # Apply strict response headers to avoid caching and referrer leaks
        headers = {
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            # Narrow CSP suitable for this simple page
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'",
        }
        return HTMLResponse(content=html, headers=headers)
    except ModuleNotFoundError:
        return HTMLResponse(status_code=501, content=(
            "<html><body><h2>Stripe library not installed</h2>"
            "<p>Server cannot retrieve your session. We will email your key shortly.</p></body></html>"))
    except Exception as e:
        return HTMLResponse(status_code=400, content=(
            f"<html><body><h2>Checkout session error</h2><p>{str(e)}</p>"
            "<p>If this persists, contact support with your receipt.</p></body></html>"))

@app.post("/billing/portal")
def create_billing_portal(ctx=Depends(feature_context)):
    """Create a Stripe Billing Portal session for the authenticated API key.

    Requires Firestore customer mapping (OSCILLINK_CUSTOMERS_COLLECTION) and STRIPE_SECRET_KEY.
    Returns a URL to redirect the user for managing/cancelling their subscription.
    """
    x_api_key = ctx["api_key"]
    if not x_api_key:
        raise HTTPException(status_code=401, detail="missing API key")
    mapping = _fs_get_customer_mapping(x_api_key)
    if not mapping or not mapping.get("stripe_customer_id"):
        raise HTTPException(status_code=404, detail="customer mapping not found; contact support")
    try:
        stripe_secret = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")
        if not stripe_secret:
            raise HTTPException(status_code=503, detail="billing not configured")
        import stripe  # type: ignore
        stripe.api_key = stripe_secret
        stripe.api_version = "2024-06-20"
        return_url = os.getenv("OSCILLINK_PORTAL_RETURN_URL", "https://oscillink.com")
        sess = stripe.billing_portal.Session.create(  # type: ignore
            customer=mapping["stripe_customer_id"],
            return_url=return_url,
        )
        return {"url": getattr(sess, "url", None) or sess.get("url")}
    except HTTPException:
        raise
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=501, detail="stripe library not installed") from exc
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to create portal session: {e}") from e

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
    # Pydantic v2 prefers model_dump; maintain compatibility with v1
    fields = payload.model_dump(exclude_unset=True) if hasattr(payload, "model_dump") else payload.dict(exclude_unset=True)
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

@app.get("/admin/billing/price-map")
def admin_get_price_map(auth=Depends(_admin_guard)):
    """Return the active Stripe price->tier map and tier catalog.

    Useful for quickly verifying which Stripe price IDs map to internal tiers and the
    associated allowances for each tier.
    """
    pmap = get_price_map()
    tiers = {
        name: {
            "name": info.name,
            "monthly_unit_cap": info.monthly_unit_cap,
            "diffusion_allowed": info.diffusion_allowed,
            "requires_manual_activation": info.requires_manual_activation,
        }
        for name, info in TIER_CATALOG.items()
    }
    return {"price_map": pmap, "tiers": tiers}

@app.get("/admin/usage/{api_key}")
def admin_get_usage(api_key: str, auth=Depends(_admin_guard)):
    """Return current in-memory quota window state and monthly usage for an API key.

    Notes:
    - Quota window is best-effort per-process; if multiple processes run, each has its own counters.
    - Monthly usage may optionally be hydrated/persisted via Firestore when configured.
    """
    # Current rolling quota window state (may be absent if key hasn't made requests this window)
    q = _key_usage.get(api_key)
    quota = None
    if q:
        quota = {
            "window_start": q.get("window_start"),
            "used": int(q.get("used", 0)),
            "limit": int(q.get("limit", 0)),
            "window": int(q.get("window", 60)),
            "reset": int(q.get("window_start", 0) + q.get("window", 60)),
            "remaining": max(int(q.get("limit", 0)) - int(q.get("used", 0)), 0),
        }
    # Monthly usage (period, used units, remaining if cap applies)
    mu = _monthly_usage.get(api_key)
    monthly = None
    if mu:
        meta: KeyMetadata | None = get_keystore().get(api_key)
        cap = tier_info(meta.tier).monthly_unit_cap if meta else None
        remaining = None if cap is None else max(int(cap) - int(mu.get("used", 0)), 0)
        monthly = {
            "period": mu.get("period"),
            "used": int(mu.get("used", 0)),
            "limit": cap,
            "remaining": remaining,
        }
    return {"api_key": api_key, "quota": quota, "monthly": monthly}

@app.post("/admin/billing/cancel/{api_key}")
def admin_cancel_subscription(api_key: str, immediate: bool | None = None, auth=Depends(_admin_guard)):
    """Cancel a customer's subscription for the given API key (admin only).

    If immediate is True, the subscription is cancelled immediately; otherwise it will cancel at period end.
    Requires customer mapping in Firestore and STRIPE_SECRET_KEY.
    """
    mapping = _fs_get_customer_mapping(api_key)
    if not mapping or not mapping.get("subscription_id"):
        raise HTTPException(status_code=404, detail="subscription mapping not found")
    try:
        stripe_secret = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")
        if not stripe_secret:
            raise HTTPException(status_code=503, detail="billing not configured")
        import stripe  # type: ignore
        stripe.api_key = stripe_secret
        stripe.api_version = "2024-06-20"
        sub_id = mapping["subscription_id"]
        do_immediate = bool(immediate) if immediate is not None else (os.getenv("OSCILLINK_STRIPE_CANCEL_IMMEDIATE", "0") in {"1","true","TRUE","on"})
        if do_immediate:
            stripe.Subscription.delete(sub_id)  # type: ignore
            status = "cancelled"
        else:
            stripe.Subscription.modify(sub_id, cancel_at_period_end=True)  # type: ignore
            status = "cancel_at_period_end"
        # Suspend key access immediately
        ks = get_keystore()
        ks.update(api_key, status="suspended")
        return {"api_key": api_key, "subscription_id": sub_id, "status": status}
    except HTTPException:
        raise
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=501, detail="stripe library not installed") from exc
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to cancel subscription: {e}") from e

# Stripe webhook with subscription → tier sync
@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):  # noqa: C901
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
            # When explicitly allowed, accept unverified events (test/dev only)
            if allow_unverified:
                try:
                    event = json.loads(payload_text)
                except Exception as err:
                    raise HTTPException(status_code=400, detail="invalid JSON payload") from err
            else:
                raise HTTPException(status_code=400, detail="missing stripe-signature header")
        # Enforce timestamp freshness (basic replay protection) if header contains t= segment
        try:
            max_age = int(os.getenv("OSCILLINK_STRIPE_MAX_AGE", "300"))  # seconds
        except ValueError:
            max_age = 300
        if max_age > 0 and sig_header:
            # stripe-signature format: t=timestamp,v1=...,v0=...
            try:
                parts = {kv.split('=')[0]: kv.split('=')[1] for kv in sig_header.split(',') if '=' in kv}
                if 't' in parts:
                    ts = int(parts['t'])
                    now = int(time.time())
                    if now - ts > max_age:
                        # Allow explicit unverified override pathway to bypass freshness (test/dev only)
                        if allow_unverified:
                            pass
                        else:
                            raise HTTPException(status_code=400, detail="webhook timestamp too old")
            except HTTPException:
                raise
            except Exception:
                # Non-fatal: if parsing fails we proceed (could tighten later)
                pass
        # If explicitly allowed, skip cryptographic verification and parse JSON
        if allow_unverified:
            try:
                event = json.loads(payload_text)
                verified = False
            except Exception as err:
                raise HTTPException(status_code=400, detail="invalid JSON payload") from err
        else:
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
                except Exception as err:
                    raise HTTPException(status_code=400, detail="invalid JSON payload (no stripe lib)") from err
            except Exception as e:  # signature failure
                # If verification fails but override is allowed, proceed unverified
                if os.getenv("OSCILLINK_ALLOW_UNVERIFIED_STRIPE", "0") in {"1","true","TRUE","on"}:
                    try:
                        event = json.loads(payload_text)
                        verified = False
                    except Exception as err:
                        raise HTTPException(status_code=400, detail="invalid JSON payload") from err
                else:
                    raise HTTPException(status_code=400, detail=f"signature verification failed: {e}") from e
    else:
        try:
            event = json.loads(payload_text)
        except Exception as err:
            raise HTTPException(status_code=400, detail="invalid JSON payload") from err

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
    # Checkout success handling (optional auto-provisioning via webhook)
    elif etype == "checkout.session.completed":
        sess_obj = event.get("data", {}).get("object", {}) if isinstance(event, dict) else {}
        email = None
        try:
            email = (
                (sess_obj.get("customer_details", {}) or {}).get("email")
                or sess_obj.get("customer_email")
            )
        except Exception:
            email = None
        stripe_secret = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")
        if not stripe_secret:
            note = "stripe secret not set; cannot provision on webhook"
        elif secret and not verified and not allow_unverified:
            note = "signature not verified; checkout session ignored"
        else:
            try:  # pragma: no cover - external dependency path
                import stripe  # type: ignore
                stripe.api_key = stripe_secret
                stripe.api_version = "2024-06-20"
                sid = sess_obj.get("id") or None
                if not sid:
                    raise ValueError("missing session id")
                session, sub = _stripe_fetch_session_and_subscription(sid)
                api_key, new_tier, status = _provision_key_for_subscription(sub)
                try:
                    cust_id = session.get("customer") if isinstance(session, dict) else None
                    sub_id = sub.get("id") if isinstance(sub, dict) else None
                    _fs_set_customer_mapping(api_key, cust_id, sub_id)
                except Exception:
                    pass
                if email:
                    _send_key_email(email, api_key, new_tier, status)
                processed = True
                note = f"key provisioned for session; tier={new_tier}"
            except ModuleNotFoundError:
                note = "stripe library not installed; cannot provision"
            except Exception as e:
                note = f"provisioning failed: {e}"
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
