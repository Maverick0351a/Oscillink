from __future__ import annotations

# Standard library
import json
import logging
import os
import random
import time
import uuid
from typing import Any

# Third-party
import numpy as np
from fastapi import Depends, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, ORJSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from oscillink import OscillinkLattice, __version__

# Local application imports
from .billing import (
    current_period,
    get_price_map,
    tier_info,
)
from .config import get_settings
from .factory import create_app
from .features import resolve_features
from .keystore import InMemoryKeyStore, KeyMetadata, get_keystore  # type: ignore
from .learners import propose_overrides, record_observation
from .models import HealthResponse, ReceiptResponse, SettleRequest
from .redis_backend import incr_with_window, redis_enabled
from .runtime_config import get_api_keys, get_quota_config, get_rate_limit
from .services import cli as cli_service
from .services.billing import (
    fs_get_customer_mapping as _fs_get_customer_mapping,
)
from .services.billing import (
    fs_set_customer_mapping as _fs_set_customer_mapping,
)
from .services.billing import (
    provision_key_for_subscription as _provision_key_for_subscription,
)
from .services.billing import (
    stripe_fetch_session_and_subscription as _stripe_fetch_session_and_subscription,
)
from .services.cache import (
    bundle_cache_get as _bundle_cache_get,
)
from .services.cache import (
    bundle_cache_put as _bundle_cache_put,
)
from .services.events import webhook_get_persistent, webhook_store_persistent
from .services.usage_log import append_usage as _append_usage
from .services.webhook_mem import get_webhook_events_mem
from .settings import get_app_settings

app = create_app()

s = get_app_settings()

MAX_BODY_BYTES = int(s.max_body_bytes)  # 1MB default


def _truthy(val: str | None) -> bool:
    return val in {"1", "true", "TRUE", "on", "On", "yes", "YES"}


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
CLI_SESSIONS_CREATED: Any
CLI_SESSIONS_PROVISIONED: Any
CLI_SESSIONS_ACTIVE: Any

if "oscillink_settle_requests_total" in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
    SETTLE_COUNTER = REGISTRY._names_to_collectors["oscillink_settle_requests_total"]  # type: ignore
    SETTLE_LATENCY = REGISTRY._names_to_collectors["oscillink_settle_latency_seconds"]  # type: ignore
    SETTLE_N_GAUGE = REGISTRY._names_to_collectors["oscillink_settle_last_N"]  # type: ignore
    SETTLE_D_GAUGE = REGISTRY._names_to_collectors["oscillink_settle_last_D"]  # type: ignore
    USAGE_NODES = REGISTRY._names_to_collectors["oscillink_usage_nodes_total"]  # type: ignore
    USAGE_NODE_DIM_UNITS = REGISTRY._names_to_collectors["oscillink_usage_node_dim_units_total"]  # type: ignore
    JOB_QUEUE_DEPTH = REGISTRY._names_to_collectors.get("oscillink_job_queue_depth")  # type: ignore
    if "oscillink_stripe_webhook_events_total" in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
        STRIPE_WEBHOOK_EVENTS = REGISTRY._names_to_collectors[
            "oscillink_stripe_webhook_events_total"
        ]  # type: ignore
    else:
        STRIPE_WEBHOOK_EVENTS = Counter(
            "oscillink_stripe_webhook_events_total", "Stripe webhook events", ["result"]
        )
    # CLI metrics in reload-safe manner
    if "oscillink_cli_sessions_created_total" in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
        CLI_SESSIONS_CREATED = REGISTRY._names_to_collectors["oscillink_cli_sessions_created_total"]  # type: ignore
    else:
        CLI_SESSIONS_CREATED = Counter(
            "oscillink_cli_sessions_created_total", "CLI pairing sessions created"
        )
    if "oscillink_cli_sessions_provisioned_total" in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
        CLI_SESSIONS_PROVISIONED = REGISTRY._names_to_collectors[
            "oscillink_cli_sessions_provisioned_total"
        ]  # type: ignore
    else:
        CLI_SESSIONS_PROVISIONED = Counter(
            "oscillink_cli_sessions_provisioned_total", "CLI pairing sessions provisioned"
        )
    if "oscillink_cli_sessions_active" in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
        CLI_SESSIONS_ACTIVE = REGISTRY._names_to_collectors["oscillink_cli_sessions_active"]  # type: ignore
    else:
        CLI_SESSIONS_ACTIVE = Gauge(
            "oscillink_cli_sessions_active",
            "Active CLI sessions (memory mode only, best-effort)",
        )
else:
    SETTLE_COUNTER = Counter("oscillink_settle_requests_total", "Total settle requests", ["status"])
    SETTLE_LATENCY = Histogram(
        "oscillink_settle_latency_seconds",
        "Settle latency",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    SETTLE_N_GAUGE = Gauge("oscillink_settle_last_N", "N of last settle")
    SETTLE_D_GAUGE = Gauge("oscillink_settle_last_D", "D of last settle")
    USAGE_NODES = Counter("oscillink_usage_nodes_total", "Total nodes processed")
    USAGE_NODE_DIM_UNITS = Counter(
        "oscillink_usage_node_dim_units_total", "Total node-dimension units processed (sum N*D)"
    )
    JOB_QUEUE_DEPTH = Gauge(
        "oscillink_job_queue_depth", "Number of jobs currently queued or running"
    )
    STRIPE_WEBHOOK_EVENTS = Counter(
        "oscillink_stripe_webhook_events_total", "Stripe webhook events", ["result"]
    )
    CLI_SESSIONS_CREATED = Counter(
        "oscillink_cli_sessions_created_total", "CLI pairing sessions created"
    )
    CLI_SESSIONS_PROVISIONED = Counter(
        "oscillink_cli_sessions_provisioned_total", "CLI pairing sessions provisioned"
    )
    CLI_SESSIONS_ACTIVE = Gauge(
        "oscillink_cli_sessions_active", "Active CLI sessions (memory mode only, best-effort)"
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


def _update_monthly_usage_doc(
    api_key: str, period: str, used: int
):  # pragma: no cover - external dependency
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
                tx.set(
                    ref,
                    {
                        "api_key": api_key,
                        "period": period,
                        "used": used,
                        "updated_at": time.time(),
                        "created_at": time.time(),
                    },
                )

        try:
            tx = client.transaction()
            _tx_update(tx, doc_ref)
        except Exception:
            # Blind overwrite (eventual consistency acceptable for quota enforcement best-effort)
            doc_ref.set(
                {"api_key": api_key, "period": period, "used": used, "updated_at": time.time()},
                merge=True,
            )
    except Exception:
        pass


def _monthly_cap_from_env() -> int:
    try:
        return int(os.getenv("OSCILLINK_MONTHLY_CAP", "0"))
    except ValueError:
        return 0


def _resolve_monthly_cap(meta: KeyMetadata | None) -> int:
    """Return effective monthly cap in units for a key metadata.

    Env override (OSCILLINK_MONTHLY_CAP>0) takes precedence; 0/invalid means use tier default.
    Returns 0 when unlimited/disabled.
    """
    if not meta:
        return 0
    cap_env = _monthly_cap_from_env()
    if cap_env > 0:
        return cap_env
    try:
        return int(tier_info(meta.tier).monthly_unit_cap)  # type: ignore[arg-type]
    except Exception:
        return 0


def _get_monthly_rec(key: str, period: str) -> dict[str, int | str]:
    """Get or initialize the per-key monthly usage record for the given period."""
    rec = _monthly_usage.get(key)
    if rec and rec.get("period") == period:
        return rec
    used_val = 0
    if _MONTHLY_USAGE_COLLECTION:
        persisted = _load_monthly_usage_doc(key, period)
        if persisted and isinstance(persisted.get("used"), (int, float)):
            used_val = int(persisted.get("used", 0))
    rec = {"period": period, "used": used_val}
    _monthly_usage[key] = rec  # type: ignore[index]
    return rec


def _check_monthly_cap(key: str | None, units: int):
    """Enforce monthly unit caps; return usage context or None when unlimited.

    Raises HTTPException(429/413) when exceeding caps.
    """
    if key is None:
        return None
    meta = get_keystore().get(key)
    cap = _resolve_monthly_cap(meta)
    if cap <= 0:
        return None

    period = current_period()
    rec = _get_monthly_rec(key, period)
    used = int(rec.get("used", 0))

    if units > cap:
        raise HTTPException(
            status_code=413, detail=f"request units {units} exceed monthly cap {cap}"
        )

    if used + units > cap:
        remaining = max(cap - used, 0)
        headers = {"X-MonthCap-Limit": str(cap), "X-MonthCap-Remaining": str(remaining)}
        raise HTTPException(
            status_code=429,
            detail=f"monthly cap exceeded (cap={cap}, used={used})",
            headers=headers,
        )

    rec["used"] = used + units  # type: ignore[index]
    if _MONTHLY_USAGE_COLLECTION:
        _update_monthly_usage_doc(key, period, int(rec["used"]))  # type: ignore[index]
    remaining = cap - int(rec["used"])  # type: ignore[index]
    return {"limit": cap, "used": int(rec["used"]), "remaining": remaining, "period": period}


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
            q_window = (
                int(meta.quota_window_seconds)
                if meta.quota_window_seconds is not None
                else q.window
            )
        else:
            q_limit, q_window = q.limit, q.window
    else:
        q_limit, q_window = q.limit, q.window
    if q_limit <= 0 or key is None:
        # Quota disabled OR unauthenticated (open mode)
        return -1, 0, 0
    now = time.time()
    rec = _key_usage.get(key)
    if (
        not rec
        or now - rec["window_start"] >= q_window
        or rec.get("limit") != q_limit
        or rec.get("window") != q_window
    ):
        rec = {"window_start": now, "used": 0.0, "limit": q_limit, "window": q_window}
        _key_usage[key] = rec
    if units > q_limit:
        raise HTTPException(
            status_code=413, detail=f"request units {units} exceed per-key limit {q_limit}"
        )
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


@app.middleware("http")
async def json_access_log_mw(request: Request, call_next):
    """Optional structured JSON access log with sampling.

    Enable with OSCILLINK_JSON_LOGS=1 and configure sample via OSCILLINK_LOG_SAMPLE (0.0..1.0).
    Defaults to disabled. Avoids logging bodies; focuses on request/response metadata only.
    """
    if not _truthy(os.getenv("OSCILLINK_JSON_LOGS")):
        return await call_next(request)
    try:
        sample = float(os.getenv("OSCILLINK_LOG_SAMPLE", "1"))
    except ValueError:
        sample = 1.0
    t0 = time.time()
    try:
        response = await call_next(request)
        status = int(getattr(response, "status_code", 200))
    except Exception:
        status = 500
        # still log the exception pathway; re-raise
        raise
    finally:
        if random.random() <= max(0.0, min(sample, 1.0)):
            rid = request.headers.get(REQUEST_ID_HEADER, "")
            client_ip = _client_ip(
                request, os.getenv("OSCILLINK_TRUST_XFF", "0") in {"1", "true", "TRUE", "on"}
            )
            rec = {
                "ts": time.time(),
                "level": "info",
                "event": "access",
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "duration_ms": int(1000.0 * (time.time() - t0)),
                "request_id": rid,
                "ip": client_ip,
            }
            try:
                logging.getLogger("oscillink").info(json.dumps(rec))
            except Exception:
                # best-effort: fallback to stdout
                print(json.dumps(rec))
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


def _endpoint_rate_limit(request: Request, limit_env: str, window_env: str) -> dict | None:
    """Apply per-endpoint rate limits using the existing Redis/in-memory primitives.

    Returns headers to set on success, or raises HTTPException 429 when exceeded.
    Disabled when limit<=0.
    """
    try:
        limit = int(os.getenv(limit_env, "0"))
    except ValueError:
        limit = 0
    try:
        window = int(os.getenv(window_env, "60"))
    except ValueError:
        window = 60
    if limit <= 0:
        return None
    trust_xff = os.getenv("OSCILLINK_TRUST_XFF", "0") in {"1", "true", "TRUE", "on"}
    ip = _client_ip(request, trust_xff)
    now = time.time()
    if redis_enabled():
        key = f"eprl:{request.url.path}:{ip}:{window}"
        count, ttl = incr_with_window(key, window, amount=1)
        if count > limit and ttl != -2:
            reset_at = int(now + (ttl if ttl >= 0 else window))
            headers = {
                "Retry-After": str(int(max(reset_at - now, 0)) + 1),
                "X-EPRL-Limit": str(limit),
                "X-EPRL-Remaining": "0",
                "X-EPRL-Reset": str(reset_at),
            }
            raise HTTPException(
                status_code=429, detail="endpoint rate limit exceeded", headers=headers
            )
        remaining = max(limit - int(count), 0)
        reset_at = int(now + (ttl if ttl >= 0 else window))
        return {
            "X-EPRL-Limit": str(limit),
            "X-EPRL-Remaining": str(remaining),
            "X-EPRL-Reset": str(reset_at),
        }
    # In-memory fallback (basic per-endpoint/IP)
    key = f"{request.url.path}:{ip}"
    rec = _ip_rl_counters.get(key)
    if not rec or now - rec["window_start"] >= window:
        rec = {"window_start": now, "count": 0.0}
        _ip_rl_counters[key] = rec  # type: ignore
    if rec["count"] >= limit:
        reset_at = rec["window_start"] + window
        headers = {
            "Retry-After": str(int(reset_at - now) + 1),
            "X-EPRL-Limit": str(limit),
            "X-EPRL-Remaining": "0",
            "X-EPRL-Reset": str(int(reset_at)),
        }
        raise HTTPException(status_code=429, detail="endpoint rate limit exceeded", headers=headers)
    rec["count"] += 1
    remaining = max(limit - int(rec["count"]), 0)
    return {
        "X-EPRL-Limit": str(limit),
        "X-EPRL-Remaining": str(remaining),
        "X-EPRL-Reset": str(int(rec["window_start"] + window)),
    }


def _apply_eprl_headers(
    request: Request | None, response: Response | None, limit_env: str, window_env: str
) -> None:
    """Apply endpoint rate limit and set response headers when enabled.

    Raises HTTPException(429) when limit exceeded. Silently ignores unexpected errors.
    """
    if request is None:
        return
    try:
        hdrs = _endpoint_rate_limit(request, limit_env, window_env)
        if hdrs and response is not None:
            for k, v in hdrs.items():
                response.headers.setdefault(k, v)
    except HTTPException:
        raise
    except Exception:
        # Best-effort only
        pass


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
                return ORJSONResponse(
                    status_code=429, content={"detail": "ip rate limit exceeded"}, headers=headers
                )
            response = await call_next(request)
            remaining = max(limit - int(count), 0)
            reset_at = int(now + (ttl if ttl >= 0 else window))
            response.headers.setdefault("X-IPLimit-Limit", str(limit))
            response.headers.setdefault("X-IPLimit-Remaining", str(remaining))
            response.headers.setdefault("X-IPLimit-Reset", str(reset_at))
            return response
    # Fallback to in-memory counters
    rec = _ip_rl_counters.get(ip)
    if (
        not rec
        or now - rec["window_start"] >= window
        or rec.get("limit") != float(limit)
        or rec.get("window") != float(window)
    ):
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
        return ORJSONResponse(
            status_code=429, content={"detail": "ip rate limit exceeded"}, headers=headers
        )
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
        if (
            request.url.path not in ("/health", "/metrics")
            and count > _rl_state["limit"]
            and ttl != -2
        ):
            reset_at = int(now + (ttl if ttl >= 0 else _rl_state["window"]))
            headers = {
                "Retry-After": str(int(max(reset_at - now, 0)) + 1),
                "X-RateLimit-Limit": str(_rl_state["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_at),
            }
            return ORJSONResponse(
                status_code=429, content={"detail": "rate limit exceeded"}, headers=headers
            )
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
            "Retry-After": f"{int(reset_in) + 1}",
            "X-RateLimit-Limit": str(_rl_state["limit"]),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(_rl_state["window_start"] + _rl_state["window"])),
        }
        return ORJSONResponse(
            status_code=429, content={"detail": "rate limit exceeded"}, headers=headers
        )
    _rl_state["count"] += 1
    resp = await call_next(request)
    remaining = max(_rl_state["limit"] - _rl_state["count"], 0)
    resp.headers.setdefault("X-RateLimit-Limit", str(_rl_state["limit"]))
    resp.headers.setdefault("X-RateLimit-Remaining", str(remaining))
    resp.headers.setdefault(
        "X-RateLimit-Reset", str(int(_rl_state["window_start"] + _rl_state["window"]))
    )
    return resp


_API_VERSION = (
    get_settings().api_version
)  # capture at import for routing; other settings fetched dynamically
_ENV_KEYS_FINGERPRINT = {
    "api_keys": os.getenv("OSCILLINK_API_KEYS", ""),
    "tiers": os.getenv("OSCILLINK_KEY_TIERS", ""),
}

# ---------------- Bundle Cache now provided by services.cache -----------------


## Jobs moved to cloud.app.jobs router

# Usage logging (optional JSONL)
USAGE_LOG_PATH = os.getenv("OSCILLINK_USAGE_LOG")  # retained for backward compat in env


# ---------------- Webhook Event Logging / Idempotency -----------------
# Shared in-memory webhook events store (stable across TestClient instances)
_webhook_events_mem = get_webhook_events_mem(app)


def _webhook_events_collection():
    return os.getenv("OSCILLINK_WEBHOOK_EVENTS_COLLECTION", "").strip()


def _webhook_get(event_id: str):
    # Memory first
    if event_id in _webhook_events_mem:
        return _webhook_events_mem[event_id]
    # Persistent stores (Redis/Firestore) best-effort
    return webhook_get_persistent(event_id)


def _webhook_store(event_id: str, record: dict):
    # Always store in memory for fast duplicate checks
    _webhook_events_mem[event_id] = record
    # Persist best-effort (Redis/Firestore) without failing request
    webhook_store_persistent(event_id, record)


## _purge_old_jobs moved to cloud.app.jobs


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
    current_fp = {
        "api_keys": os.getenv("OSCILLINK_API_KEYS", ""),
        "tiers": os.getenv("OSCILLINK_KEY_TIERS", ""),
    }
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
                if not getattr(ks, "_keys", {}):
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


def _check_diffusion_allowed(req: SettleRequest, feats) -> None:
    if req.gates is not None:
        if os.getenv("OSCILLINK_DIFFUSION_GATES_ENABLED", "1") not in {"1", "true", "TRUE", "on"}:
            raise HTTPException(status_code=403, detail="diffusion gating temporarily disabled")
        if not feats.diffusion_allowed:
            raise HTTPException(
                status_code=403, detail="diffusion gating not enabled for this tier"
            )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version=__version__)


@app.get("/license/status")
def license_status():
    """Report licensed-container status based on exported entitlements.

    Behavior:
    - Reads entitlements JSON written by entrypoint verifier.
    - Returns {status: ok, exp, iss, sub, tier} if present and not expired.
    - If expired or missing:
      - When OSCILLINK_LICENSE_REQUIRED=1|true|on, return 503 to fail readiness.
      - Otherwise return 200 with status "stale" or "unknown".
    """
    ent_path = os.getenv("OSCILLINK_ENTITLEMENTS_PATH", "/run/oscillink_entitlements.json")
    leeway = 0
    try:
        leeway = int(os.getenv("OSCILLINK_JWT_LEEWAY", "300"))
    except ValueError:
        leeway = 300
    require = os.getenv("OSCILLINK_LICENSE_REQUIRED", "0").lower() in {"1", "true", "on"}
    try:
        with open(ent_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        if require:
            return ORJSONResponse(status_code=503, content={"status": "unlicensed"})
        return ORJSONResponse({"status": "unknown"})
    exp = None
    try:
        if isinstance(data.get("exp"), (int, float)):
            exp = int(data.get("exp"))
    except Exception:
        exp = None
    now = int(time.time())
    if exp is not None and (now - leeway) > exp:
        if require:
            return ORJSONResponse(status_code=503, content={"status": "expired", "exp": exp})
        return ORJSONResponse({"status": "stale", "exp": exp})
    return ORJSONResponse(
        {
            "status": "ok",
            "iss": data.get("iss"),
            "sub": data.get("sub") or data.get("license_id"),
            "tier": data.get("tier"),
            "exp": exp,
        }
    )


def _build_lattice(
    req: SettleRequest, api_key: str | None = None
) -> tuple[OscillinkLattice, int, int, int, dict, str]:
    Y = np.array(req.Y, dtype=np.float32)
    N, D = Y.shape
    if N == 0 or D == 0:
        raise HTTPException(status_code=400, detail="Empty matrix")
    s = get_settings()
    if s.max_nodes < N:
        raise HTTPException(status_code=413, detail=f"N>{s.max_nodes} exceeds limit")
    if s.max_dim < D:
        raise HTTPException(status_code=413, detail=f"D>{s.max_dim} exceeds limit")
    # Load adaptive profile overrides (cloud-only, optional) with exploration
    profile_id, overrides = propose_overrides(
        api_key,
        base={
            "lamG": req.params.lamG,
            "lamC": req.params.lamC,
            "lamQ": req.params.lamQ,
            "kneighbors": req.params.kneighbors,
        },
    )
    # Apply safe overrides
    lamG = float(overrides.get("lamG", req.params.lamG))
    lamC = float(overrides.get("lamC", req.params.lamC))
    lamQ = float(overrides.get("lamQ", req.params.lamQ))
    k_req = int(overrides.get("kneighbors", req.params.kneighbors))
    # Clamp kneighbors to avoid argpartition errors when requested >= N
    k_eff = min(k_req, max(1, N - 1))
    lat = OscillinkLattice(
        Y,
        kneighbors=k_eff,
        lamG=lamG,
        lamC=lamC,
        lamQ=lamQ,
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
    return (
        lat,
        N,
        D,
        k_eff,
        {"lamG": lamG, "lamC": lamC, "lamQ": lamQ, "kneighbors": k_eff},
        profile_id,
    )


def _safe_record_observation(
    api_key: str | None, profile_id: str, eff_params: dict, stats: dict, tol: float
):
    if not api_key:
        return
    try:
        record_observation(
            api_key,
            profile_id,
            {
                "lamG": eff_params["lamG"],
                "lamC": eff_params["lamC"],
                "lamQ": eff_params["lamQ"],
                "kneighbors": eff_params["kneighbors"],
            },
            {
                "duration_ms": float(stats.get("duration_ms", 0.0)),
                "iters": int(stats.get("iters", 0)),
                "residual": float(stats.get("residual", 0.0)),
                "tol": float(tol),
            },
        )
    except Exception:
        pass


def _compute_bundle_with_cache(
    lat: OscillinkLattice,
    k: int,
    api_key: str | None,
    profile_id: str,
    eff_params: dict,
    dt: float,
    max_iters: int,
    tol: float,
):
    """Return (bundle, elapsed_seconds, cache_status, state_sig, cache_rec_optional).

    Uses in-memory TTL LRU per-key cache keyed by state signature. On HIT, skips compute.
    On MISS, runs settle+bundle, records learning best-effort, stores in cache, and updates metrics.
    """
    state_sig = lat._signature()
    cache_rec = _bundle_cache_get(api_key, state_sig)
    if cache_rec is not None and isinstance(cache_rec.get("bundle"), list):
        return cache_rec["bundle"], 0.0, "HIT", state_sig, cache_rec
    # Compute path
    t0 = time.time()
    settle_stats = lat.settle(dt=dt, max_iters=max_iters, tol=tol)
    elapsed = time.time() - t0
    SETTLE_COUNTER.labels(status="ok").inc()
    SETTLE_LATENCY.observe(elapsed)
    # Gauges will be set by caller for N/D
    b = lat.bundle(k=k)
    try:
        _bundle_cache_put(api_key, state_sig, b)
    except Exception:
        pass
    # Learning hook (best-effort)
    try:
        record_observation(
            api_key,
            profile_id,
            {
                "lamG": eff_params["lamG"],
                "lamC": eff_params["lamC"],
                "lamQ": eff_params["lamQ"],
                "kneighbors": eff_params.get("kneighbors", k),
            },
            {
                "duration_ms": 1000.0 * elapsed,
                "iters": int(settle_stats.get("iters", 0)),
                "residual": float(settle_stats.get("res", 0.0)),
                "tol": float(tol),
            },
        )
    except Exception:
        pass
    return b, elapsed, "MISS", state_sig, None


@app.post(f"/{_API_VERSION}/settle", response_model=ReceiptResponse)
def settle(req: SettleRequest, request: Request, response: Response, ctx=Depends(feature_context)):
    x_api_key = ctx["api_key"]
    feats = ctx["features"]
    _check_diffusion_allowed(req, feats)
    lat, N, D, k_eff, eff_params, profile_id = _build_lattice(req, x_api_key)
    units = N * D
    # Monthly cap enforcement (before quota window since it is a higher level allowance)
    monthly_ctx = _check_monthly_cap(x_api_key, units)
    remaining, limit, reset_at = _check_and_consume_quota(x_api_key, units)

    t0 = time.time()
    try:
        settle_stats = lat.settle(
            dt=req.options.dt, max_iters=req.options.max_iters, tol=req.options.tol
        )
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
        "lam": {
            "G": eff_params["lamG"],
            "C": eff_params["lamC"],
            "Q": eff_params["lamQ"],
            "P": req.params.lamP,
        },
        "profile_id": profile_id,
    }
    sig_meta = (
        receipt.get("meta", {}).get("state_sig")
        if (receipt and isinstance(receipt.get("meta"), dict))
        else None
    )
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
    # Learning hook (best-effort): record observation for EMA updates
    try:
        record_observation(
            x_api_key,
            profile_id,
            {**eff_params},
            {
                "duration_ms": t_settle,
                "iters": int(settle_stats.get("iters", 0)),
                "residual": float(settle_stats.get("res", 0.0)),
                "tol": float(req.options.tol),
            },
        )
    except Exception:
        pass

    resp = ReceiptResponse(
        state_sig=state_sig,
        receipt=receipt,
        bundle=bundle,
        timings_ms={"total_settle_ms": t_settle},
        meta={
            **meta,
            "request_id": request.headers.get(REQUEST_ID_HEADER, ""),
            "usage": {"nodes": N, "node_dim_units": units, "monthly": monthly_usage_block},
            "quota": None
            if limit == 0
            else {"limit": limit, "remaining": remaining, "reset": int(reset_at)},
        },
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
    response.headers.setdefault("X-Profile-Id", profile_id)
    _append_usage(
        {
            "ts": time.time(),
            "event": "settle",
            "api_key": x_api_key,
            "N": N,
            "D": D,
            "units": units,
            "duration_ms": t_settle,
            "quota": None
            if limit == 0
            else {"limit": limit, "remaining": remaining, "reset": int(reset_at)},
            "monthly": monthly_usage_block,
        }
    )
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
            raise HTTPException(
                status_code=403, detail="diffusion gating not enabled for this tier"
            )
    lat, N, D, k_eff, eff_params, profile_id = _build_lattice(req, x_api_key)
    units = N * D
    # Enforce monthly/quota BEFORE doing any compute to prevent free riding via failures after compute
    monthly_ctx = _check_monthly_cap(x_api_key, units)
    remaining, limit, reset_at = _check_and_consume_quota(x_api_key, units)
    t0 = time.time()
    settle_stats = lat.settle(
        dt=req.options.dt, max_iters=req.options.max_iters, tol=req.options.tol
    )
    elapsed = time.time() - t0
    SETTLE_COUNTER.labels(status="ok").inc()
    SETTLE_LATENCY.observe(elapsed)
    SETTLE_N_GAUGE.set(N)
    SETTLE_D_GAUGE.set(D)
    rec = lat.receipt()
    # Learning hook (best-effort)
    try:
        record_observation(
            x_api_key,
            profile_id,
            {
                "lamG": eff_params["lamG"],
                "lamC": eff_params["lamC"],
                "lamQ": eff_params["lamQ"],
                "kneighbors": k_eff,
            },
            {
                "duration_ms": 1000.0 * elapsed,
                "iters": int(settle_stats.get("iters", 0)),
                "residual": float(settle_stats.get("res", 0.0)),
                "tol": float(req.options.tol),
            },
        )
    except Exception:
        pass
    headers = _quota_headers(remaining, limit, reset_at)
    if monthly_ctx:
        response.headers.setdefault("X-Monthly-Cap", str(monthly_ctx["limit"]))
        response.headers.setdefault("X-Monthly-Used", str(monthly_ctx["used"]))
        response.headers.setdefault("X-Monthly-Remaining", str(monthly_ctx["remaining"]))
        response.headers.setdefault("X-Monthly-Period", str(monthly_ctx["period"]))
    for k, v in headers.items():
        response.headers.setdefault(k, v)
    response.headers.setdefault("X-Profile-Id", profile_id)
    _append_usage(
        {
            "ts": time.time(),
            "event": "receipt",
            "api_key": x_api_key,
            "N": N,
            "D": D,
            "units": units,
            "duration_ms": 1000.0 * elapsed,
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
    return {
        "state_sig": rec.get("meta", {}).get("state_sig"),
        "receipt": rec,
        "timings_ms": {"total_settle_ms": 1000.0 * elapsed},
        "meta": {
            "N": N,
            "D": D,
            "kneighbors_requested": req.params.kneighbors,
            "kneighbors_effective": k_eff,
            "profile_id": profile_id,
            "request_id": request.headers.get(REQUEST_ID_HEADER, ""),
            "usage": {
                "nodes": N,
                "node_dim_units": units,
                "monthly": None
                if not monthly_ctx
                else {
                    "limit": monthly_ctx["limit"],
                    "used": monthly_ctx["used"],
                    "remaining": monthly_ctx["remaining"],
                    "period": monthly_ctx["period"],
                },
            },
            "quota": None
            if limit == 0
            else {"limit": limit, "remaining": remaining, "reset": int(reset_at)},
        },
    }


@app.post(f"/{_API_VERSION}/bundle")
def bundle(req: SettleRequest, request: Request, response: Response, ctx=Depends(feature_context)):
    """Return only the bundle (requires options.bundle_k)."""
    x_api_key = ctx["api_key"]
    feats = ctx["features"]
    _check_diffusion_allowed(req, feats)
    if not req.options.bundle_k:
        raise HTTPException(status_code=400, detail="options.bundle_k must be set for /bundle")
    lat, N, D, k_eff, eff_params, profile_id = _build_lattice(req, x_api_key)
    units = N * D
    # Quota + monthly first (no compute before cost authorization)
    monthly_ctx = _check_monthly_cap(x_api_key, units)
    remaining, limit, reset_at = _check_and_consume_quota(x_api_key, units)
    # N/D gauges are per-request characteristics; set before compute/helper
    SETTLE_N_GAUGE.set(N)
    SETTLE_D_GAUGE.set(D)
    b, elapsed, cache_status, state_sig, cache_rec = _compute_bundle_with_cache(
        lat,
        req.options.bundle_k,
        x_api_key,
        profile_id,
        eff_params,
        req.options.dt,
        req.options.max_iters,
        req.options.tol,
    )
    headers = _quota_headers(remaining, limit, reset_at)
    if monthly_ctx:
        response.headers.setdefault("X-Monthly-Cap", str(monthly_ctx["limit"]))
        response.headers.setdefault("X-Monthly-Used", str(monthly_ctx["used"]))
        response.headers.setdefault("X-Monthly-Remaining", str(monthly_ctx["remaining"]))
        response.headers.setdefault("X-Monthly-Period", str(monthly_ctx["period"]))
    for k, v in headers.items():
        response.headers.setdefault(k, v)
    response.headers.setdefault("X-Profile-Id", profile_id)
    # Cache headers
    try:
        response.headers.setdefault("X-Cache", cache_status)
        if cache_status == "HIT" and cache_rec is not None:
            response.headers.setdefault("X-Cache-Hits", str(int(cache_rec.get("hits", 0))))
            age = time.time() - float(cache_rec.get("created", time.time()))
            response.headers.setdefault("X-Cache-Age", str(int(age)))
    except Exception:
        pass
    _append_usage(
        {
            "ts": time.time(),
            "event": "bundle",
            "api_key": x_api_key,
            "N": N,
            "D": D,
            "units": units,
            "duration_ms": 1000.0 * elapsed,
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
    return {
        "state_sig": state_sig,
        "bundle": b,
        "timings_ms": {"total_settle_ms": 1000.0 * elapsed},
        "meta": {
            "N": N,
            "D": D,
            "kneighbors_requested": req.params.kneighbors,
            "kneighbors_effective": k_eff,
            "profile_id": profile_id,
            "request_id": request.headers.get(REQUEST_ID_HEADER, ""),
            "usage": {
                "nodes": N,
                "node_dim_units": units,
                "monthly": None
                if not monthly_ctx
                else {
                    "limit": monthly_ctx["limit"],
                    "used": monthly_ctx["used"],
                    "remaining": monthly_ctx["remaining"],
                    "period": monthly_ctx["period"],
                },
            },
            "quota": None
            if limit == 0
            else {"limit": limit, "remaining": remaining, "reset": int(reset_at)},
        },
    }


## Removed earlier draft Stripe webhook stub; consolidated full implementation later in file.


@app.post(f"/{_API_VERSION}/chain/receipt")
def chain_receipt(
    req: SettleRequest, request: Request, response: Response, ctx=Depends(feature_context)
):
    """Return settle plus chain receipt (requires chain)."""
    x_api_key = ctx["api_key"]
    feats = ctx["features"]
    if req.gates is not None:
        if os.getenv("OSCILLINK_DIFFUSION_GATES_ENABLED", "1") not in {"1", "true", "TRUE", "on"}:
            raise HTTPException(status_code=403, detail="diffusion gating temporarily disabled")
        if not feats.diffusion_allowed:
            raise HTTPException(
                status_code=403, detail="diffusion gating not enabled for this tier"
            )
    if not req.chain:
        raise HTTPException(status_code=400, detail="chain must be provided")
    lat, N, D, k_eff, eff_params, profile_id = _build_lattice(req, x_api_key)
    units = N * D
    # Enforce billing constraints prior to compute
    monthly_ctx = _check_monthly_cap(x_api_key, units)
    remaining, limit, reset_at = _check_and_consume_quota(x_api_key, units)
    t0 = time.time()
    settle_stats = lat.settle(
        dt=req.options.dt, max_iters=req.options.max_iters, tol=req.options.tol
    )
    elapsed = time.time() - t0
    SETTLE_COUNTER.labels(status="ok").inc()
    SETTLE_LATENCY.observe(elapsed)
    SETTLE_N_GAUGE.set(N)
    SETTLE_D_GAUGE.set(D)
    rec = lat.chain_receipt(req.chain)
    # Learning hook (best-effort)
    try:
        record_observation(
            x_api_key,
            profile_id,
            {
                "lamG": eff_params["lamG"],
                "lamC": eff_params["lamC"],
                "lamQ": eff_params["lamQ"],
                "kneighbors": k_eff,
            },
            {
                "duration_ms": 1000.0 * elapsed,
                "iters": int(settle_stats.get("iters", 0)),
                "residual": float(settle_stats.get("res", 0.0)),
                "tol": float(req.options.tol),
            },
        )
    except Exception:
        pass
    headers = _quota_headers(remaining, limit, reset_at)
    if monthly_ctx:
        response.headers.setdefault("X-Monthly-Cap", str(monthly_ctx["limit"]))
        response.headers.setdefault("X-Monthly-Used", str(monthly_ctx["used"]))
        response.headers.setdefault("X-Monthly-Remaining", str(monthly_ctx["remaining"]))
        response.headers.setdefault("X-Monthly-Period", str(monthly_ctx["period"]))
    for k, v in headers.items():
        response.headers.setdefault(k, v)
    response.headers.setdefault("X-Profile-Id", profile_id)
    _append_usage(
        {
            "ts": time.time(),
            "event": "chain_receipt",
            "api_key": x_api_key,
            "N": N,
            "D": D,
            "units": units,
            "duration_ms": 1000.0 * elapsed,
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
    return {
        "state_sig": lat._signature(),
        "chain_receipt": rec,
        "timings_ms": {"total_settle_ms": 1000.0 * elapsed},
        "meta": {
            "N": N,
            "D": D,
            "kneighbors_requested": req.params.kneighbors,
            "kneighbors_effective": k_eff,
            "profile_id": profile_id,
            "request_id": request.headers.get(REQUEST_ID_HEADER, ""),
            "usage": {
                "nodes": N,
                "node_dim_units": units,
                "monthly": None
                if not monthly_ctx
                else {
                    "limit": monthly_ctx["limit"],
                    "used": monthly_ctx["used"],
                    "remaining": monthly_ctx["remaining"],
                    "period": monthly_ctx["period"],
                },
            },
            "quota": None
            if limit == 0
            else {"limit": limit, "remaining": remaining, "reset": int(reset_at)},
        },
    }


@app.get("/metrics")
def metrics(request: Request, x_admin_secret: str | None = Header(default=None)):
    # Optional protection: require admin secret for metrics only when enabled AND a secret is set
    if _truthy(os.getenv("OSCILLINK_METRICS_PROTECTED", "0")):
        required = os.getenv("OSCILLINK_ADMIN_SECRET")
        if required and x_admin_secret != required:
            return ORJSONResponse(status_code=403, content={"detail": "forbidden"})
    data = generate_latest()  # type: ignore
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


def _new_api_key() -> str:
    # Generate a URL-safe API key (32 bytes -> 43 chars base64url). Prefix for readability.
    try:
        import secrets

        return "ok_" + secrets.token_urlsafe(32)
    except Exception:
        return "ok_" + uuid.uuid4().hex


## CLI signup sessions moved to services.cli


## Billing helpers moved to cloud.app.services.billing


@app.post("/billing/cli/start")
def billing_cli_start(
    request: Request,
    response: Response,
    tier: str = "beta",
    email: str | None = None,
):  # pragma: no cover - external dependency path
    """Start a CLI signup session and return a Checkout URL plus a short code.

    The user completes checkout in a browser; the server provisions an API key on webhook
    and makes it available for retrieval via /billing/cli/poll/{code}.
    """
    # Pick a price for the requested tier using the configured price->tier map
    pmap = get_price_map()  # maps price_id -> tier
    price_id = next((pid for pid, t in pmap.items() if t == tier), None)
    if not price_id:
        raise HTTPException(status_code=400, detail=f"no Stripe price configured for tier '{tier}'")
    stripe_secret = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")
    if not stripe_secret:
        raise HTTPException(status_code=503, detail="billing not configured")
    # Per-endpoint rate limit (feature-flag via env)
    _apply_eprl_headers(
        request, response, "OSCILLINK_EPRL_CLI_START_LIMIT", "OSCILLINK_EPRL_CLI_START_WINDOW"
    )
    code = cli_service.new_code()
    try:
        import stripe  # type: ignore

        stripe.api_key = stripe_secret
        stripe.api_version = "2024-06-20"
        success_url = os.getenv("OSCILLINK_CLI_SUCCESS_URL", "https://oscillink.com/thanks")
        # Create a Checkout Session with the code embedded for correlation
        _kwargs = dict(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            client_reference_id=code,
            metadata={"cli_code": code},
            success_url=success_url,
            allow_promotion_codes=True,
        )
        if email:
            _kwargs["customer_email"] = email
        sess = stripe.checkout.Session.create(**_kwargs)  # type: ignore
        cli_service.purge_expired()
        cli_service.set_session(
            code,
            {
                "status": "pending",
                "created": time.time(),
                "session_id": getattr(sess, "id", None) or sess.get("id"),
                "tier": tier,
                "email": email,
            },
        )
        try:
            CLI_SESSIONS_CREATED.inc()
            # Best-effort: track active sessions in memory mode
            CLI_SESSIONS_ACTIVE.inc()
        except Exception:
            pass
        return {
            "code": code,
            "checkout_url": getattr(sess, "url", None) or sess.get("url"),
            "expires_in": cli_service.ttl_seconds(),
        }
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=501, detail="stripe library not installed") from exc
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to start CLI session: {e}") from e


@app.get("/billing/cli/poll/{code}")
def billing_cli_poll(code: str, request: Request, response: Response):
    """Poll the status of a CLI signup session and return the API key when ready."""
    # Per-endpoint rate limit (feature-flag via env)
    _apply_eprl_headers(
        request, response, "OSCILLINK_EPRL_CLI_POLL_LIMIT", "OSCILLINK_EPRL_CLI_POLL_WINDOW"
    )
    cli_service.purge_expired()
    rec = cli_service.get_session(code)
    if not rec:
        return {"status": "expired"}
    if rec.get("status") == "provisioned":
        # Mark as claimed to allow eventual purge
        cli_service.update_session(code, {"status": "claimed"})
        try:
            CLI_SESSIONS_PROVISIONED.inc()
            CLI_SESSIONS_ACTIVE.dec()
        except Exception:
            pass
        return {"status": "ready", "api_key": rec.get("api_key"), "tier": rec.get("tier")}
    return {"status": "pending"}


@app.get("/billing/success")
def billing_success(session_id: str | None = None):  # pragma: no cover - external dependency path
    """Stripe Checkout success landing page."""
    if not session_id:
        return HTMLResponse(
            status_code=400,
            content=(
                "<html><body><h2>Missing session</h2><p>session_id is required. "
                "If you reached this page from Stripe Checkout, contact support.</p></body></html>"
            ),
        )
    try:
        _ = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")
        if not _:
            return HTMLResponse(
                status_code=503,
                content=(
                    "<html><body><h2>Billing not configured</h2>"
                    "<p>Stripe secret not set on server. Your payment likely succeeded, but we can't "
                    "provision a key automatically. Please email contact@oscillink.com with your receipt.</p>"
                    "</body></html>"
                ),
            )
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
        return HTMLResponse(
            status_code=501,
            content=(
                "<html><body><h2>Stripe library not installed</h2>"
                "<p>Server cannot retrieve your session. We will email your key shortly.</p></body></html>"
            ),
        )
    except Exception as e:
        return HTMLResponse(
            status_code=400,
            content=(
                f"<html><body><h2>Checkout session error</h2><p>{str(e)}</p>"
                "<p>If this persists, contact support with your receipt.</p></body></html>"
            ),
        )


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


## Job submission and management endpoints moved to cloud.app.jobs router


# Admin endpoints moved to cloud.app.admin router


# CLI entrypoint for uvicorn
# uvicorn cloud.app.main:app --reload --port 8000
