from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from cloud.app.redis_backend import get_redis

_CLI_SESSIONS: dict[str, dict[str, Any]] = {}


def _backend() -> str:
    return os.getenv("OSCILLINK_CLI_SESSIONS_BACKEND", "memory").lower()


def get_sessions() -> dict[str, dict[str, Any]]:
    """Return the in-memory sessions mapping.

    Only meaningful in memory mode. In redis mode, prefer get_session/set_session/update_session.
    """
    return _CLI_SESSIONS


def ttl_seconds() -> int:
    try:
        return int(os.getenv("OSCILLINK_CLI_TTL", "900"))
    except Exception:
        return 900


def new_code() -> str:
    try:
        import secrets

        return secrets.token_hex(4)
    except Exception:
        return uuid.uuid4().hex[:8]


def _redis_key(code: str) -> str:
    return f"cli:{code}"


def set_session(code: str, record: dict[str, Any]) -> bool:
    """Create or replace a CLI session record.

    Returns True on success.
    """
    record = dict(record)
    record.setdefault("created", time.time())
    if _backend() == "redis":
        r = get_redis()
        if not r:
            # Fallback to memory
            _CLI_SESSIONS[code] = record
            return True
        ttl = ttl_seconds()
        try:
            r.set(_redis_key(code), json.dumps(record), ex=ttl)
            return True
        except Exception:
            return False
    _CLI_SESSIONS[code] = record
    return True


def get_session(code: str) -> dict[str, Any] | None:
    if _backend() == "redis":
        r = get_redis()
        if not r:
            return _CLI_SESSIONS.get(code)
        try:
            val = r.get(_redis_key(code))
            if val is None:
                return None
            rec = json.loads(val)
            if isinstance(rec, dict):
                return rec
        except Exception:
            return None
        return None
    return _CLI_SESSIONS.get(code)


def session_exists(code: str) -> bool:
    if _backend() == "redis":
        r = get_redis()
        if not r:
            return code in _CLI_SESSIONS
        try:
            return bool(r.exists(_redis_key(code)))
        except Exception:
            return False
    return code in _CLI_SESSIONS


def update_session(code: str, patch: dict[str, Any]) -> bool:
    """Merge patch into session record if present.

    Returns True if updated; False if session missing or error.
    """
    if _backend() == "redis":
        r = get_redis()
        if not r:
            if code in _CLI_SESSIONS:
                _CLI_SESSIONS[code].update(patch)
                return True
            return False
        try:
            key = _redis_key(code)
            val = r.get(key)
            if val is None:
                return False
            rec = json.loads(val) if val else {}
            if not isinstance(rec, dict):
                rec = {}
            rec.update(patch)
            # Preserve existing TTL by reading it; if unavailable, reset to default
            try:
                ttl = int(r.ttl(key))
                # If TTL missing, negative, or rounded down to 0, reset to default
                if ttl is None or ttl <= 0:
                    ttl = ttl_seconds()
            except Exception:
                ttl = ttl_seconds()
            r.set(key, json.dumps(rec), ex=ttl)
            return True
        except Exception:
            return False
    if code in _CLI_SESSIONS:
        _CLI_SESSIONS[code].update(patch)
        return True
    return False


def purge_expired() -> None:
    if _backend() == "redis":
        # Redis expiry is managed by EX; nothing to purge explicitly when Redis is available.
        # If Redis is not available (fallback path), proceed with in-memory purge.
        r = get_redis()
        if r:
            return
    now = time.time()
    ttl = ttl_seconds()
    expired = [
        c
        for c, rec in list(_CLI_SESSIONS.items())
        if now - rec.get("created", now) > ttl or rec.get("status") == "claimed"
    ]
    for c in expired:
        _CLI_SESSIONS.pop(c, None)
