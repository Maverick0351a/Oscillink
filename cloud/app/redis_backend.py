from __future__ import annotations

import os
from typing import Any, Optional

_redis_client: Any = None


def redis_enabled() -> bool:
    return os.getenv("OSCILLINK_STATE_BACKEND", "memory").lower() == "redis"


def get_redis() -> Optional[Any]:
    global _redis_client
    if not redis_enabled():
        return None
    try:  # lazy import
        import redis  # type: ignore
    except Exception:
        return None
    if _redis_client is not None:
        return _redis_client
    url = os.getenv("OSCILLINK_REDIS_URL") or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
    _redis_client = redis.Redis.from_url(url, decode_responses=True)
    try:
        _redis_client.ping()
    except Exception:
        # Treat as disabled if unreachable
        return None
    return _redis_client


def incr_with_window(key: str, window_seconds: int, amount: int = 1) -> tuple[int, int]:
    """Increment a counter with a TTL window and return (count, ttl).

    ttl is seconds remaining in the window (>=0 when active, -1 if no ttl, -2 if missing).
    """
    r = get_redis()
    if not r:
        return 0, -2
    pipe = r.pipeline()
    # INCR then set EXPIRE if new
    pipe.incrby(key, amount)
    pipe.ttl(key)
    vals = pipe.execute()
    count = int(vals[0])
    ttl = int(vals[1])
    if ttl < 0:
        r.expire(key, window_seconds)
        ttl = window_seconds
    return count, ttl


def get_with_ttl(key: str) -> tuple[Optional[str], int]:
    r = get_redis()
    if not r:
        return None, -2
    pipe = r.pipeline()
    pipe.get(key)
    pipe.ttl(key)
    val, ttl = pipe.execute()
    return val, int(ttl)


def set_with_ttl(key: str, value: str, ttl: int) -> bool:
    r = get_redis()
    if not r:
        return False
    try:
        r.set(key, value, ex=ttl)
        return True
    except Exception:
        return False
