from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import Any

_BUNDLE_CACHE: dict[str, OrderedDict[str, dict[str, Any]]] = {}


def cache_enabled() -> bool:
    return os.getenv("OSCILLINK_CACHE_ENABLE", "0").lower() in {"1", "true", "on", "yes"}


def cache_ttl() -> int:
    try:
        return max(1, int(os.getenv("OSCILLINK_CACHE_TTL", "300")))
    except Exception:
        return 300


def cache_cap() -> int:
    try:
        return max(1, int(os.getenv("OSCILLINK_CACHE_CAP", "128")))
    except Exception:
        return 128


def bundle_cache_get(api_key: str | None, sig: str):
    if not (cache_enabled() and api_key and sig):
        return None
    od = _BUNDLE_CACHE.get(api_key)
    if not isinstance(od, OrderedDict):
        return None
    rec = od.get(sig)
    if not rec:
        return None
    try:
        created = float(rec.get("created", 0.0))
    except Exception:
        created = 0.0
    if time.time() - created > cache_ttl():
        try:
            od.pop(sig, None)
        except Exception:
            pass
        return None
    try:
        od.move_to_end(sig)
    except Exception:
        pass
    try:
        rec["hits"] = int(rec.get("hits", 0)) + 1
    except Exception:
        pass
    return rec


def bundle_cache_put(api_key: str | None, sig: str, bundle: list[dict]) -> None:
    if not (cache_enabled() and api_key and sig and isinstance(bundle, list)):
        return
    od = _BUNDLE_CACHE.get(api_key)
    if not isinstance(od, OrderedDict):
        od = OrderedDict()
        _BUNDLE_CACHE[api_key] = od
    od[sig] = {"bundle": bundle, "created": time.time(), "hits": 0}
    od.move_to_end(sig)
    cap = cache_cap()
    while len(od) > cap:
        try:
            od.popitem(last=False)
        except Exception:
            break
