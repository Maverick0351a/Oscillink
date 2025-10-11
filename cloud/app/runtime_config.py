from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitConfig:
    limit: int
    window: int


@dataclass(frozen=True)
class QuotaConfig:
    limit: int
    window: int


def get_api_keys_raw() -> str | None:
    # No caching to allow dynamic test + runtime rotation in single-process deployment
    return os.getenv("OSCILLINK_API_KEYS")


def get_api_keys() -> set[str]:
    raw = get_api_keys_raw()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def get_rate_limit() -> RateLimitConfig:
    try:
        return RateLimitConfig(
            limit=int(os.getenv("OSCILLINK_RATE_LIMIT", "0")),
            window=int(os.getenv("OSCILLINK_RATE_WINDOW", "60")),
        )
    except ValueError:
        return RateLimitConfig(limit=0, window=60)


def get_quota_config() -> QuotaConfig:
    try:
        return QuotaConfig(
            limit=int(os.getenv("OSCILLINK_KEY_NODE_UNITS_LIMIT", "0")),
            window=int(os.getenv("OSCILLINK_KEY_NODE_UNITS_WINDOW", "3600")),
        )
    except ValueError:
        return QuotaConfig(limit=0, window=3600)


# Helper to force refresh in tests after monkeypatching


def refresh_runtime_caches():
    # Retained for API compatibility; no-op now that API keys are uncached
    return None
