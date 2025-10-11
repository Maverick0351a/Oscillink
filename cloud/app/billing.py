"""Billing / subscription tier utilities.

Maps Stripe price IDs to internal tiers and provides helpers to resolve tier
from subscription events.

Environment configuration:
- OSCILLINK_STRIPE_PRICE_MAP: JSON string or semicolon list mapping price_id->tier
    Examples:
      '{"price_123":"free","price_456":"pro"}'
      'price_123:free;price_456:pro;price_789:enterprise'

If a price ID is not present, a default tier (free) is returned.

Future extensions: feature allowances, unit caps per tier.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

DEFAULT_TIER = "free"


@dataclass
class TierInfo:
    name: str
    # Month cap (node_dim_units) None = unlimited
    monthly_unit_cap: Optional[int] = None
    diffusion_allowed: bool = False
    requires_manual_activation: bool = False


# Static fallback tier catalog; real caps can be refined later
TIER_CATALOG: Dict[str, TierInfo] = {
    "free": TierInfo(name="free", monthly_unit_cap=5_000_000, diffusion_allowed=False),
    # Beta access: lower price, hard monthly cap, diffusion allowed; support is limited operationally (docs-level)
    "beta": TierInfo(name="beta", monthly_unit_cap=25_000_000, diffusion_allowed=True),
    "pro": TierInfo(name="pro", monthly_unit_cap=50_000_000, diffusion_allowed=True),
    # Enterprise requires manual activation by an admin (pending until approved)
    "enterprise": TierInfo(
        name="enterprise",
        monthly_unit_cap=None,
        diffusion_allowed=True,
        requires_manual_activation=True,
    ),
}


def _parse_price_map(raw: str) -> Dict[str, str]:
    if not raw:
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            return {str(k): str(v) for k, v in data.items()}
        except Exception:
            return {}
    # Fallback semicolon list
    mapping: Dict[str, str] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pid, tier = part.split(":", 1)
            mapping[pid.strip()] = tier.strip()
    return mapping


# Cached mapping
_price_map_cache: Dict[str, str] | None = None

# Built-in fallback price mapping so tests and dev env work without configuring
# OSCILLINK_STRIPE_PRICE_MAP. Environment mapping (when present) overrides these.
# NOTE: Keep keys stable; changing them will break existing subscriptions.
_DEFAULT_PRICE_MAP: Dict[str, str] = {
    # Beta monthly subscription price id (placeholder ID used in tests/dev)
    "price_cloud_beta_monthly": "beta",
    # Pro monthly subscription price id (placeholder ID used in tests)
    "price_cloud_pro_monthly": "pro",
    # Enterprise subscription price id (requires manual activation -> pending status initially)
    "price_cloud_enterprise": "enterprise",
}


def get_price_map(refresh: bool = False) -> Dict[str, str]:
    global _price_map_cache
    if _price_map_cache is None or refresh:
        env_val = os.getenv("OSCILLINK_STRIPE_PRICE_MAP", "")
        env_map = _parse_price_map(env_val)
        # Merge built-in defaults with env overrides (env wins)
        merged: Dict[str, str] = {**_DEFAULT_PRICE_MAP, **env_map}
        _price_map_cache = merged
    return _price_map_cache


def tier_for_price(price_id: str) -> str:
    return get_price_map().get(price_id, DEFAULT_TIER)


def tier_info(name: str) -> TierInfo:
    return TIER_CATALOG.get(name, TIER_CATALOG[DEFAULT_TIER])


def resolve_tier_from_subscription(sub: dict) -> str:
    """Derive tier from a Stripe subscription object.

    Strategy:
      - Inspect subscription['items']['data'] list; take first active item's price.id
      - Map price.id via OSCILLINK_STRIPE_PRICE_MAP
      - Fallback to DEFAULT_TIER when missing
    """
    try:
        items = sub.get("items", {}).get("data", [])  # type: ignore
        if not items:
            return DEFAULT_TIER
        first = items[0]
        price = first.get("price", {})
        pid = price.get("id")
        if not pid:
            return DEFAULT_TIER
        return tier_for_price(pid)
    except Exception:
        return DEFAULT_TIER


def current_period() -> str:
    """Return current billing period identifier (YYYYMM)."""
    import datetime as _dt

    now = _dt.datetime.utcnow()
    return f"{now.year:04d}{now.month:02d}"
