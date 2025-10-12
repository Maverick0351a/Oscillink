"""Tier -> feature resolution.

Simple static mapping plus a resolver that merges:
1. Static defaults by tier
2. Per-key metadata.features overrides

Future: dynamic override doc in Firestore.
"""

from __future__ import annotations

from .keystore import KeyMetadata

# Static map (align with docs/FIRESTORE_USAGE_MODEL.md & STRIPE_INTEGRATION.md)
TIER_FEATURES: dict[str, dict[str, bool]] = {
    "free": {
        "diffusion_gates": False,
        "async_jobs": True,
        "signed_usage": False,
        "priority_queue": False,
    },
    "beta": {
        "diffusion_gates": True,
        "async_jobs": True,
        "signed_usage": True,
        "priority_queue": False,
    },
    "pro": {
        "diffusion_gates": True,
        "async_jobs": True,
        "signed_usage": True,
        "priority_queue": False,
    },
    "enterprise": {
        "diffusion_gates": True,
        "async_jobs": True,
        "signed_usage": True,
        "priority_queue": True,
    },
}

DEFAULT_TIER = "free"


class FeatureBundle(dict):
    @property
    def diffusion_allowed(self) -> bool:
        return bool(self.get("diffusion_gates"))


def resolve_features(meta: KeyMetadata | None) -> FeatureBundle:
    tier = DEFAULT_TIER
    if meta:
        tier = meta.tier or DEFAULT_TIER
    base = dict(TIER_FEATURES.get(tier, TIER_FEATURES[DEFAULT_TIER]))
    # Overlay explicit per-key overrides
    if meta and meta.features:
        base.update(meta.features)
    # Overlay environment-driven feature flags (e.g., from licensed container entitlements)
    import os as _os

    try:
        for k in list(base.keys()):
            env_name = f"OSCILLINK_FEAT_{str(k).upper()}"
            v = _os.getenv(env_name)
            if v is None:
                continue
            base[k] = v in {"1", "true", "TRUE", "on", "On", "yes", "YES"}
    except Exception:
        # Safety: never fail feature resolution due to env parsing issues
        pass
    fb = FeatureBundle(base)
    fb["tier"] = tier
    return fb
