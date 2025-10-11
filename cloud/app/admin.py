from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException

from .billing import TIER_CATALOG, get_price_map, tier_info
from .keystore import get_keystore
from .models import AdminKeyResponse, AdminKeyUpdate
from .services.webhook_mem import get_webhook_events_mem

router = APIRouter()


def _admin_guard(x_admin_secret: str | None = Header(default=None)):
    required = os.getenv("OSCILLINK_ADMIN_SECRET")
    if not required:
        raise HTTPException(status_code=503, detail="admin secret not configured")
    if x_admin_secret != required:
        raise HTTPException(status_code=401, detail="invalid admin secret")
    return True


@router.get("/admin/keys/{api_key}", response_model=AdminKeyResponse)
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


@router.put("/admin/keys/{api_key}", response_model=AdminKeyResponse)
def admin_put_key(api_key: str, payload: AdminKeyUpdate, auth=Depends(_admin_guard)):
    ks = get_keystore()
    fields = (
        payload.model_dump(exclude_unset=True)
        if hasattr(payload, "model_dump")
        else payload.dict(exclude_unset=True)
    )
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


@router.get("/admin/webhook/events")
def admin_list_webhook_events(limit: int = 50, auth=Depends(_admin_guard)):
    from .main import app as _app

    lim = max(1, min(limit, 500))
    events_mem = get_webhook_events_mem(_app)
    events = list(events_mem.values())
    events.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return {"events": events[:lim], "count": len(events), "returned": len(events[:lim])}


@router.get("/admin/billing/price-map")
def admin_get_price_map(auth=Depends(_admin_guard)):
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


@router.get("/admin/usage/{api_key}")
def admin_get_usage(api_key: str, auth=Depends(_admin_guard)):
    from .main import _key_usage, _monthly_usage

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
    mu = _monthly_usage.get(api_key)
    monthly = None
    if mu:
        from .keystore import KeyMetadata  # local type import

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


@router.post("/admin/billing/cancel/{api_key}")
def admin_cancel_subscription(
    api_key: str, immediate: bool | None = None, auth=Depends(_admin_guard)
):
    from .main import _fs_get_customer_mapping

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
        do_immediate = (
            bool(immediate)
            if immediate is not None
            else (
                os.getenv("OSCILLINK_STRIPE_CANCEL_IMMEDIATE", "0") in {"1", "true", "TRUE", "on"}
            )
        )
        if do_immediate:
            stripe.Subscription.delete(sub_id)  # type: ignore
            status = "cancelled"
        else:
            stripe.Subscription.modify(sub_id, cancel_at_period_end=True)  # type: ignore
            status = "cancel_at_period_end"
        ks = get_keystore()
        ks.update(api_key, status="suspended")
        return {"api_key": api_key, "subscription_id": sub_id, "status": status}
    except HTTPException:
        raise
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=501, detail="stripe library not installed") from exc
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to cancel subscription: {e}") from e
