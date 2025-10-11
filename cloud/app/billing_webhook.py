from __future__ import annotations

import hashlib
import json
import os
import time

from fastapi import APIRouter, HTTPException, Request

# Local imports safe at module-level (do not import main here to avoid cycles)
from cloud.app.billing import resolve_tier_from_subscription, tier_info
from cloud.app.keystore import get_keystore

router = APIRouter()


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request):  # noqa: C901
    # Lazy import from main to avoid circular import during app startup
    from .main import (  # noqa: E402
        CLI_SESSIONS_PROVISIONED,
        STRIPE_WEBHOOK_EVENTS,
        _webhook_get,
        _webhook_store,
    )
    from .services import cli as cli_service  # noqa: E402
    from .services.billing import (  # noqa: E402
        fs_set_customer_mapping as _fs_set_customer_mapping,
    )
    from .services.billing import (
        provision_key_for_subscription as _provision_key_for_subscription,
    )
    from .services.billing import (
        send_key_email as _send_key_email,
    )
    from .services.billing import (
        stripe_fetch_session_and_subscription as _stripe_fetch_session_and_subscription,
    )

    secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    body = await request.body()
    payload_text = body.decode("utf-8", errors="replace")
    event = None
    allow_unverified = os.getenv("OSCILLINK_ALLOW_UNVERIFIED_STRIPE", "0") in {
        "1",
        "true",
        "TRUE",
        "on",
    }
    if allow_unverified:
        try:
            print(
                "[warn] OSCILLINK_ALLOW_UNVERIFIED_STRIPE is enabled — webhook signature verification may be bypassed. Do NOT enable in production."
            )
        except Exception:
            pass
    verified = False
    if secret:
        sig_header = request.headers.get("stripe-signature")
        if not sig_header:
            if allow_unverified:
                try:
                    event = json.loads(payload_text)
                except Exception as err:
                    raise HTTPException(status_code=400, detail="invalid JSON payload") from err
            else:
                raise HTTPException(status_code=400, detail="missing stripe-signature header")
        try:
            max_age = int(os.getenv("OSCILLINK_STRIPE_MAX_AGE", "300"))
        except ValueError:
            max_age = 300
        if max_age > 0 and sig_header:
            try:
                parts = {
                    kv.split("=")[0]: kv.split("=")[1] for kv in sig_header.split(",") if "=" in kv
                }
                if "t" in parts:
                    ts = int(parts["t"])
                    now = int(time.time())
                    if now - ts > max_age and not allow_unverified:
                        raise HTTPException(status_code=400, detail="webhook timestamp too old")
            except HTTPException:
                raise
            except Exception:
                pass
        if allow_unverified:
            try:
                event = json.loads(payload_text)
                verified = False
            except Exception as err:
                raise HTTPException(status_code=400, detail="invalid JSON payload") from err
        else:
            try:  # pragma: no cover
                import stripe  # type: ignore

                stripe.api_version = "2024-06-20"
                event = stripe.Webhook.construct_event(payload_text, sig_header, secret)
                verified = True
            except ModuleNotFoundError:
                try:
                    event = json.loads(payload_text)
                except Exception as err:
                    raise HTTPException(
                        status_code=400, detail="invalid JSON payload (no stripe lib)"
                    ) from err
            except Exception as e:
                if os.getenv("OSCILLINK_ALLOW_UNVERIFIED_STRIPE", "0") in {
                    "1",
                    "true",
                    "TRUE",
                    "on",
                }:
                    try:
                        event = json.loads(payload_text)
                        verified = False
                    except Exception as err:
                        raise HTTPException(status_code=400, detail="invalid JSON payload") from err
                else:
                    raise HTTPException(
                        status_code=400, detail=f"signature verification failed: {e}"
                    ) from e
    else:
        try:
            event = json.loads(payload_text)
        except Exception as err:
            raise HTTPException(status_code=400, detail="invalid JSON payload") from err

    etype = (
        event.get("type", "unknown")
        if isinstance(event, dict)
        else getattr(event, "type", "unknown")
    )
    event_id = event.get("id") if isinstance(event, dict) else getattr(event, "id", None)
    if not event_id:
        raise HTTPException(status_code=400, detail="event missing id")

    existing = _webhook_get(event_id)
    if existing:
        try:
            STRIPE_WEBHOOK_EVENTS.labels(result="duplicate").inc()  # type: ignore
        except Exception:
            pass
        return {
            "received": True,
            "id": event_id,
            "type": etype,
            "processed": False,
            "duplicate": True,
            "note": "duplicate ignored",
        }

    processed = False
    note = None
    if etype.startswith("customer.subscription."):
        sub_obj = event.get("data", {}).get("object", {}) if isinstance(event, dict) else {}
        api_key = None
        try:
            metadata = sub_obj.get("metadata", {}) or {}
            api_key = metadata.get("api_key")
        except Exception:
            api_key = None
        if api_key:
            ks = get_keystore()
            if not verified and secret and not allow_unverified:
                note = "signature not verified; subscription event ignored"
            else:
                if etype in {"customer.subscription.created", "customer.subscription.updated"}:
                    new_tier = resolve_tier_from_subscription(sub_obj)
                    tinfo = tier_info(new_tier)
                    status = (
                        "pending"
                        if getattr(tinfo, "requires_manual_activation", False)
                        else "active"
                    )
                    ks.update(
                        api_key,
                        create=True,
                        tier=new_tier,
                        status=status,
                        features={"diffusion_gates": tinfo.diffusion_allowed},
                    )
                    processed = True
                    note = f"tier set to {new_tier} (status={status})"
                elif etype in {"customer.subscription.deleted", "customer.subscription.cancelled"}:
                    ks.update(api_key, status="suspended")
                    processed = True
                    note = "subscription cancelled; key suspended"
        else:
            note = "subscription missing api_key metadata"
    elif etype == "checkout.session.completed":
        sess_obj = event.get("data", {}).get("object", {}) if isinstance(event, dict) else {}
        email = None
        try:
            email = (sess_obj.get("customer_details", {}) or {}).get("email") or sess_obj.get(
                "customer_email"
            )
        except Exception:
            email = None
        stripe_secret = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")
        if not stripe_secret:
            note = "stripe secret not set; cannot provision on webhook"
        elif secret and not verified and not allow_unverified:
            note = "signature not verified; checkout session ignored"
        else:
            try:  # pragma: no cover
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
                try:
                    cli_code = (
                        session.get("client_reference_id") if isinstance(session, dict) else None
                    ) or (
                        (session.get("metadata") or {}).get("cli_code")
                        if isinstance(session, dict)
                        else None
                    )
                except Exception:
                    cli_code = None
                if cli_code:
                    cli_service.update_session(
                        cli_code,
                        {
                            "status": "provisioned",
                            "api_key": api_key,
                            "tier": new_tier,
                            "updated": time.time(),
                        },
                    )
                    try:
                        CLI_SESSIONS_PROVISIONED.inc()
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
        "api_key": api_key if "api_key" in locals() else None,
        "payload_sha256": hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
        "freshness_max_age": os.getenv("OSCILLINK_STRIPE_MAX_AGE", "300"),
    }
    _webhook_store(event_id, record)
    try:
        STRIPE_WEBHOOK_EVENTS.labels(result="processed" if processed else "ignored").inc()  # type: ignore
    except Exception:
        pass
    return record
