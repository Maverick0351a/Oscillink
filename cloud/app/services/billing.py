from __future__ import annotations

import os
import smtplib
import time
from email.message import EmailMessage

from cloud.app.billing import resolve_tier_from_subscription, tier_info
from cloud.app.keystore import get_keystore


# --- Optional Firestore mapping: api_key -> (stripe_customer_id, subscription_id) ---
def fs_get_customer_mapping(api_key: str):  # pragma: no cover - external dependency
    collection = os.getenv("OSCILLINK_CUSTOMERS_COLLECTION", "").strip()
    if not collection:
        return None
    try:
        from google.cloud import firestore  # type: ignore

        client = firestore.Client()
        snap = client.collection(collection).document(api_key).get()
        if snap.exists:
            return snap.to_dict() or None
    except Exception:
        return None
    return None


def fs_set_customer_mapping(
    api_key: str, customer_id: str | None, subscription_id: str | None
):  # pragma: no cover - external dependency
    collection = os.getenv("OSCILLINK_CUSTOMERS_COLLECTION", "").strip()
    if not collection or not api_key or not (customer_id or subscription_id):
        return
    try:
        from google.cloud import firestore  # type: ignore

        client = firestore.Client()
        doc_ref = client.collection(collection).document(api_key)
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
        pass


def stripe_fetch_session_and_subscription(session_id: str) -> tuple[dict, dict]:  # pragma: no cover
    secret = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")
    if not secret:
        raise RuntimeError("stripe secret not configured")
    import stripe  # type: ignore

    stripe.api_key = secret
    stripe.api_version = "2024-06-20"
    session = stripe.checkout.Session.retrieve(session_id, expand=["subscription", "customer"])  # type: ignore
    if not session:
        raise ValueError("session not found")
    sub = (
        session.get("subscription")
        if isinstance(session, dict)
        else getattr(session, "subscription", None)
    )
    if isinstance(sub, str):
        sub = stripe.Subscription.retrieve(sub)  # type: ignore
    if not isinstance(sub, dict):
        sub_id = session.get("subscription") if isinstance(session, dict) else None
        if sub_id:
            sub = stripe.Subscription.retrieve(sub_id)  # type: ignore
    if not isinstance(sub, dict):
        raise ValueError("subscription not found for session")
    return session, sub


def send_key_email(to_email: str, api_key: str, tier: str, status: str) -> bool:
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
            return False
    return False


def provision_key_for_subscription(sub: dict) -> tuple[str, str, str]:
    meta = sub.get("metadata", {}) or {}
    api_key = meta.get("api_key") if isinstance(meta, dict) else None
    new_tier = resolve_tier_from_subscription(sub)
    tinfo = tier_info(new_tier)
    status = "pending" if getattr(tinfo, "requires_manual_activation", False) else "active"
    if not api_key:
        api_key = _new_api_key()
        try:
            import stripe  # type: ignore

            stripe.Subscription.modify(sub.get("id"), metadata={**meta, "api_key": api_key})  # type: ignore
        except Exception:
            pass
    ks = get_keystore()
    ks.update(
        api_key,
        create=True,
        tier=new_tier,
        status=status,
        features={"diffusion_gates": tinfo.diffusion_allowed},
    )
    return api_key, new_tier, status


def _new_api_key() -> str:
    try:
        import secrets

        return "ok_" + secrets.token_urlsafe(32)
    except Exception:
        import uuid

        return "ok_" + uuid.uuid4().hex
