from __future__ import annotations

import json
import os
from typing import Any

from ..redis_backend import get_with_ttl, redis_enabled, set_with_ttl


def _webhook_events_collection() -> str:
    return os.getenv("OSCILLINK_WEBHOOK_EVENTS_COLLECTION", "").strip()


def webhook_get_persistent(event_id: str) -> dict[str, Any] | None:
    """Fetch a webhook event from persistent stores (Redis/Firestore) if configured.

    Returns a dict or None. Does not consult in-memory state (handled by caller).
    """
    # Redis
    if redis_enabled():
        val, _ttl = get_with_ttl(f"stripe_evt:{event_id}")
        if val:
            try:
                return json.loads(val)
            except Exception:
                return {"id": event_id, "source": "redis"}
    # Firestore (optional)
    coll = _webhook_events_collection()
    if not coll:
        return None
    try:  # pragma: no cover - external dependency path
        from google.cloud import firestore  # type: ignore

        client = firestore.Client()
        snap = client.collection(coll).document(event_id).get()
        if snap.exists:
            return snap.to_dict() or None
    except Exception:
        return None
    return None


def webhook_store_persistent(event_id: str, record: dict[str, Any]) -> None:
    """Persist a webhook event to Redis (with TTL) and optionally Firestore.

    Caller is responsible for maintaining in-memory state.
    """
    # Redis with TTL
    if redis_enabled():
        try:
            ttl = int(os.getenv("OSCILLINK_WEBHOOK_TTL", "604800"))  # 7 days default
        except ValueError:
            ttl = 604800
        try:
            set_with_ttl(f"stripe_evt:{event_id}", json.dumps(record, separators=(",", ":")), ttl)
        except Exception:
            pass
    # Firestore (optional)
    coll = _webhook_events_collection()
    if not coll:
        return
    try:  # pragma: no cover - external dependency path
        from google.cloud import firestore  # type: ignore

        client = firestore.Client()
        # Use create to preserve idempotency (do not overwrite existing)
        doc_ref = client.collection(coll).document(event_id)
        if not doc_ref.get().exists:
            doc_ref.set(record, merge=False)
    except Exception:
        # Swallow errors silently (observability layer can catch later)
        pass
