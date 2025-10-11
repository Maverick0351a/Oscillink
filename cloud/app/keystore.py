"""Simple in-memory key store abstraction.

Phase: foundational (Absolutely Needed).

Provides a pluggable interface so later we can swap in Firestore without rewriting
handlers or quota logic.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class KeyMetadata:
    api_key: str
    tier: str = "free"
    status: str = "active"  # active|pending|revoked|suspended
    quota_limit_units: Optional[int] = None  # override (N*D units per window)
    quota_window_seconds: Optional[int] = None
    features: Dict[str, bool] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())

    def is_active(self) -> bool:
        return self.status == "active"


class KeyStore:
    """Abstract base for key lookups.

    Implementations should be lightweight; Firestore backend is optional and lazily imported.
    """

    def get(self, api_key: str) -> Optional[KeyMetadata]:  # pragma: no cover - interface
        raise NotImplementedError

    def update(
        self, api_key: str, **fields
    ) -> Optional[KeyMetadata]:  # pragma: no cover - interface
        """Update mutable fields for a key; returns updated metadata or None if absent.

        Implementations should create the key if it does not exist when explicit create=True passed.
        """
        raise NotImplementedError


class InMemoryKeyStore(KeyStore):
    """Key store seeded from environment.

    Environment variable OSCILLINK_API_KEYS may contain a comma-separated list of keys.
    Optionally per-key tier override via OSCILLINK_KEY_TIERS (format: key:tier;key2:tier2).
    This is intentionally lightweight; Firestore/Stripe integration will replace it.
    """

    def __init__(self):
        self._keys: Dict[str, KeyMetadata] = {}
        raw = os.getenv("OSCILLINK_API_KEYS", "").strip()
        if raw:
            for k in [x.strip() for x in raw.split(",") if x.strip()]:
                self._keys[k] = KeyMetadata(api_key=k)
        tiers = os.getenv("OSCILLINK_KEY_TIERS", "").strip()
        if tiers:
            for part in [x.strip() for x in tiers.split(";") if x.strip()]:
                if ":" in part:
                    k, t = part.split(":", 1)
                    meta = self._keys.get(k)
                    if meta:
                        meta.tier = t
                        meta.updated_at = time.time()
                    else:
                        self._keys[k] = KeyMetadata(api_key=k, tier=t)

    def get(self, api_key: str) -> Optional[KeyMetadata]:
        return self._keys.get(api_key)

    def update(self, api_key: str, create: bool = False, **fields) -> Optional[KeyMetadata]:
        meta = self._keys.get(api_key)
        if not meta and not create:
            return None
        if not meta:
            meta = KeyMetadata(api_key=api_key)
            self._keys[api_key] = meta
        # Apply updates
        for k, v in fields.items():
            if hasattr(meta, k) and v is not None:
                setattr(meta, k, v)
        meta.updated_at = time.time()
        return meta


class FirestoreKeyStore(KeyStore):  # pragma: no cover - covered indirectly when firestore installed
    """Firestore-backed key store.

    Expected document structure (collection configurable via OSCILLINK_FIRESTORE_COLLECTION, default 'oscillink_api_keys'):
        key (document id) -> {
            tier: str ('free'|'pro'|'enterprise' ...),
            status: 'active'|'revoked'|'suspended',
            quota_limit_units: int | null,
            quota_window_seconds: int | null,
            features: { feature_flag: bool, ... },
            created_at: float (epoch seconds),
            updated_at: float (epoch seconds)
        }

    Only fields present are applied; missing fields fall back to KeyMetadata defaults.
    """

    def __init__(self):
        try:
            from google.cloud import firestore  # type: ignore
        except Exception as e:  # pragma: no cover - import error path
            raise RuntimeError("FirestoreKeyStore requires google-cloud-firestore package") from e
        self._client = firestore.Client()  # relies on GOOGLE_APPLICATION_CREDENTIALS or env auth
        self._collection = os.getenv("OSCILLINK_FIRESTORE_COLLECTION", "oscillink_api_keys")

    def get(self, api_key: str) -> Optional[KeyMetadata]:
        doc_ref = self._client.collection(self._collection).document(api_key)
        snap = doc_ref.get()
        if not snap.exists:
            return None
        data: Dict[str, Any] = snap.to_dict() or {}
        # Construct metadata with safe fallbacks
        meta = KeyMetadata(
            api_key=api_key,
            tier=data.get("tier", "free"),
            status=data.get("status", "active"),
            quota_limit_units=data.get("quota_limit_units"),
            quota_window_seconds=data.get("quota_window_seconds"),
            features=data.get("features", {}) or {},
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )
        return meta

    def update(
        self, api_key: str, create: bool = False, **fields
    ) -> Optional[KeyMetadata]:  # pragma: no cover - network path
        doc_ref = self._client.collection(self._collection).document(api_key)
        now = time.time()
        existing = doc_ref.get()
        if not existing.exists and not create:
            return None
        if not existing.exists:
            base = {"api_key": api_key, "tier": "free", "status": "active", "created_at": now}
        else:
            base = existing.to_dict() or {}
        # Merge
        for k, v in fields.items():
            if v is not None:
                base[k] = v
        base["updated_at"] = now
        # Persist
        doc_ref.set(base, merge=True)
        return self.get(api_key)


# Singleton accessor (simple for now)
_key_store: Optional[KeyStore] = None


def get_keystore() -> KeyStore:
    """Return singleton keystore instance.

    Selects backend via OSCILLINK_KEYSTORE_BACKEND env var: 'memory' (default) or 'firestore'.
    Firestore backend requires google-cloud-firestore dependency and proper GCP credentials.
    """
    global _key_store
    if _key_store is None:
        backend = os.getenv("OSCILLINK_KEYSTORE_BACKEND", "memory").lower()
        _key_store = FirestoreKeyStore() if backend == "firestore" else InMemoryKeyStore()
    return _key_store


# --- Tier / Feature Mutation Helpers (for Stripe or admin paths) ---


def update_key_tier(api_key: str, tier: str, *, create: bool = False) -> KeyMetadata | None:
    """Update (or create) a key's tier and touch updated_at.

    Does NOT adjust feature flags directly; feature resolution layer should map tier -> features.
    Returns updated metadata or None if key absent and create=False.
    """
    ks = get_keystore()
    meta = ks.update(api_key, create=create, tier=tier)
    return meta
