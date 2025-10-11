from __future__ import annotations

import time
from typing import Any

import pytest

from cloud.app.services import cli


class _FakeRedis:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        rec: dict[str, Any] = {"value": value, "expire_at": None}
        if isinstance(ex, int) and ex > 0:
            rec["expire_at"] = time.time() + ex
        self._store[key] = rec
        return True

    def get(self, key: str) -> str | None:
        rec = self._store.get(key)
        if not rec:
            return None
        exp = rec.get("expire_at")
        if exp is not None and time.time() >= float(exp):
            # expired
            self._store.pop(key, None)
            return None
        return rec.get("value")

    def exists(self, key: str) -> int:
        return 1 if self.get(key) is not None else 0

    def ttl(self, key: str) -> int:
        rec = self._store.get(key)
        if not rec:
            return -2  # missing
        exp = rec.get("expire_at")
        if exp is None:
            return -1  # no ttl
        remaining = int(exp - time.time())
        return remaining if remaining >= 0 else -2

    def expire(self, key: str, seconds: int) -> bool:
        rec = self._store.get(key)
        if not rec:
            return False
        rec["expire_at"] = time.time() + max(0, int(seconds))
        return True


@pytest.fixture
def fake_redis(monkeypatch):
    fr = _FakeRedis()
    # Force Redis mode and patch the get_redis symbol used inside services.cli
    monkeypatch.setenv("OSCILLINK_CLI_SESSIONS_BACKEND", "redis")
    monkeypatch.setenv("OSCILLINK_STATE_BACKEND", "redis")
    monkeypatch.setenv("OSCILLINK_CLI_TTL", "1")  # short TTL for tests
    monkeypatch.setattr(cli, "get_redis", lambda: fr, raising=True)
    yield fr


def test_redis_set_get_update_and_expiry(fake_redis, monkeypatch):
    # Create session
    code = "abcd1234"
    ok = cli.set_session(code, {"status": "pending", "created": time.time(), "tier": "beta"})
    assert ok
    # Read back
    rec = cli.get_session(code)
    assert rec and rec.get("status") == "pending" and rec.get("tier") == "beta"
    assert cli.session_exists(code) is True
    # Update
    assert cli.update_session(code, {"status": "provisioned", "api_key": "ok_test"})
    rec2 = cli.get_session(code)
    assert rec2 and rec2.get("status") == "provisioned" and rec2.get("api_key") == "ok_test"
    # purge_expired is a no-op under Redis
    cli.purge_expired()
    assert cli.session_exists(code) is True
    # Wait for TTL expiry
    time.sleep(1.1)
    assert cli.get_session(code) is None
    assert cli.session_exists(code) is False


def test_redis_mode_without_client_falls_back_to_memory(monkeypatch):
    # Enable redis mode but make get_redis return None -> memory fallback
    monkeypatch.setenv("OSCILLINK_CLI_SESSIONS_BACKEND", "redis")
    monkeypatch.setenv("OSCILLINK_STATE_BACKEND", "redis")
    monkeypatch.setattr(cli, "get_redis", lambda: None, raising=True)
    # Ensure we start from a clean memory map
    cli.get_sessions().clear()
    code = cli.new_code()
    assert cli.set_session(code, {"status": "pending"})
    assert cli.session_exists(code) is True
    assert cli.update_session(code, {"status": "claimed"})
    rec = cli.get_session(code)
    assert rec is not None and rec.get("status") == "claimed"
    # purge_expired should remove claimed sessions in memory fallback
    monkeypatch.setenv("OSCILLINK_CLI_TTL", "900")
    cli.purge_expired()
    assert cli.session_exists(code) is False
