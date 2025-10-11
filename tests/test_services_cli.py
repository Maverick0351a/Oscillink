from __future__ import annotations

import time

import pytest

from cloud.app.services import cli


@pytest.fixture(autouse=True)
def _clear_cli_sessions():
    sess = cli.get_sessions()
    sess.clear()
    yield
    sess.clear()


def test_cli_ttl_seconds_default_and_override(monkeypatch):
    # Default when unset
    monkeypatch.delenv("OSCILLINK_CLI_TTL", raising=False)
    assert cli.ttl_seconds() == 900
    # Valid override
    monkeypatch.setenv("OSCILLINK_CLI_TTL", "123")
    assert cli.ttl_seconds() == 123
    # Invalid should fall back to default
    monkeypatch.setenv("OSCILLINK_CLI_TTL", "not-a-number")
    assert cli.ttl_seconds() == 900


def test_cli_new_code_uniqueness_and_length():
    codes = {cli.new_code() for _ in range(16)}
    assert len(codes) == 16
    for c in codes:
        assert isinstance(c, str)
        assert len(c) == 8  # secrets.token_hex(4) -> 8 hex chars; uuid fallback uses first 8


def test_cli_purge_expired_and_claimed(monkeypatch):
    # Use a small TTL to make timing easy
    monkeypatch.setenv("OSCILLINK_CLI_TTL", "10")
    ttl = cli.ttl_seconds()
    now = time.time()
    sessions = cli.get_sessions()
    sessions.clear()
    sessions["expired"] = {"status": "pending", "created": now - (ttl + 1)}
    sessions["active"] = {"status": "pending", "created": now}
    sessions["claimed"] = {"status": "claimed", "created": now - 1}
    cli.purge_expired()
    assert "expired" not in sessions
    assert "claimed" not in sessions
    assert "active" in sessions
