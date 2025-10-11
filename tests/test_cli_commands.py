import importlib
import json
import os
from email.message import Message
from pathlib import Path


def _reload_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("OSCILLINK_CONFIG_DIR", str(tmp_path / ".oscillink_test"))
    # Ensure fresh import using the env var
    import oscillink.cli as cli  # noqa: F401

    importlib.reload(cli)
    return cli


def test_signup_wait_false(monkeypatch, tmp_path, capsys):
    cli = _reload_cli(tmp_path, monkeypatch)

    def fake_http(method, url, data=None, headers=None):
        return {"code": "abcd", "checkout_url": "https://checkout", "expires_in": 900}

    monkeypatch.setattr(cli, "_http_request", fake_http)
    rc = cli.main(["signup", "--tier", "beta", "--base", "http://x"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Checkout URL" in out and "Code:" in out
    assert "oscillink login --code" in out


def test_signup_wait_ready(monkeypatch, tmp_path):
    cli = _reload_cli(tmp_path, monkeypatch)

    def fake_http(method, url, data=None, headers=None):
        if url.endswith("/billing/cli/start"):
            return {"code": "abcd", "checkout_url": "https://checkout", "expires_in": 900}
        assert "/billing/cli/poll/abcd" in url
        return {"status": "ready", "api_key": "k123", "tier": "beta"}

    monkeypatch.setattr(cli, "_http_request", fake_http)
    rc = cli.main(["signup", "--wait", "--timeout", "10", "--base", "http://x"])
    assert rc == 0
    cfg_path = Path(os.environ["OSCILLINK_CONFIG_DIR"]) / "config.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["api_key"] == "k123" and data["tier"] == "beta"


class _FakeResp:
    def __init__(self, data: bytes, ct: str):
        self._data = data
        self.headers = {"Content-Type": ct}

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_http_request_json(monkeypatch, tmp_path):
    cli = _reload_cli(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli.urllib.request,
        "urlopen",
        lambda req, timeout=30: _FakeResp(b'{\n "ok": true\n}', "application/json"),
    )
    out = cli._http_request("GET", "http://x")
    assert out == {"ok": True}


def test_http_request_text(monkeypatch, tmp_path):
    cli = _reload_cli(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli.urllib.request, "urlopen", lambda req, timeout=30: _FakeResp(b"hello", "text/plain")
    )
    out = cli._http_request("GET", "http://x")
    assert out == "hello"


def test_http_request_http_error(monkeypatch, tmp_path):
    cli = _reload_cli(tmp_path, monkeypatch)

    def raise_http(*args, **kwargs):
        hdrs = Message()
        raise cli.urllib.error.HTTPError(url="http://x", code=500, msg="boom", hdrs=hdrs, fp=None)

    monkeypatch.setattr(cli.urllib.request, "urlopen", raise_http)
    try:
        cli._http_request("GET", "http://x")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "HTTP 500" in str(e)


def test_http_request_url_error(monkeypatch, tmp_path):
    cli = _reload_cli(tmp_path, monkeypatch)

    def raise_url(*args, **kwargs):
        raise cli.urllib.error.URLError("nope")

    monkeypatch.setattr(cli.urllib.request, "urlopen", raise_url)
    try:
        cli._http_request("GET", "http://x")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "Request failed" in str(e)


def test_login_ready_and_whoami(monkeypatch, tmp_path, capsys):
    cli = _reload_cli(tmp_path, monkeypatch)

    def fake_http(method, url, data=None, headers=None):
        assert url.endswith("/billing/cli/poll/ok")
        return {"status": "ready", "api_key": "k_abc", "tier": "beta"}

    monkeypatch.setattr(cli, "_http_request", fake_http)
    rc = cli.main(["login", "--code", "ok", "--base", "http://x"])
    assert rc == 0
    # whoami shows current info
    rc2 = cli.main(["whoami"])
    assert rc2 == 0
    out = capsys.readouterr().out
    assert "API base:" in out and "API key:" in out and "Tier:" in out


def test_login_expired(monkeypatch, tmp_path, capsys):
    cli = _reload_cli(tmp_path, monkeypatch)

    def fake_http(method, url, data=None, headers=None):
        return {"status": "expired"}

    monkeypatch.setattr(cli, "_http_request", fake_http)
    rc = cli.main(["login", "--code", "any", "--base", "http://x"])
    assert rc == 2
    assert "Code expired" in capsys.readouterr().out


def test_main_no_args_prints_help(monkeypatch, tmp_path, capsys):
    cli = _reload_cli(tmp_path, monkeypatch)
    rc = cli.main([])
    assert rc == 1
    assert "usage:" in capsys.readouterr().out.lower()


def test_cli_handles_http_error(monkeypatch, tmp_path):
    cli = _reload_cli(tmp_path, monkeypatch)

    def fake_http(method, url, data=None, headers=None):
        raise RuntimeError("HTTP 500: boom")

    monkeypatch.setattr(cli, "_http_request", fake_http)
    # command will run and catch the runtime error; returns code 2
    rc = cli.main(["signup", "--base", "http://x"])
    assert rc == 2


def test_logout_idempotent(monkeypatch, tmp_path, capsys):
    cli = _reload_cli(tmp_path, monkeypatch)
    # When no config present or no api_key key, it should print Already logged out.
    rc = cli.main(["logout"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Already logged out" in out


def test_portal_requires_login(monkeypatch, tmp_path, capsys):
    cli = _reload_cli(tmp_path, monkeypatch)
    rc = cli.main(["portal"])  # no stored key
    assert rc == 1
    assert "Not logged in" in capsys.readouterr().out


def test_portal_success(monkeypatch, tmp_path, capsys):
    cli = _reload_cli(tmp_path, monkeypatch)
    # save a config with api_key first
    cfg_dir = Path(os.environ["OSCILLINK_CONFIG_DIR"])
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text(
        json.dumps({"api_key": "k", "api_base": "http://x"}), encoding="utf-8"
    )

    def fake_http(method, url, data=None, headers=None):
        assert headers and headers.get("X-API-Key") == "k"
        return {"url": "https://portal"}

    monkeypatch.setattr(cli, "_http_request", fake_http)
    rc = cli.main(["portal"])
    assert rc == 0
    assert "Open Portal URL: https://portal" in capsys.readouterr().out


def test_signup_wait_timeout(monkeypatch, tmp_path, capsys):
    cli = _reload_cli(tmp_path, monkeypatch)

    # start returns code, poll returns pending forever
    def fake_http(method, url, data=None, headers=None):
        if url.endswith("/billing/cli/start"):
            return {"code": "c1", "checkout_url": "https://co", "expires_in": 900}
        return {"status": "pending"}

    monkeypatch.setattr(cli, "_http_request", fake_http)
    # fake time to trigger immediate timeout without sleeping
    t = [0]

    def fake_time():
        t[0] += 1
        return t[0]

    monkeypatch.setattr(cli.time, "time", fake_time)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    rc = cli.main(["signup", "--wait", "--timeout", "1", "--base", "http://x"])
    assert rc == 3
    assert "Timed out" in capsys.readouterr().out
