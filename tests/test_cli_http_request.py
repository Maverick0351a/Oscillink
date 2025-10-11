import io
import urllib.error
from email.message import Message

from oscillink import cli


class _Resp:
    def __init__(self, body: bytes, content_type: str):
        self._body = body
        self.headers = {"Content-Type": content_type}

    def read(self):  # noqa: D401
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_http_request_returns_json(monkeypatch):
    def fake_urlopen(req, timeout=30):
        return _Resp(b'{\n  "ok": 1\n}', "application/json; charset=utf-8")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = cli._http_request("GET", "http://example")
    assert out == {"ok": 1}


def test_http_request_returns_text(monkeypatch):
    def fake_urlopen(req, timeout=30):
        return _Resp(b"hello world", "text/plain; charset=utf-8")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = cli._http_request("GET", "http://example")
    assert out == "hello world"


def test_http_request_http_error_raises_runtime_error(monkeypatch):
    def fake_urlopen(req, timeout=30):
        # Simulate HTTP 400 with a readable body for detail
        headers = Message()
        headers["Content-Type"] = "text/plain"
        return (_ for _ in ()).throw(
            urllib.error.HTTPError(
                url="http://example",
                code=400,
                msg="Bad Request",
                hdrs=headers,
                fp=io.BytesIO(b"oops"),
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    try:
        cli._http_request("GET", "http://example")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "HTTP 400" in str(e)
