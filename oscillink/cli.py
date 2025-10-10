from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_API_BASE = os.environ.get("OSCILLINK_API_BASE", "http://localhost:8000")
CONFIG_DIR = Path(os.environ.get("OSCILLINK_CONFIG_DIR", Path.home() / ".oscillink"))
CONFIG_FILE = CONFIG_DIR / "config.json"


def _http_request(method: str, url: str, data: dict | None = None, headers: dict | None = None):
    body = None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    if data is not None:
        body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            ct = resp.headers.get("Content-Type", "")
            raw = resp.read()
            if "application/json" in ct:
                return json.loads(raw.decode("utf-8"))
            return raw.decode("utf-8")
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = str(e)
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Request failed: {e}") from e


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def cmd_signup(args: argparse.Namespace) -> int:
    base = args.base or DEFAULT_API_BASE
    tier = args.tier
    email = args.email
    resp = _http_request(
        "POST",
        urllib.parse.urljoin(base, "/billing/cli/start"),
        {"tier": tier, **({"email": email} if email else {})},
    )
    code = resp.get("code")
    checkout_url = resp.get("checkout_url")
    print(f"Checkout URL: {checkout_url}")
    print(f"Code: {code} (expires in {resp.get('expires_in')}s)")
    if args.wait:
        # poll until ready
        t0 = time.time()
        while True:
            poll = _http_request("GET", urllib.parse.urljoin(base, f"/billing/cli/poll/{code}"))
            if poll.get("status") == "ready":
                api_key = poll.get("api_key")
                cfg = _load_config()
                cfg["api_key"] = api_key
                cfg["tier"] = poll.get("tier")
                cfg["api_base"] = base
                _save_config(cfg)
                print("Your API key:", api_key)
                return 0
            if poll.get("status") == "expired":
                print("Session expired. Please run signup again.")
                return 2
            if time.time() - t0 > args.timeout:
                print("Timed out waiting for key. Use --wait --timeout N to extend.")
                return 3
            time.sleep(2)
    else:
        print("Complete checkout in your browser, then run: oscillink login --code", code)
    return 0


def cmd_login(args: argparse.Namespace) -> int:
    base = args.base or DEFAULT_API_BASE
    code = args.code
    poll = _http_request("GET", urllib.parse.urljoin(base, f"/billing/cli/poll/{code}"))
    if poll.get("status") == "ready":
        cfg = _load_config()
        cfg["api_key"] = poll.get("api_key")
        cfg["tier"] = poll.get("tier")
        cfg["api_base"] = base
        _save_config(cfg)
        print("Logged in. Key stored in", CONFIG_FILE)
        return 0
    elif poll.get("status") == "expired":
        print("Code expired. Run 'oscillink signup' again.")
        return 2
    else:
        print("Not ready yet. Complete checkout and try again.")
        return 1


def cmd_whoami(_: argparse.Namespace) -> int:
    cfg = _load_config()
    api_key = cfg.get("api_key")
    api_base = cfg.get("api_base", DEFAULT_API_BASE)
    if not api_key:
        print("Not logged in.")
        return 1
    print("API base:", api_base)
    print("API key:", api_key)
    print("Tier:", cfg.get("tier", "unknown"))
    return 0


def cmd_logout(_: argparse.Namespace) -> int:
    cfg = _load_config()
    if "api_key" in cfg:
        cfg.pop("api_key")
        _save_config(cfg)
        print("Logged out.")
        return 0
    print("Already logged out.")
    return 0


def cmd_portal(args: argparse.Namespace) -> int:
    cfg = _load_config()
    api_key = cfg.get("api_key")
    base = args.base or cfg.get("api_base", DEFAULT_API_BASE)
    if not api_key:
        print("Not logged in. Run 'oscillink signup --wait' first.")
        return 1
    resp = _http_request(
        "POST",
        urllib.parse.urljoin(base, "/billing/portal"),
        headers={"X-API-Key": api_key},
    )
    url = resp.get("url")
    print("Open Portal URL:", url)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="oscillink", description="Oscillink CLI")
    sub = p.add_subparsers(dest="cmd")

    ps = sub.add_parser("signup", help="Start signup and get key (optionally wait)")
    ps.add_argument("--tier", default="beta")
    ps.add_argument("--email", default=None)
    ps.add_argument("--base", default=None, help="API base, default from OSCILLINK_API_BASE")
    ps.add_argument("--wait", action="store_true", help="Wait and print key when ready")
    ps.add_argument("--timeout", type=int, default=600)
    ps.set_defaults(func=cmd_signup)

    pl = sub.add_parser("login", help="Claim a code to store the key locally")
    pl.add_argument("--code", required=True)
    pl.add_argument("--base", default=None)
    pl.set_defaults(func=cmd_login)

    pw = sub.add_parser("whoami", help="Show current login info")
    pw.set_defaults(func=cmd_whoami)

    po = sub.add_parser("logout", help="Remove stored key")
    po.set_defaults(func=cmd_logout)

    pp = sub.add_parser("portal", help="Open customer portal URL")
    pp.add_argument("--base", default=None)
    pp.set_defaults(func=cmd_portal)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = build_parser()
    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    try:
        return int(args.func(args))
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
