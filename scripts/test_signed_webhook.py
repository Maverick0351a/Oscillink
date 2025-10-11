"""
Send a signed Stripe-style webhook test event to the deployed Cloud Run service.

It reads STRIPE_WEBHOOK_SECRET from the environment first; if missing, it attempts
to parse cloud/env.private.yaml or cloud/env.stripe.yaml for the secret line.

It posts a small JSON payload with an event id and computes the Stripe-Signature
header as: t={timestamp},v1=HMAC_SHA256(secret, f"{t}.{payload}")

Endpoints tested (in order):
 - https://oscillink-cloud-963096600449.us-central1.run.app/stripe/webhook
 - https://api2.odinprotocol.dev/stripe/webhook

The script handles 307 redirects by re-POSTing to the Location target.
Output: one line per endpoint with status, verified flag, id and type.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import os
import sys
import time
import uuid
from typing import Optional
from urllib.parse import urlparse

SERVICE_URL = "https://oscillink-cloud-963096600449.us-central1.run.app/stripe/webhook"
CUSTOM_URL = "https://api2.odinprotocol.dev/stripe/webhook"


def _read_secret_from_yaml(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("STRIPE_WEBHOOK_SECRET:"):
                    # Expected formats:
                    # STRIPE_WEBHOOK_SECRET: "whsec_xxx"
                    # STRIPE_WEBHOOK_SECRET: whsec_xxx
                    _, val = s.split(":", 1)
                    val = val.strip().strip('"').strip("'")
                    return val
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return None


def get_secret() -> str:
    env = os.getenv("STRIPE_WEBHOOK_SECRET")
    if env:
        return env
    # fallbacks to local files
    for p in (
        os.path.join("cloud", "env.private.yaml"),
        os.path.join("cloud", "env.stripe.yaml"),
    ):
        s = _read_secret_from_yaml(p)
        if s:
            return s
    raise RuntimeError("STRIPE_WEBHOOK_SECRET not found in env or cloud/env.*.yaml")


def sign_payload(secret: str, payload: str, ts: int) -> str:
    to_sign = f"{ts}.{payload}".encode()
    sig = hmac.new(secret.encode(), to_sign, hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


def _post_once(url: str, payload: str, sig_header: str):
    parsed = urlparse(url)
    assert parsed.scheme == "https", "https required"
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    host = parsed.hostname or ""
    conn = http.client.HTTPSConnection(host, parsed.port or 443, timeout=10)
    conn.request(
        "POST",
        path,
        body=payload.encode(),
        headers={
            "Content-Type": "application/json",
            "Stripe-Signature": sig_header,
        },
    )
    resp = conn.getresponse()
    data = resp.read()
    headers = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    return resp.status, headers, data


def post_with_redirect(url: str, payload: str, sig_header: str):
    status, headers, data = _post_once(url, payload, sig_header)
    if status in (301, 302, 303, 307, 308) and "location" in headers:
        # Re-POST to the new location
        url2 = headers["location"]
        status, headers, data = _post_once(url2, payload, sig_header)
        return url2, status, headers, data
    return url, status, headers, data


def main():
    try:
        secret = get_secret()
    except Exception as e:
        print(f"error: {e}")
        return 2
    results = []
    for url in (SERVICE_URL, CUSTOM_URL):
        ts = int(time.time())
        evt_id = f"evt_test_{uuid.uuid4().hex[:8]}"
        payload_obj = {"id": evt_id, "type": "payment_intent.succeeded", "data": {"object": {}}}
        payload = json.dumps(payload_obj, separators=(",", ":"))
        sig = sign_payload(secret, payload, ts)
        try:
            final_url, status, _headers, body = post_with_redirect(url, payload, sig)
            try:
                j = json.loads(body.decode("utf-8"))
            except Exception:
                j = {"raw": body.decode("utf-8", errors="replace")}
            results.append((final_url, status, j))
        except Exception as e:
            results.append((url, 0, {"error": str(e)}))

    for final_url, status, j in results:
        verified = j.get("verified") if isinstance(j, dict) else None
        etype = j.get("type") if isinstance(j, dict) else None
        eid = j.get("id") if isinstance(j, dict) else None
        print(f"url={final_url} status={status} verified={verified} type={etype} id={eid}")

    # Non-zero exit if both failed or both not verified
    ok = any(isinstance(j, dict) and j.get("verified") for _, _, j in results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
