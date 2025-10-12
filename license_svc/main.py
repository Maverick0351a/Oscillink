from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI(title="Oscillink License Service")

# In-memory demo keys (replace with secure store)
PUB_JWKS = {
    "keys": [
        # Example: Ed25519 public key in JWK form with kid
        # {"kty":"OKP","crv":"Ed25519","kid":"k1","x":"<base64url>"}
    ]
}


class RenewBody(BaseModel):
    sub: str


@app.get("/.well-known/jwks.json")
async def jwks():
    return PUB_JWKS


@app.post("/v1/license/renew")
async def renew(body: RenewBody):
    if not body.sub:
        raise HTTPException(400, "missing sub")
    # TODO: look up entitlements by sub in Firestore/DB
    ent = {
        "tier": "beta",
        "limits": {"monthly_units": 25_000_000, "max_nodes": 5000, "max_dim": 4096},
        "features": {"diffusion_gates": True, "advisor": False, "chain_prior": True},
        "telemetry": "minimal",
    }
    now = datetime.now(timezone.utc)
    # TODO: sign a real JWT with Ed25519 private key
    fake_token = json.dumps(
        {
            "sub": body.sub,
            "tier": ent["tier"],
            "limits": ent["limits"],
            "features": ent["features"],
            "telemetry": ent["telemetry"],
            "nbf": int(now.timestamp()),
            "exp": int((now + timedelta(days=30)).timestamp()),
            "kid": "k1",
        }
    )
    return {"token": fake_token}


@app.post("/v1/usage/report")
async def usage_report(req: Request):
    body = await req.json()
    lic = body.get("license_id")
    lines = body.get("lines", [])
    mac = body.get("hmac", "")
    if not lic:
        raise HTTPException(400, "missing license_id")
    # Demo check: if a per-license secret is set, verify HMAC
    secret = os.getenv("USAGE_HMAC_SECRET")
    if secret is not None:
        msg = json.dumps({"license_id": lic, "lines": lines}, separators=(",", ":")).encode()
        calc = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(calc, mac):
            raise HTTPException(401, "bad hmac")
    # TODO: persist aggregated usage
    return {"ok": True, "count": len(lines)}
