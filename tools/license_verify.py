from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def _parse_jwt(token: str) -> tuple[str, str, str, dict, dict]:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError as e:
        raise SystemExit(f"Invalid JWT format: {e}") from e
    try:
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception as e:  # noqa: BLE001 - keep broad for robustness at startup
        raise SystemExit(f"Invalid token encoding: {e}") from e
    return header_b64, payload_b64, sig_b64, header, payload


def _now() -> int:
    return int(time.time())


def verify_times(payload: dict, leeway: int = 300) -> None:
    now = _now()
    nbf = int(payload.get("nbf", 0))
    exp = int(payload.get("exp", 0))
    iat = int(payload.get("iat", 0))
    if nbf and now + leeway < nbf:
        raise SystemExit("Token not valid yet (nbf)")
    if exp and now - leeway > exp:
        raise SystemExit("Token expired (exp)")
    if iat and iat - leeway > now:
        raise SystemExit("Token issued in the future (iat)")


def verify_claims(payload: dict, expect_iss: str | None, expect_aud: str | None) -> None:
    if expect_iss is not None and payload.get("iss") != expect_iss:
        raise SystemExit("Issuer mismatch (iss)")
    if expect_aud is not None:
        aud = payload.get("aud")
        if isinstance(aud, list):
            if expect_aud not in aud:
                raise SystemExit("Audience mismatch (aud)")
        elif isinstance(aud, str):
            if expect_aud != aud:
                raise SystemExit("Audience mismatch (aud)")
        else:
            raise SystemExit("Audience claim missing (aud)")


def _read_cached_jwks(cache_path: str) -> tuple[dict | None, str | None, int | None]:
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("jwks"), data.get("etag"), int(data.get("fetched_at", 0))
    except Exception:
        return None, None, None


def _write_cached_jwks(cache_path: str, jwks: dict, etag: str | None) -> None:
    payload = {"jwks": jwks, "etag": etag, "fetched_at": _now()}
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        # Non-fatal; continue without cache persistence
        pass


def load_jwks_with_cache(url: str, cache_path: str, cache_ttl: int, offline_grace: int) -> dict:
    cached, etag, fetched_at = _read_cached_jwks(cache_path)
    # Use cache if within TTL
    if cached and fetched_at and (_now() - fetched_at) < cache_ttl:
        return cached
    # Try conditional fetch with ETag
    req = urllib.request.Request(url)
    if etag:
        req.add_header("If-None-Match", etag)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # nosec - ops-controlled URL
            jwks = json.load(resp)
            new_etag = resp.headers.get("ETag")
            _write_cached_jwks(cache_path, jwks, new_etag)
            return jwks
    except urllib.error.HTTPError as e:
        if e.code == 304 and cached:
            # Not modified
            _write_cached_jwks(cache_path, cached, etag)
            return cached
        # Other HTTP errors fall through to offline grace
    except Exception:
        # Network or JSON error; fall through
        pass
    # Offline grace
    if cached and fetched_at and (_now() - fetched_at) < offline_grace:
        return cached
    raise SystemExit("Failed to load JWKS and no valid cache available")


def match_jwk_by_kid(header: dict, jwks: dict) -> dict | None:
    kid = header.get("kid")
    if not kid:
        return None
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            return k
    return None


def verify_eddsa_signature(header_b64: str, payload_b64: str, sig_b64: str, jwk: dict) -> None:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "cryptography package is required for Ed25519 verification in licensed container"
        ) from e

    if jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519" or not jwk.get("x"):
        raise SystemExit("JWKS key is not Ed25519 (OKP/Ed25519)")
    try:
        pub_bytes = _b64url_decode(jwk["x"])
        sig = _b64url_decode(sig_b64)
        signing_input = (header_b64 + "." + payload_b64).encode("ascii")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Failed to decode key/signature: {e}") from e

    try:
        Ed25519PublicKey.from_public_bytes(pub_bytes).verify(sig, signing_input)
    except Exception as e:  # noqa: BLE001
        raise SystemExit("Invalid signature (Ed25519 verification failed)") from e


def _build_env_lines(payload: dict) -> list[str]:
    lines: list[str] = []
    sub = payload.get("sub") or payload.get("license_id")
    tier = payload.get("tier")
    limits = payload.get("limits", {}) or {}
    features = payload.get("features", {}) or {}
    if tier:
        lines.append(f"OSCILLINK_TIER={tier}")
    if sub:
        lines.append(f"OSCILLINK_API_KEYS={sub}")
        if tier:
            lines.append(f"OSCILLINK_KEY_TIERS={sub}:{tier}")
    max_nodes = limits.get("max_nodes")
    max_dim = limits.get("max_dim")
    if isinstance(max_nodes, int):
        lines.append(f"OSCILLINK_MAX_NODES={max_nodes}")
    if isinstance(max_dim, int):
        lines.append(f"OSCILLINK_MAX_DIM={max_dim}")
    qps = limits.get("qps")
    qps_window = limits.get("qps_window") or 60
    if isinstance(qps, int) and qps > 0:
        lines.append(f"OSCILLINK_RATE_LIMIT={qps}")
        lines.append(f"OSCILLINK_RATE_WINDOW={int(qps_window)}")
    quota_units = limits.get("quota_units")
    quota_window = limits.get("quota_window") or 3600
    if isinstance(quota_units, int) and quota_units > 0:
        lines.append(f"OSCILLINK_KEY_NODE_UNITS_LIMIT={quota_units}")
        lines.append(f"OSCILLINK_KEY_NODE_UNITS_WINDOW={int(quota_window)}")
    # Monthly cap override (units per calendar month). If present, app will honor via env.
    monthly_cap = limits.get("monthly_cap") or limits.get("monthly_units")
    if isinstance(monthly_cap, int) and monthly_cap > 0:
        lines.append(f"OSCILLINK_MONTHLY_CAP={monthly_cap}")
    for fname, on in features.items():
        enabled = "1" if bool(on) else "0"
        lines.append(f"OSCILLINK_FEAT_{str(fname).upper()}={enabled}")
    return lines


def _verify_and_decode(
    token: str,
    jwks_url: str,
    jwks_cache: str,
    ttl: int,
    grace: int,
    leeway: int,
    iss: str | None,
    aud: str | None,
) -> dict:
    header_b64, payload_b64, sig_b64, header, payload = _parse_jwt(token)
    alg = (header.get("alg") or "").upper()
    typ = (header.get("typ") or "").upper()
    if alg not in {"EDDSA"}:
        raise SystemExit(f"Unsupported alg: {alg}")
    if typ and typ != "JWT":
        raise SystemExit(f"Unsupported typ: {typ}")
    verify_times(payload, leeway=leeway)
    verify_claims(payload, expect_iss=iss, expect_aud=aud)
    jwks = load_jwks_with_cache(jwks_url, cache_path=jwks_cache, cache_ttl=ttl, offline_grace=grace)
    jwk = match_jwk_by_kid(header, jwks)
    if not jwk:
        raise SystemExit("No matching key in JWKS (kid)")
    verify_eddsa_signature(header_b64, payload_b64, sig_b64, jwk)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify Oscillink license JWT and export entitlements")
    ap.add_argument("--license", required=True, help="Path to license JWT file")
    ap.add_argument("--jwks", required=True, help="JWKS URL")
    ap.add_argument("--entitlements-path", default="/run/oscillink_entitlements.json")
    ap.add_argument("--env-path", default="/run/oscillink_entitlements.env")
    ap.add_argument("--leeway", type=int, default=int(os.getenv("OSCILLINK_JWT_LEEWAY", "300")))
    ap.add_argument("--iss", default=os.getenv("OSCILLINK_JWT_ISS"))
    ap.add_argument("--aud", default=os.getenv("OSCILLINK_JWT_AUD"))
    ap.add_argument(
        "--jwks-cache", default=os.getenv("OSCILLINK_JWKS_CACHE", "/run/jwks_cache.json")
    )
    ap.add_argument(
        "--jwks-cache-ttl", type=int, default=int(os.getenv("OSCILLINK_JWKS_TTL", "3600"))
    )
    ap.add_argument(
        "--jwks-offline-grace",
        type=int,
        default=int(os.getenv("OSCILLINK_JWKS_OFFLINE_GRACE", "86400")),
    )
    args = ap.parse_args()

    with open(args.license, encoding="utf-8") as fh:
        token = fh.read().strip()

    payload = _verify_and_decode(
        token,
        jwks_url=args.jwks,
        jwks_cache=args.jwks_cache,
        ttl=args.jwks_cache_ttl,
        grace=args.jwks_offline_grace,
        leeway=args.leeway,
        iss=args.iss,
        aud=args.aud,
    )

    with open(args.entitlements_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    env_lines = _build_env_lines(payload)
    if env_lines:
        with open(args.env_path, "w", encoding="utf-8") as f:
            f.write("\n".join(env_lines) + "\n")

    print("OK: license verified and entitlements exported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
