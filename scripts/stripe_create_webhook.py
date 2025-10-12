"""
Create a Stripe Webhook Endpoint and securely handle the signing secret.

Usage:
  python scripts/stripe_create_webhook.py --url https://your.domain/stripe/webhook \
      --events checkout.session.completed customer.subscription.created \
      --output-env .env

Requirements:
    - STRIPE_SECRET_KEY or STRIPE_API_KEY in the environment, or pass --api-key
Notes:
    - The webhook signing secret is returned only at creation time by Stripe.
    - This script will NOT print or write the secret to disk. It will attempt to store it in the OS keyring if available.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import stripe  # type: ignore
except ImportError:  # pragma: no cover
    print("stripe package required. Install with: python -m pip install stripe", file=sys.stderr)
    sys.exit(1)


def _mask(secret: str) -> str:
    """Return a masked representation of a secret for safe display."""
    if not secret:
        return "****"
    if len(secret) <= 8:
        return "****"
    return f"{secret[:4]}…{secret[-4:]}"


def _write_env_placeholder(path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n# STRIPE_WEBHOOK_SECRET=<set securely via your secret manager or CI/CD>\n")


def _store_secret_keyring(wid_or_url: str, secret: str) -> bool:
    """Best-effort secure storage using system keyring. Returns True on success."""
    try:
        import keyring  # type: ignore

        service_name = "oscillink/stripe_webhook"
        username = f"webhook:{wid_or_url}"
        keyring.set_password(service_name, username, secret)
        print(
            "Stored signing secret in OS keyring under service 'oscillink/stripe_webhook'. "
            f"Account: {username}"
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"warn: failed to store secret in keyring: {e}", file=sys.stderr)
        print(
            "Tip: install the 'keyring' package to store the secret in your OS keychain "
            "(avoids clear-text files)."
        )
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Public URL to receive webhooks (https)")
    ap.add_argument(
        "--events",
        nargs="+",
        default=[
            "checkout.session.completed",
            "customer.subscription.created",
            "customer.subscription.updated",
            "customer.subscription.deleted",
        ],
        help="Event types to subscribe to",
    )
    ap.add_argument("--api-key", default=None, help="Stripe secret key (overrides env)")
    ap.add_argument("--output-env", default=None, help="Path to .env to update (placeholder only)")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")
    if not api_key:
        print(
            "error: STRIPE_SECRET_KEY/STRIPE_API_KEY not set and --api-key not provided",
            file=sys.stderr,
        )
        return 2

    if not args.url.startswith("https://"):
        print("error: webhook URL must be https", file=sys.stderr)
        return 2

    stripe.api_key = api_key
    stripe.api_version = "2024-06-20"

    try:  # pragma: no cover - external API call
        wh = stripe.WebhookEndpoint.create(url=args.url, enabled_events=args.events)  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"error: failed to create webhook endpoint: {e}", file=sys.stderr)
        return 1

    # Newer Stripe versions return 'secret' only on creation
    secret = getattr(wh, "secret", None) or getattr(wh, "_secret", None) or wh.get("secret")  # type: ignore
    wid = getattr(wh, "id", None) or wh.get("id")  # type: ignore
    print("Created Webhook Endpoint:")
    print(f"  id: {wid}")
    if not secret:
        print("  signing secret: <not returned by API; copy from Stripe Dashboard>")
    else:
        # Do not print masked or full secret to avoid clear-text logging findings.
        print("  signing secret: <captured; not printed> (stored securely if keyring available)")

    if args.output_env:
        try:
            _write_env_placeholder(args.output_env)
            print(
                f"Wrote placeholder STRIPE_WEBHOOK_SECRET to {args.output_env} (not storing secret in clear text)."
            )
        except Exception as e:  # noqa: BLE001
            print(f"warn: failed to write {args.output_env}: {e}", file=sys.stderr)

    if secret:
        _store_secret_keyring(wid or args.url, secret)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
