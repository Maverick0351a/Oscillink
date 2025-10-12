"""
Create a Stripe Webhook Endpoint and print the signing secret.

Usage:
  python scripts/stripe_create_webhook.py --url https://your.domain/stripe/webhook \
      --events checkout.session.completed customer.subscription.created \
      --output-env .env

Requirements:
  - STRIPE_SECRET_KEY or STRIPE_API_KEY in the environment, or pass --api-key
Notes:
  - The webhook signing secret is returned only at creation time by Stripe.
  - This script will print the secret and optionally append STRIPE_WEBHOOK_SECRET=... to an env file.
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
    ap.add_argument(
        "--output-env", default=None, help="Path to .env to append STRIPE_WEBHOOK_SECRET"
    )
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
    if secret:
        print(f"  signing secret: {secret}")
    else:
        print("  signing secret: <not returned by API; copy from Stripe Dashboard>")

    if args.output_env and secret:
        try:
            with open(args.output_env, "a", encoding="utf-8") as f:
                f.write(f"\nSTRIPE_WEBHOOK_SECRET={secret}\n")
            print(f"Appended STRIPE_WEBHOOK_SECRET to {args.output_env}")
        except Exception as e:  # noqa: BLE001
            print(f"warn: failed to write {args.output_env}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
