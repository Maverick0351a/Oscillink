"""Create a Stripe webhook endpoint for Oscillink and print the signing secret.

Usage:
  python scripts/stripe_setup_webhook.py --url https://api2.odinprotocol.dev/stripe/webhook --api-key sk_live_...

Notes:
- The signing secret is ONLY returned by Stripe at creation time.
- If an endpoint with the same URL already exists, this script will not recreate it and will exit with code 3.
- In that case, retrieve the secret from Stripe Dashboard or delete/recreate the endpoint there.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

try:
    import stripe  # type: ignore
except Exception:  # pragma: no cover
    print("stripe package required. Install with: python -m pip install stripe", file=sys.stderr)
    sys.exit(2)

stripe.api_version = "2024-06-20"

DEFAULT_EVENTS: List[str] = [
    "checkout.session.completed",
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--url", required=True, help="Webhook endpoint URL (must be publicly reachable)"
    )
    ap.add_argument(
        "--api-key",
        default=None,
        help="Stripe secret key (fallback to STRIPE_API_KEY/STRIPE_SECRET_KEY env)",
    )
    ap.add_argument(
        "--events", nargs="*", default=None, help="Override enabled events (space-separated)"
    )
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("STRIPE_API_KEY") or os.getenv("STRIPE_SECRET_KEY")
    if not api_key:
        print(
            "STRIPE_API_KEY or STRIPE_SECRET_KEY is required (pass --api-key or set env)",
            file=sys.stderr,
        )
        return 2

    stripe.api_key = api_key

    url = args.url
    events = args.events if args.events else DEFAULT_EVENTS

    # Check for existing endpoint with exact URL
    existing = stripe.WebhookEndpoint.list(limit=100)
    for ep in existing.auto_paging_iter():  # type: ignore
        if getattr(ep, "url", None) == url and getattr(ep, "status", "enabled") == "enabled":
            # Secret cannot be retrieved after creation
            print("", end="")
            return 3

    # Create new endpoint
    ep = stripe.WebhookEndpoint.create(
        url=url,
        enabled_events=events,  # type: ignore[arg-type]
        description="Oscillink webhook",
        metadata={"oscillink": "1"},
        connect=False,
    )
    # Print only the secret to stdout
    secret = getattr(ep, "secret", None)
    if not secret:
        print("", end="")
        return 4
    print(secret, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
