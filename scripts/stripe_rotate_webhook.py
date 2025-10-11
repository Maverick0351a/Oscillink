"""Rotate the Stripe webhook endpoint by URL and print the new signing secret.

Usage:
  python scripts/stripe_rotate_webhook.py --url https://api2.odinprotocol.dev/stripe/webhook --api-key sk_live_...

Notes:
- This script deletes any existing enabled endpoint at the given URL, then recreates it
  with the default set of events used by Oscillink.
- The new signing secret will be printed to stdout exactly once; capture and store it
  in your deployment environment immediately.
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

    # Find existing enabled endpoint with exact URL
    existing = stripe.WebhookEndpoint.list(limit=100)
    target = None
    for ep in existing.auto_paging_iter():  # type: ignore
        if getattr(ep, "url", None) == url and getattr(ep, "status", "enabled") == "enabled":
            target = ep
            break

    # Delete existing endpoint first (rotate)
    if target is not None:
        try:
            stripe.WebhookEndpoint.delete(target.id)  # type: ignore
        except Exception as e:
            print(f"failed to delete existing endpoint: {e}", file=sys.stderr)
            return 4

    # Create new endpoint
    try:
        ep = stripe.WebhookEndpoint.create(
            url=url,
            enabled_events=events,  # type: ignore[arg-type]
            description="Oscillink webhook",
            metadata={"oscillink": "1"},
            connect=False,
        )
    except Exception as e:
        print(f"failed to create endpoint: {e}", file=sys.stderr)
        return 5

    secret = getattr(ep, "secret", None)
    if not secret:
        print("no secret returned by Stripe", file=sys.stderr)
        return 6
    print(secret, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
