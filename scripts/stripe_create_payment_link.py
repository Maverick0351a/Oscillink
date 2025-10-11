"""Create a Stripe Payment Link for an Oscillink tier.

Usage (requires STRIPE_API_KEY env var):
    python scripts/stripe_create_payment_link.py --tier beta --base-url https://api.yourdomain.com

This will create a Payment Link that:
- Charges the configured price for the tier
- After completion, redirects to: {base-url}/billing/success?session_id={CHECKOUT_SESSION_ID}

Notes:
- Early beta: Pro is hidden by default. To allow creating Pro links, pass --allow-pro.
- Ensure your server exposes GET /billing/success (implemented in cloud/app/main.py)
- The success URL uses the special token {CHECKOUT_SESSION_ID} so your server can retrieve the
    session and display the API key immediately.
- You can create multiple links for different tiers by running with different --tier values.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

try:
    import stripe  # type: ignore
except ImportError:  # pragma: no cover
    print("stripe package required. Install with: pip install stripe", file=sys.stderr)
    sys.exit(1)

stripe.api_version = "2024-06-20"

# Fallback names used by scripts/stripe_setup.py
TIER_NAMES = {
    "free": "Oscillink Free",
    "beta": "Oscillink Beta",
    "pro": "Oscillink Pro",
    "enterprise": "Oscillink Enterprise",
}


def _find_price_for_tier(tier: str) -> str:
    """Return a price ID whose metadata.tier matches the requested tier.

    Assumes products/prices were created with scripts/stripe_setup.py and metadata.tier set.
    """
    # Search products by metadata.tier
    products = stripe.Product.list(limit=100, active=True)
    product_id = None
    for p in products.auto_paging_iter():  # type: ignore
        md = getattr(p, "metadata", {}) or {}
        if md.get("tier") == tier:
            product_id = p.id
            break
    if not product_id:
        raise RuntimeError(
            f"No product found for tier '{tier}'. Run scripts/stripe_setup.py --create first."
        )
    # Find a price with matching metadata.tier
    prices = stripe.Price.list(product=product_id, active=True, limit=100)
    for pr in prices.auto_paging_iter():  # type: ignore
        md = getattr(pr, "metadata", {}) or {}
        if md.get("tier") == tier:
            return pr.id
    raise RuntimeError(f"No price found for tier '{tier}'. Ensure prices have metadata.tier set.")


def create_payment_link(tier: str, base_url: str) -> Dict[str, str]:
    price_id = _find_price_for_tier(tier)
    # Create Payment Link with redirect to our success endpoint
    success_url = base_url.rstrip("/") + "/billing/success?session_id={CHECKOUT_SESSION_ID}"
    link = stripe.PaymentLink.create(
        line_items=[{"price": price_id, "quantity": 1}],
        after_completion={
            "type": "redirect",
            "redirect": {"url": success_url},
        },
        # optional niceties
        allow_promotion_codes=False,
        metadata={"tier": tier, "oscillink": "1"},
    )
    return {"url": link.url, "price_id": price_id, "tier": tier}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["free", "beta", "pro", "enterprise"], default="beta")
    ap.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("OSCILLINK_PUBLIC_BASE_URL"),
        help="Public base URL of your API (e.g., https://api.example.com)",
    )
    ap.add_argument("--api-key", type=str, default=None, help="Stripe secret key (overrides env)")
    ap.add_argument(
        "--allow-pro",
        action="store_true",
        help="Allow creating links for the Pro tier (hidden by default during beta)",
    )
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("STRIPE_API_KEY") or os.getenv("STRIPE_SECRET_KEY")
    if not api_key:
        print("STRIPE_API_KEY (or STRIPE_SECRET_KEY) env var required", file=sys.stderr)
        sys.exit(2)
    if not args.base_url:
        print("--base-url or OSCILLINK_PUBLIC_BASE_URL is required", file=sys.stderr)
        sys.exit(2)
    stripe.api_key = api_key

    if args.tier == "pro" and not args.allow_pro:
        raise SystemExit(
            "Pro tier is hidden during early beta. Re-run with --allow-pro if you intend to generate a Pro link."
        )

    res = create_payment_link(args.tier, args.base_url)
    print("Created payment link:")
    print(f"  Tier:      {res['tier']}")
    print(f"  Price ID:  {res['price_id']}")
    print(f"  URL:       {res['url']}")
    print()
    print("Add this URL to README as the 'Get API Key →' link.")


if __name__ == "__main__":  # pragma: no cover
    main()
