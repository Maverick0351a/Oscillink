"""Utility script to create Stripe products/prices for Oscillink tiers.

Usage (requires STRIPE_API_KEY env var):
  python scripts/stripe_setup.py --create --output price_map.json

Outputs a JSON mapping of price_id -> tier suitable for OSCILLINK_STRIPE_PRICE_MAP.
If products/prices already exist (matched by metadata.tier) it reuses them.

You can also run in dry mode to just print the existing mapping:
  python scripts/stripe_setup.py --print
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

try:
    import stripe  # type: ignore
except ImportError:  # pragma: no cover
    print("stripe package required. Install with pip install stripe", file=sys.stderr)
    sys.exit(1)

stripe.api_version = "2024-06-20"

TIERS = [
    {"name": "Oscillink Free", "tier": "free", "amount": 0, "interval": "month", "currency": "usd"},
    {"name": "Oscillink Pro", "tier": "pro", "amount": 4900, "interval": "month", "currency": "usd"},
    {"name": "Oscillink Enterprise", "tier": "enterprise", "amount": 0, "interval": "month", "currency": "usd", "custom": True},
]


def ensure_products_and_prices(live: bool = False) -> Dict[str, str]:
    """Create or reuse products+prices; return price_id -> tier mapping."""
    mode = "live" if live else "test"
    price_map: Dict[str, str] = {}
    for spec in TIERS:
        tier = spec["tier"]
        # Attempt to find existing product by metadata.tier
        existing = stripe.Product.list(limit=100, active=True)
        product_id = None
        for p in existing.auto_paging_iter():  # type: ignore
            if p.metadata.get("tier") == tier:
                product_id = p.id
                break
        if not product_id:
            product = stripe.Product.create(name=spec["name"], metadata={"tier": tier, "oscillink": "1"})
            product_id = product.id
        # Enterprise may be custom pricing; skip creating price amount 0 recurring for real live mode
        if spec.get("custom"):
            # Attempt to find a $0 placeholder price (test mode easier for mapping)
            prices = stripe.Price.list(product=product_id, limit=10)
            placeholder = None
            for pr in prices.auto_paging_iter():  # type: ignore
                if pr.metadata.get("tier") == tier:
                    placeholder = pr
                    break
            if not placeholder:
                placeholder = stripe.Price.create(product=product_id, unit_amount=0, currency=spec["currency"], recurring={"interval": spec["interval"]}, metadata={"tier": tier})
            price_map[placeholder.id] = tier
            continue
        # Find or create price
        prices = stripe.Price.list(product=product_id, limit=10)
        price_id = None
        for pr in prices.auto_paging_iter():  # type: ignore
            if pr.recurring and pr.recurring.get("interval") == spec["interval"] and pr.unit_amount == spec["amount"]:
                price_id = pr.id
                break
        if not price_id:
            price = stripe.Price.create(product=product_id, unit_amount=spec["amount"], currency=spec["currency"], recurring={"interval": spec["interval"]}, metadata={"tier": tier})
            price_id = price.id
        price_map[price_id] = tier
    return price_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true", help="Create/reuse products and prices")
    ap.add_argument("--output", type=str, default=None, help="Path to write price map JSON")
    ap.add_argument("--print", action="store_true", help="Print existing mapping from created products/prices")
    args = ap.parse_args()

    api_key = os.getenv("STRIPE_API_KEY")
    if not api_key:
        print("STRIPE_API_KEY env var required", file=sys.stderr)
        sys.exit(2)
    stripe.api_key = api_key

    if args.create:
        mapping = ensure_products_and_prices()
    else:
        # Derive mapping from existing products (best effort)
        mapping = {}
        products = stripe.Product.list(limit=100, active=True)
        for p in products.auto_paging_iter():  # type: ignore
            t = p.metadata.get("tier") if hasattr(p, "metadata") else None
            if not t:
                continue
            prices = stripe.Price.list(product=p.id, limit=10)
            for pr in prices.auto_paging_iter():  # type: ignore
                if pr.metadata.get("tier") == t:
                    mapping[pr.id] = t
                    break
    if args.print or True:
        print(json.dumps(mapping, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        print(f"Wrote price map to {args.output}")
    print("Set OSCILLINK_STRIPE_PRICE_MAP to either this JSON or a semicolon list, e.g.:")
    print("OSCILLINK_STRIPE_PRICE_MAP=\"" + ";".join(f"{pid}:{tier}" for pid, tier in mapping.items()) + "\"")

if __name__ == "__main__":  # pragma: no cover
    main()
