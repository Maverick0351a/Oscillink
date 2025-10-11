#!/usr/bin/env python
"""Check for breaking OpenAPI changes (field removals) vs baseline.

Usage: python scripts/check_openapi_diff.py --baseline openapi_baseline.json --current openapi_current.json
Generate current spec with `python -m scripts.export_openapi --out openapi_current.json` first.
Exit codes:
 0 = OK (no breaking removals)
 1 = Baseline missing / load error
 2 = Breaking change detected
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def flatten_keys(obj, prefix=""):
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_p = f"{prefix}.{k}" if prefix else k
            keys.add(new_p)
            keys |= flatten_keys(v, new_p)
    elif isinstance(obj, list):
        # We don't index list positions; treat list as a single node
        keys.add(prefix + "[]")
        for item in obj:
            keys |= flatten_keys(item, prefix + "[]")
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", required=True)
    args = ap.parse_args()
    try:
        base = json.loads(Path(args.baseline).read_text())
        cur = json.loads(Path(args.current).read_text())
    except Exception as e:
        print(f"ERROR: unable to load specs: {e}")
        sys.exit(1)
    base_keys = flatten_keys(base)
    cur_keys = flatten_keys(cur)
    missing = sorted(k for k in base_keys if k not in cur_keys)
    if missing:
        print("BREAKING: keys removed:")
        for k in missing[:50]:
            print(" -", k)
        if len(missing) > 50:
            print(f" ... (+{len(missing) - 50} more)")
        sys.exit(2)
    print("OpenAPI diff check passed (no removals)")


if __name__ == "__main__":
    main()
