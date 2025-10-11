#!/usr/bin/env python
"""Compare two OpenAPI schema JSON files focusing on path/method presence.

Usage:
  python -m scripts.check_openapi_diff_simple --prev prev.json --current current.json

Exits non-zero if a path+method present in prev is missing in current (breaking).
Add --allow-removed PATH1,PATH2 to tolerate specific removals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", type=Path, required=True)
    ap.add_argument("--current", type=Path, required=True)
    ap.add_argument(
        "--allow-removed",
        type=str,
        default="",
        help="Comma list of path patterns to ignore (exact match)",
    )
    args = ap.parse_args()
    prev = load(args.prev)
    cur = load(args.current)
    allow = {s.strip() for s in args.allow_removed.split(",") if s.strip()}
    missing = []
    for path, _ops in prev.get("paths", {}).items():
        if path not in cur.get("paths", {}):
            if path not in allow:
                missing.append(path)
            continue
    for method in _ops:
        if method not in cur["paths"][path]:
            combo = f"{path}:{method}"
            if combo not in allow:
                missing.append(combo)
    if missing:
        sys.stderr.write(
            "Breaking changes detected (removed paths/methods):\n" + "\n".join(missing) + "\n"
        )
        sys.exit(1)
    print("No breaking path/method removals detected.")


if __name__ == "__main__":
    main()
