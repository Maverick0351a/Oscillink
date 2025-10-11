#!/usr/bin/env python
"""Export the FastAPI OpenAPI schema to a JSON file.

Usage:
  python -m scripts.export_openapi [--out openapi.json]

If --out is omitted, writes ./openapi.json (overwrites).

Intended for CI / release automation to publish a stable schema artifact.
"""

from __future__ import annotations

import argparse
import json
from importlib import reload
from pathlib import Path

# Ensure app import picks up any env changes (API version etc.)
from cloud.app import main as mainmod  # type: ignore

reload(mainmod)

app = mainmod.app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("openapi.json"), help="Output path")
    args = p.parse_args()
    schema = app.openapi()
    # Remove unused servers list or other transient keys if desired (keep full fidelity for now)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, sort_keys=True)
    print(f"Wrote OpenAPI schema to {args.out}")


if __name__ == "__main__":
    main()
