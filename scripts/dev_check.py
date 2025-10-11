#!/usr/bin/env python
"""Convenience dev check script.

Runs: ruff (fix), mypy (errors only), pytest with coverage summary.

Usage:
  python scripts/dev_check.py
"""

from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> int:
    print(f"\n>> {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> int:
    failures = 0
    failures += run([sys.executable, "-m", "ruff", "check", ".", "--fix"])
    failures += run([sys.executable, "-m", "mypy", "oscillink"])
    failures += run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--maxfail=1",
            "-q",
            "--cov=oscillink",
            "--cov-report=term",
        ]
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
