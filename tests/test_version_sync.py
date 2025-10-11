from __future__ import annotations

import re
from pathlib import Path

import oscillink


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    m = re.search(r"^version\s*=\s*\"([^\"]+)\"", text, flags=re.M)
    assert m, "pyproject.toml missing project version"
    py_version = m.group(1)
    # When running from source in editable mode, importlib.metadata should resolve
    assert oscillink.__version__ == py_version
