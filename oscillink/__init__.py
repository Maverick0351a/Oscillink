import re
from pathlib import Path

from .core.lattice import OscillinkLattice  # noqa: F401
from .core.perf import compare_perf  # noqa: F401
from .core.provenance import compare_provenance  # noqa: F401
from .core.receipts import verify_receipt, verify_receipt_mode  # noqa: F401
from .preprocess.diffusion import compute_diffusion_gates  # noqa: F401

# Public alias preferred in docs
Oscillink = OscillinkLattice

__all__ = [
    "Oscillink",
    "OscillinkLattice",
    "verify_receipt",
    "verify_receipt_mode",
    "compare_perf",
    "compare_provenance",
    "compute_diffusion_gates",
]
try:
    # Prefer package metadata when installed (including editable installs)
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("oscillink")
except Exception:
    # Fallback default (will be overridden by pyproject probe below when available)
    __version__ = "0.1.9"

# In development/editable mode, a local pyproject.toml may be present; prefer its version
try:
    _root = Path(__file__).resolve().parents[1]
    _py = _root / "pyproject.toml"
    if _py.exists():
        _text = _py.read_text(encoding="utf-8")
        _m = re.search(r"(?m)^version\s*=\s*\"([^\"]+)\"", _text)
        if _m:
            __version__ = _m.group(1)
except Exception:
    pass
