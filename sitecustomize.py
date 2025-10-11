"""
Test/runtime shims loaded automatically if this directory is on sys.path.

We pre-register python_multipart under the legacy 'multipart' name to avoid
Starlette's PendingDeprecationWarning during import time in tests.
"""

from __future__ import annotations

import sys
import warnings

try:  # pragma: no cover - behavior-only import
    import python_multipart as _pm  # type: ignore

    sys.modules.setdefault("multipart", _pm)  # type: ignore[assignment]
except Exception:
    pass

# Silence upstream PendingDeprecationWarning emitted by Starlette when importing
# the legacy 'multipart' name. This is informational and does not affect runtime
# behavior when python-multipart is installed.
warnings.filterwarnings(
    "ignore",
    message=r"Please use `import python_multipart` instead\.",
    category=PendingDeprecationWarning,
    module=r"starlette\.formparsers",
)
