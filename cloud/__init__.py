import sys

# Optional shim to avoid PendingDeprecationWarning from Starlette importing
# the legacy 'multipart' package name. If python_multipart is installed,
# pre-register it under the legacy name so Starlette finds it without
# triggering the deprecation warning.
try:  # pragma: no cover - behavior-only import
    import python_multipart as _pm  # type: ignore

    sys.modules.setdefault("multipart", _pm)  # type: ignore[assignment]
except Exception:
    # It's an optional dependency; cloud extras install includes it.
    pass

__all__: list[str] = []  # Marker file to ensure 'cloud' package inclusion in builds
