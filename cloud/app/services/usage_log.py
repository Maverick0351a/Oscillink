from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any


def append_usage(record: dict[str, Any]) -> None:
    """Append a JSON line to the usage log if configured.

    Adds a compact signature field when OSCILLINK_USAGE_SIGNING_SECRET is present.
    Failures are swallowed — logging is best-effort.
    """
    log_path = os.getenv("OSCILLINK_USAGE_LOG")
    if not log_path:
        return
    try:
        payload = json.dumps(record, separators=(",", ":"), sort_keys=True)
        signing_secret = os.getenv("OSCILLINK_USAGE_SIGNING_SECRET")
        if signing_secret:
            sig = hmac.new(
                signing_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            record = {**record, "sig": {"alg": "HS256", "h": sig}}
        dir_part = os.path.dirname(log_path)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception:
        pass
