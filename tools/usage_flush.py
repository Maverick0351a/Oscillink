from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import time
from typing import Any
from urllib import error, request

"""
Background task that tails OSCILLINK_USAGE_LOG and periodically sends batches to
OSCILLINK_USAGE_FLUSH_URL as JSON. Best-effort only; failures are retried with backoff.

Environment:
- OSCILLINK_USAGE_LOG: path to local JSONL file (written by cloud.app.services.usage_log)
- OSCILLINK_USAGE_FLUSH_URL: remote endpoint (e.g., https://license.oscillink.com/v1/usage/report)
- OSCILLINK_LICENSE_ID: license id (sub) to include
- OSCILLINK_USAGE_SIGNING_SECRET: optional HMAC secret to sign the batch
- OSCILLINK_FLUSH_INTERVAL: seconds between idle polls (default 120)
- OSCILLINK_FLUSH_BATCH_MAX: max lines per batch (default 200)
- OSCILLINK_FLUSH_MAX_RETRY: max retries per batch (default 5)
"""


def _post_json(url: str, payload: dict[str, Any]) -> int:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=5) as resp:  # nosec - controlled URL
            _ = resp.read()
            return int(getattr(resp, "status", 200))
    except error.HTTPError as e:
        return int(getattr(e, "code", 500))
    except Exception:
        return 0


def _send_with_backoff(url: str, body: dict[str, Any], max_retry: int) -> bool:
    attempt = 0
    while True:
        status = _post_json(url, body)
        if status and 200 <= status < 300:
            return True
        attempt += 1
        if attempt > max_retry:
            return False
        backoff = min(30.0, (2 ** min(attempt, 5)) + random.uniform(0, 0.5))
        time.sleep(backoff)


def _build_batch(
    lines_raw: list[str], lic: str, interval: int, sent: int, failed: int
) -> dict[str, Any]:
    try:
        parsed = [json.loads(x) for x in lines_raw]
    except Exception:
        parsed = []
    raw_concat = ("\n".join(lines_raw)).encode("utf-8")
    bucket = int(time.time() // max(1, interval))
    idem = hashlib.sha256(raw_concat + str(bucket).encode()).hexdigest()
    body: dict[str, Any] = {
        "license_id": lic,
        "ts": int(time.time()),
        "lines": parsed,
        "idempotency_key": idem,
        "counters": {"sent": sent, "failed": failed},
    }
    secret = os.getenv("OSCILLINK_USAGE_SIGNING_SECRET")
    if secret:
        msg = json.dumps({"license_id": lic, "lines": parsed}, separators=(",", ":")).encode()
        mac = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
        body["hmac"] = mac
    return body


def _read_new_lines(path: str, start: int, limit: int) -> tuple[list[str], int]:
    with open(path, encoding="utf-8") as f:
        f.seek(start)
        out: list[str] = []
        for _ in range(limit):
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if s:
                out.append(s)
        pos = f.tell()
    return out, pos


def _should_sleep(log_path: str, last_size: int) -> tuple[bool, int]:
    if not os.path.exists(log_path):
        return True, last_size
    size = os.path.getsize(log_path)
    if size == last_size:
        return True, last_size
    return False, size


def run_loop() -> None:
    log_path = os.getenv("OSCILLINK_USAGE_LOG")
    flush_url = os.getenv("OSCILLINK_USAGE_FLUSH_URL")
    lic = os.getenv("OSCILLINK_LICENSE_ID")
    if not log_path or not flush_url or not lic:
        return
    try:
        interval = int(os.getenv("OSCILLINK_FLUSH_INTERVAL", "120"))
    except ValueError:
        interval = 120
    try:
        batch_max = int(os.getenv("OSCILLINK_FLUSH_BATCH_MAX", "200"))
    except ValueError:
        batch_max = 200

    last_size = 0
    sent_batches = 0
    failed_batches = 0
    while True:
        try:
            sleep, _ = _should_sleep(log_path, last_size)
            if sleep:
                time.sleep(interval)
                continue
            lines_raw, new_pos = _read_new_lines(log_path, last_size, batch_max)
            if lines_raw:
                body = _build_batch(lines_raw, lic, interval, sent_batches, failed_batches)
                max_retry = int(os.getenv("OSCILLINK_FLUSH_MAX_RETRY", "5") or "5")
                ok = _send_with_backoff(flush_url, body, max_retry)
                if ok:
                    sent_batches += 1
                else:
                    failed_batches += 1
            last_size = new_pos
        except Exception:
            pass
        time.sleep(interval)


if __name__ == "__main__":
    run_loop()
