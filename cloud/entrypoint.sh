#!/usr/bin/env bash
set -euo pipefail

LICENSE_PATH="${OSCILLINK_LICENSE_PATH:-}"
JWKS_URL="${OSCILLINK_JWKS_URL:-}"

if [[ -z "${LICENSE_PATH:-}" ]] || [[ ! -f "${LICENSE_PATH}" ]]; then
  echo "ERROR: License file not found at ${LICENSE_PATH:-<unset>}" >&2
  exit 90
fi
if [[ -z "${JWKS_URL:-}" ]]; then
  echo "ERROR: JWKS URL is not set (OSCILLINK_JWKS_URL)" >&2
  exit 91
fi

# Verify license; writes entitlements to /run/oscillink_entitlements.json
python /app/tools/license_verify.py --license "${LICENSE_PATH}" --jwks "${JWKS_URL}" || exit 92

# Optionally export select entitlements as env for the app
if [[ -f /run/oscillink_entitlements.env ]]; then
  set -a
  source /run/oscillink_entitlements.env
  set +a
fi

# Start background usage flusher if configured
if [[ -n "${OSCILLINK_USAGE_LOG:-}" ]] && [[ -n "${OSCILLINK_USAGE_FLUSH_URL:-}" ]]; then
  python /app/tools/usage_flush.py &
fi

# Run the API
exec uvicorn cloud.app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
