from __future__ import annotations

# Standard library
import os

# Third-party
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Local application imports
from .admin import router as admin_router
from .autocorrect import router as autocorrect_router
from .benchmarks import router as benchmarks_router
from .billing_webhook import router as billing_webhook_router
from .jobs import router as jobs_router
from .settings import AppSettings, get_app_settings


def _truthy(val: str | None) -> bool:
    return val in {"1", "true", "TRUE", "on", "On", "yes", "YES"}


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    This function encapsulates middleware and router wiring so tests and
    entrypoints can import a single factory. Behavior is intentionally kept
    identical to the previous in-module setup in main.py.
    """
    s = settings or get_app_settings()

    app = FastAPI(title="Oscillink Cloud API", default_response_class=ORJSONResponse)

    # CORS from env/settings
    allow_origins_raw = (
        s.cors_allow_origins_raw or os.getenv("OSCILLINK_CORS_ALLOW_ORIGINS", "").strip()
    )
    if allow_origins_raw:
        origins = [o.strip() for o in allow_origins_raw.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["POST", "GET", "OPTIONS", "DELETE"],
            allow_headers=["*"],
            max_age=600,
        )

    # Trusted hosts
    trusted_hosts_raw = s.trusted_hosts_raw or os.getenv("OSCILLINK_TRUSTED_HOSTS", "").strip()
    if trusted_hosts_raw:
        hosts = [h.strip() for h in trusted_hosts_raw.split(",") if h.strip()]
        # Append localhost defaults unless explicitly disabled
        if (
            _truthy(os.getenv("OSCILLINK_TRUSTED_ADD_LOCAL", "1"))
            if settings is None
            else s.trusted_add_local
        ):
            for h in ("localhost", "127.0.0.1"):
                if h not in hosts:
                    hosts.append(h)
        # Optionally allow Cloud Run host wildcard patterns
        if (
            _truthy(os.getenv("OSCILLINK_TRUSTED_ALLOW_CLOUDRUN", "0"))
            if settings is None
            else s.trusted_allow_cloudrun
        ):
            for h in ("*.a.run.app", "*.run.app"):
                if h not in hosts:
                    hosts.append(h)
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    # HTTPS redirect
    if s.force_https:
        app.add_middleware(HTTPSRedirectMiddleware)

    # Routers (same as before)
    app.include_router(autocorrect_router)
    app.include_router(benchmarks_router)
    app.include_router(billing_webhook_router)
    app.include_router(jobs_router)
    app.include_router(admin_router)

    return app
