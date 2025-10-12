from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _truthy(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val in {"1", "true", "TRUE", "on", "On", "yes", "YES"}


@dataclass(frozen=True)
class AppSettings:
    cors_allow_origins_raw: str
    trusted_hosts_raw: str
    trusted_add_local: bool
    trusted_allow_cloudrun: bool
    force_https: bool
    max_body_bytes: int

    @property
    def cors_allow_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins_raw.split(",") if o.strip()]

    @property
    def trusted_hosts(self) -> list[str]:
        hosts = [h.strip() for h in self.trusted_hosts_raw.split(",") if h.strip()]
        if self.trusted_add_local:
            for h in ("localhost", "127.0.0.1"):
                if h not in hosts:
                    hosts.append(h)
        if self.trusted_allow_cloudrun:
            for h in ("*.a.run.app", "*.run.app"):
                if h not in hosts:
                    hosts.append(h)
        return hosts


@lru_cache
def get_app_settings() -> AppSettings:
    return AppSettings(
        cors_allow_origins_raw=os.getenv("OSCILLINK_CORS_ALLOW_ORIGINS", "").strip(),
        trusted_hosts_raw=os.getenv("OSCILLINK_TRUSTED_HOSTS", "").strip(),
        trusted_add_local=_truthy(os.getenv("OSCILLINK_TRUSTED_ADD_LOCAL", "1"), True),
        trusted_allow_cloudrun=_truthy(os.getenv("OSCILLINK_TRUSTED_ALLOW_CLOUDRUN", "0"), False),
        force_https=_truthy(os.getenv("OSCILLINK_FORCE_HTTPS", "0"), False),
        max_body_bytes=int(os.getenv("OSCILLINK_MAX_BODY_BYTES", "1048576")),
    )
