from __future__ import annotations

import os
from functools import lru_cache


class Settings:
    project_name: str = "oscillink-cloud"
    api_version: str = "v1"
    max_nodes: int = int(os.getenv("OSCILLINK_MAX_NODES", "5000"))
    max_dim: int = int(os.getenv("OSCILLINK_MAX_DIM", "2048"))
    enable_signature: bool = os.getenv("OSCILLINK_ENABLE_SIGNATURE", "1") == "1"
    receipt_secret: str | None = os.getenv("OSCILLINK_RECEIPT_SECRET")
    api_keys_raw: str | None = os.getenv("OSCILLINK_API_KEYS")  # comma-separated allowed keys

    @property
    def api_keys(self) -> set[str]:
        if not self.api_keys_raw:
            return set()
        return {k.strip() for k in self.api_keys_raw.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
