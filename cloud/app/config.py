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

@lru_cache
def get_settings() -> Settings:
    return Settings()
