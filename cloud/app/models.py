from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Params(BaseModel):
    lamG: float = 1.0
    lamC: float = 0.5
    lamQ: float = 4.0
    lamP: float = 0.0
    kneighbors: int = 6
    deterministic_k: bool = False
    neighbor_seed: Optional[int] = None


class SettleOptions(BaseModel):
    max_iters: int = 12
    tol: float = 1e-3
    dt: float = 1.0
    bundle_k: int | None = None
    include_receipt: bool = True


class SettleRequest(BaseModel):
    # Y is a list of rows (each a list[float]); shape validation performed in endpoint logic.
    Y: list[list[float]] = Field(..., description="Matrix N x D (list of rows)")
    psi: Optional[list[float]] = None
    gates: Optional[list[float]] = None
    chain: Optional[list[int]] = None
    params: Params = Params()
    options: SettleOptions = SettleOptions()


class ReceiptResponse(BaseModel):
    state_sig: str
    receipt: dict | None = None
    bundle: list[dict] | None = None
    timings_ms: dict
    meta: dict


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class AdminKeyUpdate(BaseModel):
    tier: str | None = None
    status: str | None = None  # active|revoked|suspended|pending
    quota_limit_units: int | None = None
    quota_window_seconds: int | None = None
    features: dict[str, bool] | None = None


class AdminKeyResponse(BaseModel):
    api_key: str
    tier: str
    status: str
    quota_limit_units: int | None = None
    quota_window_seconds: int | None = None
    features: dict[str, bool] = {}
    created_at: float
    updated_at: float
