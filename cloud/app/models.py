from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, conlist

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
    Y: list[conlist(float, min_items=1)] = Field(..., description="Matrix N x D")
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
