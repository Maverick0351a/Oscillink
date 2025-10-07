from __future__ import annotations
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from .models import SettleRequest, ReceiptResponse, HealthResponse
from .config import get_settings
from oscillink import OscillinkLattice, __version__
import numpy as np

app = FastAPI(title="Oscillink Cloud API", default_response_class=ORJSONResponse)
settings = get_settings()

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version=__version__)

@app.post(f"/{settings.api_version}/settle", response_model=ReceiptResponse)
def settle(req: SettleRequest):
    Y = np.array(req.Y, dtype=np.float32)
    N, D = Y.shape
    if N == 0 or D == 0:
        raise HTTPException(status_code=400, detail="Empty matrix")
    if N > settings.max_nodes:
        raise HTTPException(status_code=413, detail=f"N>{settings.max_nodes} exceeds limit")
    if D > settings.max_dim:
        raise HTTPException(status_code=413, detail=f"D>{settings.max_dim} exceeds limit")

    lat = OscillinkLattice(
        Y,
        kneighbors=req.params.kneighbors,
        lamG=req.params.lamG,
        lamC=req.params.lamC,
        lamQ=req.params.lamQ,
        deterministic_k=req.params.deterministic_k,
        neighbor_seed=req.params.neighbor_seed,
    )
    if req.psi is not None:
        psi = np.array(req.psi, dtype=np.float32)
        if psi.shape[0] != D:
            raise HTTPException(status_code=400, detail="psi dimension mismatch")
        lat.set_query(psi)
    if req.gates is not None:
        gates = np.array(req.gates, dtype=np.float32)
        if gates.shape[0] != N:
            raise HTTPException(status_code=400, detail="gates length mismatch")
        lat.set_gates(gates)
    if req.chain:
        if len(req.chain) < 2:
            raise HTTPException(status_code=400, detail="chain must have >=2 nodes")
        lat.add_chain(req.chain, lamP=req.params.lamP)

    t0 = time.time()
    lat.settle(dt=req.options.dt, max_iters=req.options.max_iters, tol=req.options.tol)
    t_settle = 1000.0 * (time.time() - t0)

    receipt = None
    bundle = None
    if req.options.include_receipt:
        receipt = lat.receipt()
    if req.options.bundle_k:
        bundle = lat.bundle(k=req.options.bundle_k)

    # derive minimal meta subset
    meta = {
        "N": int(N),
        "D": int(D),
        "kneighbors": req.params.kneighbors,
        "lam": {"G": req.params.lamG, "C": req.params.lamC, "Q": req.params.lamQ, "P": req.params.lamP},
    }
    state_sig = receipt.get("meta", {}).get("state_sig") if receipt else lat._signature()

    return ReceiptResponse(
        state_sig=state_sig,
        receipt=receipt,
        bundle=bundle,
        timings_ms={"total_settle_ms": t_settle},
        meta=meta,
    )

# CLI entrypoint for uvicorn
# uvicorn cloud.app.main:app --reload --port 8000
