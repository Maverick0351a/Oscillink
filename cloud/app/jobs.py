from __future__ import annotations

# Standard library
import time
import uuid
from typing import Any

# Third-party
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from .config import get_settings
from .features import resolve_features
from .keystore import get_keystore  # type: ignore
from .models import SettleRequest

router = APIRouter()

_API_VERSION = get_settings().api_version


# In-memory async job store (non-persistent, single-process)
_jobs: dict[str, dict[str, Any]] = {}
try:
    _JOB_TTL_SEC = int(__import__("os").getenv("OSCILLINK_JOB_TTL", "3600"))
except Exception:
    _JOB_TTL_SEC = 3600


def _purge_old_jobs() -> None:
    now = time.time()
    expired = [jid for jid, rec in _jobs.items() if now - rec.get("created", now) > _JOB_TTL_SEC]
    for jid in expired:
        _jobs.pop(jid, None)
    # Update gauge if available
    try:
        from cloud.app import main as m

        if "JOB_QUEUE_DEPTH" in m.__dict__:
            m.JOB_QUEUE_DEPTH.set(len(_jobs))  # type: ignore
    except Exception:
        pass


def _get_feature_context(request: Request) -> dict[str, Any]:
    """Rebuild feature context using main's guard and local resolvers to avoid import cycles."""
    try:
        from cloud.app import main as m

        x_api_key = m.api_key_guard(x_api_key=request.headers.get("x-api-key"))
    except HTTPException:
        # Propagate auth errors
        raise
    except Exception:
        x_api_key = None
    meta = get_keystore().get(x_api_key) if x_api_key else None
    feats = resolve_features(meta)
    return {"api_key": x_api_key, "features": feats}


def _guard_diffusion(req: SettleRequest, feats: Any) -> None:
    if req.gates is None:
        return
    if __import__("os").getenv("OSCILLINK_DIFFUSION_GATES_ENABLED", "1") not in {
        "1",
        "true",
        "TRUE",
        "on",
    }:
        raise HTTPException(status_code=403, detail="diffusion gating temporarily disabled")
    if not feats.diffusion_allowed:
        raise HTTPException(status_code=403, detail="diffusion gating not enabled for this tier")


def _run_job_task(
    job_id: str, created: float, req: SettleRequest, x_api_key: str | None, request: Request
):
    try:
        # Lazy import to avoid circular dependencies
        from cloud.app import main as m

        lat, N, D, k_eff, eff_params, profile_id = m._build_lattice(req, x_api_key)
        # Quota checks at execution time
        try:
            units = N * D
            monthly_ctx = m._check_monthly_cap(x_api_key, units)
            remaining, limit, reset_at = m._check_and_consume_quota(x_api_key, units)
        except HTTPException as he:
            _jobs[job_id] = {
                "status": "error",
                "error": he.detail,
                "created": created,
                "quota_error": True,
            }
            return
        t0 = time.time()
        settle_stats = lat.settle(
            dt=req.options.dt, max_iters=req.options.max_iters, tol=req.options.tol
        )
        elapsed = time.time() - t0
        rec = lat.receipt() if req.options.include_receipt else None
        bundle = lat.bundle(k=req.options.bundle_k) if req.options.bundle_k else None
        # Metrics
        try:
            m.USAGE_NODES.inc(N)  # type: ignore
            m.USAGE_NODE_DIM_UNITS.inc(N * D)  # type: ignore
        except Exception:
            pass
        _jobs[job_id] = {
            "status": "done",
            "created": created,
            "completed": time.time(),
            "result": {
                "state_sig": rec.get("meta", {}).get("state_sig") if rec else lat._signature(),
                "receipt": rec,
                "bundle": bundle,
                "timings_ms": {"total_settle_ms": 1000.0 * elapsed},
                "meta": {
                    "N": N,
                    "D": D,
                    "kneighbors_requested": req.params.kneighbors,
                    "kneighbors_effective": k_eff,
                    "profile_id": profile_id,
                    "request_id": request.headers.get(m.REQUEST_ID_HEADER, ""),
                    "usage": {
                        "nodes": N,
                        "node_dim_units": units,
                        "monthly": None
                        if not monthly_ctx
                        else {
                            "limit": monthly_ctx["limit"],
                            "used": monthly_ctx["used"],
                            "remaining": monthly_ctx["remaining"],
                            "period": monthly_ctx["period"],
                        },
                    },
                    "quota": None
                    if limit == 0
                    else {"limit": limit, "remaining": remaining, "reset": int(reset_at)},
                },
            },
        }
        # Learning hook (best-effort)
        try:
            from cloud.app.learners import record_observation as _record_observation

            _record_observation(
                x_api_key,
                profile_id,
                {
                    "lamG": eff_params["lamG"],
                    "lamC": eff_params["lamC"],
                    "lamQ": eff_params["lamQ"],
                    "kneighbors": k_eff,
                },
                {
                    "duration_ms": 1000.0 * elapsed,
                    "iters": int(settle_stats.get("iters", 0)),
                    "residual": float(settle_stats.get("res", 0.0)),
                    "tol": float(req.options.tol),
                },
            )
        except Exception:
            pass
        # Usage logging (best-effort)
        try:
            from cloud.app.services.usage_log import append_usage as _append_usage

            _append_usage(
                {
                    "ts": time.time(),
                    "event": "job_settle",
                    "api_key": x_api_key,
                    "job_id": job_id,
                    "N": N,
                    "D": D,
                    "units": units,
                    "duration_ms": 1000.0 * elapsed,
                    "quota": None
                    if limit == 0
                    else {"limit": limit, "remaining": remaining, "reset": int(reset_at)},
                    "monthly": None
                    if not monthly_ctx
                    else {
                        "limit": monthly_ctx["limit"],
                        "used": monthly_ctx["used"],
                        "remaining": monthly_ctx["remaining"],
                        "period": monthly_ctx["period"],
                    },
                }
            )
        except Exception:
            pass
    except Exception as e:
        _jobs[job_id] = {"status": "error", "error": str(e), "created": created}
    try:
        from cloud.app import main as m

        m.JOB_QUEUE_DEPTH.set(len(_jobs))  # type: ignore
    except Exception:
        pass


@router.post(f"/{_API_VERSION}/jobs/settle")
def submit_job(req: SettleRequest, background: BackgroundTasks, request: Request):
    ctx = _get_feature_context(request)
    x_api_key = ctx["api_key"]
    feats = ctx["features"]
    _guard_diffusion(req, feats)

    job_id = uuid.uuid4().hex
    created = time.time()
    _purge_old_jobs()
    _jobs[job_id] = {"status": "queued", "created": created}
    try:
        from cloud.app import main as m

        m.JOB_QUEUE_DEPTH.set(len(_jobs))  # type: ignore
    except Exception:
        pass

    background.add_task(_run_job_task, job_id, created, req, x_api_key, request)
    return {"job_id": job_id, "status": "queued"}


@router.get(f"/{_API_VERSION}/jobs/{{job_id}}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.delete(f"/{_API_VERSION}/jobs/{{job_id}}")
def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") in {"done", "error"}:
        return {"job_id": job_id, "status": job["status"], "note": "already finished"}
    job["status"] = "cancelled"
    try:
        from cloud.app import main as m

        m.JOB_QUEUE_DEPTH.set(len(_jobs))  # type: ignore
    except Exception:
        pass
    return {"job_id": job_id, "status": "cancelled"}
