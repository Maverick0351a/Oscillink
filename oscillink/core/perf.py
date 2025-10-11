from __future__ import annotations

from typing import Any, Dict


def compare_perf(
    baseline: Dict[str, Any], current: Dict[str, Any], metrics=None, tolerance_pct: float = 20.0
):
    """Compare aggregate means in benchmark JSON objects.

    Returns dict with deviations and failures list.
    """
    if metrics is None:
        metrics = ["build_ms", "settle_ms", "receipt_ms"]
    failures = []
    deviations = {}
    for m in metrics:
        bmean = baseline["aggregates"][m]["mean"]
        cmean = current["aggregates"][m]["mean"]
        if bmean <= 0:
            continue
        pct = 100.0 * (cmean - bmean) / bmean
        deviations[m] = pct
        if pct > tolerance_pct:
            failures.append({"metric": m, "pct": pct, "baseline": bmean, "current": cmean})
    return {"deviations": deviations, "failures": failures, "tolerance_pct": tolerance_pct}
