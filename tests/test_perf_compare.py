from oscillink.core.perf import compare_perf


def test_compare_perf_default_metrics_and_failures():
    baseline = {
        "aggregates": {
            "build_ms": {"mean": 10.0},
            "settle_ms": {"mean": 10.0},
            "receipt_ms": {"mean": 5.0},
        }
    }
    current = {
        "aggregates": {
            "build_ms": {"mean": 11.0},
            "settle_ms": {"mean": 15.0},
            "receipt_ms": {"mean": 7.0},
        }
    }
    out = compare_perf(baseline, current, tolerance_pct=20.0)
    # build within 20%, settle and receipt > 20%
    assert out["failures"] and any(f["metric"] == "settle_ms" for f in out["failures"])


def test_compare_perf_custom_metrics():
    baseline = {"aggregates": {"x": {"mean": 100.0}}}
    current = {"aggregates": {"x": {"mean": 110.0}}}
    out = compare_perf(baseline, current, metrics=["x"], tolerance_pct=5.0)
    assert out["deviations"]["x"] == 10.0


def test_compare_perf_baseline_nonpositive():
    baseline = {"aggregates": {"z": {"mean": 0.0}}}
    current = {"aggregates": {"z": {"mean": 999.0}}}
    out = compare_perf(baseline, current, metrics=["z"], tolerance_pct=1.0)
    # No failure recorded because baseline mean <= 0 skips comparison
    assert out["failures"] == []
