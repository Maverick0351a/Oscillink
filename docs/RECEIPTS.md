This document has moved.

New location: [docs/reference/RECEIPTS.md](./reference/RECEIPTS.md)
# Receipts Schema (JSON)

Top-level fields:
- `deltaH_total` (float): exact energy gap (non‑negative).
- `coh_drop_sum`, `anchor_pen_sum`, `query_term_sum` (floats): component aggregates.
- `cg_iters`, `residual`, `t_ms` (diagnostics).
- `null_points`: list of edges with high z‑score residuals.

## Receipt detail modes

You can control which diagnostics are computed to trade detail for latency:

- full (default): includes per‑node diagnostics (coh_drop, anchor_pen, query_term) and the full list of `null_points`.
- light: computes only the core energy delta (`deltaH_total`) and metadata; skips per‑node diagnostics and `null_points` discovery. This reduces receipt time substantially on large N.

How to set (SDK):

```python
lat = Oscillink(Y, kneighbors=6)
lat.set_receipt_detail("light")
rec = lat.receipt()
```

The receipt `meta` block includes `receipt_detail` and a `null_points_summary` indicating whether a cap was applied.

## Chain Receipt
- `verdict` (bool): pass iff all chain edges' z‑scores <= threshold and coherence gain >= 0.
- `weakest_link`: `{k, edge: [i,j], zscore}`.
- `coherence_gain` (float): chain coherence improvement vs anchors.
- `edges`: per‑edge `{k, edge, z_struct, z_path, r_struct, r_path}`.
