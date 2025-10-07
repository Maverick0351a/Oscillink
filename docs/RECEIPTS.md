# Receipts Schema (JSON)

Top-level fields:
- `deltaH_total` (float): exact energy gap (non‑negative).
- `coh_drop_sum`, `anchor_pen_sum`, `query_term_sum` (floats): component aggregates.
- `cg_iters`, `residual`, `t_ms` (diagnostics).
- `null_points`: list of edges with high z‑score residuals.

## Chain Receipt
- `verdict` (bool): pass iff all chain edges' z‑scores <= threshold and coherence gain >= 0.
- `weakest_link`: `{k, edge: [i,j], zscore}`.
- `coherence_gain` (float): chain coherence improvement vs anchors.
- `edges`: per‑edge `{k, edge, z_struct, z_path, r_struct, r_path}`.
