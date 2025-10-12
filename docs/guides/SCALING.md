# Scaling Oscillink: From Single Lattice to Lattice of Lattices

This document outlines how Oscillink Lattice scales from a single in–process working memory to hierarchical, horizontally distributed “lattices of lattices” while preserving the same **physics–based SPD energy contract**.

## 1. Core Scaling Principles

| Aspect | Single Lattice | Horizontal / Hierarchical Extension |
|--------|----------------|--------------------------------------|
| Data Unit | Anchor node (embedding) | Shard summary node / super–node |
| Objective | Minimize ΔH over (Y, ψ, gates) | Minimize ΔH locally + ΔH over shard summaries |
| Coherence Bridge | Normalized Laplacian L_sym | Inter–shard Laplacian over summary embeddings |
| Trust/Gating | Per node gates in [0,1] | Gate both raw nodes & shard summaries |
| Chain Priors | Path Laplacian on node indices | Path across super–nodes + optional intra–path refinement |
| Explainability | Single receipt (ΔH, null points, chain verdict) | Per–shard receipts + folded summary receipt |

## 2. Vertical Scaling (Increasing N, D)

The baseline implementation targets fast construction for mid–scale (N ≲ 50k) with:
- Mutual kNN (O(kN log N) typical with partial sorts)
- CG over SPD system (iterations depend on spectrum; preconditioning improves with future releases)

Optimizations (planned / optional):
- Approximate neighbor graph (NN-descent / HNSW) – must preserve determinism gates or supply receipt note.
- Blocked CG / multi-RHS fusion for batching multiple query attraction vectors.
- Mixed precision (float32 anchors + float16 internal ops) with residual correction.

## 3. Horizontal Scaling (Partitioning)

Partition the anchor set into disjoint shards S_i. For each shard:
1. Build local lattice L_i and settle → receipt R_i (local ΔH_i, null points, chain verdicts).
2. Extract a **summary embedding** per shard (e.g., weighted centroid of top coherent nodes or stationary solution mean). Optionally attach summary diagnostics (average gate, anomaly density).
3. Form a super–lattice over the shard summaries (size m = number of shards). Solve once to obtain inter–shard coherent ordering.
4. Merge: select top shards by super–lattice bundle, then widen within each shard using local bundles.

Receipt Composition:
- Global ΔH_total ≈ Σ ΔH_i (local improvements) + ΔH_super (cross–shard energy) – overlap corrections.
- Provide layered receipt structure:
```jsonc
{
  "version": "1.0",
  "layers": {
    "shards": [ { "id": 0, "deltaH": 123.4, "null_points": 4, ... }, ... ],
    "super_lattice": { "deltaH": 17.8, "null_points": 1 }
  },
  "composed": { "deltaH_total": 141.2 }
}
```

## 4. Lattice of Lattices (Recursive)

Apply the horizontal pattern recursively:
- Level 0: raw nodes
- Level 1: shard summaries
- Level 2: region summaries (e.g., topical clusters, time buckets)

Each level remains an SPD solve; invariants:
- λ_G > 0 at every level ensures SPD
- Summaries encoded as pseudo–anchors with provenance back–pointers
- Gates can propagate downward (if a shard summary is suppressed, its children’s effective gates are scaled)

## 5. Streaming & Incremental Updates

For append–only streams:
1. Buffer new nodes
2. Periodically integrate via micro–lattice solve
3. Update summary embedding; re–solve only super–level (cheap: small m)
4. Provide delta receipts (ΔH_since_last)

Potential optimization: maintain low–rank updates to Laplacian rather than rebuilding full neighbor graph. Determinism preserved by stable seeding.

## 6. Fault Domains & Isolation

When deployed across processes / machines:
- Each shard produces independently verifiable receipts (HMAC optional)
- Super–lattice orchestrator only needs summary vectors + minimal statistics (privacy boundary)
- Failed shard can be omitted; receipt marks it missing (integrity transparency)

## 7. Scaling Limits & Practical Guidance

| Dimension | Consideration | Mitigation |
|-----------|---------------|------------|
| Memory | O(ND + kN) | Shard; compress anchors (PQ / fp16); lazy load |
| Solve Time | CG iteration growth with condition number | Preconditioner (diag / ILU), better ordering |
| Graph Build | kNN cost | Approximate kNN + audit sample; reuse previous graph |
| Explainability Cost | Aggregating receipts | Compress node diagnostics; sample null points |

## 8. Future Enhancements
- Adaptive shard sizing (variance-aware)
- Cross–level constraint priors (enforce thematic continuity across hierarchy)
- Multi–query joint lattice (simultaneous ψ set; shared solve via block system)
- Fully incremental Laplacian updates (rank–k corrections)

## 9. API Sketch (Future)

```python
from oscillink.scaling import HierarchicalLattice
hl = HierarchicalLattice(levels=[{'max_nodes': 5000}, {'max_shards': 32}])
hl.ingest(batch_embeddings, ids=batch_ids)
receipt = hl.receipt()
final_bundle = hl.bundle(k=20)
```

_Status: conceptual; not yet in Phase 1 core. This doc sets terminology and invariants so future implementation aligns with existing receipts._
