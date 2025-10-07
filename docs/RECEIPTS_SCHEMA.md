# Oscillink Receipts JSON Schema (v1)

`$id`: `https://oscillink.org/schema/receipts-v1.json`

`$schema`: `https://json-schema.org/draft/2020-12/schema`

Purpose: Stable, extensible contract for programmatic analysis, storage, and signing of lattice receipts.

## Top-Level Receipt (Base)
```json
{
  "version": "1.0.0",        // schema version
  "deltaH_total": 12.345,      // non-negative energy gap
  "coh_drop_sum": 5.12,        // aggregate coherence drop
  "anchor_pen_sum": 3.77,      // anchor fidelity penalty sum
  "query_term_sum": 3.45,      // query alignment penalty sum
  "cg_iters": 8,               // iterations in last implicit settle
  "residual": 0.00092,         // final CG residual (max norm over RHS)
  "t_ms": 1.72,                // time of last settle (milliseconds)
  "null_points": [             // anomalous structural edges
    { "edge": [2, 7], "z": 4.1, "residual": 0.123 }
  ],
  "meta": {                    // OPTIONAL: extra diagnostic / signatures
    "ustar_cached": true,
    "engine": "numpy-cg",
    "sig": "sha256:..."        // content hash for signing
  }
}
```

## Chain Receipt Extension
```json
{
  "version": "1.0.0",
  "verdict": true,                   // all chain edges within z threshold
  "weakest_link": {"k": 2, "edge": [5,7], "zscore": 2.91},
  "coherence_gain": 0.8123,          // positive is good
  "edges": [
    {"k":0, "edge":[2,5], "z_struct":1.1, "z_path":0.8, "r_struct":0.02, "r_path":0.02}
  ]
}
```

## JSON Schema (Concise)
```json
{
  "$id": "https://oscillink.org/schema/receipts-v1.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Oscillink Receipt v1",
  "type": "object",
  "required": ["deltaH_total"],
  "properties": {
    "version": {"type": "string"},
    "deltaH_total": {"type": "number", "minimum": 0},
    "coh_drop_sum": {"type": "number"},
    "anchor_pen_sum": {"type": "number"},
    "query_term_sum": {"type": "number"},
    "cg_iters": {"type": "integer", "minimum": 0},
    "residual": {"type": "number", "minimum": 0},
    "t_ms": {"type": "number", "minimum": 0},
    "null_points": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["edge", "z", "residual"],
        "properties": {
          "edge": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems":2, "maxItems":2},
          "z": {"type": "number"},
            "residual": {"type": "number"}
        }
      }
    },
    "meta": {"type": "object"}
  },
  "additionalProperties": true
}
```

## Versioning Policy
- Backward-compatible additive fields bump patch (`1.0.x`).
- Semantic changes / field meaning shifts bump minor.
- Incompatible removals bump major.

## Recommended Hashing for Signing
Compute SHA-256 over canonical JSON with keys sorted, excluding `meta.sig` itself. Example pseudocode:
```
obj2 = deepcopy(receipt)
remove obj2.meta.sig
payload = json.dumps(obj2, sort_keys=True, separators=(",", ":")).encode("utf-8")
sha256(payload)
```

## Future Extensions (Reserved Keys)
- `coherence_profile`: per-iteration ΔH descent curve.
- `chain_templates`: named chain prior references.
- `optimizer_hint`: recommended λ adjustments.

---
This document is part of Phase‑1 hardening. Update when adding new diagnostic fields.
