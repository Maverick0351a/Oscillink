# Hallucination Suppression — Reproducible Proof Harness

This document explains the small, reproducible harness that demonstrates controllable hallucination suppression using Oscillink Lattice gates.

- Goal: Show how near-zero gating on low-trust sources eliminates trap selections while improving F1 on a toy fact task.
- Reproducibility: Single Python script, deterministic RNG, no external models or data required.
- Scope: Demonstration only (controlled synthetic corpus). Not a generalized benchmark.

## What the Harness Does

The script `scripts/proof_hallucination.py` runs multiple trials over a tiny synthetic corpus that includes:

- Positive items (facts relevant to a query topic)
- Trap items (contain tokens like "fake", "spurious")
- Filler/off-topic items

For each trial:

1. Generate random embeddings for corpus items and a query vector anchored to a true positive.
2. Baseline: Select top-k by cosine similarity.
3. Lattice (Gated): Run Oscillink Lattice with near-zero gates for trap items and mild damping for off-topic items; then select top-k from `lat.bundle(k)`.
4. Record:
   - Hallucination occurrence: any top-k containing a trap item
   - Precision, recall, and F1 against a tiny ground-truth set

The summary reports mean F1 and hallucination rate for both methods across trials.

## Run It

Human-readable output:

```powershell
python scripts/proof_hallucination.py --trials 20 --k 3 --seed 0
```

JSON summary for automation:

```powershell
python scripts/proof_hallucination.py --trials 50 --k 3 --seed 0 --json
```

Example (illustrative):
```json
{"trials":20,"k":3,"baseline_hallucination_rate":0.45,"lattice_hallucination_rate":0.0,"baseline_f1_mean":0.32,"lattice_f1_mean":0.54}
```

Interpretation:
- Baseline occasionally picks traps ("fake"/"spurious" lines) → non-zero hallucination rate.
- Gated lattice suppresses traps and increases average F1 on this toy task.

## Method Notes

- Embeddings: Gaussian random vectors (D=96). This keeps the demo lightweight and deterministic; real use should replace with semantic embeddings.
- Query: Taken from a true positive vector to simulate a relevant question.
- Gates: Manual heuristic for the demo (near-zero for trap tokens, mild damp for off-topic). In practice, gates can be informed by provenance, classifiers, or diffusion preprocessing.
- Top-k: Uses `bundle(k)` from Oscillink which blends coherence and alignment with MMR-style diversification.

## Caveats

- The corpus and trap labeling are synthetic; results are specific to this setup.
- F1 and hallucination rate are computed with small ground truth and may be sensitive to k and RNG.
- This demonstrates controllability, not universal hallucination elimination.

## Next Steps (Optional Enhancements)

- Swap random embeddings for `oscillink.adapters.text.embed_texts` (sentence-transformers optional) for a semantic variant.
- Parameter sweep: Try different `lamC`, `lamQ`, and `kneighbors` to observe stability vs uplift.
- Add a small CSV input option to let users plug their own labeled toy corpora.
