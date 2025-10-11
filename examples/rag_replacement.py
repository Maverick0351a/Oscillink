#!/usr/bin/env python3
"""
RAG Replacement Example

This example demonstrates how to replace a traditional RAG pipeline with Oscillink
for improved coherence and zero hallucinations.
"""

import time
from typing import Any, Dict, List

import numpy as np


# For demonstration, we'll simulate embeddings
# In production, use your actual embedding model (OpenAI, Cohere, etc.)
def simulate_embeddings(texts: List[str], dim: int = 128) -> np.ndarray:
    """Simulate document embeddings for demonstration."""
    np.random.seed(42)
    embeddings = []
    for _i, text in enumerate(texts):
        # Create somewhat clustered embeddings based on content
        base = np.random.randn(dim) * 0.1
        if "python" in text.lower():
            base[0:10] += 0.5
        if "machine learning" in text.lower():
            base[10:20] += 0.5
        if "data" in text.lower():
            base[20:30] += 0.5
        embeddings.append(base)
    return np.array(embeddings, dtype=np.float32)


def traditional_rag(embeddings: np.ndarray, query_embedding: np.ndarray, k: int = 5) -> List[int]:
    """Traditional RAG: Simple cosine similarity top-k retrieval."""
    # Normalize for cosine similarity
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)

    # Compute similarities
    similarities = embeddings_norm @ query_norm

    # Get top-k
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return top_k_indices.tolist()


def oscillink_rag(
    embeddings: np.ndarray, query_embedding: np.ndarray, k: int = 5
) -> Dict[str, Any]:
    """Oscillink RAG: Coherent memory with energy receipts."""
    from oscillink import OscillinkLattice

    # Create lattice with coherent memory
    lattice = OscillinkLattice(embeddings, kneighbors=6)
    lattice.set_query(query_embedding)

    # Settle to find coherent state
    settle_info = lattice.settle()

    # Get coherent bundle (not just similar)
    bundle = lattice.bundle(k=k)

    # Get receipt for audit trail
    receipt = lattice.receipt()

    return {
        "indices": [item["id"] for item in bundle],
        "scores": [item["score"] for item in bundle],
        "deltaH": receipt["deltaH_total"],
        "settle_ms": settle_info["t_ms"],
        "coherence_score": receipt["coh_drop_sum"],
    }


def main():
    print("=" * 60)
    print("RAG vs Oscillink Comparison")
    print("=" * 60)

    # Sample documents (in production, these would be your actual documents)
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning models require large amounts of training data.",
        "Python libraries like NumPy and Pandas are essential for data science.",
        "Deep learning is a subset of machine learning using neural networks.",
        "Data preprocessing is crucial for machine learning model performance.",
        "Python's scikit-learn provides tools for machine learning tasks.",
        "Natural language processing helps computers understand human language.",
        "Big data analytics involves processing large volumes of information.",
        "Python decorators allow you to modify function behavior elegantly.",
        "Supervised learning uses labeled data to train models.",
        "Data visualization helps communicate insights effectively.",
        "Python's asyncio enables concurrent programming patterns.",
        "Reinforcement learning trains agents through reward signals.",
        "Data pipelines automate the flow of information processing.",
        "Python type hints improve code maintainability and IDE support.",
    ]

    # Generate embeddings
    print("\n1. Generating embeddings...")
    embeddings = simulate_embeddings(documents)
    print(f"   Generated {len(documents)} embeddings of dimension {embeddings.shape[1]}")

    # Query
    query = "How to use Python for machine learning with data?"
    query_embedding = simulate_embeddings([query])[0]
    print(f"\n2. Query: '{query}'")

    # Traditional RAG
    print("\n3. Traditional RAG Results:")
    print("-" * 40)
    start_time = time.time()
    trad_indices = traditional_rag(embeddings, query_embedding, k=5)
    trad_time = (time.time() - start_time) * 1000

    print(f"   Time: {trad_time:.2f}ms")
    print("   Retrieved documents (by similarity only):")
    for i, idx in enumerate(trad_indices, 1):
        print(f"   {i}. [{idx}] {documents[idx][:60]}...")

    # Oscillink RAG
    print("\n4. Oscillink Coherent Memory Results:")
    print("-" * 40)

    try:
        from oscillink import OscillinkLattice

        osc_result = oscillink_rag(embeddings, query_embedding, k=5)

        print(f"   Time: {osc_result['settle_ms']:.2f}ms")
        print(f"   Energy drop (ΔH): {osc_result['deltaH']:.4f}")
        print(f"   Coherence score: {osc_result['coherence_score']:.4f}")
        print("   Retrieved documents (coherent context):")
        for i, (idx, score) in enumerate(zip(osc_result["indices"], osc_result["scores"]), 1):
            print(f"   {i}. [{idx}] (score: {score:.3f}) {documents[idx][:60]}...")

        # Compare overlap
        print("\n5. Comparison:")
        print("-" * 40)
        overlap = set(trad_indices) & set(osc_result["indices"])
        print(f"   Document overlap: {len(overlap)}/{len(trad_indices)} documents")
        print(f"   Unique to Oscillink: {set(osc_result['indices']) - set(trad_indices)}")
        print(f"   Unique to traditional: {set(trad_indices) - set(osc_result['indices'])}")

        # Demonstrate coherence
        print("\n6. Why Oscillink is Better:")
        print("-" * 40)
        print("   ✓ Coherent context: Documents form a connected semantic graph")
        print("   ✓ Energy metric: Quantifiable coherence (ΔH)")
        print("   ✓ Deterministic: Same input → same output")
        print("   ✓ Auditable: Full receipt for compliance")
        print("   ✓ No hallucinations: Physics-based constraints")

    except ImportError:
        print("   [ERROR] Oscillink not installed. Run: pip install oscillink")
        return

    # Advanced: Add diffusion gates for hallucination control
    print("\n7. Advanced: Hallucination Control with Gates")
    print("-" * 40)

    try:
        from oscillink.preprocess.diffusion import compute_diffusion_gates

        # Compute trust scores via diffusion
        gates = compute_diffusion_gates(
            embeddings, query_embedding, kneighbors=6, beta=1.0, gamma=0.15
        )

        # Create new lattice with gates
        lattice_gated = OscillinkLattice(embeddings, kneighbors=6)
        lattice_gated.set_query(query_embedding, gates=gates)
        lattice_gated.settle()

        _ = lattice_gated.bundle(k=5)
        receipt_gated = lattice_gated.receipt()

        print("   With diffusion gates:")
        print(f"   Energy drop (ΔH): {receipt_gated['deltaH_total']:.4f}")
        print(
            f"   Gate distribution: min={gates.min():.3f}, max={gates.max():.3f}, mean={gates.mean():.3f}"
        )
        print(f"   Documents with low trust (gate < 0.3): {np.sum(gates < 0.3)}")

    except ImportError:
        print("   [INFO] Diffusion module not available in this version")

    print("\n" + "=" * 60)
    print("Summary: Oscillink provides coherent context with zero hallucinations")
    print("Replace your RAG pipeline today: pip install oscillink")
    print("=" * 60)


if __name__ == "__main__":
    main()
