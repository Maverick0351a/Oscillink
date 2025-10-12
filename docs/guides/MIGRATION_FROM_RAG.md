# Migrating from RAG to Oscillink

This guide helps you transition from traditional RAG (Retrieval-Augmented Generation) pipelines to Oscillink's coherent memory system.

## Why Migrate?

### RAG's Fundamental Problems

Traditional RAG systems suffer from:
- **Disconnected chunks**: Top-k similarity returns related but incoherent pieces
- **No semantic continuity**: Each chunk is evaluated independently
- **Hallucination prone**: LLMs struggle to synthesize disconnected context
- **No explainability**: Black box retrieval with no audit trail

### Oscillink's Solution

- **Coherent context**: Physics-based system ensures semantic coherence
- **Proven results**: 42.9% → 0% hallucination rate in controlled tests
- **Deterministic receipts**: Every decision has an auditable energy metric
- **Drop-in replacement**: Same API pattern, better results

## Migration Patterns

### Pattern 1: Direct Replacement (Simplest)

#### Before (Traditional RAG)
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Traditional RAG pipeline
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(k=5)

# Retrieve disconnected chunks
docs = retriever.get_relevant_documents(query)
context = "\n".join([d.page_content for d in docs])
response = llm(f"Context: {context}\n\nQuestion: {query}")
```

#### After (Oscillink)
```python
from oscillink import OscillinkLattice
import numpy as np

# Get embeddings (same as before)
doc_embeddings = [embeddings.embed_query(d.page_content) for d in documents]
query_embedding = embeddings.embed_query(query)

# Create coherent memory
lattice = OscillinkLattice(
    np.array(doc_embeddings).astype(np.float32),
    kneighbors=6
)
lattice.set_query(np.array(query_embedding).astype(np.float32))
lattice.settle()

# Get coherent context
bundle = lattice.bundle(k=5)
coherent_docs = [documents[item["id"]] for item in bundle]
context = "\n".join([d.page_content for d in coherent_docs])
response = llm(f"Context: {context}\n\nQuestion: {query}")

# Bonus: Audit trail
receipt = lattice.receipt()
print(f"Coherence score: {receipt['deltaH_total']}")
```

### Pattern 2: Cloud API Integration

#### Before (Vector Database)
```python
import weaviate

client = weaviate.Client("http://localhost:8080")
results = client.query.get("Document", ["content"]) \
    .with_near_text({"concepts": [query]}) \
    .with_limit(5) \
    .do()
```

#### After (Oscillink Cloud)
```python
import httpx
import os

# Fetch embeddings from your vector DB
embeddings = fetch_all_embeddings()  # Your implementation
query_embedding = get_query_embedding(query)

# Call Oscillink Cloud
response = httpx.post(
    "https://api.oscillink.com/v1/settle",
    json={
        "Y": embeddings,
        "psi": query_embedding,
        "options": {"bundle_k": 5, "include_receipt": True}
    },
    headers={"X-API-Key": os.environ["OSCILLINK_API_KEY"]}
)

result = response.json()
coherent_indices = [item["id"] for item in result["bundle"]]
```

### Pattern 3: Hybrid Approach (Best of Both)

Use your existing vector store for initial retrieval, then apply Oscillink for coherence:

```python
from oscillink import OscillinkLattice

# Step 1: Use existing RAG for initial retrieval (larger set)
initial_docs = vectorstore.similarity_search(query, k=50)

# Step 2: Get embeddings for retrieved docs
doc_embeddings = [embeddings.embed_query(d.page_content) for d in initial_docs]
query_embedding = embeddings.embed_query(query)

# Step 3: Apply Oscillink for coherent reranking
lattice = OscillinkLattice(
    np.array(doc_embeddings).astype(np.float32),
    kneighbors=10
)
lattice.set_query(np.array(query_embedding).astype(np.float32))
lattice.settle()

# Step 4: Get coherent top-k
bundle = lattice.bundle(k=5)
final_docs = [initial_docs[item["id"]] for item in bundle]
```

## Advanced Features

### Hallucination Gating

Suppress low-trust sources using diffusion gates:

```python
from oscillink import compute_diffusion_gates

# Compute trust scores based on source reliability
gates = compute_diffusion_gates(
    doc_embeddings,
    query_embedding,
    kneighbors=6,
    beta=1.0,  # Diffusion strength
    gamma=0.15  # Suppression threshold
)

# Apply gates to suppress unreliable sources
lattice.set_query(query_embedding, gates=gates)
lattice.settle()
```

### Chain Reasoning

Enforce logical flow through documents:

```python
# Define reasoning chain (e.g., chronological order)
chain = [0, 5, 12, 18, 23]  # Document indices

# Add chain prior
lattice.add_chain(chain, lamP=0.2)
lattice.settle()

# Verify chain coherence
chain_receipt = lattice.chain_receipt(chain)
print(f"Chain coherent: {chain_receipt['verdict']}")
```

### Audit Trail

Every Oscillink operation produces a deterministic receipt:

```python
receipt = lattice.receipt()

# Energy metrics (lower = more coherent)
print(f"Total energy drop: {receipt['deltaH_total']}")
print(f"Coherence improvement: {receipt['coh_drop_sum']}")

# Null points (semantic discontinuities)
if receipt['null_points']:
    print("Warning: Semantic gaps detected")
    for null in receipt['null_points']:
        print(f"  Edge {null['edge']}: z-score {null['z']}")

# Signed receipt for compliance
print(f"State signature: {receipt['meta']['state_sig']}")
```

## Performance Comparison

| Metric | Traditional RAG | Oscillink |
|--------|----------------|-----------|
| Retrieval Time | 5-10ms | 10-40ms |
| Hallucination Rate | 20-40% | 0-5% |
| Context Coherence | Low | High |
| Explainability | None | Full receipts |
| Deterministic | No | Yes |

## Migration Checklist

- [ ] **Inventory your embeddings**: Oscillink works with any embedding model
- [ ] **Choose integration pattern**: Direct, Cloud API, or Hybrid
- [ ] **Set up Oscillink**:
  - [ ] Install: `pip install oscillink`
  - [ ] Get API key (for cloud): [oscillink.com](https://oscillink.com)
- [ ] **Update retrieval code**: Replace similarity search with Oscillink
- [ ] **Add monitoring**: Use receipts for quality metrics
- [ ] **Test hallucination reduction**: Compare before/after on your test set
- [ ] **Enable gating** (optional): Suppress low-trust sources
- [ ] **Add chain priors** (optional): Enforce reasoning paths

## Common Patterns

### LangChain Integration
```python
from langchain.retrievers.base import BaseRetriever
from oscillink import OscillinkLattice

class OscillinkRetriever(BaseRetriever):
    def __init__(self, embeddings, documents, kneighbors=6):
        self.embeddings = embeddings
        self.documents = documents
        self.kneighbors = kneighbors

    def get_relevant_documents(self, query):
        # Compute embeddings
        doc_embeddings = [self.embeddings.embed_query(d.page_content)
                         for d in self.documents]
        query_embedding = self.embeddings.embed_query(query)

        # Create coherent memory
        lattice = OscillinkLattice(
            np.array(doc_embeddings).astype(np.float32),
            kneighbors=self.kneighbors
        )
        lattice.set_query(np.array(query_embedding).astype(np.float32))
        lattice.settle()

        # Return coherent documents
        bundle = lattice.bundle(k=5)
        return [self.documents[item["id"]] for item in bundle]

# Use as drop-in replacement
retriever = OscillinkRetriever(embeddings, documents)
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

### LlamaIndex Integration
```python
from llama_index import VectorStoreIndex, ServiceContext
from oscillink import OscillinkLattice

class OscillinkVectorStore:
    def query(self, query_embedding, top_k=5):
        lattice = OscillinkLattice(self.embeddings, kneighbors=6)
        lattice.set_query(query_embedding)
        lattice.settle()
        return lattice.bundle(k=top_k)

# Use with LlamaIndex
vector_store = OscillinkVectorStore()
index = VectorStoreIndex.from_vector_store(
    vector_store,
    service_context=ServiceContext.from_defaults()
)
```

## FAQ

**Q: Do I need to re-embed my documents?**
A: No, Oscillink works with your existing embeddings from any model.

**Q: How much latency does Oscillink add?**
A: Typically 10-40ms for 1,200 documents. The coherence improvement is worth the small latency cost.

**Q: Can I use Oscillink with streaming?**
A: Yes, process chunks in batches and maintain coherence across batches using the lattice state.

**Q: Does it work with multi-modal embeddings?**
A: Yes, Oscillink works with any embedding space (text, image, audio, etc.).

## Support

- Documentation: see the repo’s `docs/README.md` index
- Examples: `examples/`
- Notebooks: `notebooks/`
- Email: support@oscillink.com
- Discord: [Join our community](https://discord.gg/oscillink)

## Next Steps

1. Try the quickstart example at `examples/quickstart.py`
2. Run the hallucination reduction notebook at `notebooks/04_hallucination_reduction.ipynb`
3. Benchmark on your data using `scripts/benchmark.py`
4. Deploy with confidence knowing you have deterministic, auditable results

---

© 2025 Odin Protocol Inc. (Oscillink brand)
