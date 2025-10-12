# Benchmarks: BEIR and RAG

This folder describes how to run retrieval benchmarks comparing cosine baseline vs Oscillink post-processing.

## Installation (extras)

Install bench extras:

```bash
pip install -e .[bench]
```

## BEIR retrieval (CPU only)

Example (FiQA, MiniLM, 500 queries):

```bash
python scripts/bench_beir.py --dataset beir/fiqa/dev --model sentence-transformers/all-MiniLM-L6-v2 --top-k 10 --kneighbors 6 --max-queries 500 --out benchmarks/fiqa_minilm_oscillink.json
```

Outputs a JSON summary with ndcg@10 and recall@50 for cosine vs oscillink.

Recommended datasets to showcase diversity:
- beir/fiqa/dev
- beir/scifact/dev
- beir/trec-covid/dev
- beir/dbpedia-entity/dev
- beir/nq/dev

## Notes

- Embeddings are computed with sentence-transformers; results are CPU-friendly.
- Oscillink runs on the top-K documents returned by cosine baseline and refines their scores coherently.
- For plots, you can adapt `scripts/plot_benchmarks.py` or consume the JSON in your own notebooks.
