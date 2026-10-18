# Evaluation report

Golden set: 44 cases (factual, paraphrase, access-control allowed/denied, prompt injection, out-of-scope).  
Embedder: `hashing-1024` · LLM: `offline-extractive`

| Config | hit@1 | hit@5 | mrr | answer_accuracy | citation_accuracy | abstention_accuracy | acl_leak_rate | latency_p50_ms | latency_p95_ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.971 | 1.0 | 0.985 | 0.824 | 0.824 | 1.0 | 0.0 | 0.4 | 0.7 |
| vector | 0.971 | 1.0 | 0.985 | 0.824 | 0.824 | 1.0 | 0.0 | 2.5 | 3.2 |
| hybrid | 1.0 | 1.0 | 1.0 | 0.824 | 0.824 | 1.0 | 0.0 | 3.1 | 4.6 |
| hybrid+rerank | 0.971 | 1.0 | 0.985 | 0.882 | 0.882 | 1.0 | 0.0 | 5.6 | 6.1 |

## Failures (best config)

- `f07` (factual): I don't know based on the documents you have access to.
- `f12` (factual): I don't know based on the documents you have access to.
- `p02` (paraphrase): I don't know based on the documents you have access to.
- `p04` (paraphrase): I don't know based on the documents you have access to.
