---
title: Enterprise Knowledge Copilot
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: RAG copilot with citations and role-based access control
---

# Enterprise Knowledge Copilot — live demo

Ask questions about a fictional company's policies and runbooks. Answers cite their sources, and
**switching the user in the sidebar changes what the copilot is allowed to read**: an engineer can't
see board financials or salary bands, the CEO can.

Things to try:
- As **Priya (Engineer)**: "What was Q3 revenue?" → abstains (no access)
- As **Rohan (CEO)**: "What was Q3 revenue?" → answers with a citation
- "What is the hotel limit in Mumbai?" · "Who do I escalate to if a P1 page isn't acknowledged?"

Runs on free CPU with bge-small embeddings, hybrid search (Qdrant + BM25) and a cross-encoder reranker.
Answers are extractive (no LLM key needed).

Source code, architecture and evaluation results: https://github.com/Tanzil-Ali-Rizvi/enterprise-knowledge-copilot
