# Reference resume gap analysis

This checklist translates the supplied resume screenshot into verifiable ServicePilot deliverables. A box is checked only when source, tests, and a runnable entry point exist. Measured claims additionally require a fixed evaluation set and saved raw results.

## 1. Agent Runtime and business workflow

- [x] Typed session state and explicit business stages.
- [x] Tool result validation, retries, circuit breaker, idempotency, and human review.
- [x] Agent Trace UI and durable event replay.
- [x] LangGraph implementation with conditional edges and bounded execution.
- [x] Tool registry with dynamic allowlists by intent and prompt profile.
- [x] MCP exposure for order, policy, refund, and ticket tools.

## 2. Multi-route intent recognition

- [x] Rule router, structured LLM router, confidence threshold, and fallback.
- [x] Offline intent evaluation with per-class Precision, Recall, and Macro-F1.
- [x] Embedding similarity route and weighted voting across three routes.
- [x] Ten after-sales intents and four urgency levels.
- [x] Bounded LRU cache for repeated intent queries.
- [x] Dynamic tool allowlist and prompt selection from the routing result.

## 3. Hybrid RAG

- [x] BM25, vector retrieval, metadata filtering, and RRF fusion.
- [x] API Embedding adapter and reproducible retrieval comparison.
- [x] Chroma persistent vector index.
- [x] Local BGE semantic Embedding with a saved frozen-set comparison against BM25.
- [x] Query rewrite, parallel recall, deduplication, DeepSeek LLM reranking, and TTL cache.
- [x] Nine policy domains and 37 policy chunks.
- [x] Retrieval confidence threshold and explicit degradation behavior.
- [x] MCP policy retrieval tool.

## 4. Evaluation

- [x] Intent, retrieval, and business-flow evaluation entry points.
- [x] Recall@K, MRR, NDCG@K, tool constraints, and raw predictions.
- [x] 35 fixed, manually authored retrieval queries.
- [x] Direct, hybrid, and rewrite strategy comparison with saved results.
- [x] 20 end-to-end multi-turn cases and rule-based quality checks.
- [x] Optional LLM-as-Judge adapter with a disclosed rubric and model configuration.

## 5. Layered memory

- [x] Durable short-term session state and ordered event audit trail.
- [x] Redis session store with TTL and graceful SQLite fallback.
- [x] Long-term episodic summaries in a vector index.
- [x] Structured user profile memory with explicit writable fields.
- [x] Token-budgeted context assembly and dynamic history selection.
- [x] Cross-session profile reuse tests and privacy controls.

## Resume gate

Do not copy claims or metrics from the reference. The final resume entry must be generated from checked capabilities and committed experiment artifacts produced by this repository.
