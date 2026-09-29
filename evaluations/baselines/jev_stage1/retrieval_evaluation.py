from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .providers import (
    APIEmbeddingEncoder,
    OnlineModelSettings,
    ResponsesPolicyReranker,
    UrllibJsonTransport,
)
from .retrieval import (
    AdvancedHybridPolicyRetriever,
    BM25PolicyRetriever,
    FastEmbedTextEncoder,
    HashingTextEncoder,
    HybridPolicyRetriever,
    PolicySearch,
    TextEncoder,
    VectorPolicyRetriever,
    load_policy_chunks,
)


def _case_slices(case: dict[str, object]) -> tuple[str, ...]:
    raw = case.get("slice", "unspecified")
    if isinstance(raw, list):
        return tuple(str(item) for item in raw) or ("unspecified",)
    return (str(raw),)


def _aggregate(items: list[dict[str, object]], *, limit: int) -> dict[str, object]:
    count = len(items)
    return {
        "queries": count,
        f"recall@{limit}": round(
            sum(float(item["recall"]) for item in items) / count, 4
        ) if count else 0.0,
        "mrr": round(sum(float(item["reciprocal_rank"]) for item in items) / count, 4)
        if count else 0.0,
        f"ndcg@{limit}": round(sum(float(item["ndcg"]) for item in items) / count, 4)
        if count else 0.0,
        "zero_hit_rate": round(sum(bool(item["zero_hit"]) for item in items) / count, 4)
        if count else 0.0,
    }


def evaluate_retriever(
    retriever: PolicySearch, cases: list[dict[str, object]], *, limit: int = 3
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    for case in cases:
        relevant = set(case["relevant_policy_ids"])
        hits = retriever.search(
            str(case["query"]),
            category=str(case["category"]) if case.get("category") else None,
            limit=limit,
        )
        retrieved = [hit.chunk.policy_id for hit in hits]
        recall = len(relevant.intersection(retrieved)) / len(relevant)

        first_relevant_rank = next(
            (rank for rank, policy_id in enumerate(retrieved, start=1) if policy_id in relevant),
            None,
        )
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, policy_id in enumerate(retrieved, start=1)
            if policy_id in relevant
        )
        ideal_hits = min(len(relevant), limit)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
        predictions.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "relevant": sorted(relevant),
                "retrieved": retrieved,
                "recall": recall,
                "reciprocal_rank": reciprocal_rank,
                "ndcg": ndcg,
                "zero_hit": not bool(relevant.intersection(retrieved)),
                "slices": _case_slices(case),
                "difficulty": str(case.get("difficulty", "unspecified")),
            }
        )

    report = _aggregate(predictions, limit=limit)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    difficulties: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in predictions:
        for slice_name in item["slices"]:
            grouped[str(slice_name)].append(item)
        difficulties[str(item["difficulty"])].append(item)
    report["slice_metrics"] = {
        name: _aggregate(items, limit=limit) for name, items in sorted(grouped.items())
    }
    report["difficulty_metrics"] = {
        name: _aggregate(items, limit=limit) for name, items in sorted(difficulties.items())
    }
    report["predictions"] = predictions
    if isinstance(retriever, AdvancedHybridPolicyRetriever):
        report["reranker"] = retriever.last_diagnostics.reranker
        report["reranker_failure_count"] = retriever.reranker_failure_count
    return report


def compare(
    cases_path: Path,
    policies_path: Path,
    *,
    limit: int = 3,
    semantic_encoder: TextEncoder | None = None,
    reranker=None,
) -> dict[str, object]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    chunks = load_policy_chunks(policies_path)
    encoder = HashingTextEncoder()
    lexical = BM25PolicyRetriever(chunks)
    vector = VectorPolicyRetriever(chunks, encoder)
    hybrid = HybridPolicyRetriever(lexical, vector)
    rewrite_hybrid = AdvancedHybridPolicyRetriever(lexical, vector)
    report: dict[str, object] = {
        "note": "hashing-vector is an offline engineering baseline, not a semantic embedding model",
        "bm25": evaluate_retriever(lexical, cases, limit=limit),
        "hashing_vector": evaluate_retriever(vector, cases, limit=limit),
        "hybrid_rrf": evaluate_retriever(hybrid, cases, limit=limit),
        "hybrid_rewrite_rerank": evaluate_retriever(rewrite_hybrid, cases, limit=limit),
    }
    if semantic_encoder is not None:
        semantic = VectorPolicyRetriever(chunks, semantic_encoder)
        semantic_hybrid = HybridPolicyRetriever(lexical, semantic)
        semantic_advanced = AdvancedHybridPolicyRetriever(
            lexical, semantic, reranker=reranker
        )
        report["semantic_vector"] = evaluate_retriever(semantic, cases, limit=limit)
        report["hybrid_semantic_rrf"] = evaluate_retriever(
            semantic_hybrid, cases, limit=limit
        )
        report["semantic_rewrite_llm_rerank"] = evaluate_retriever(
            semantic_advanced, cases, limit=limit
        )
        report["semantic_configuration"] = {
            "encoder": type(semantic_encoder).__name__,
            "model": getattr(semantic_encoder, "model_name", None),
            "reranker": type(reranker).__name__ if reranker else None,
        }
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compare policy retrieval strategies")
    parser.add_argument(
        "--cases", type=Path, default=project_root / "evaluations" / "retrieval_cases.json"
    )
    parser.add_argument("--policies", type=Path, default=project_root / "data" / "policies.json")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--online", action="store_true", help="Also evaluate the configured Embeddings API"
    )
    parser.add_argument("--output", type=Path, help="Write the complete reproducible JSON report")
    args = parser.parse_args()
    semantic_encoder: TextEncoder | None = None
    reranker = None
    if args.online:
        settings = OnlineModelSettings.from_env()
        transport = UrllibJsonTransport(settings)
        if settings.embedding_mode == "api":
            semantic_encoder = APIEmbeddingEncoder(transport, settings.embedding_model)
        elif settings.embedding_mode == "local":
            semantic_encoder = FastEmbedTextEncoder(settings.embedding_model)
        reranker = ResponsesPolicyReranker(transport, settings.intent_model)
    report = compare(
        args.cases,
        args.policies,
        limit=args.limit,
        semantic_encoder=semantic_encoder,
        reranker=reranker,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{serialized}\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
