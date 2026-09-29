import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from servicepilot.models import RetrievalHit
from servicepilot.retrieval import (
    AdvancedHybridPolicyRetriever,
    BM25PolicyRetriever,
    ChromaPolicyRetriever,
    HashingTextEncoder,
    QueryRewriter,
    load_policy_chunks,
)

ROOT = Path(__file__).resolve().parents[1]


class FailingRetriever:
    def search(self, query: str, *, category: str | None = None, limit: int = 3):
        raise RuntimeError("vector service unavailable")


class CountingRetriever:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.calls = 0

    def search(
        self, query: str, *, category: str | None = None, limit: int = 3
    ) -> list[RetrievalHit]:
        self.calls += 1
        return self.delegate.search(query, category=category, limit=limit)


class ReverseReranker:
    def rerank(self, query, candidates, *, limit):
        return list(reversed(candidates))[:limit]


class FailingReranker:
    def rerank(self, query, candidates, *, limit):
        raise RuntimeError("reranker unavailable")


class AdvancedRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = load_policy_chunks(ROOT / "data" / "policies.json")
        self.lexical = BM25PolicyRetriever(self.chunks)

    def test_query_rewrite_expands_business_colloquialism(self) -> None:
        variants = QueryRewriter().rewrite("我的钱没到")

        self.assertEqual(variants[0], "我的钱没到")
        self.assertIn("退款到账时效", variants[1])

    def test_vector_failure_degrades_to_bm25(self) -> None:
        retriever = AdvancedHybridPolicyRetriever(self.lexical, FailingRetriever())

        hits = retriever.search("商品破损质量问题")

        self.assertEqual(hits[0].chunk.policy_id, "POLICY-DAMAGE-30D")
        self.assertEqual(retriever.last_diagnostics.degraded_sources, ("vector",))

    def test_ttl_cache_avoids_repeated_recall(self) -> None:
        lexical = CountingRetriever(self.lexical)
        vector = CountingRetriever(self.lexical)
        retriever = AdvancedHybridPolicyRetriever(lexical, vector, cache_ttl_seconds=60)

        retriever.search("七天无理由退货")
        retriever.search("七天无理由退货")

        self.assertEqual(lexical.calls, 1)
        self.assertEqual(vector.calls, 1)
        self.assertTrue(retriever.last_diagnostics.cache_hit)

    def test_chroma_index_persists_and_filters_category(self) -> None:
        with TemporaryDirectory() as directory:
            first = ChromaPolicyRetriever(
                self.chunks,
                HashingTextEncoder(),
                persist_directory=directory,
            )
            first_hits = first.search("试穿鞋底吊牌", category="运动鞋服")
            second = ChromaPolicyRetriever(
                self.chunks,
                HashingTextEncoder(),
                persist_directory=directory,
            )
            second_hits = second.search("试穿鞋底吊牌", category="运动鞋服")

        self.assertEqual(first_hits[0].chunk.policy_id, "POLICY-SPORTSWEAR")
        self.assertEqual(second_hits[0].chunk.policy_id, "POLICY-SPORTSWEAR")

    def test_external_reranker_reorders_fused_candidates(self) -> None:
        retriever = AdvancedHybridPolicyRetriever(
            self.lexical, self.lexical, reranker=ReverseReranker()
        )

        hits = retriever.search("退款到账时效", limit=3)

        self.assertEqual(retriever.last_diagnostics.reranker, "ReverseReranker")
        self.assertFalse(retriever.last_diagnostics.reranker_degraded)
        self.assertEqual(len(hits), 3)

    def test_reranker_failure_degrades_to_fused_order(self) -> None:
        retriever = AdvancedHybridPolicyRetriever(
            self.lexical, self.lexical, reranker=FailingReranker()
        )

        hits = retriever.search("退款到账时效", limit=3)

        self.assertTrue(hits)
        self.assertTrue(retriever.last_diagnostics.reranker_degraded)
        self.assertEqual(retriever.reranker_failure_count, 1)


if __name__ == "__main__":
    unittest.main()
