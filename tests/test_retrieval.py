import unittest
from pathlib import Path

from servicepilot.retrieval import (
    HashingTextEncoder,
    HybridPolicyRetriever,
    PolicyRetriever,
    VectorPolicyRetriever,
    load_policy_chunks,
)

ROOT = Path(__file__).resolve().parents[1]


class PolicyRetrieverTests(unittest.TestCase):
    def test_damage_query_retrieves_damage_policy(self) -> None:
        retriever = PolicyRetriever.from_json(ROOT / "data" / "policies.json")
        hits = retriever.search("商品破损，需要上传照片", limit=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.policy_id, "POLICY-DAMAGE-EVIDENCE")

    def test_category_filter_keeps_global_and_matching_policies(self) -> None:
        retriever = PolicyRetriever.from_json(ROOT / "data" / "policies.json")
        hits = retriever.search("退款 退货", category="运动鞋服", limit=5)
        ids = {hit.chunk.policy_id for hit in hits}
        self.assertNotIn("POLICY-DIGITAL", ids)

    def test_hybrid_retriever_fuses_and_deduplicates_results(self) -> None:
        chunks = load_policy_chunks(ROOT / "data" / "policies.json")
        lexical = PolicyRetriever(chunks)
        vector = VectorPolicyRetriever(chunks, HashingTextEncoder())
        hybrid = HybridPolicyRetriever(lexical, vector)
        hits = hybrid.search("耳机质量问题需要照片", category="数码产品", limit=3)
        ids = [hit.chunk.policy_id for hit in hits]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("POLICY-DIGITAL", ids)
        self.assertIn("POLICY-DAMAGE-30D", ids)


if __name__ == "__main__":
    unittest.main()
