import unittest
from pathlib import Path

from servicepilot.retrieval_evaluation import compare

ROOT = Path(__file__).resolve().parents[1]


class RetrievalEvaluationTests(unittest.TestCase):
    def test_comparison_reports_all_retrievers(self) -> None:
        report = compare(
            ROOT / "evaluations" / "retrieval_cases.json",
            ROOT / "data" / "policies.json",
        )
        self.assertEqual(report["bm25"]["queries"], 35)
        self.assertIn("hashing_vector", report)
        self.assertIn("hybrid_rrf", report)
        self.assertIn("hybrid_rewrite_rerank", report)
        self.assertIn("slice_metrics", report["hybrid_rewrite_rerank"])
        self.assertIn("zero_hit_rate", report["hybrid_rewrite_rerank"])

    def test_comparison_accepts_an_online_encoder(self) -> None:
        class ConstantEncoder:
            def encode(self, texts):
                return [[1.0, 0.0] for _ in texts]

        report = compare(
            ROOT / "evaluations" / "retrieval_cases.json",
            ROOT / "data" / "policies.json",
            semantic_encoder=ConstantEncoder(),
        )
        self.assertIn("semantic_vector", report)
        self.assertIn("hybrid_semantic_rrf", report)
        self.assertIn("semantic_rewrite_llm_rerank", report)


if __name__ == "__main__":
    unittest.main()
