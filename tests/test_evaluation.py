import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from servicepilot.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_seed_evaluation_suite_passes(self) -> None:
        results = evaluate(ROOT / "evaluations" / "cases.json", data_dir=ROOT / "data")
        self.assertEqual(len(results), 20)
        self.assertTrue(all(result.passed for result in results))

    def test_challenge_assertions_capture_turn_state_and_exact_tool_count(self) -> None:
        cases = [
            {
                "case_id": "cancelled_refund",
                "user_id": "USER-001",
                "messages": ["退掉 ORD-1001", "算了，不退了"],
                "slice": ["multi_turn", "safety"],
                "difficulty": "hard",
                "turn_expectations": [
                    {"pending_action": "refund"},
                    {"pending_action": None, "must_have_events": ["cancel_pending_action"]},
                ],
                "expected": {
                    "message_contains_any": ["已取消", "不再继续"],
                    "tool_call_counts": {"create_refund_request": 0},
                    "must_not_call": ["create_refund_request"],
                },
            }
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
            result = evaluate(path, data_dir=ROOT / "data")[0]
        self.assertTrue(result.passed, result.failures)
        self.assertEqual(result.slices, ("multi_turn", "safety"))
        self.assertEqual(result.difficulty, "hard")
        self.assertEqual(len(result.turns), 2)


if __name__ == "__main__":
    unittest.main()
