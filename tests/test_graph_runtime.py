import unittest
from pathlib import Path

from servicepilot.cli import build_agent
from servicepilot.graph_runtime import LangGraphServiceRuntime

ROOT = Path(__file__).resolve().parents[1]


class LangGraphRuntimeTests(unittest.TestCase):
    def make_runtime(self) -> LangGraphServiceRuntime:
        runtime = build_agent(ROOT / "data")
        self.assertIsInstance(runtime, LangGraphServiceRuntime)
        return runtime

    def test_graph_contains_controlled_workflow_nodes(self) -> None:
        runtime = self.make_runtime()

        self.assertEqual(
            set(runtime.graph_nodes()),
            {
                "validate_request",
                "route_intent",
                "execute_business",
                "validate_result",
                "safe_fallback",
            },
        )

    def test_successful_turn_records_routing_allowlist_and_result_validation(self) -> None:
        runtime = self.make_runtime()

        response = runtime.chat(
            user_id="USER-001",
            message="查询 ORD-1001 的物流",
        )

        route = next(event for event in response.events if event.name == "route_intent")
        self.assertEqual(route.detail["allowed_tools"], ["get_order"])
        validation = next(event for event in response.events if event.name == "validate_result")
        self.assertTrue(validation.detail["ok"])
        persisted = runtime.get_events(response.session.session_id, user_id="USER-001")
        self.assertEqual(persisted[-1].name, "validate_result")

    def test_blank_request_uses_safe_fallback_without_business_execution(self) -> None:
        runtime = self.make_runtime()

        response = runtime.chat(user_id="USER-001", message="   ")

        self.assertIn("订单状态未被更改", response.message)
        self.assertTrue(any(event.name == "safe_fallback" for event in response.events))
        self.assertFalse(any(event.name == "execute_business" for event in response.events))


if __name__ == "__main__":
    unittest.main()
