import unittest
from pathlib import Path

from servicepilot.agent import ServiceAgent
from servicepilot.models import Intent
from servicepilot.repository import InMemoryCommerceRepository
from servicepilot.retrieval import HybridPolicyRetriever
from servicepilot.tool_policy import ToolAccessPolicy

ROOT = Path(__file__).resolve().parents[1]


class ToolAccessPolicyTests(unittest.TestCase):
    def test_each_intent_has_least_privilege_allowlist(self) -> None:
        policy = ToolAccessPolicy()

        self.assertEqual(policy.allowed_tools(Intent.LOGISTICS), frozenset({"get_order"}))
        self.assertIn("create_refund_request", policy.allowed_tools(Intent.REFUND))
        self.assertNotIn("create_refund_request", policy.allowed_tools(Intent.DAMAGED_ITEM))
        self.assertEqual(policy.allowed_tools(Intent.UNKNOWN), frozenset())

    def test_agent_blocks_tool_outside_intent_allowlist(self) -> None:
        repository = InMemoryCommerceRepository.from_json(ROOT / "data" / "orders.json")
        retriever = HybridPolicyRetriever.from_json(ROOT / "data" / "policies.json")
        policy = ToolAccessPolicy(allowlist={intent: frozenset() for intent in Intent})
        agent = ServiceAgent(repository, retriever, tool_policy=policy)

        response = agent.chat(
            user_id="USER-001",
            message="查询 ORD-1001 的物流",
        )

        self.assertIn("不在 logistics 意图白名单", response.message)
        self.assertTrue(any(event.name == "tool_blocked" for event in response.events))


if __name__ == "__main__":
    unittest.main()
