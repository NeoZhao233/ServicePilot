import unittest
from pathlib import Path

from servicepilot.agent import ServiceAgent
from servicepilot.models import ToolResult
from servicepilot.repository import InMemoryCommerceRepository
from servicepilot.retrieval import PolicyRetriever

ROOT = Path(__file__).resolve().parents[1]


def make_agent() -> ServiceAgent:
    repository = InMemoryCommerceRepository.from_json(ROOT / "data" / "orders.json")
    retriever = PolicyRetriever.from_json(ROOT / "data" / "policies.json")
    return ServiceAgent(repository, retriever)


class FailingTicketRepository(InMemoryCommerceRepository):
    def create_ticket(self, *, order_id: str, user_id: str, description: str) -> ToolResult:
        return ToolResult(ok=False, error="工单服务暂时不可用")


class ServiceAgentTests(unittest.TestCase):
    def test_logistics_query_uses_owned_order(self) -> None:
        response = make_agent().chat(user_id="USER-001", message="帮我查一下 ORD-1003 的物流")
        self.assertIn("运输中", response.message)
        self.assertEqual(response.events[-1].name, "get_order")

    def test_refund_requires_confirmation_before_mutation(self) -> None:
        agent = make_agent()
        first = agent.chat(user_id="USER-001", message="我要退掉 ORD-1001")
        self.assertIn("回复“确认”", first.message)
        self.assertFalse(any(event.name == "create_refund_request" for event in first.events))

        second = agent.chat(
            user_id="USER-001",
            message="确认",
            session_id=first.session.session_id,
        )
        self.assertIn("REF-0001", second.message)
        self.assertTrue(any(event.name == "create_refund_request" for event in second.events))

    def test_user_cannot_read_someone_elses_order(self) -> None:
        response = make_agent().chat(user_id="USER-001", message="查一下 ORD-2001 的物流")
        self.assertIn("无权查看", response.message)

    def test_damaged_item_requires_evidence_then_creates_ticket(self) -> None:
        agent = make_agent()
        first = agent.chat(user_id="USER-001", message="ORD-1001 收到就坏了")
        self.assertIn("上传", first.message)

        second = agent.chat(
            user_id="USER-001",
            message="已上传照片",
            session_id=first.session.session_id,
        )
        self.assertIn("TICKET-0001", second.message)
        self.assertIn("POLICY-DAMAGE-30D", second.citations)

    def test_high_value_refund_is_sent_to_human_review(self) -> None:
        response = make_agent().chat(user_id="USER-001", message="ORD-1002 的耳机有质量问题")
        self.assertIn("人工审核", response.message)
        self.assertIn("TICKET-0001", response.message)

    def test_pending_refund_can_be_cancelled(self) -> None:
        agent = make_agent()
        first = agent.chat(user_id="USER-001", message="我要退掉 ORD-1001")
        second = agent.chat(
            user_id="USER-001", message="算了，不退了", session_id=first.session.session_id
        )
        self.assertIn("已取消", second.message)
        self.assertIsNone(second.session.pending_action)
        self.assertFalse(any(event.name == "create_refund_request" for event in second.events))

    def test_switching_order_clears_pending_refund(self) -> None:
        agent = make_agent()
        first = agent.chat(user_id="USER-001", message="我要退掉 ORD-1001")
        second = agent.chat(
            user_id="USER-001",
            message="我改成退 ORD-1003",
            session_id=first.session.session_id,
        )
        self.assertIn("尚未签收", second.message)
        self.assertTrue(
            any(event.name == "clear_pending_action_on_order_change" for event in second.events)
        )

    def test_food_without_quality_issue_cannot_be_returned(self) -> None:
        response = make_agent().chat(user_id="USER-001", message="不要了，退款 ORD-1004")
        self.assertIn("食品类商品", response.message)
        self.assertFalse(any(event.name == "create_refund_request" for event in response.events))

    def test_tool_failure_is_not_reported_as_success(self) -> None:
        repository = FailingTicketRepository.from_json(ROOT / "data" / "orders.json")
        retriever = PolicyRetriever.from_json(ROOT / "data" / "policies.json")
        agent = ServiceAgent(repository, retriever)
        response = agent.chat(user_id="USER-001", message="ORD-1002 的耳机坏了")
        self.assertIn("创建失败", response.message)
        self.assertIn("尚未发生变更", response.message)
        self.assertNotIn("TICKET-", response.message)
        tool_event = next(event for event in response.events if event.name == "create_ticket")
        self.assertEqual(tool_event.detail["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
