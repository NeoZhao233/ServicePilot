import unittest
from pathlib import Path

from mcp.server.fastmcp.exceptions import ToolError

from servicepilot.mcp_server import create_mcp_server

ROOT = Path(__file__).resolve().parents[1]


class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_registers_four_business_tools(self) -> None:
        server = create_mcp_server(ROOT / "data")

        tools = await server.list_tools()

        self.assertEqual(
            {tool.name for tool in tools},
            {"get_order", "search_policy", "create_refund_request", "create_ticket"},
        )

    async def test_order_tool_enforces_ownership(self) -> None:
        server = create_mcp_server(ROOT / "data")

        with self.assertRaises(ToolError):
            await server.call_tool(
                "get_order",
                {"order_id": "ORD-1001", "user_id": "USER-002"},
            )

    async def test_refund_tool_requires_explicit_confirmation(self) -> None:
        server = create_mcp_server(ROOT / "data")

        with self.assertRaises(ToolError):
            await server.call_tool(
                "create_refund_request",
                {
                    "order_id": "ORD-1001",
                    "user_id": "USER-001",
                    "reason": "不需要",
                    "user_confirmed": False,
                },
            )


if __name__ == "__main__":
    unittest.main()
