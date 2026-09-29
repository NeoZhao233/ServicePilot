from pathlib import Path

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from servicepilot.cli import build_agent
from servicepilot.mcp_server import create_mcp_server

ROOT = Path(__file__).resolve().parents[1]


def test_mcp_uses_shared_ledger_and_does_not_trust_boolean(tmp_path):
    import asyncio

    async def scenario():
        db = tmp_path / "shared.sqlite3"
        runtime = build_agent(ROOT / "data", session_db=db, commerce_db=db)
        server = create_mcp_server(
            ROOT / "data", database=db, trusted_user_id="USER-001", enable_recovery_tools=True
        )
        args = {
            "order_id": "ORD-1001",
            "user_id": "USER-001",
            "reason": "test",
            "user_confirmed": True,
        }
        with pytest.raises(ToolError):
            await server.call_tool("create_refund_request", args)
        first = runtime.chat(user_id="USER-001", message="退掉 ORD-1001")
        args["authorization"] = first.session.authorization["token"]
        with pytest.raises(ToolError):
            await server.call_tool("create_refund_request", args)
        runtime.agent.repository.confirm_authorization(
            token=args["authorization"], user_id="USER-001"
        )
        await server.call_tool("create_refund_request", args)
        assert (
            runtime.agent.repository.get_order(order_id="ORD-1001", user_id="USER-001").data[
                "status"
            ]
            == "refunding"
        )
        await server.call_tool(
            "get_operation",
            {"operation_id": first.session.authorization["operation_id"], "user_id": "USER-001"},
        )
        with pytest.raises(ToolError):
            await server.call_tool("get_order", {"order_id": "ORD-2001", "user_id": "USER-002"})

    asyncio.run(scenario())
