from __future__ import annotations

import argparse
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .observability import Span, TraceService
from .repository import SQLiteCommerceRepository
from .retrieval import (
    AdvancedHybridPolicyRetriever,
    BM25PolicyRetriever,
    HashingTextEncoder,
    VectorPolicyRetriever,
    load_policy_chunks,
)


def create_mcp_server(
    data_dir: Path,
    *,
    database: Path | None = None,
    trusted_user_id: str | None = None,
    enable_recovery_tools: bool = False,
) -> FastMCP:
    repository = SQLiteCommerceRepository.from_json(
        data_dir / "orders.json",
        database=database or os.getenv("SERVICEPILOT_COMMERCE_DB", ":memory:"),
    )

    def check_identity(user_id):
        if trusted_user_id is not None and user_id != trusted_user_id:
            raise PermissionError("工具身份与受信连接不匹配")

    chunks = load_policy_chunks(data_dir / "policies.json")
    retriever = AdvancedHybridPolicyRetriever(
        BM25PolicyRetriever(chunks),
        VectorPolicyRetriever(chunks, HashingTextEncoder()),
    )
    from .corrective_rag import configure_rag

    retriever = configure_rag(retriever, chunks, retriever.vector)
    server = FastMCP(
        "ServicePilot",
        instructions="电商售后工具服务。所有订单工具必须传入已认证的 user_id。",
    )

    trace_service = TraceService.create(
        database
        or os.getenv("SERVICEPILOT_TRACE_DB")
        or os.getenv("SERVICEPILOT_COMMERCE_DB", ":memory:")
    )

    def trace_tool(function):
        import functools
        import inspect

        signature = inspect.signature(function)

        @functools.wraps(function)
        def call(*args, **kwargs):
            inputs = dict(signature.bind(*args, **kwargs).arguments)
            with trace_service.trace("mcp.request", user_id=trusted_user_id):
                with Span(
                    function.__name__, "mcp", inputs, {"transport": "mcp_server", "retry_count": 0}
                ) as span:
                    result = function(*args, **kwargs)
                    span.output(result)
                    return result

        return call

    @server.tool(description="查询当前用户拥有的订单；禁止跨用户读取。")
    @trace_tool
    def get_order(order_id: str, user_id: str) -> dict[str, object]:
        check_identity(user_id)
        result = repository.get_order(order_id=order_id, user_id=user_id)
        if not result.ok:
            raise ValueError(result.error)
        return result.data

    @server.tool(description="从售后政策库进行混合检索，并返回可引用的政策 ID。")
    @trace_tool
    def search_policy(query: str, category: str | None = None, limit: int = 3) -> list[dict]:
        from .evidence import EvidenceContext, evidence_context

        with evidence_context(EvidenceContext(question=query, category=category, safe_empty=True)):
            return [
                {
                    "policy_id": hit.chunk.policy_id,
                    "title": hit.chunk.title,
                    "content": hit.chunk.content,
                    "score": round(hit.score, 6),
                }
                for hit in retriever.search(query, category=category, limit=min(max(limit, 1), 5))
            ]

    @server.tool(description="经用户明确确认后创建退款申请；工具内部再次校验订单归属和规则。")
    @trace_tool
    def create_refund_request(
        order_id: str,
        user_id: str,
        reason: str,
        user_confirmed: bool,
        authorization: str | None = None,
    ) -> dict[str, object]:
        check_identity(user_id)
        if not user_confirmed:
            raise ValueError("退款属于敏感操作，缺少用户明确确认")
        # A model-supplied boolean cannot mint authority. The trusted confirmation
        # boundary must already have confirmed this persisted, scoped grant.
        result = repository.create_refund_request(
            order_id=order_id, user_id=user_id, reason=reason, authorization=authorization
        )
        if not result.ok:
            raise ValueError(result.error)
        return result.data

    @server.tool(description="为已校验归属的订单创建人工售后工单。")
    @trace_tool
    def create_ticket(order_id: str, user_id: str, description: str) -> dict[str, object]:
        check_identity(user_id)
        lookup = repository.get_order(order_id=order_id, user_id=user_id)
        if not lookup.ok:
            raise ValueError(lookup.error)
        result = repository.create_ticket(
            order_id=order_id,
            user_id=user_id,
            description=description,
        )
        if not result.ok:
            raise ValueError(result.error)
        return result.data

    if enable_recovery_tools:

        @server.tool(description="查询原业务操作结果；查询不到不能作为重复提交的依据。")
        @trace_tool
        def get_operation(operation_id: str, user_id: str) -> dict[str, object]:
            check_identity(user_id)
            result = repository.get_operation(operation_id=operation_id, user_id=user_id)
            if not result.ok:
                raise ValueError(result.error)
            return result.data

    return server


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run ServicePilot as an MCP server")
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
    )
    parser.add_argument("--commerce-db", type=Path)
    parser.add_argument("--user", default=os.getenv("SERVICEPILOT_MCP_USER", "USER-001"))
    args = parser.parse_args()
    create_mcp_server(
        args.data_dir,
        database=args.commerce_db,
        trusted_user_id=args.user,
        enable_recovery_tools=True,
    ).run(transport=args.transport)


if __name__ == "__main__":
    main()
