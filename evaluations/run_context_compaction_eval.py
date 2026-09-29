"""Twelve 20-turn B0/B1 after-sales cases with an offline extractive summary stub."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from servicepilot.agent import ServiceAgent
from servicepilot.context_manager import ContextCompactor, DeepSeekContextSummaryModel
from servicepilot.models import ToolResult
from servicepilot.providers import OnlineModelSettings, UrllibJsonTransport
from servicepilot.repository import InMemoryCommerceRepository
from servicepilot.retrieval import HybridPolicyRetriever

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    name: str
    first: str
    last: str
    expected_text: str
    expected_order: str
    expected_tools: tuple[str, ...]
    fail_ticket: bool = False


CASES = (
    Case(
        "logistics-delivered-1",
        "查询 ORD-1001 的物流",
        "查询 ORD-1001 的物流",
        "已于 9 月 12 日签收",
        "ORD-1001",
        ("get_order",),
    ),
    Case(
        "logistics-delivered-2",
        "查询 ORD-1002 的物流",
        "查询 ORD-1002 的物流",
        "已签收",
        "ORD-1002",
        ("get_order",),
    ),
    Case(
        "logistics-shipped",
        "查询 ORD-1003 的物流",
        "查询 ORD-1003 的物流",
        "运输中",
        "ORD-1003",
        ("get_order",),
    ),
    Case(
        "refund-confirm",
        "我要退掉 ORD-1001",
        "确认",
        "退款申请已提交",
        "ORD-1001",
        ("get_order", "create_refund_request"),
    ),
    Case("refund-cancel", "我要退掉 ORD-1001", "算了，不退了", "已取消当前操作", "ORD-1001", ()),
    Case(
        "refund-change-order",
        "我要退掉 ORD-1001",
        "我改成退 ORD-1003",
        "尚未签收",
        "ORD-1003",
        ("get_order",),
    ),
    Case(
        "damage-evidence",
        "ORD-1001 收到就坏了",
        "已上传照片",
        "创建售后审核工单",
        "ORD-1001",
        ("get_order", "create_ticket"),
    ),
    Case(
        "damage-tool-failure",
        "ORD-1001 收到就坏了",
        "已上传照片",
        "工单暂时创建失败",
        "ORD-1001",
        ("get_order", "create_ticket"),
        True,
    ),
    Case(
        "high-value-review",
        "ORD-1002 的耳机有质量问题",
        "ORD-1002 的耳机有质量问题",
        "人工审核",
        "ORD-1002",
        ("get_order", "create_ticket"),
    ),
    Case(
        "food-policy-block",
        "不要了，退款 ORD-1004",
        "不要了，退款 ORD-1004",
        "食品类商品",
        "ORD-1004",
        ("get_order",),
    ),
    Case(
        "undelivered-policy-block",
        "我要退掉 ORD-1003",
        "我要退掉 ORD-1003",
        "尚未签收",
        "ORD-1003",
        ("get_order",),
    ),
    Case(
        "exchange-ticket",
        "我想换货 ORD-1001",
        "我想换货 ORD-1001",
        "创建核验工单",
        "ORD-1001",
        ("get_order", "create_ticket"),
    ),
)


class FailingTicketRepository(InMemoryCommerceRepository):
    def create_ticket(self, *, order_id: str, user_id: str, description: str) -> ToolResult:
        return ToolResult(ok=False, error="工单服务暂时不可用")


class ExtractiveStub:
    def __init__(self):
        self.calls = 0

    def summarize(self, old_summary, messages):
        self.calls += 1
        source = [item["content"] for item in messages]
        safe = next((text for text in source if text.startswith("补充说明")), "")
        return {
            "current_goal": safe,
            "confirmed_facts": [],
            "completed_steps": [],
            "unresolved_issues": [],
            "important_constraints": [],
            "relevant_policy_refs": [],
            "important_tool_results": [],
        }


def run_case(case: Case, compact: bool, live_settings: OnlineModelSettings | None = None) -> dict:
    repository_type = FailingTicketRepository if case.fail_ticket else InMemoryCommerceRepository
    repository = repository_type.from_json(ROOT / "data" / "orders.json")
    retriever = HybridPolicyRetriever.from_json(ROOT / "data" / "policies.json")
    model = (
        DeepSeekContextSummaryModel(
            UrllibJsonTransport(live_settings),
            os.getenv("SERVICEPILOT_CONTEXT_MODEL") or "deepseek-chat",
        )
        if live_settings is not None
        else ExtractiveStub()
    )
    compactor = (
        ContextCompactor(model, max_messages=12, max_tokens=4000, keep_recent_messages=4)
        if compact
        else None
    )
    agent = ServiceAgent(repository, retriever, context_compactor=compactor)
    session_id = None
    latencies = []
    prompts = []
    responses = []
    for turn in range(20):
        message = (
            case.first
            if turn == 0
            else case.last
            if turn == 19
            else f"补充说明 {case.name}：第 {turn} 轮，请继续处理"
        )
        start = time.perf_counter()
        response = agent.chat(user_id="USER-001", message=message, session_id=session_id)
        latencies.append((time.perf_counter() - start) * 1000)
        responses.append(response)
        session_id = response.session.session_id
        prompts.append(agent.layered_memory.assemble(response.session, message).estimated_tokens)
    final = responses[-1]
    tools = tuple(event.name for event in final.events if event.event_type == "tool")
    mutations = [
        event.name
        for response in responses
        for event in response.events
        if event.event_type == "tool"
        and event.name in {"create_refund_request", "create_ticket"}
        and event.detail.get("ok")
    ]
    summary_outcomes = [
        event.detail.get("status")
        for response in responses
        for event in response.events
        if event.name == "context_compaction"
    ]
    summary_failure_reasons = [
        event.detail.get("reason_code", event.detail.get("error_type", "unknown"))
        for response in responses
        for event in response.events
        if event.name == "context_compaction" and event.detail.get("status") == "failed"
    ]
    return {
        "case": case.name,
        "mode": "B1" if compact else "B0",
        "final_reply": final.message,
        "task_success": case.expected_text in final.message and tools == case.expected_tools,
        "wrong_state_reference": final.session.order_id != case.expected_order,
        "final_order_id": final.session.order_id,
        "final_goal_version": final.session.goal_version,
        "final_plan_version": final.session.plan_version,
        "final_tool_names": tools,
        "successful_business_writes": mutations,
        "retained_messages": len(final.session.turns),
        "retained_text_tokens_estimate": ContextCompactor.tokens(final.session.turns),
        "context_pack_tokens_mean_estimate": round(statistics.mean(prompts), 1),
        "context_pack_tokens_max_estimate": max(prompts),
        "turn_latency_ms_mean": round(statistics.mean(latencies), 2),
        "summary_calls": model.calls,
        "summary_successes": summary_outcomes.count("completed"),
        "summary_failures": summary_outcomes.count("failed"),
        "summary_failure_reasons": summary_failure_reasons,
        "actual_prompt_tokens": (
            model.usage.get("prompt_tokens", 0) if live_settings is not None else None
        ),
        "actual_completion_tokens": (
            model.usage.get("completion_tokens", 0) if live_settings is not None else None
        ),
        "max_prompt_tokens_per_request": (
            max((record.get("prompt_tokens", 0) for record in model.usage_records), default=0)
            if live_settings is not None
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-summary", action="store_true")
    parser.add_argument("--case", choices=[case.name for case in CASES])
    args = parser.parse_args()
    live_settings = None
    if args.live_summary:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    parsed = shlex.split(value, comments=True)
                    os.environ.setdefault(key.strip(), parsed[0] if parsed else "")
        live_settings = OnlineModelSettings.from_env()
        try:
            DeepSeekContextSummaryModel(
                UrllibJsonTransport(live_settings),
                os.getenv("SERVICEPILOT_CONTEXT_MODEL") or "deepseek-chat",
            ).summarize(None, [{"role": "user", "content": "售后"}])
        except Exception as exc:
            status = "HTTP 401" if "HTTP 401" in str(exc) else type(exc).__name__
            raise SystemExit(f"DeepSeek preflight failed: {status}; no live result saved") from exc
    rows = []
    cases = tuple(case for case in CASES if args.case in (None, case.name))
    for case in cases:
        rows.extend((run_case(case, False, live_settings), run_case(case, True, live_settings)))
    pairs = list(zip(rows[::2], rows[1::2], strict=True))
    metrics = {
        "cases": len(pairs),
        "turns_per_case": 20,
        "reply_parity_rate": sum(a["final_reply"] == b["final_reply"] for a, b in pairs)
        / len(pairs),
        "mechanical_state_parity_rate": sum(
            (a["final_order_id"], a["final_goal_version"], a["final_plan_version"])
            == (b["final_order_id"], b["final_goal_version"], b["final_plan_version"])
            for a, b in pairs
        )
        / len(pairs),
        "tool_and_write_parity_rate": sum(
            (a["final_tool_names"], a["successful_business_writes"])
            == (b["final_tool_names"], b["successful_business_writes"])
            for a, b in pairs
        )
        / len(pairs),
    }
    for mode in ("B0", "B1"):
        subset = [row for row in rows if row["mode"] == mode]
        metrics[mode] = {
            key: round(statistics.mean(row[key] for row in subset), 2)
            for key in (
                "retained_messages",
                "retained_text_tokens_estimate",
                "context_pack_tokens_mean_estimate",
                "context_pack_tokens_max_estimate",
                "turn_latency_ms_mean",
                "summary_calls",
                "summary_successes",
                "summary_failures",
            )
        }
        metrics[mode]["task_success_rate"] = sum(row["task_success"] for row in subset) / len(
            subset
        )
        metrics[mode]["wrong_state_references"] = sum(
            row["wrong_state_reference"] for row in subset
        )
        metrics[mode]["correct_tool_rate"] = sum(
            row["final_tool_names"] == case.expected_tools
            for row, case in zip(subset, cases, strict=True)
        ) / len(subset)
        if live_settings is not None:
            metrics[mode]["actual_prompt_tokens_total"] = sum(
                row["actual_prompt_tokens"] for row in subset
            )
            metrics[mode]["actual_prompt_tokens_per_case_mean"] = round(
                statistics.mean(row["actual_prompt_tokens"] for row in subset), 2
            )
            metrics[mode]["actual_completion_tokens_total"] = sum(
                row["actual_completion_tokens"] for row in subset
            )
            metrics[mode]["max_prompt_tokens_per_request"] = max(
                row["max_prompt_tokens_per_request"] for row in subset
            )
            calls = sum(row["summary_calls"] for row in subset)
            metrics[mode]["actual_prompt_tokens_per_request_mean"] = (
                round(metrics[mode]["actual_prompt_tokens_total"] / calls, 2) if calls else 0
            )
            metrics[mode]["actual_prompt_tokens_per_turn_mean"] = round(
                metrics[mode]["actual_prompt_tokens_total"] / (len(subset) * 20), 2
            )
    result = {
        "notes": (
            "DeepSeek summary calls only; B0 has no model calls in this deterministic runtime."
            if live_settings is not None
            else "Offline extractive stub; no DeepSeek calls or actual prompt-token usage."
        ),
        "summary": metrics,
        "cases": rows,
    }
    name = "context_compaction_live.json" if live_settings else "context_compaction_offline.json"
    output = ROOT / "evaluations" / "results" / name
    if args.case:
        output = output.with_name(output.stem + "_single.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
