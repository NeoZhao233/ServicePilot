from __future__ import annotations

from dataclasses import dataclass

from .models import AgentEvent, ToolResult
from .observability import event, observed


@dataclass(frozen=True)
class FailureAnalysis:
    category: str
    strategy: str


class FailureAnalyzer:
    @staticmethod
    def analyze(result: ToolResult) -> FailureAnalysis:
        code = result.data.get("code", "tool_failure")
        categories = {
            "schema": ("schema", "repair_plan"),
            "precondition": ("precondition", "reconcile"),
            "authorization": ("authorization", "reconfirm"),
            "permission": ("permission", "escalate"),
            "policy_evidence": ("policy_evidence", "escalate"),
            "outcome_unknown": ("outcome_unknown", "query_operation"),
            "goal_changed": ("goal_changed", "reconcile"),
        }
        return FailureAnalysis(*categories.get(code, ("tool_failure", "escalate")))


class RecoveryPolicy:
    """Reconcile a write once; never interpret an unavailable query as failure."""

    @staticmethod
    @observed("reconcile_refund", "recovery")
    def reconcile_refund(
        repository, *, operation_id: str, user_id: str, events: list[AgentEvent]
    ) -> ToolResult:
        event("RECOVERY_STARTED", operation_id=operation_id, strategy="query_operation")
        try:
            observed = repository.get_operation(operation_id=operation_id, user_id=user_id)
        except Exception:
            observed = ToolResult(False, error="业务操作查询不可用")
        events.append(
            AgentEvent(
                "tool",
                "get_operation",
                {
                    "ok": observed.ok,
                    "operation_id": operation_id,
                    **observed.data,
                    "error": observed.error,
                },
            )
        )
        if observed.ok and observed.data.get("state") == "succeeded":
            events.append(
                AgentEvent(
                    "recovery",
                    "response_lost_reconciled",
                    {
                        "operation_id": operation_id,
                        "strategy": "continue_without_resubmit",
                    },
                )
            )
            event("RECOVERY_COMPLETED", operation_id=operation_id, outcome="succeeded")
            return ToolResult(True, observed.data)
        event("EXECUTION_UNCERTAIN", operation_id=operation_id, outcome="unknown")
        repository.freeze_operation(operation_id, user_id, "结果无法确认，需人工对账")
        return ToolResult(
            False,
            {"code": "outcome_unknown", "outcome": "unknown", "operation_id": operation_id},
            "结果尚未确认，已冻结并转人工核对",
        )
