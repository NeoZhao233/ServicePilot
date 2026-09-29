from __future__ import annotations

import sqlite3
from pathlib import Path

from servicepilot.agent import ServiceAgent
from servicepilot.context_manager import ContextCompactor, DeepSeekContextSummaryModel
from servicepilot.graph_runtime import LangGraphServiceRuntime
from servicepilot.memory import SQLiteSessionStore, deserialize_state, serialize_state
from servicepilot.models import SessionState
from servicepilot.planning import TaskPlanner, TaskState
from servicepilot.repository import InMemoryCommerceRepository
from servicepilot.retrieval import HybridPolicyRetriever


class ExtractiveModel:
    def __init__(self, fail=False, invalid=False):
        self.calls = 0
        self.fail = fail
        self.invalid = invalid

    def summarize(self, old_summary, messages):
        self.calls += 1
        if self.fail:
            raise RuntimeError("offline")
        result = {
            "current_goal": "退货",
            "confirmed_facts": [],
            "completed_steps": ["核对照片"],
            "unresolved_issues": ["等待客服回复"],
            "important_constraints": [],
            "relevant_policy_refs": [],
            "important_tool_results": [],
        }
        if self.invalid:
            result.pop("completed_steps")
        return result


def state():
    return SessionState(
        "S",
        "U",
        order_id="ORD-1001",
        goal_version=4,
        plan_version=3,
        authorization={"token": "real"},
        turns=[
            {"role": "user", "content": "退货"},
            {"role": "assistant", "content": "核对照片"},
            {"role": "user", "content": "等待客服回复"},
            {"role": "assistant", "content": "处理中"},
            {"role": "user", "content": "继续"},
        ],
    )


def test_short_conversation_is_not_compacted():
    model = ExtractiveModel()
    compact = ContextCompactor(model, max_messages=10, max_tokens=100)
    assert compact.compact(state()) is None
    assert model.calls == 0


def test_message_threshold_preserves_recent_and_mechanical_state():
    model = ExtractiveModel()
    compact = ContextCompactor(model, max_messages=4, max_tokens=1000, keep_recent_messages=2)
    session = state()
    recent = session.turns[-2:]
    result = compact.compact(session)
    assert result["status"] == "completed"
    assert session.turns == recent
    assert session.context_summary["content"]["current_goal"] == "退货"
    assert session.context_summary["content"]["unresolved_issues"] == ["等待客服回复"]
    assert session.context_summary["content"]["completed_steps"] == ["核对照片"]
    assert (session.order_id, session.goal_version, session.plan_version) == ("ORD-1001", 4, 3)
    assert session.authorization == {"token": "real"}
    assert model.calls == 1


def test_token_threshold_and_rolling_summary():
    model = ExtractiveModel()
    compact = ContextCompactor(model, max_messages=100, max_tokens=6, keep_recent_messages=2)
    session = state()
    assert compact.compact(session)["status"] == "completed"
    session.turns.extend(
        [
            {"role": "user", "content": "退货"},
            {"role": "assistant", "content": "核对照片"},
            {"role": "user", "content": "等待客服回复"},
        ]
    )
    assert compact.compact(session)["status"] == "completed"
    assert session.context_summary["summary_version"] == 2
    assert session.context_summary["source_message_range"] == [0, 6]


def test_model_failure_or_invalid_output_keeps_history_and_old_summary():
    for model in (ExtractiveModel(fail=True), ExtractiveModel(invalid=True)):
        session = state()
        session.context_summary = {"content": {"current_goal": "退货"}, "summary_version": 2}
        before = list(session.turns)
        assert (
            ContextCompactor(model, max_messages=4, keep_recent_messages=2).compact(session)[
                "status"
            ]
            == "failed"
        )
        assert session.turns == before
        assert session.context_summary["summary_version"] == 2


def test_failed_summary_waits_for_new_history_before_retry():
    model = ExtractiveModel(fail=True)
    session = state()
    compact = ContextCompactor(model, max_messages=4, keep_recent_messages=2)
    assert compact.compact(session)["status"] == "failed"
    assert model.calls == 1
    assert compact.compact(session) is None
    assert model.calls == 1
    session.turns.extend(
        [
            {"role": "assistant", "content": "继续处理"},
            {"role": "user", "content": "还有进展吗"},
        ]
    )
    assert compact.compact(session)["status"] == "failed"
    assert model.calls == 2


def test_fabricated_mechanical_values_are_removed_from_summary():
    class BadModel(ExtractiveModel):
        def summarize(self, old_summary, messages):
            result = super().summarize(old_summary, messages)
            result["confirmed_facts"] = ["order_id=ORD-9999"]
            return result

    session = state()
    assert (
        ContextCompactor(BadModel(), max_messages=4, keep_recent_messages=2).compact(session)[
            "status"
        ]
        == "completed"
    )
    assert session.order_id == "ORD-1001"
    assert session.context_summary["content"]["confirmed_facts"] == []


def test_stale_policy_and_failure_memory_do_not_enter_summary():
    class StaleModel(ExtractiveModel):
        def summarize(self, old_summary, messages):
            result = super().summarize(old_summary, messages)
            result["relevant_policy_refs"] = ["old-policy"]
            return result

    session = state()
    session.turns[0]["content"] = "退货 old-policy FM-12"
    assert (
        ContextCompactor(StaleModel(), max_messages=4, keep_recent_messages=2).compact(session)[
            "status"
        ]
        == "completed"
    )
    assert session.context_summary["content"]["relevant_policy_refs"] == []


def test_unsupported_model_claim_is_removed_without_changing_goal_state():
    class HallucinatingModel(ExtractiveModel):
        def summarize(self, old_summary, messages):
            result = super().summarize(old_summary, messages)
            result["current_goal"] = "换货"
            result["confirmed_facts"] = ["已收到付款"]
            return result

    session = state()
    result = ContextCompactor(HallucinatingModel(), max_messages=4, keep_recent_messages=2).compact(
        session
    )
    assert result["status"] == "completed"
    assert session.context_summary["content"]["current_goal"] == ""
    assert session.context_summary["content"]["confirmed_facts"] == []
    assert session.order_id == "ORD-1001"


def test_repeated_model_steps_are_deduplicated_and_bounded():
    class RepetitiveModel(ExtractiveModel):
        def summarize(self, old_summary, messages):
            result = super().summarize(old_summary, messages)
            result["completed_steps"] = ["核对照片"] * 12
            result["important_tool_results"] = ["核对照片"] * 12
            return result

    session = state()
    result = ContextCompactor(RepetitiveModel(), max_messages=4, keep_recent_messages=2).compact(
        session
    )
    assert result["status"] == "completed"
    assert session.context_summary["content"]["completed_steps"] == ["核对照片"]
    assert session.context_summary["content"]["important_tool_results"] == ["核对照片"]


def test_planner_ignores_summary_for_tool_order_id():
    class PlannerModel:
        def propose(self, template, goal, observation):
            assert observation["conversation_context"]["summary"]["current_goal"] == "ORD-9999"
            return {
                "steps": [
                    {
                        key: value
                        for key, value in step.items()
                        if key not in {"execution_status", "result"}
                    }
                    for step in template["steps"]
                ]
            }

    task = TaskState("T", "USER-001", {"intent": "refund", "order_id": "ORD-1001"})
    plan = TaskPlanner(PlannerModel()).plan(
        task, context={"summary": {"current_goal": "ORD-9999"}, "recent_messages": []}
    )
    assert {step.parameters["order_id"] for step in plan.steps} == {"ORD-1001"}
    assert (plan.goal_version, plan.plan_version) == (task.goal_version, task.plan_version + 1)


def test_summary_metadata_survives_session_storage():
    session = state()
    compact = ContextCompactor(ExtractiveModel(), max_messages=4, keep_recent_messages=2)
    assert compact.compact(session)["status"] == "completed"
    session.turn_count = 3
    restored = deserialize_state(serialize_state(session))
    assert restored.context_summary == session.context_summary
    assert restored.turn_count == 3


def test_goal_change_drops_stale_history_but_keeps_current_user_message():
    root = Path(__file__).resolve().parents[1]
    agent = ServiceAgent(
        InMemoryCommerceRepository.from_json(root / "data" / "orders.json"),
        HybridPolicyRetriever.from_json(root / "data" / "policies.json"),
    )
    session = state()
    session.turns.append({"role": "user", "content": "改为换货"})
    session.context_summary = {"content": {"current_goal": "退货"}}
    agent.invalidate_goal(session)
    assert session.turns == [{"role": "user", "content": "改为换货"}]
    assert session.context_summary is None
    assert (session.goal_version, session.plan_version) == (5, 4)


def test_deepseek_adapter_tracks_usage_without_storing_prompt_text():
    class Transport:
        def post(self, path, payload):
            assert path == "/chat/completions"
            assert payload["response_format"] == {"type": "json_object"}
            return {
                "choices": [{"message": {"content": '{"current_goal":"退货"}'}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5},
            }

    model = DeepSeekContextSummaryModel(Transport(), "deepseek-chat")
    model.summarize(None, [{"role": "user", "content": "退货"}])
    model.summarize(None, [{"role": "user", "content": "退货"}])
    assert model.usage == {"prompt_tokens": 22, "completion_tokens": 10}
    assert model.last_usage == {"prompt_tokens": 11, "completion_tokens": 5}
    assert len(model.usage_records) == 2
    assert model.calls == 2


def test_compaction_keeps_audit_turn_numbers_monotonic(tmp_path):
    class EmptySummaryModel:
        def summarize(self, old_summary, messages):
            return {
                "current_goal": "",
                "confirmed_facts": [],
                "completed_steps": [],
                "unresolved_issues": [],
                "important_constraints": [],
                "relevant_policy_refs": [],
                "important_tool_results": [],
            }

    root = Path(__file__).resolve().parents[1]
    database = tmp_path / "sessions.sqlite3"
    agent = ServiceAgent(
        InMemoryCommerceRepository.from_json(root / "data" / "orders.json"),
        HybridPolicyRetriever.from_json(root / "data" / "policies.json"),
        session_store=SQLiteSessionStore(database),
        context_compactor=ContextCompactor(
            EmptySummaryModel(), max_messages=4, keep_recent_messages=2
        ),
    )
    runtime = LangGraphServiceRuntime(agent)
    session_id = None
    for index in range(8):
        response = runtime.chat(
            user_id="USER-001", message=f"退货 ORD-1001 第 {index} 次跟进", session_id=session_id
        )
        session_id = response.session.session_id
    with sqlite3.connect(database) as connection:
        indexes = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT turn_index FROM agent_events "
                "WHERE session_id = ? ORDER BY turn_index",
                (session_id,),
            )
        ]
    assert indexes == list(range(1, 9))
    assert response.session.turn_count == 8
    assert response.session.context_summary["summary_version"] >= 1
