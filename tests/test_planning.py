from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from servicepilot.cli import build_agent
from servicepilot.planning import ChatCompletionPlannerModel, TaskPlanner, TaskState

ROOT = Path(__file__).resolve().parents[1]


def candidate():
    task = TaskState("task", "USER-001", {"intent": "refund", "order_id": "ORD-1001"})
    plan = TaskPlanner().plan(task)
    rows = [
        {k: v for k, v in asdict(s).items() if k not in {"execution_status", "result"}}
        for s in plan.steps
    ]
    return task, {"steps": rows}


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "cycle", "missing", "identity", "authority", "effect", "duplicate", "completed"],
)
def test_model_cannot_bypass_execution_contract(mutation):
    task, plan = candidate()
    if mutation == "unknown":
        plan["steps"][2]["action"] = "transfer_money"
    elif mutation == "cycle":
        plan["steps"][0]["dependencies"] = ["submit"]
    elif mutation == "missing":
        plan["steps"][2]["dependencies"] = []
    elif mutation == "identity":
        plan["steps"][2]["parameters"]["user_id"] = "USER-002"
    elif mutation == "authority":
        plan["steps"][2]["preconditions"].remove("authorized")
    elif mutation == "effect":
        plan["steps"][2]["expected_effect"] = "money_received"
    elif mutation == "duplicate":
        plan["steps"][1] = deepcopy(plan["steps"][0])
    else:
        plan["steps"][2]["execution_status"] = "succeeded"
    with pytest.raises(ValueError):
        TaskPlanner.validate(plan, task, expected_action="create_refund_request")


def test_valid_non_list_order_is_scheduled_by_dependencies():
    task, plan = candidate()
    plan["steps"].reverse()
    parsed = TaskPlanner.validate(plan, task, expected_action="create_refund_request")
    assert parsed.steps[0].dependencies == ["order", "policy"]


def test_malformed_model_falls_back_once_without_authorizing():
    class Model:
        calls = 0

        def propose(self, *args):
            self.calls += 1
            return {"steps": []}

    model = Model()
    runtime = build_agent(ROOT / "data", dynamic=True, planner=TaskPlanner(model))
    response = runtime.chat(user_id="USER-001", message="退掉 ORD-1001")
    assert model.calls == 1
    assert "确认" in response.message
    task = runtime.agent.repository.load_task(response.session.session_id)
    assert task["failure_events"][0]["category"] == "schema"
    assert task["planning_count"] == 1
    assert not any(e.name == "create_refund_request" for e in response.events)


def test_deepseek_adapter_uses_json_mode_and_records_actual_usage():
    import json

    _, plan = candidate()

    class Transport:
        def post(self, path, payload):
            assert path == "/chat/completions"
            assert payload["response_format"] == {"type": "json_object"}
            return {
                "choices": [{"message": {"content": json.dumps(plan)}}],
                "usage": {"total_tokens": 123},
            }

    model = ChatCompletionPlannerModel(Transport(), "test-model")
    result = model.propose({"steps": plan["steps"]}, {}, {})
    assert result == plan
    assert model.usage["total_tokens"] == 123


def test_simple_query_bypasses_dynamic_planner():
    class Model:
        def propose(self, *args):
            raise AssertionError("simple query must not call planner")

    runtime = build_agent(ROOT / "data", dynamic=True, planner=TaskPlanner(Model()))
    response = runtime.chat(user_id="USER-001", message="ORD-1003 物流")
    assert "运输中" in response.message
    assert runtime.agent.repository.load_task(response.session.session_id) is None
