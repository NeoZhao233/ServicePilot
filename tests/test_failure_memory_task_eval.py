import importlib.util
import sys
from pathlib import Path

EVALUATIONS = Path(__file__).resolve().parents[1] / "evaluations"
sys.path.insert(0, str(EVALUATIONS))
spec = importlib.util.spec_from_file_location(
    "failure_memory_task_eval", EVALUATIONS / "run_failure_memory_task_eval.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_clarification_requires_no_business_side_effects():
    case = {
        "task": "clarify",
        "expected_intent": "unknown",
        "expected_counts": {"refunds": 0, "tickets": 0},
    }
    turn = {"intent": "unknown", "reply": "请告诉我你的售后问题。", "tools": []}
    assert module.judge(case, [turn], {"refunds": 0, "tickets": 0})[0]
    assert not module.judge(case, [turn], {"refunds": 0, "tickets": 1})[0]


def test_refund_requires_confirmed_single_submission_and_ledger_entry():
    case = {
        "task": "refund",
        "expected_intent": "refund",
        "expected_counts": {"refunds": 1, "tickets": 0},
    }
    turns = [
        {"intent": "refund", "reply": "请确认", "tools": ["get_order"], "pending_action": "refund"},
        {
            "intent": "refund",
            "reply": "退款申请已提交",
            "tools": ["get_order", "create_refund_request"],
            "pending_action": None,
        },
    ]
    assert module.judge(case, turns, {"refunds": 1, "tickets": 0})[0]
    assert not module.judge(case, turns, {"refunds": 0, "tickets": 0})[0]
    turns[0]["pending_action"] = None
    assert not module.judge(case, turns, {"refunds": 1, "tickets": 0})[0]


def test_paired_summary_counts_outcomes_and_regressions():
    rows = [
        {"case_id": "a", "task": "clarify", "group": "B0", "passed": False},
        {"case_id": "a", "task": "clarify", "group": "B2", "passed": True},
        {"case_id": "b", "task": "refund", "group": "B0", "passed": True},
        {"case_id": "b", "task": "refund", "group": "B2", "passed": False},
    ]
    summary = module.summarize(rows)
    assert summary["B0_passed"] == summary["B2_passed"] == 1
    assert summary["slices"]["clarify"]["B2_better"] == 1
    assert summary["slices"]["refund"]["B2_worse"] == 1
