import importlib.util
import sys
from pathlib import Path

EVALUATIONS = Path(__file__).resolve().parents[1] / "evaluations"
sys.path.insert(0, str(EVALUATIONS))
spec = importlib.util.spec_from_file_location(
    "dynamic_policy_ablation", EVALUATIONS / "run_dynamic_policy_ablation.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_missing_policy_requires_handoff_without_business_write():
    case = {"task": "refund", "evidence": "missing"}
    turns = [{"reply": "缺少可信政策依据，需人工核对。"}]
    assert module.judge(case, turns, {"refunds": 0, "tickets": 0})[0]
    assert not module.judge(case, turns, {"refunds": 1, "tickets": 0})[0]


def test_present_policy_refund_requires_confirmed_ledger_result():
    case = {"task": "refund", "evidence": "present"}
    turns = [{"reply": "请回复确认"}, {"reply": "退款申请已提交"}]
    assert module.judge(case, turns, {"refunds": 1, "tickets": 0})[0]
    assert not module.judge(case, turns, {"refunds": 0, "tickets": 0})[0]


def test_summary_keeps_missing_and_present_slices_separate():
    rows = [
        {"case_id": "a", "group": "STATIC", "evidence": "missing", "passed": False},
        {"case_id": "a", "group": "DYNAMIC", "evidence": "missing", "passed": True},
        {"case_id": "b", "group": "STATIC", "evidence": "present", "passed": True},
        {"case_id": "b", "group": "DYNAMIC", "evidence": "present", "passed": True},
    ]
    summary = module.summarize(rows)
    assert summary["overall"]["STATIC_passed"] == 1
    assert summary["overall"]["DYNAMIC_passed"] == 2
    assert summary["evidence"]["missing"]["DYNAMIC_better"] == 1
    assert summary["evidence"]["present"]["DYNAMIC_better"] == 0
