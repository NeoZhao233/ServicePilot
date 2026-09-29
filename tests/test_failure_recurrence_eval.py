import importlib.util
import sys
from pathlib import Path

import pytest

EVALUATIONS = Path(__file__).resolve().parents[1] / "evaluations"
sys.path.insert(0, str(EVALUATIONS))
spec = importlib.util.spec_from_file_location(
    "analyze_failure_recurrence", EVALUATIONS / "analyze_failure_recurrence.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_recurrence_denominator_excludes_controls_and_counts_paired_change():
    cases = [
        {
            "case_id": "p",
            "failure_pattern": "role_spoof_no_request",
            "task": "clarify",
            "expected_intent": "unknown",
            "expected_counts": {"refunds": 0, "tickets": 0},
        },
        {"case_id": "c", "failure_pattern": "control"},
    ]
    experiment = {
        "training": {"expected": "unknown", "predicted": "human_handoff"},
        "runs": [
            {
                "case_id": "p",
                "group": "B0",
                "passed": False,
                "failures": ["wrong_final_intent"],
                "memory_injections": 0,
            },
            {
                "case_id": "p",
                "group": "B2",
                "passed": True,
                "failures": [],
                "memory_injections": 1,
            },
            {
                "case_id": "c",
                "group": "B0",
                "passed": True,
                "failures": [],
                "memory_injections": 0,
            },
            {
                "case_id": "c",
                "group": "B2",
                "passed": True,
                "failures": [],
                "memory_injections": 1,
            },
        ],
    }
    summary = module.analyze(cases, experiment)
    assert summary["opportunities"] == 1
    assert summary["B0_recurrence_rate"] == 1
    assert summary["B2_recurrence_rate"] == 0
    assert summary["B2_corrected_cases"] == ["p"]
    assert summary["controls"] == {"n": 1, "B0_passed": 1, "B2_passed": 1}
    assert summary["B2_actual_injections"] == 2


def test_unrelated_failure_cannot_be_counted_as_pattern_recurrence():
    cases = [
        {
            "case_id": "p",
            "failure_pattern": "role_spoof_no_request",
            "task": "clarify",
            "expected_intent": "unknown",
            "expected_counts": {"refunds": 0, "tickets": 0},
        },
        {"case_id": "c", "failure_pattern": "control"},
    ]
    experiment = {
        "training": {"expected": "unknown", "predicted": "human_handoff"},
        "runs": [
            {
                "case_id": case_id,
                "group": group,
                "passed": case_id == "c",
                "failures": ["timeout"] if case_id == "p" else [],
                "memory_injections": 0,
            }
            for case_id in ("p", "c")
            for group in ("B0", "B2")
        ],
    }
    with pytest.raises(ValueError, match="outside the verified pattern"):
        module.analyze(cases, experiment)
