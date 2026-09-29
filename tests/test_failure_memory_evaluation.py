import pytest

from servicepilot.failure_memory_evaluation import EvaluationRun, summarize


def run(group, case_id, *, success, failures=(), retrieved=0, applied=0):
    return EvaluationRun(
        group=group,
        case_id=case_id,
        scenario="same_pattern",
        failure_pattern="PLANNING_ERROR",
        seed=42,
        model_id="frozen-planner-v1",
        policy_version="policy-v1",
        tool_version="tool-v1",
        oracle_ref="oracle:case1",
        post_failure=True,
        first_followup=True,
        task_success=success,
        verified_agent_failure_patterns=list(failures),
        retrieved_count=retrieved,
        invalid_retrieval_count=0,
        applied_count=applied,
        wrong_memory_use_count=0,
        input_tokens=100,
        output_tokens=20,
        latency_ms=50,
        tool_calls=2,
        replans=0,
    )


def test_repeated_failure_rate_uses_only_verified_same_pattern():
    rows = [
        run("B0", "case1", success=False, failures=["PLANNING_ERROR"]),
        run("B1", "case1", success=True, failures=["TOOL_EXECUTION_ERROR"], retrieved=2, applied=2),
        run("B2", "case1", success=True, retrieved=1, applied=1),
    ]
    report = summarize(rows)
    assert report["groups"]["B0"]["repeated_failure_rate"] == 1
    assert report["groups"]["B1"]["repeated_failure_rate"] == 0
    assert report["groups"]["B2"]["mean_memories_injected"] == 1


def test_missing_or_misaligned_baseline_is_rejected():
    with pytest.raises(ValueError):
        summarize([run("B0", "a", success=True)])
    with pytest.raises(ValueError):
        summarize(
            [
                run("B0", "a", success=True),
                run("B1", "b", success=True),
                run("B2", "a", success=True),
            ]
        )
