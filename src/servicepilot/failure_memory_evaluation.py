"""Evaluate paired B0/B1/B2 Failure Memory runs from independent task-oracle logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

SCENARIOS = {
    "same_pattern",
    "different_order",
    "same_tool_different_cause",
    "invalidated_memory",
    "irrelevant_memory",
    "conflicting_memories",
}
GROUPS = {"B0", "B1", "B2"}


class EvaluationRun(BaseModel):
    group: str
    case_id: str
    scenario: str
    failure_pattern: str
    seed: int
    model_id: str
    policy_version: str
    tool_version: str
    oracle_ref: str
    post_failure: bool
    first_followup: bool = False
    task_success: bool
    verified_agent_failure_patterns: list[str] = Field(default_factory=list)
    retrieved_count: int = Field(ge=0)
    invalid_retrieval_count: int = Field(ge=0)
    applied_count: int = Field(ge=0)
    wrong_memory_use_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    tool_calls: int = Field(ge=0)
    replans: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_protocol(self):
        if self.group not in GROUPS or self.scenario not in SCENARIOS:
            raise ValueError("unknown group or scenario")
        if not all((self.model_id, self.policy_version, self.tool_version, self.oracle_ref)):
            raise ValueError("evaluation provenance is required")
        if self.invalid_retrieval_count > self.retrieved_count:
            raise ValueError("invalid retrieval count exceeds retrieved count")
        if self.applied_count > self.retrieved_count:
            raise ValueError("applied count exceeds retrieved count")
        if self.wrong_memory_use_count > self.applied_count:
            raise ValueError("wrong use count exceeds applied count")
        if self.first_followup and not self.post_failure:
            raise ValueError("first followup must be post-failure")
        return self


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(runs: list[EvaluationRun], *, require_paired: bool = True) -> dict:
    if not runs:
        raise ValueError("no evaluation runs")
    grouped = defaultdict(dict)
    for run in runs:
        if run.case_id in grouped[run.group]:
            raise ValueError(f"duplicate case {run.group}/{run.case_id}")
        grouped[run.group][run.case_id] = run
    if require_paired:
        if set(grouped) != GROUPS:
            raise ValueError("B0, B1 and B2 are all required")
        case_sets = [set(grouped[group]) for group in sorted(GROUPS)]
        if len({frozenset(ids) for ids in case_sets}) != 1:
            raise ValueError("case IDs must match across B0/B1/B2")
        for case_id in case_sets[0]:
            scenarios = {grouped[group][case_id].scenario for group in GROUPS}
            patterns = {grouped[group][case_id].failure_pattern for group in GROUPS}
            controls = {
                (
                    grouped[group][case_id].seed,
                    grouped[group][case_id].model_id,
                    grouped[group][case_id].policy_version,
                    grouped[group][case_id].tool_version,
                    grouped[group][case_id].oracle_ref,
                )
                for group in GROUPS
            }
            if len(scenarios) != 1 or len(patterns) != 1 or len(controls) != 1:
                raise ValueError(f"paired case {case_id} differs across groups")
    report = {"sample_size": len(runs), "groups": {}}
    for group, by_case in sorted(grouped.items()):
        rows = list(by_case.values())
        followups = [r for r in rows if r.post_failure]
        repeats = sum(r.failure_pattern in r.verified_agent_failure_patterns for r in followups)
        first = [r for r in followups if r.first_followup]
        retrieved = sum(r.retrieved_count for r in rows)
        applied = sum(r.applied_count for r in rows)
        report["groups"][group] = {
            "n": len(rows),
            "post_failure_n": len(followups),
            "repeated_failure_rate": _rate(repeats, len(followups)),
            "task_success_rate": _rate(sum(r.task_success for r in rows), len(rows)),
            "first_followup_success_rate": _rate(sum(r.task_success for r in first), len(first)),
            "invalid_memory_retrieval_rate": _rate(
                sum(r.invalid_retrieval_count for r in rows), retrieved
            ),
            "wrong_memory_use_rate": _rate(sum(r.wrong_memory_use_count for r in rows), applied),
            "mean_memories_injected": round(applied / len(rows), 3),
            "mean_tokens": round(
                sum(r.input_tokens + r.output_tokens for r in rows) / len(rows), 2
            ),
            "mean_latency_ms": round(sum(r.latency_ms for r in rows) / len(rows), 2),
            "mean_tool_calls": round(sum(r.tool_calls for r in rows) / len(rows), 2),
            "mean_replans": round(sum(r.replans for r in rows) / len(rows), 2),
            "scenario_counts": {
                scenario: sum(r.scenario == scenario for r in rows)
                for scenario in sorted(SCENARIOS)
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired Failure Memory experiment logs")
    parser.add_argument("runs", type=Path, help="JSONL of independent oracle-scored task runs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [
        EvaluationRun.model_validate_json(line)
        for line in args.runs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = summarize(rows)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
