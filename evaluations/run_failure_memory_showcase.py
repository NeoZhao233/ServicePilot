"""Small, reproducible Failure Memory portfolio demo using the real runtime.

The planner is scripted. Results demonstrate wiring and safety, not model quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from servicepilot.cli import build_agent
from servicepilot.failure_memory import FailureMemoryStore
from servicepilot.observability import TraceService
from servicepilot.planning import TaskPlanner, TaskState
from servicepilot.trace_store import TraceStore

ROOT = Path(__file__).resolve().parents[1]
USER = "USER-001"
ORDER = "ORD-1001"


class LessonAwareScriptedPlanner:
    """Repeat a malformed plan until a verified lesson appears in context."""

    def propose(self, template, _goal, observation):
        if not observation.get("verified_failure_lessons"):
            return {"steps": []}
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


def runtime_at(directory: Path, *, planner=None, memory=None):
    directory.mkdir(parents=True, exist_ok=True)
    business = directory / "business.sqlite"
    trace = TraceService(TraceStore(directory / "trace.sqlite"))
    # The demo passes B2 memory explicitly; ambient user settings cannot change B0.
    with patch.dict(
        os.environ, {"SERVICEPILOT_FAILURE_MEMORY_MODE": "B0", "SERVICEPILOT_FAILURE_MEMORY": "0"}
    ):
        runtime = build_agent(
            ROOT / "data",
            dynamic=True,
            session_db=business,
            commerce_db=business,
            trace_service=trace,
            planner=planner,
            failure_memory=memory,
        )
    return runtime, trace, business


def trace_events(trace: TraceService, session_id: str) -> list[str]:
    owner = trace.redactor.pseudonym(USER)
    rows = trace.store.list(owner, thread_id=session_id)["items"]
    return [
        event["event_type"]
        for row in rows
        for event in trace.store.detail(row["trace_id"], owner)["events"]
    ]


def refund_count(business: Path) -> int:
    with sqlite3.connect(business) as db:
        return db.execute(
            "SELECT count(*) FROM commerce_refunds WHERE order_id=?", (ORDER,)
        ).fetchone()[0]


def normal_refund(directory: Path) -> dict:
    runtime, trace, business = runtime_at(directory, planner=TaskPlanner())
    first = runtime.chat(user_id=USER, message=f"退掉 {ORDER}")
    before = refund_count(business)
    second = runtime.chat(user_id=USER, message="确认", session_id=first.session.session_id)
    task = TaskState.from_dict(runtime.dynamic.repository.load_task(first.session.session_id))
    return {
        "confirmation_requested": "确认" in first.message,
        "refunds_before_confirmation": before,
        "refunds_after_confirmation": refund_count(business),
        "task_completed": task.execution_status == "completed",
        "response_reports_submission": "已提交" in second.message,
        "ledger_checked": "get_operation" in [event.name for event in second.events],
    }


def stale_plan(directory: Path) -> dict:
    memory = FailureMemoryStore(directory / "memory.sqlite")
    runtime, trace, business = runtime_at(directory / "runtime", memory=memory)
    first = runtime.chat(user_id=USER, message=f"退掉 {ORDER}")
    task = TaskState.from_dict(runtime.dynamic.repository.load_task(first.session.session_id))
    task.goal_version += 1
    step = next(s for s in task.current_plan.steps if s.action == "create_refund_request")
    state = {
        "task": task,
        "session": first.session,
        "events": [],
        "reply": "",
        "citations": (),
        "next": "",
        "step_id": step.step_id,
    }
    with trace.trace(user_id=USER, thread_id=first.session.session_id):
        runtime.dynamic._execute_step(state)
    kinds = [record.failure_type.value for record in memory.list_active()]
    events = trace_events(trace, first.session.session_id)
    return {
        "stale_action_blocked": state["task"].execution_status == "escalated",
        "refunds_created": refund_count(business),
        "verified_failure_types": kinds,
        "candidate_traced": "FAILURE_CANDIDATE_CREATED" in events,
        "verification_traced": "FAILURE_VERIFIED" in events,
    }


def paired_followups(directory: Path) -> dict:
    groups = {}
    for group in ("B0", "B2"):
        group_dir = directory / group
        memory = FailureMemoryStore(group_dir / "failure.sqlite") if group == "B2" else None
        planner = TaskPlanner(LessonAwareScriptedPlanner())
        training, training_trace, _ = runtime_at(
            group_dir / "training", planner=planner, memory=memory
        )
        first = training.chat(user_id=USER, message=f"退掉 {ORDER}")
        training_events = trace_events(training_trace, first.session.session_id)
        rows = []
        for index in range(1, 4):
            runtime, trace, business = runtime_at(
                group_dir / f"followup_{index}",
                planner=TaskPlanner(LessonAwareScriptedPlanner()),
                memory=memory,
            )
            response = runtime.chat(user_id=USER, message=f"退掉 {ORDER}")
            runtime.chat(user_id=USER, message="确认", session_id=response.session.session_id)
            task = TaskState.from_dict(
                runtime.dynamic.repository.load_task(response.session.session_id)
            )
            events = trace_events(trace, response.session.session_id)
            rows.append(
                {
                    "case_id": f"followup_{index}",
                    "plan_validation_errors": events.count("PLAN_VALIDATION_FAILED"),
                    "memories_injected": events.count("FAILURE_MEMORY_APPLIED"),
                    "task_completed": task.execution_status == "completed",
                    "refunds_created": refund_count(business),
                }
            )
        groups[group] = {
            "training_plan_validation_errors": training_events.count("PLAN_VALIDATION_FAILED"),
            "verified_memories_after_training": len(memory.list_active()) if memory else 0,
            "followups": rows,
            "repeated_plan_errors": sum(row["plan_validation_errors"] for row in rows),
            "completed_tasks": sum(row["task_completed"] for row in rows),
            "duplicate_refunds": sum(row["refunds_created"] > 1 for row in rows),
        }
    return groups


def run() -> dict:
    with TemporaryDirectory(prefix="servicepilot-failure-showcase-") as temporary:
        work = Path(temporary)
        return {
            "method": "offline scripted planner; real LangGraph runtime, Trace and SQLite ledger",
            "scope": "fixed functional showcase, not an online model effectiveness estimate",
            "normal_refund": normal_refund(work / "normal"),
            "stale_plan_fault_injection": stale_plan(work / "stale"),
            "paired_followups": paired_followups(work / "paired"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
