"""Paired real-model B0/B2 planner evaluation on synthetic held-out cases."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from servicepilot.failure_memory import (
    ExperienceExtractor,
    FailureAnalyzer,
    FailureMemoryRetriever,
    FailureMemoryStore,
    record_verified_failure,
)
from servicepilot.planning import (
    ChatCompletionPlannerModel,
    PlanValidationError,
    TaskPlanner,
    TaskState,
)
from servicepilot.providers import OnlineModelSettings, UrllibJsonTransport

ROOT = Path(__file__).resolve().parents[1]


def load_local_settings(env_file: Path) -> OnlineModelSettings:
    # Only load the settings needed for this run; never print or persist credentials.
    allowed = {
        "SERVICEPILOT_API_KEY",
        "SERVICEPILOT_BASE_URL",
        "SERVICEPILOT_INTENT_MODEL",
        "SERVICEPILOT_EMBEDDING_MODE",
        "SERVICEPILOT_EMBEDDING_MODEL",
        "SERVICEPILOT_TIMEOUT_SECONDS",
    }
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                if key.strip() in allowed:
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    return OnlineModelSettings.from_env()


def synthetic_task(case_id: str, note: str) -> TaskState:
    return TaskState(
        task_id=f"synthetic-{case_id}",
        user_id="synthetic-user",
        current_goal={"intent": "refund", "order_id": f"SYNTHETIC-{case_id.upper()}"},
        observed_state={"category": "synthetic-category", "untrusted_note": note},
    )


def seed_synthetic_prior_failure(store: FailureMemoryStore) -> str:
    """Create one independently checked fixture; it is not a real-model failure."""
    task = synthetic_task("training", "fictional training context")
    try:
        TaskPlanner.validate({"steps": []}, task, expected_action="create_refund_request")
    except ValueError:
        pass
    else:
        raise AssertionError("training fixture unexpectedly passed validation")
    analysis = FailureAnalyzer.analyze(
        trace={"status": "ok", "events": [{"event_type": "PLAN_VALIDATION_FAILED"}]},
        task_state=task,
        assertions=[
            {
                "rule": "plan_schema",
                "source": "test_oracle",
                "passed": False,
                "step": "plan",
                "actual": "empty steps",
                "evidence_ref": "oracle:synthetic-training",
            }
        ],
    )
    record = record_verified_failure(
        store,
        ExperienceExtractor(),
        analysis,
        task_id=task.task_id,
        trace_id="synthetic-training-trace",
        task_type="refund",
    )
    if not record:
        raise AssertionError("synthetic training experience was not recorded")
    return record.failure_id


def evaluate_one(transport, model_id, case, group, store):
    task = synthetic_task(case["case_id"], case["untrusted_note"])
    lessons = []
    if store:
        lessons = FailureMemoryRetriever(store).retrieve(
            goal="refund",
            task_type="refund",
            stage="planning",
            conditions={"refund"},
            task_id=task.task_id,
        )
        if not lessons:
            raise AssertionError("B2 experience was not retrieved")
    model = ChatCompletionPlannerModel(transport, model_id)
    injected = []

    def mark_injected():
        for item in lessons:
            store.record_usage(
                item.memory_id,
                trace_id=None,
                task_id=task.task_id,
                stage="planning",
                score=item.score,
                applied=True,
            )
            injected.append(item.memory_id)

    started = time.monotonic()
    try:
        plan = TaskPlanner(model).plan(
            task,
            lessons=tuple(item.lesson for item in lessons),
            on_lessons_injected=mark_injected if lessons else None,
        )
        valid = True
        actions = [step.action for step in plan.steps]
        validation_error = None
    except PlanValidationError as exc:
        valid = False
        actions = []
        validation_error = str(exc)
    usage = model.usage
    return {
        "case_id": case["case_id"],
        "group": group,
        "valid_plan": valid,
        "validation_error": validation_error,
        "actions": actions,
        "memories_injected": len(injected),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "latency_ms": round((time.monotonic() - started) * 1000),
    }


def summarize(rows):
    pairs = {}
    for row in rows:
        pairs.setdefault((row["repeat"], row["case_id"]), {})[row["group"]] = row
    if any(set(pair) != {"B0", "B2"} for pair in pairs.values()):
        raise AssertionError("all cases must have B0/B2 results")
    b0_errors = sum(not pair["B0"]["valid_plan"] for pair in pairs.values())
    b2_errors = sum(not pair["B2"]["valid_plan"] for pair in pairs.values())
    improved = sum(not p["B0"]["valid_plan"] and p["B2"]["valid_plan"] for p in pairs.values())
    worsened = sum(p["B0"]["valid_plan"] and not p["B2"]["valid_plan"] for p in pairs.values())
    return {
        "paired_cases": len(pairs),
        "B0_validation_errors": b0_errors,
        "B2_validation_errors": b2_errors,
        "B0_total_tokens": sum(
            (row["input_tokens"] or 0) + (row["output_tokens"] or 0)
            for row in rows
            if row["group"] == "B0"
        ),
        "B2_total_tokens": sum(
            (row["input_tokens"] or 0) + (row["output_tokens"] or 0)
            for row in rows
            if row["group"] == "B2"
        ),
        "B2_memories_injected": sum(
            row["memories_injected"] for row in rows if row["group"] == "B2"
        ),
        "B2_better_pairs": improved,
        "B2_worse_pairs": worsened,
        "conclusion": (
            "no observed improvement on this set"
            if b0_errors <= b2_errors
            else "observed difference on this small set; replicate before claiming benefit"
        ),
    }


def run(settings, model_id, cases, repeats, seed):
    transport = UrllibJsonTransport(settings)
    store = FailureMemoryStore(":memory:")
    seed_synthetic_prior_failure(store)
    ordered = [(repeat, case) for repeat in range(repeats) for case in cases]
    random.Random(seed).shuffle(ordered)
    rows = []
    for index, (repeat, case) in enumerate(ordered):
        # Alternate order to limit drift and cache effects between conditions.
        groups = ("B0", "B2") if index % 2 == 0 else ("B2", "B0")
        for group in groups:
            row = evaluate_one(transport, model_id, case, group, store if group == "B2" else None)
            row["repeat"] = repeat
            rows.append(row)
        print(f"completed pair {index + 1}/{len(ordered)}", flush=True)
    return {
        "method": "real remote model, temperature=0, paired synthetic held-out planner tasks",
        "model_id": model_id,
        "seed": seed,
        "repeats": repeats,
        "memory_source": "independently validated synthetic prior failure; not a model failure",
        "oracle": "TaskPlanner.validate structural and safety checks",
        "limits": "Planner-only test; does not measure customer outcomes or broad intelligence",
        "summary": summarize(rows),
        "runs": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-online", action="store_true", required=False)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "evaluations/failure_memory_real_cases.json"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evaluations/results/failure_memory_real_eval.json"
    )
    args = parser.parse_args()
    if not args.allow_online:
        parser.error("pass --allow-online to make paid model requests")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        parser.error("cases must be nonempty with unique case_id values")
    settings = load_local_settings(args.env_file)
    report = run(settings, args.model, cases, args.repeats, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
