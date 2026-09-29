"""Paired end-to-end customer-service outcome evaluation with a real intent model.

All orders, policies, and messages used by this experiment are synthetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import tempfile
from pathlib import Path

from run_intent_failure_memory_eval import (
    TRAINING_MESSAGE,
    seed_from_observed_failure,
)
from run_real_failure_memory_eval import load_local_settings

from servicepilot.cli import build_agent
from servicepilot.failure_memory import FailureMemoryStore
from servicepilot.providers import ResponsesIntentModel, UrllibJsonTransport

ROOT = Path(__file__).resolve().parents[1]


def synthetic_data(directory: Path) -> None:
    (directory / "orders.json").write_text(
        json.dumps(
            [
                {
                    "order_id": "ORD-9001",
                    "user_id": "SYNTHETIC-USER",
                    "product_name": "虚构运动鞋",
                    "category": "运动鞋服",
                    "amount": 100,
                    "status": "delivered",
                    "days_since_delivery": 2,
                    "logistics_status": "虚构包裹已签收",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (directory / "policies.json").write_text(
        json.dumps(
            [
                {
                    "policy_id": "SYNTHETIC-RETURN",
                    "title": "虚构退货规则",
                    "content": "签收七天内、商品完好可申请退款；必须由本人确认后提交。",
                    "categories": [],
                    "tags": ["退款", "退货"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def database_counts(path: Path) -> dict[str, int]:
    with sqlite3.connect(path) as db:
        return {
            "refunds": db.execute("SELECT count(*) FROM commerce_refunds").fetchone()[0],
            "tickets": db.execute("SELECT count(*) FROM commerce_tickets").fetchone()[0],
        }


def judge(case: dict, turns: list[dict], counts: dict[str, int]) -> tuple[bool, list[str]]:
    """Fixed oracle over workflow output and persisted business state."""
    last = turns[-1]
    all_tools = [tool for turn in turns for tool in turn["tools"]]
    failures = []
    if last["intent"] != case["expected_intent"]:
        failures.append("wrong_final_intent")
    if counts != case["expected_counts"]:
        failures.append("wrong_business_state")
    if case["task"] == "clarify":
        if all_tools or "请告诉我" not in last["reply"]:
            failures.append("did_not_clarify_without_tools")
    elif case["task"] == "logistics":
        if "虚构包裹已签收" not in last["reply"] or "get_order" not in all_tools:
            failures.append("logistics_not_delivered")
    elif case["task"] == "refund":
        if len(turns) != 2 or turns[0]["pending_action"] != "refund":
            failures.append("confirmation_not_requested")
        if "退款申请已提交" not in last["reply"] or all_tools.count("create_refund_request") != 1:
            failures.append("refund_not_completed_once")
    elif case["task"] == "handoff":
        if "人工客服工单" not in last["reply"] or all_tools.count("create_ticket") != 1:
            failures.append("handoff_not_completed_once")
    else:
        raise ValueError(f"unknown task: {case['task']}")
    return not failures, failures


def run_one(case: dict, group: str, data_dir: Path, store: FailureMemoryStore, directory: Path):
    db = directory / f"{case['case_id']}-{group}.sqlite"
    applied_before = store.db.execute(
        "SELECT count(*) FROM failure_memory_usage WHERE applied=1"
    ).fetchone()[0]
    os.environ["SERVICEPILOT_FAILURE_MEMORY_ROUTING"] = "1" if group == "B2" else "0"
    agent = build_agent(
        data_dir,
        online=True,
        commerce_db=db,
        failure_memory=store if group == "B2" else None,
    )
    session_id = None
    turns = []
    for message in case["messages"]:
        response = agent.chat(user_id="SYNTHETIC-USER", message=message, session_id=session_id)
        session_id = response.session.session_id
        turns.append(
            {
                "message": message,
                "reply": response.message,
                "intent": response.session.intent.value,
                "pending_action": response.session.pending_action,
                "tools": [e.name for e in response.events if e.event_type == "tool"],
            }
        )
    applied_after = store.db.execute(
        "SELECT count(*) FROM failure_memory_usage WHERE applied=1"
    ).fetchone()[0]
    injections = applied_after - applied_before
    if injections != (len(case["messages"]) if group == "B2" else 0):
        raise AssertionError(
            f"unexpected memory injections: {case['case_id']} {group} {injections}"
        )
    counts = database_counts(db)
    passed, failures = judge(case, turns, counts)
    return {
        "case_id": case["case_id"],
        "task": case["task"],
        "group": group,
        "passed": passed,
        "failures": failures,
        "memory_injections": injections,
        "turns": turns,
        "business_state": counts,
    }


def summarize(rows: list[dict]) -> dict:
    pairs = {}
    for row in rows:
        pairs.setdefault(row["case_id"], {})[row["group"]] = row
    if any(set(pair) != {"B0", "B2"} for pair in pairs.values()):
        raise AssertionError("missing paired result")
    slices = {}
    for task in sorted({pair["B0"]["task"] for pair in pairs.values()}):
        subset = [pair for pair in pairs.values() if pair["B0"]["task"] == task]
        slices[task] = {
            "n": len(subset),
            "B0_passed": sum(pair["B0"]["passed"] for pair in subset),
            "B2_passed": sum(pair["B2"]["passed"] for pair in subset),
            "B2_better": sum(not pair["B0"]["passed"] and pair["B2"]["passed"] for pair in subset),
            "B2_worse": sum(pair["B0"]["passed"] and not pair["B2"]["passed"] for pair in subset),
        }
    return {
        "n": len(pairs),
        "B0_passed": sum(pair["B0"]["passed"] for pair in pairs.values()),
        "B2_passed": sum(pair["B2"]["passed"] for pair in pairs.values()),
        "slices": slices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-online", action="store_true")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "evaluations/failure_memory_task_cases.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evaluations/results/failure_memory_task_eval.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.allow_online:
        parser.error("pass --allow-online to make paid model requests")
    settings = load_local_settings(args.env_file)
    os.environ["SERVICEPILOT_EMBEDDING_MODE"] = "hashing"
    os.environ["SERVICEPILOT_INTENT_MODEL"] = "deepseek-flash"
    os.environ["SERVICEPILOT_DYNAMIC_PLANNING"] = "0"
    os.environ["SERVICEPILOT_FAILURE_MEMORY"] = "0"
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    baseline = ResponsesIntentModel(UrllibJsonTransport(settings), "deepseek-flash")
    training = baseline.classify_intent(TRAINING_MESSAGE)
    store = FailureMemoryStore(":memory:")
    record = seed_from_observed_failure(store, training["intent"])
    ordered = list(cases)
    random.Random(args.seed).shuffle(ordered)
    rows = []
    with tempfile.TemporaryDirectory(prefix="servicepilot-task-eval-") as tmp:
        directory = Path(tmp)
        data_dir = directory / "data"
        data_dir.mkdir()
        synthetic_data(data_dir)
        for index, case in enumerate(ordered):
            groups = ("B0", "B2") if index % 2 == 0 else ("B2", "B0")
            for group in groups:
                rows.append(run_one(case, group, data_dir, store, directory))
            print(f"completed pair {index + 1}/{len(ordered)}", flush=True)
    report = {
        "method": "paired online full ServicePilot workflow and synthetic commerce ledger",
        "model_id": "deepseek-flash",
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "training": {
            "message": TRAINING_MESSAGE,
            "expected": "unknown",
            "predicted": training["intent"],
            "memory_id": record.failure_id,
            "lesson": record.reusable_lesson,
        },
        "summary": summarize(rows),
        "runs": rows,
        "limits": (
            "Synthetic tasks; clarify cases score correct non-action, "
            "not a completed refund or sale."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
