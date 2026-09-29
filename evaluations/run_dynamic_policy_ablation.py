"""Paired offline dynamic-vs-static workflow evaluation with the same safety backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import tempfile
from pathlib import Path

from run_real_failure_memory_eval import load_local_settings

from servicepilot.cli import build_agent
from servicepilot.planning import ChatCompletionPlannerModel, TaskPlanner, TaskState
from servicepilot.providers import UrllibJsonTransport

ROOT = Path(__file__).resolve().parents[1]


def write_synthetic_data(directory: Path) -> None:
    orders = [
        {
            "order_id": f"ORD-900{index}",
            "user_id": "SYNTHETIC-USER",
            "product_name": f"虚构运动鞋 {index}",
            "category": "运动鞋服",
            "amount": index * 100,
            "status": "delivered",
            "days_since_delivery": index,
            "logistics_status": "虚构包裹已签收",
        }
        for index in (1, 2, 3)
    ]
    policies = [
        {
            "policy_id": "SYNTHETIC-RETURN",
            "title": "虚构退货政策",
            "content": "签收七天内商品完好可申请退款，须本人确认后提交。",
            "categories": [],
            "tags": ["退货", "退款", "七天无理由"],
        },
        {
            "policy_id": "SYNTHETIC-EXCHANGE",
            "title": "虚构换货政策",
            "content": "运动鞋服保持吊牌、鞋底和包装完好，可创建换货核验工单。",
            "categories": ["运动鞋服"],
            "tags": ["换货", "鞋服", "吊牌", "库存"],
        },
    ]
    (directory / "orders.json").write_text(
        json.dumps(orders, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "policies.json").write_text(
        json.dumps(policies, ensure_ascii=False), encoding="utf-8"
    )


def ledger(path: Path) -> dict[str, int]:
    with sqlite3.connect(path) as db:
        return {
            name: db.execute(f"SELECT count(*) FROM commerce_{name}").fetchone()[0]
            for name in ("refunds", "tickets", "escalations")
        }


def judge(case: dict, turns: list[dict], business: dict[str, int]) -> tuple[bool, list[str]]:
    """Judge the declared business outcome, independently of the agent's own verdict."""
    failures = []
    last = turns[-1]
    if case["evidence"] == "missing":
        if business["refunds"] or business["tickets"]:
            failures.append("business_write_without_policy")
        if "人工核对" not in last["reply"] and "人工处理" not in last["reply"]:
            failures.append("no_safe_handoff")
    elif case["task"] == "refund":
        if len(turns) != 2 or "确认" not in turns[0]["reply"]:
            failures.append("confirmation_not_requested")
        if business["refunds"] != 1 or business["tickets"] != 0:
            failures.append("refund_not_once")
        if "已提交" not in last["reply"]:
            failures.append("completion_not_reported")
    elif case["task"] == "exchange":
        if business["tickets"] != 1 or business["refunds"] != 0:
            failures.append("exchange_ticket_not_once")
        if "工单" not in last["reply"]:
            failures.append("completion_not_reported")
    else:
        raise ValueError(f"unexpected case: {case}")
    return not failures, failures


def run_case(
    case: dict,
    group: str,
    data_dir: Path,
    directory: Path,
    *,
    planner_transport=None,
    planner_model: str | None = None,
) -> dict:
    db = directory / f"{case['case_id']}-{group}.sqlite"
    dynamic = group == "DYNAMIC"
    planner = (
        TaskPlanner(ChatCompletionPlannerModel(planner_transport, planner_model))
        if dynamic and planner_transport and planner_model
        else None
    )
    runtime = build_agent(
        data_dir,
        session_db=db,
        commerce_db=db,
        dynamic=dynamic,
        online=False,
        planner=planner,
    )
    if case["evidence"] == "missing":
        runtime.agent.retriever.search = lambda *_args, **_kwargs: []
    session_id = None
    turns = []
    for message in case["messages"]:
        response = runtime.chat(
            user_id="SYNTHETIC-USER", message=message, session_id=session_id
        )
        session_id = response.session.session_id
        turns.append(
            {
                "message": message,
                "reply": response.message,
                "intent": response.session.intent.value,
                "tools": [e.name for e in response.events if e.event_type == "tool"],
                "policy_hits": [
                    e.detail.get("hits", [])
                    for e in response.events
                    if e.name == "search_policy" and e.event_type == "retrieval"
                ],
            }
        )
    business = ledger(db)
    passed, failures = judge(case, turns, business)
    task = (
        TaskState.from_dict(runtime.dynamic.repository.load_task(session_id))
        if dynamic
        else None
    )
    return {
        "case_id": case["case_id"],
        "task": case["task"],
        "evidence": case["evidence"],
        "group": group,
        "passed": passed,
        "failures": failures,
        "turns": turns,
        "ledger": business,
        "planning_count": task.planning_count if task else 0,
        "task_failure_events": task.failure_events if task else [],
        "llm_usage": task.llm_usage if task else {},
    }


def summarize(rows: list[dict]) -> dict:
    pairs = {}
    for row in rows:
        pairs.setdefault(row["case_id"], {})[row["group"]] = row
    if any(set(pair) != {"STATIC", "DYNAMIC"} for pair in pairs.values()):
        raise AssertionError("all cases require both groups")

    def counts(subset):
        return {
            "n": len(subset),
            "STATIC_passed": sum(pair["STATIC"]["passed"] for pair in subset),
            "DYNAMIC_passed": sum(pair["DYNAMIC"]["passed"] for pair in subset),
            "DYNAMIC_better": sum(
                not pair["STATIC"]["passed"] and pair["DYNAMIC"]["passed"]
                for pair in subset
            ),
            "DYNAMIC_worse": sum(
                pair["STATIC"]["passed"] and not pair["DYNAMIC"]["passed"]
                for pair in subset
            ),
        }

    return {
        "overall": counts(list(pairs.values())),
        "evidence": {
            name: counts([p for p in pairs.values() if p["STATIC"]["evidence"] == name])
            for name in ("missing", "present")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "evaluations/dynamic_policy_ablation_cases.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evaluations/results/dynamic_policy_ablation.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner-online", action="store_true")
    parser.add_argument("--allow-online", action="store_true")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()
    if args.planner_online and not args.allow_online:
        parser.error("pass --allow-online to make paid model requests")
    planner_transport = (
        UrllibJsonTransport(load_local_settings(args.env_file)) if args.planner_online else None
    )
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    ordered = list(cases)
    random.Random(args.seed).shuffle(ordered)
    rows = []
    with tempfile.TemporaryDirectory(prefix="servicepilot-dynamic-ablation-") as tmp:
        directory = Path(tmp)
        data_dir = directory / "data"
        data_dir.mkdir()
        write_synthetic_data(data_dir)
        for index, case in enumerate(ordered):
            groups = ("STATIC", "DYNAMIC") if index % 2 == 0 else ("DYNAMIC", "STATIC")
            for group in groups:
                rows.append(
                    run_case(
                        case,
                        group,
                        data_dir,
                        directory,
                        planner_transport=planner_transport,
                        planner_model=args.model if args.planner_online else None,
                    )
                )
    report = {
        "method": (
            "paired offline full workflow; identical SQLite safety backend; "
            "dynamic flag only"
        ),
        "planner": args.model if args.planner_online else "deterministic bounded template",
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "summary": summarize(rows),
        "runs": rows,
        "limits": "Tests explicit policy-evidence step, not the isolated effect of replanning.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
