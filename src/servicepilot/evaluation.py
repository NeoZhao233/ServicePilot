from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .cli import build_agent
from .providers import OnlineModelSettings, ResponsesDialogueJudge, UrllibJsonTransport


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    final_message: str
    called_tools: tuple[str, ...]
    failures: tuple[str, ...]
    user_messages: tuple[str, ...]
    assistant_messages: tuple[str, ...]
    citations: tuple[str, ...]
    tool_results: tuple[dict[str, object], ...]
    slices: tuple[str, ...]
    difficulty: str
    turns: tuple[dict[str, object], ...]


def _case_slices(case: dict[str, object]) -> tuple[str, ...]:
    raw = case.get("slice", "unspecified")
    if isinstance(raw, list):
        return tuple(str(item) for item in raw) or ("unspecified",)
    return (str(raw),)


def _check_expectations(
    expected: dict[str, object],
    *,
    message: str,
    called_tools: list[str],
    event_names: list[str],
    citations: list[str],
    intent: str,
    pending_action: str | None,
    prefix: str = "",
) -> list[str]:
    failures: list[str] = []
    label = f"{prefix}：" if prefix else ""
    for text in expected.get("message_contains", []):
        if str(text) not in message:
            failures.append(f"{label}回复缺少：{text}")
    contains_any = [str(text) for text in expected.get("message_contains_any", [])]
    if contains_any and not any(text in message for text in contains_any):
        failures.append(f"{label}回复未包含任一候选内容：{' / '.join(contains_any)}")
    for text in expected.get("message_not_contains", []):
        if str(text) in message:
            failures.append(f"{label}回复包含禁用内容：{text}")
    for tool in expected.get("must_call", []):
        if str(tool) not in called_tools:
            failures.append(f"{label}未调用必要工具：{tool}")
    for tool in expected.get("must_not_call", []):
        if str(tool) in called_tools:
            failures.append(f"{label}错误调用工具：{tool}")
    for event in expected.get("must_have_events", []):
        if str(event) not in event_names:
            failures.append(f"{label}缺少必要事件：{event}")
    for event in expected.get("must_not_have_events", []):
        if str(event) in event_names:
            failures.append(f"{label}出现禁用事件：{event}")
    if "intent" in expected and intent != str(expected["intent"]):
        failures.append(f"{label}意图应为 {expected['intent']}，实际为 {intent}")
    if "pending_action" in expected and pending_action != expected["pending_action"]:
        failures.append(
            f"{label}pending_action 应为 {expected['pending_action']}，实际为 {pending_action}"
        )
    if expected.get("citations_any") and not citations:
        failures.append(f"{label}缺少引用")
    for citation in expected.get("citations_all", []):
        if str(citation) not in citations:
            failures.append(f"{label}缺少引用：{citation}")
    return failures


def run_case(case: dict[str, object], *, data_dir: Path, online: bool = False) -> CaseResult:
    agent = build_agent(data_dir, online=online)
    session_id: str | None = None
    called_tools: list[str] = []
    final_message = ""
    assistant_messages: list[str] = []
    citations: list[str] = []
    tool_results: list[dict[str, object]] = []
    turns: list[dict[str, object]] = []
    failures: list[str] = []

    turn_expectations = case.get("turn_expectations", [])
    for turn_index, message in enumerate(case["messages"]):
        response = agent.chat(
            user_id=str(case["user_id"]),
            message=str(message),
            session_id=session_id,
        )
        session_id = response.session.session_id
        final_message = response.message
        assistant_messages.append(response.message)
        turn_tools = [event.name for event in response.events if event.event_type == "tool"]
        event_names = [event.name for event in response.events]
        called_tools.extend(turn_tools)
        citations.extend(response.citations)
        tool_results.extend(
            {"name": event.name, "detail": event.detail}
            for event in response.events
            if event.event_type == "tool"
        )
        turn = {
            "message": response.message,
            "called_tools": turn_tools,
            "events": event_names,
            "citations": list(response.citations),
            "intent": response.session.intent.value,
            "pending_action": response.session.pending_action,
        }
        turns.append(turn)
        if turn_index < len(turn_expectations):
            failures.extend(
                _check_expectations(
                    turn_expectations[turn_index],
                    message=response.message,
                    called_tools=turn_tools,
                    event_names=event_names,
                    citations=list(response.citations),
                    intent=response.session.intent.value,
                    pending_action=response.session.pending_action,
                    prefix=f"第 {turn_index + 1} 轮",
                )
            )

    expected = case.get("expected", {})
    final_turn = turns[-1] if turns else {
        "events": [], "citations": [], "intent": "unknown", "pending_action": None
    }
    failures.extend(
        _check_expectations(
            expected,
            message=final_message,
            called_tools=called_tools,
            event_names=[str(item) for turn in turns for item in turn["events"]],
            citations=citations,
            intent=str(final_turn["intent"]),
            pending_action=final_turn["pending_action"],
            prefix="最终",
        )
    )
    tool_counts = Counter(called_tools)
    for tool, expected_count in expected.get("tool_call_counts", {}).items():
        if tool_counts[str(tool)] != int(expected_count):
            failures.append(
                f"工具 {tool} 应调用 {expected_count} 次，实际 {tool_counts[str(tool)]} 次"
            )
    if "max_tool_calls" in expected and len(called_tools) > int(expected["max_tool_calls"]):
        failures.append(
            f"工具总调用应不超过 {expected['max_tool_calls']} 次，实际 {len(called_tools)} 次"
        )

    return CaseResult(
        case_id=str(case["case_id"]),
        passed=not failures,
        final_message=final_message,
        called_tools=tuple(called_tools),
        failures=tuple(failures),
        user_messages=tuple(str(message) for message in case["messages"]),
        assistant_messages=tuple(assistant_messages),
        citations=tuple(citations),
        tool_results=tuple(tool_results),
        slices=_case_slices(case),
        difficulty=str(case.get("difficulty", "unspecified")),
        turns=tuple(turns),
    )


def evaluate(
    cases_path: Path, *, data_dir: Path, online: bool = False
) -> list[CaseResult]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    return [run_case(case, data_dir=data_dir, online=online) for case in cases]


def trajectory_fingerprint(result: CaseResult) -> str:
    payload = {
        "user_messages": result.user_messages,
        "assistant_messages": result.assistant_messages,
        "called_tools": result.called_tools,
        "citations": result.citations,
        "tool_results": result.tool_results,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def grouped_pass_rates(results: list[CaseResult], field: str) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        values = result.slices if field == "slices" else (result.difficulty,)
        for value in values:
            grouped[value].append(result)
    return {
        name: {
            "count": len(items),
            "passed": sum(item.passed for item in items),
            "pass_rate": round(sum(item.passed for item in items) / len(items), 4),
        }
        for name, items in sorted(grouped.items())
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate ServicePilot business behavior")
    parser.add_argument("--cases", type=Path, default=project_root / "evaluations" / "cases.json")
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--output", type=Path, help="Write raw trajectories and metrics as JSON")
    parser.add_argument(
        "--online", action="store_true", help="Use the configured LLM intent router"
    )
    parser.add_argument(
        "--judge-online",
        action="store_true",
        help="Score every dialogue using the configured intent model as LLM-as-Judge",
    )
    args = parser.parse_args()

    results = evaluate(args.cases, data_dir=args.data_dir, online=args.online)
    passed = sum(result.passed for result in results)
    report: dict[str, object] = {
        "evaluation_mode": "online" if args.online else "offline",
        "case_count": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "slice_metrics": grouped_pass_rates(results, "slices"),
        "difficulty_metrics": grouped_pass_rates(results, "difficulty"),
        "rule_rubric": {
            "checks": [
                "required response content",
                "required tool calls",
                "forbidden tool calls",
                "turn-level state and event assertions",
                "tool call counts and citation requirements",
            ]
        },
        "cases": [asdict(result) for result in results],
    }
    if args.judge_online:
        settings = OnlineModelSettings.from_env()
        judge = ResponsesDialogueJudge(
            UrllibJsonTransport(settings),
            settings.intent_model,
        )
        existing: dict[str, dict[str, object]] = {}
        if args.output and args.output.exists():
            try:
                saved = json.loads(args.output.read_text(encoding="utf-8"))
                for item in saved.get("llm_as_judge", {}).get("judgements", []):
                    if item.get("trajectory_fingerprint"):
                        existing[str(item["case_id"])] = item
            except (json.JSONDecodeError, OSError):
                existing = {}
        judge_report: dict[str, object] = {
            "model": settings.intent_model,
            "rubric": ["correctness", "groundedness", "safety", "coherence"],
            "judgements": [],
        }
        report["llm_as_judge"] = judge_report
        judgements: list[dict[str, object]] = []
        for result in results:
            fingerprint = trajectory_fingerprint(result)
            cached = existing.get(result.case_id)
            if cached and cached.get("trajectory_fingerprint") == fingerprint:
                judgement = cached
            else:
                judgement = {
                    "case_id": result.case_id,
                    "trajectory_fingerprint": fingerprint,
                    **judge.judge(
                        user_messages=list(result.user_messages),
                        assistant_messages=list(result.assistant_messages),
                        called_tools=list(result.called_tools),
                        citations=list(result.citations),
                        tool_results=list(result.tool_results),
                    ),
                }
            judgements.append(judgement)
            judge_report["judgements"] = judgements
            if args.output:
                write_report(args.output, report)
    if args.output:
        write_report(args.output, report)
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        print(f"[{marker}] {result.case_id}: {result.final_message}")
        for failure in result.failures:
            print(f"  - {failure}")
    print(f"\n通过率：{passed}/{len(results)} ({passed / len(results):.1%})")


if __name__ == "__main__":
    main()
