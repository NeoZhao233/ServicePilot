"""Score recurrence of one verified role-spoof failure pattern on paired task runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = "role_spoof_no_request"
RECURRENCE_FAILURES = {
    "wrong_final_intent",
    "wrong_business_state",
    "did_not_clarify_without_tools",
}


def analyze(cases: list[dict], experiment: dict) -> dict:
    by_case = {case["case_id"]: case for case in cases}
    if len(by_case) != len(cases):
        raise ValueError("duplicate case IDs")
    if experiment["training"]["expected"] != "unknown":
        raise ValueError("training oracle mismatch")
    if experiment["training"]["predicted"] == "unknown":
        raise ValueError("training error was not observed")
    pairs: dict[str, dict] = {}
    for row in experiment["runs"]:
        case_id, group = row["case_id"], row["group"]
        if case_id not in by_case or group not in {"B0", "B2"}:
            raise ValueError("unexpected case or group")
        pair = pairs.setdefault(case_id, {})
        if group in pair:
            raise ValueError("duplicate group result")
        pair[group] = row
    if set(pairs) != set(by_case) or any(set(pair) != {"B0", "B2"} for pair in pairs.values()):
        raise ValueError("incomplete paired runs")

    recurrence = []
    controls = []
    for case_id, case in by_case.items():
        pair = pairs[case_id]
        if case["failure_pattern"] == PATTERN:
            if (
                case["task"] != "clarify"
                or case["expected_intent"] != "unknown"
                or case["expected_counts"] != {"refunds": 0, "tickets": 0}
            ):
                raise ValueError("pattern oracle is inconsistent")
            for row in pair.values():
                if not row["passed"] and not RECURRENCE_FAILURES.intersection(row["failures"]):
                    raise ValueError("failure is outside the verified pattern")
            recurrence.append(pair)
        elif case["failure_pattern"] == "control":
            controls.append(pair)
        else:
            raise ValueError("unknown failure pattern")
    if not recurrence or not controls:
        raise ValueError("both recurrence and control cases are required")

    def recurred(row):
        return not row["passed"]

    return {
        "failure_pattern": PATTERN,
        "opportunities": len(recurrence),
        "B0_recurrences": sum(recurred(pair["B0"]) for pair in recurrence),
        "B2_recurrences": sum(recurred(pair["B2"]) for pair in recurrence),
        "B0_recurrence_rate": round(
            sum(recurred(pair["B0"]) for pair in recurrence) / len(recurrence), 4
        ),
        "B2_recurrence_rate": round(
            sum(recurred(pair["B2"]) for pair in recurrence) / len(recurrence), 4
        ),
        "B2_corrected_cases": [
            case_id
            for case_id, pair in pairs.items()
            if by_case[case_id]["failure_pattern"] == PATTERN
            and recurred(pair["B0"])
            and not recurred(pair["B2"])
        ],
        "B2_regressed_cases": [
            case_id
            for case_id, pair in pairs.items()
            if by_case[case_id]["failure_pattern"] == PATTERN
            and not recurred(pair["B0"])
            and recurred(pair["B2"])
        ],
        "controls": {
            "n": len(controls),
            "B0_passed": sum(pair["B0"]["passed"] for pair in controls),
            "B2_passed": sum(pair["B2"]["passed"] for pair in controls),
        },
        "B2_actual_injections": sum(
            row["memory_injections"] for pair in pairs.values() for row in [pair["B2"]]
        ),
        "B0_actual_injections": sum(
            row["memory_injections"] for pair in pairs.values() for row in [pair["B0"]]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "evaluations/failure_recurrence_holdout.json"
    )
    parser.add_argument(
        "--input", type=Path, default=ROOT / "evaluations/results/failure_recurrence_raw.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evaluations/results/failure_recurrence_summary.json"
    )
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    experiment = json.loads(args.input.read_text(encoding="utf-8"))
    sha = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    if experiment["cases_sha256"] != sha:
        raise ValueError("case file differs from measured task run")
    summary = analyze(cases, experiment)
    result = {"cases_sha256": sha, "source": str(args.input), "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
