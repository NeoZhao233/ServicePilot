"""Offline aggregation of B0-B3 observations; missing measurements stay unknown."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def mean(values):
    return statistics.mean(values) if values else None


def distribution(values):
    values = sorted(values)
    return {
        "n": len(values),
        "mean": mean(values),
        "p50": statistics.median(values) if values else None,
        "p95": values[math.ceil(len(values) * 0.95) - 1] if values else None,
    }


def evaluate(cases, runs):
    by_id = {c["case_id"]: c for c in cases}
    if len(by_id) != len(cases):
        raise ValueError("Duplicate case IDs")
    report = {}
    for group in ("B0", "B1", "B2", "B3", "B4"):
        rows = [r for r in runs if r["group"] == group]
        if len({r["case_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate group/case observations")
        recall, reciprocal, critical, incorrect, sufficiency = [], [], [], [], []
        correctness, promises, rounds, calls, latencies = [], [], [], [], []
        token_values = {"input_tokens": [], "output_tokens": [], "api_cost": []}
        failures = []
        for row in rows:
            case = by_id[row["case_id"]]
            ranked = row.get("retrieved_ids", [])[:5]
            if len(ranked) != len(set(ranked)):
                raise ValueError("Duplicate returned policy IDs")
            expected = set(case["relevant_policy_ids"])
            if expected:
                recall.append(len(expected & set(ranked)) / len(expected))
                reciprocal.append(
                    next((1 / (i + 1) for i, pid in enumerate(ranked) if pid in expected), 0)
                )
            protected = set(case.get("critical_policy_ids", []))
            if protected:
                critical.append(len(protected & set(ranked)) / len(protected))
            # Only explicitly labelled negatives; non-gold is not automatically irrelevant.
            negatives = set(case.get("irrelevant_policy_ids", [])) & set(
                row.get("candidate_ids", [])
            )
            removed = set(row.get("filtered_ids", []))
            if negatives:
                incorrect.append(len(negatives & removed) / len(negatives))
            false_filtered = expected & removed
            if false_filtered:
                failures.append({"case_id": row["case_id"], "policy_ids": sorted(false_filtered)})
            if row.get("sufficiency") is not None and case.get("expected_sufficiency") is not None:
                sufficiency.append(row["sufficiency"] == case["expected_sufficiency"])
            # Answer-level observations require explicit adjudication, never inferred from recall.
            if row.get("answer_correct") is not None:
                if type(row["answer_correct"]) is not bool:
                    raise ValueError("Answer correctness must be an adjudicated boolean")
                correctness.append(row["answer_correct"])
            for field, values in [
                ("unsupported_promises", promises),
                ("rounds", rounds),
                ("deepseek_calls", calls),
                ("latency_ms", latencies),
            ]:
                value = row.get(field)
                if value is not None:
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError("Invalid measured value")
                    values.append(value)
            for field, values in token_values.items():
                value = row.get(field)
                if value is not None and (
                    type(value) not in (int, float) or not math.isfinite(value) or value < 0
                ):
                    raise ValueError("Invalid usage")
                values.append(value)
        report[group] = {
            "status": "not_run" if not rows else "observed",
            "source": sorted({r.get("source", "unspecified") for r in rows}),
            "coverage": {"observed": len(rows), "total": len(cases)},
            "recall@5": mean(recall),
            "mrr": mean(reciprocal),
            "critical_retention": mean(critical),
            "irrelevant_filter_rate": mean(incorrect),
            "sufficiency_accuracy": mean(sufficiency),
            "sufficiency_observed": len(sufficiency),
            "answer_accuracy": mean(correctness),
            "answers_adjudicated": len(correctness),
            "unsupported_promises": sum(promises) if len(promises) == len(rows) and rows else None,
            "rounds": distribution(rounds),
            "deepseek_calls": distribution(calls),
            "latency_ms": distribution(latencies),
            "false_filtered_cases": failures,
            "usage": {
                field: sum(v) if v and all(x is not None for x in v) else None
                for field, v in token_values.items()
            },
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(json.loads(args.cases.read_text()), json.loads(args.runs.read_text()))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
