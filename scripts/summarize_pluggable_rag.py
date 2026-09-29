"""Aggregate saved live observations without API calls or invented missing metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from servicepilot.rag_evaluation import distribution, evaluate

ROOT = Path(__file__).resolve().parents[1]


def summarize(artifacts, frozen, complex_cases, baseline_rows=()):
    if not artifacts:
        raise ValueError("At least one observed run is required")
    for key in ("cases_sha256", "policies_sha256"):
        if len({a[key] for a in artifacts}) != 1:
            raise ValueError("Incompatible evaluation inputs")
    by_key = {(r["group"], r["case_id"]): r for r in baseline_rows}
    for artifact in artifacts:
        for row in artifact["runs"]:
            key = row["group"], row["case_id"]
            if key in by_key and by_key[key] != row:
                raise ValueError("Conflicting repeated observation")
            by_key[key] = row
    rows = list(by_key.values())
    frozen_ids = {c["case_id"] for c in frozen}
    complex_ids = {c["case_id"] for c in complex_cases}
    common = frozen_ids.copy()
    for group in ("B0", "B1", "B2", "B3"):
        common &= {r["case_id"] for r in rows if r["group"] == group}
    cohorts = []
    for index, artifact in enumerate(artifacts):
        own_rows = [r for r in artifact["runs"] if r["group"] != "B0"]
        cohort_rows = [r for r in rows if r["group"] == "B0"] + own_rows
        shared = frozen_ids.copy()
        for group in ("B0", "B1", "B2", "B3"):
            shared &= {r["case_id"] for r in cohort_rows if r["group"] == group}
        cohorts.append(
            {
                "artifact_index": index,
                "config": artifact.get("evaluator_config", {"max_output_tokens": 4096}),
                "common_frozen_case_ids": sorted(shared),
                "common_frozen": evaluate(
                    [c for c in frozen if c["case_id"] in shared],
                    [r for r in cohort_rows if r["case_id"] in shared],
                ),
                "complex": evaluate(
                    complex_cases, [r for r in own_rows if r["case_id"] in complex_ids]
                ),
            }
        )
    version_rows = {}
    version_configs = {}
    for artifact in artifacts:
        config = artifact.get("evaluator_config", {"max_output_tokens": 4096})
        key = json.dumps(config, sort_keys=True)
        version_configs[key] = config
        version_rows.setdefault(key, []).extend(r for r in artifact["runs"] if r["group"] != "B0")
    versions = []
    for key, own in version_rows.items():
        cohort_rows = [r for r in rows if r["group"] == "B0"] + own
        shared = frozen_ids.copy()
        for group in ("B0", "B1", "B2", "B3"):
            shared &= {r["case_id"] for r in cohort_rows if r["group"] == group}
        complex_metrics = evaluate(complex_cases, [r for r in own if r["case_id"] in complex_ids])
        for group, metrics in complex_metrics.items():
            labelled = [c for c in complex_cases if c.get("expected_sufficiency")]
            observed = {r["case_id"]: r for r in own if r["group"] == group}
            correct = sum(
                observed.get(c["case_id"], {}).get("sufficiency") == c["expected_sufficiency"]
                for c in labelled
            )
            metrics["labelled_case_completion_accuracy"] = (
                correct / len(labelled) if labelled and metrics["sufficiency_observed"] else None
            )
        versions.append(
            {
                "config": version_configs[key],
                "common_frozen_case_ids": sorted(shared),
                "common_frozen": evaluate(
                    [c for c in frozen if c["case_id"] in shared],
                    [r for r in cohort_rows if r["case_id"] in shared],
                ),
                "complex": complex_metrics,
            }
        )
    calls = [call for a in artifacts for call in a["physical_calls"]]
    attempts = sum(a["attempts"] for a in artifacts)
    known_usage = {}
    total_usage = {}
    for field in ("input_tokens", "output_tokens", "cost"):
        values = [(c.get("usage") or {}).get(field) for c in calls]
        known = [v for v in values if type(v) in (int, float) and v >= 0]
        known_usage[field] = sum(known) if known else None
        total_usage[field] = sum(known) if len(known) == len(calls) == attempts else None
    failures = [a["failure"] for a in artifacts if a.get("failure")]
    return {
        "status": (
            "in_progress"
            if any(a["status"] == "running" for a in artifacts)
            else "partial"
            if failures or any(a.get("excluded_cases") for a in artifacts)
            else "completed"
        ),
        "cohorts": cohorts,
        "provider_versions": versions,
        "artifact_statuses": [a["status"] for a in artifacts],
        "failures": failures,
        "frozen": evaluate(frozen, [r for r in rows if r["case_id"] in frozen_ids]),
        "common_frozen_case_ids": sorted(common),
        "common_frozen": evaluate(
            [c for c in frozen if c["case_id"] in common],
            [r for r in rows if r["case_id"] in common],
        ),
        "complex": evaluate(complex_cases, [r for r in rows if r["case_id"] in complex_ids]),
        "physical": {
            "attempts": attempts,
            "responses": len(calls),
            "usage": total_usage,
            "reported_usage_subtotal": known_usage,
            "response_latency_ms": distribution([c["elapsed_ms"] for c in calls]),
        },
        "method": "Shared prefixes are charged once physically, "
        "counted per logical path in groups. "
        "Group latency is reconstructed RAG latency, not end-to-end chat latency. "
        "Use cohort common_frozen for equal-coverage comparison within one prompt/config. "
        "Pooled frozen metrics mix pilot and revised configurations. "
        "Unobserved answer accuracy, promises and API cost stay null.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        [json.loads(path.read_text()) for path in args.run],
        json.loads((ROOT / "evaluations/retrieval_challenge.json").read_text()),
        json.loads((ROOT / "evaluations/corrective_rag_cases.json").read_text()),
        json.loads((ROOT / "evaluations/results/jev_corrective/observed_runs.json").read_text()),
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
