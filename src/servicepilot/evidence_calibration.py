"""Recompute calibration gates from scored observations, not a bare enable/threshold flag."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .jev import probability


@dataclass(frozen=True)
class Calibration:
    threshold: float
    confidence_floor: float
    artifact_id: str
    protected_ids: tuple[str, ...] = ()


def threshold_sweep(rows):
    if not rows or not any(r["relevant"] for r in rows) or not any(not r["relevant"] for r in rows):
        raise ValueError("Calibration requires both relevant and irrelevant evidence")
    for row in rows:
        probability(row["score"])
        if type(row["relevant"]) is not bool or type(row["critical"]) is not bool:
            raise ValueError("Labels must be booleans")
        if row["critical"] and not row["relevant"]:
            raise ValueError("Critical evidence must be relevant")
    reports = []
    for threshold in sorted({0.0, *(r["score"] for r in rows)}):
        relevant = [r for r in rows if r["relevant"]]
        critical = [r for r in rows if r["critical"]]
        irrelevant = [r for r in rows if not r["relevant"]]
        reports.append(
            {
                "threshold": threshold,
                "relevant_retention": sum(r["score"] >= threshold for r in relevant)
                / len(relevant),
                "critical_retention": (
                    sum(r["score"] >= threshold for r in critical) / len(critical)
                    if critical
                    else None
                ),
                "irrelevant_filter_rate": sum(r["score"] < threshold for r in irrelevant)
                / len(irrelevant),
                "false_filtered_ids": [r["policy_id"] for r in relevant if r["score"] < threshold],
            }
        )
    return reports


def load_calibration(
    path,
    *,
    corpus_hash,
    model,
    ranker_version,
    frozen_ids=(),
    corrective=False,
    scoring_mode="listwise",
):
    report = json.loads(Path(path).read_text())
    if report.get("source") != "jev_live" or report.get("corpus_sha256") != corpus_hash:
        raise ValueError("Calibration must contain live Jev scores for the current corpus")
    if report.get("model") != model or report.get("config_version") != ranker_version:
        raise ValueError("Calibration model/configuration mismatch")
    if report.get("scoring_mode") != scoring_mode or report.get("library_version") != "0.1.2":
        raise ValueError("Calibration scoring mode/library mismatch")
    rows = report["observations"]
    case_ids = {r["case_id"] for r in rows}
    if len(case_ids) < 20 or case_ids.intersection(frozen_ids):
        raise ValueError("Need >=20 independent calibration cases, disjoint from frozen test cases")
    keys = [(r["case_id"], r["policy_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate calibration observations")
    if len({r["policy_id"] for r in rows if r["critical"]}) < 3:
        raise ValueError("Calibration lacks coverage of rare critical policies")
    accepted = [
        r
        for r in threshold_sweep(rows)
        if r["critical_retention"] == 1 and r["relevant_retention"] >= 0.98
    ]
    threshold = max(r["threshold"] for r in accepted)
    confidence = probability(report.get("confidence_floor", 0.9))
    if confidence < 0.9:
        raise ValueError("Confidence floor below release requirement")
    if corrective:
        decisions = report.get("sufficiency_validation", [])
        decision_ids = [r["case_id"] for r in decisions]
        if (
            len(set(decision_ids)) < 20
            or len(decision_ids) != len(set(decision_ids))
            or set(decision_ids).intersection(case_ids | set(frozen_ids))
        ):
            raise ValueError("Need separate sufficiency validation cases")
        states = {"SUFFICIENT", "INSUFFICIENT", "CONFLICTING", "UNCERTAIN"}
        if {r["expected"] for r in decisions} != states:
            raise ValueError("Validation must cover all four sufficiency states")
        for row in decisions:
            if row["predicted"] not in states:
                raise ValueError("Invalid predicted sufficiency")
        accuracy = sum(r["expected"] == r["predicted"] for r in decisions) / len(decisions)
        if accuracy < 0.9 or any(
            r["predicted"] == "SUFFICIENT" and r["expected"] != "SUFFICIENT" for r in decisions
        ):
            raise ValueError("Sufficiency validation failed")
        if report.get("sufficiency_config_version") != "sufficiency-v1":
            raise ValueError("Sufficiency validation configuration mismatch")
    return Calibration(
        threshold,
        confidence,
        str(report["artifact_id"]),
        tuple(sorted({r["policy_id"] for r in rows if r["critical"]})),
    )
