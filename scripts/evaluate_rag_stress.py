"""Evaluate the frozen 100-case synthetic set without external requests.

This measures retrieval and pre-rerank candidate coverage. It does not claim
to measure DeepSeek reranking, corrective decisions, or final task completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from servicepilot.retrieval import (
    AdvancedHybridPolicyRetriever,
    BM25PolicyRetriever,
    FastEmbedTextEncoder,
    HybridPolicyRetriever,
    VectorPolicyRetriever,
    load_policy_chunks,
)

ROOT = Path(__file__).resolve().parents[1]


class CandidateTap:
    def __init__(self) -> None:
        self.all_candidates: list[str] = []

    def rerank(self, query, candidates, *, limit):
        self.all_candidates = [hit.chunk.policy_id for hit in candidates]
        return candidates[:limit]


def _metrics(cases: list[dict], predictions: dict[str, list[str]], k: int) -> dict:
    rows = []
    for case in cases:
        gold = set(case["relevant_policy_ids"])
        found = set(predictions[case["case_id"]][:k]) & gold
        rows.append(
            {
                "id": case["case_id"],
                "slice": case["slice"],
                "gold_count": len(gold),
                "found_count": len(found),
                "missing": sorted(gold - found),
                "recall": len(found) / len(gold),
                "all_evidence": len(found) == len(gold),
                "any_evidence": bool(found),
            }
        )

    def summarize(subset):
        eligible = [row for row in subset if row["gold_count"] <= k]
        return {
            "n": len(subset),
            "all_evidence_eligible_n": len(eligible),
            "macro_recall": round(statistics.mean(row["recall"] for row in subset), 4),
            "all_evidence_rate": round(statistics.mean(row["all_evidence"] for row in eligible), 4)
            if eligible
            else None,
            "any_evidence_rate": round(statistics.mean(row["any_evidence"] for row in subset), 4),
            "missing_cases": sum(not row["all_evidence"] for row in eligible),
        }

    groups = defaultdict(list)
    for row in rows:
        groups[row["slice"]].append(row)
    result = summarize(rows)
    result["by_slice"] = {name: summarize(group) for name, group in sorted(groups.items())}
    result["case_results"] = rows
    return result


def _controlled_missing(cases: list[dict], predictions: dict[str, list[str]]) -> list[dict]:
    """Freeze initial evidence with one missing gold item for later B3 tests."""
    output = []
    for case in cases:
        gold = set(case["relevant_policy_ids"])
        if len(gold) < 2:
            continue
        initial = list(predictions[case["case_id"]][:5])
        natural_missing = gold - set(initial)
        removed = None
        if not natural_missing:
            removed = next(policy_id for policy_id in reversed(initial) if policy_id in gold)
            initial.remove(removed)
        output.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "relevant_policy_ids": case["relevant_policy_ids"],
                "initial_policy_ids": initial,
                "missing_policy_ids": sorted(gold - set(initial)),
                "arm": "natural_miss" if natural_missing else "controlled_omission",
                "removed_policy_id": removed,
            }
        )
    return output


def run(*, include_semantic: bool = True) -> dict:
    cases_path = ROOT / "evaluations/rag_stress_cases.json"
    policies_path = ROOT / "data/policies.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    policies = load_policy_chunks(policies_path)
    known = {policy.policy_id for policy in policies}
    if len(cases) != 100 or len({case["case_id"] for case in cases}) != 100:
        raise ValueError("The stress set must contain 100 unique case IDs")
    if len({case["query"] for case in cases}) != 100:
        raise ValueError("The stress set must contain 100 unique queries")
    for case in cases:
        labels = case["relevant_policy_ids"]
        if (
            not labels
            or len(labels) > 5
            or len(labels) != len(set(labels))
            or not set(labels) <= known
        ):
            raise ValueError(f"Invalid gold labels: {case['case_id']}")

    lexical = BM25PolicyRetriever(policies)
    retrievers = {"bm25": lexical}
    if include_semantic:
        encoder = FastEmbedTextEncoder()
        vector = VectorPolicyRetriever(policies, encoder)
        tap = CandidateTap()
        retrievers.update(
            {
                "bge_vector": vector,
                "bge_hybrid_rrf": HybridPolicyRetriever(lexical, vector),
                "bge_rewrite_heuristic": AdvancedHybridPolicyRetriever(
                    lexical, vector, reranker=tap
                ),
            }
        )
    result = {
        "scope": (
            "synthetic offline retrieval; no DeepSeek calls; "
            "not an end-to-end completion rate"
        ),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "policies_sha256": hashlib.sha256(policies_path.read_bytes()).hexdigest(),
        "parameters": {
            "top_k": [3, 5],
            "candidate_limit": 12,
            "encoder": "BAAI/bge-small-zh-v1.5" if include_semantic else None,
        },
        "groups": {},
    }
    for name, retriever in retrievers.items():
        predictions = {}
        candidate_predictions = {}
        for case in cases:
            hits = retriever.search(case["query"], limit=5)
            predictions[case["case_id"]] = [hit.chunk.policy_id for hit in hits]
            if name == "bge_rewrite_heuristic":
                candidate_predictions[case["case_id"]] = list(tap.all_candidates)
        group = {
            "top3": _metrics(cases, predictions, 3),
            "top5": _metrics(cases, predictions, 5),
            "predictions": predictions,
        }
        if candidate_predictions:
            group["pre_rerank_candidate_pool"] = _metrics(
                cases, candidate_predictions, max(len(p) for p in candidate_predictions.values())
            )
            group["candidate_predictions"] = candidate_predictions
        result["groups"][name] = group
        if name == "bge_rewrite_heuristic":
            controlled = _controlled_missing(cases, predictions)
            result["corrective_stress_fixture"] = {
                "scope": "initial evidence fixture only; B3 recovery not run",
                "n": len(controlled),
                "natural_miss_n": sum(item["arm"] == "natural_miss" for item in controlled),
                "controlled_omission_n": sum(
                    item["arm"] == "controlled_omission" for item in controlled
                ),
                "cases": controlled,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()
    report = run(include_semantic=not args.bm25_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, group in report["groups"].items():
        print(
            name,
            "top5",
            {k: v for k, v in group["top5"].items() if k != "case_results" and k != "by_slice"},
        )


if __name__ == "__main__":
    main()
