"""Measure the unchanged offline retrieval pipeline; never calls a remote API."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from servicepilot.observability import TraceService
from servicepilot.retrieval import (
    AdvancedHybridPolicyRetriever,
    BM25PolicyRetriever,
    HashingTextEncoder,
    HybridPolicyRetriever,
    VectorPolicyRetriever,
    load_policy_chunks,
)
from servicepilot.retrieval_evaluation import evaluate_retriever

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    values = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "p50": statistics.median(values) if values else None,
        "p95": values[math.ceil(0.95 * len(values)) - 1] if values else None,
    }


class CandidateTap:
    """Record pre-rerank candidates and preserve the original no-model ordering."""

    def __init__(self):
        self.records = []

    def rerank(self, query, candidates, *, limit):
        self.records.append(
            {
                "query": query,
                "candidates": [
                    {
                        "policy_id": hit.chunk.policy_id,
                        "score": hit.score,
                        "content_sha256": hashlib.sha256(hit.chunk.content.encode()).hexdigest(),
                    }
                    for hit in candidates
                ],
            }
        )
        return candidates[:limit]


def run(repeats=3):
    cases_path = ROOT / "evaluations/retrieval_challenge.json"
    policies_path = ROOT / "data/policies.json"
    cases = json.loads(cases_path.read_text())
    chunks = load_policy_chunks(policies_path)
    if len(cases) != 24:
        raise ValueError("Frozen challenge must contain 24 cases")
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scope": "offline retrieval only, not full chat",
        "cases_sha256": digest(cases_path),
        "policies_sha256": digest(policies_path),
        "code_sha256": {
            name: digest(ROOT / "src/servicepilot" / name)
            for name in (
                "retrieval.py",
                "providers.py",
                "retrieval_evaluation.py",
                "models.py",
            )
        },
        "benchmark_sha256": digest(Path(__file__)),
        "parameters": {
            "candidate_limit": 12,
            "rank_constant": 60,
            "minimum_score": 0.01,
            "cache_ttl_seconds": 300,
            "cache_capacity": 256,
            "hashing_dimensions": 256,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
        },
        "timing": "3 independent cold-cache passes; init/indexing excluded; Trace enabled",
        "repeats": repeats,
        "groups": {},
        "b0_online": {"status": "not_run"},
        "packages": {},
    }
    for package in ("fastembed", "langgraph", "chromadb", "httpx", "python-dotenv"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    service = TraceService.create()
    for limit in (3, 5):
        for name in ("bm25", "hashing_vector", "hybrid_rrf", "hybrid_rewrite_heuristic"):
            runs, durations, rerank_durations, snapshots = [], [], [], []
            for _repetition in range(repeats):
                lexical = BM25PolicyRetriever(chunks)
                vector = VectorPolicyRetriever(chunks, HashingTextEncoder())
                choices = {
                    "bm25": lexical,
                    "hashing_vector": vector,
                    "hybrid_rrf": HybridPolicyRetriever(lexical, vector),
                    "hybrid_rewrite_heuristic": AdvancedHybridPolicyRetriever(lexical, vector),
                }
                retriever = choices[name]
                tap = CandidateTap()
                if name == "hybrid_rewrite_heuristic":
                    retriever.reranker = tap

                class Measured:
                    def search(
                        self,
                        query,
                        *,
                        category=None,
                        limit=3,
                        _retriever=retriever,
                        _durations=durations,
                        _rerank_durations=rerank_durations,
                    ):
                        with service.trace("baseline", user_id="benchmark") as collector:
                            start = time.perf_counter()
                            result = _retriever.search(query, category=category, limit=limit)
                            _durations.append((time.perf_counter() - start) * 1000)
                            _rerank_durations.extend(
                                s["duration_ms"]
                                for s in collector.spans
                                if s["name"] == "rag.rerank"
                            )
                            return result

                runs.append(evaluate_retriever(Measured(), cases, limit=limit))
                if tap.records:
                    original = AdvancedHybridPolicyRetriever(lexical, vector)
                    control = evaluate_retriever(original, cases, limit=limit)
                    if control["predictions"] != runs[-1]["predictions"]:
                        raise AssertionError("Candidate tap changed original rankings")
                    snapshots.append(tap.records)
            report["groups"][f"{name}_k{limit}"] = {
                "runs": runs,
                "search_latency_ms": distribution(durations),
                "heuristic_rerank_latency_ms": distribution(rerank_durations),
                "remote_rerank_latency_ms": None,
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "api_cost": 0,
                "candidate_snapshots": snapshots,
                "tap_equivalence_checked": bool(snapshots),
                "stable_rankings": all(r["predictions"] == runs[0]["predictions"] for r in runs),
                "answer_correctness": None,
                "unsupported_policy_promises": None,
            }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, group in report["groups"].items():
        metrics = group["runs"][0]
        print(
            name,
            {k: v for k, v in metrics.items() if k.startswith("recall") or k == "mrr"},
            group["search_latency_ms"],
        )


if __name__ == "__main__":
    main()
