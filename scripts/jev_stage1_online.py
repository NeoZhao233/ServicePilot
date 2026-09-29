"""Bounded original BGE + Responses rerank baseline; explicit network opt-in required."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from jev_stage1_baseline import digest, distribution

from servicepilot.observability import TraceService
from servicepilot.providers import (
    OnlineModelSettings,
    ResponsesPolicyReranker,
    UrllibJsonTransport,
)
from servicepilot.retrieval import (
    AdvancedHybridPolicyRetriever,
    BM25PolicyRetriever,
    FastEmbedTextEncoder,
    VectorPolicyRetriever,
    load_policy_chunks,
)
from servicepilot.retrieval_evaluation import evaluate_retriever

ROOT = Path(__file__).resolve().parents[1]


def summarize(report, cases):
    completed = [row for row in report["cases"] if row["status"] == "ok"]
    indexed = {case["case_id"]: case for case in cases}
    selected = [indexed[row["case_id"]] for row in completed]
    chunks = {c.policy_id: c for c in load_policy_chunks(ROOT / "data/policies.json")}
    from servicepilot.models import RetrievalHit

    class Replay:
        def __init__(self, field):
            self.rows = iter(completed)
            self.field = field

        def search(self, query, *, category=None, limit=3):
            row = next(self.rows)
            return [
                RetrievalHit(chunks[pid], 1.0 / (i + 1), ())
                for i, pid in enumerate(row[self.field][:limit])
            ]

    report["coverage"] = {"completed": len(completed), "total": len(cases)}
    report["metrics"] = {
        f"{field}_k{k}": evaluate_retriever(Replay(field), selected, limit=k)
        for field in ("ranked_ids", "heuristic_ids")
        for k in (3, 5)
    }
    report["latency_ms"] = {
        field: distribution([row[field] for row in completed])
        for field in ("search_ms", "rerank_ms")
    }
    usages = [row["usage"] for row in completed]
    report["usage_totals"] = {}
    for field in ("input_tokens", "output_tokens"):
        values = [u.get(field) for u in usages]
        report["usage_totals"][field] = (
            sum(values) if values and all(type(v) is int for v in values) else None
        )
    report["api_cost"] = None  # Provider bill is not inferred from token counts.
    report["answer_accuracy"] = None
    report["full_chat_latency_ms"] = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--max-calls", type=int, default=23)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.allow_network or not 1 <= args.max_calls <= 24:
        parser.error("Explicit --allow-network and max-calls 1..24 required")
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    settings = OnlineModelSettings.from_env()
    if settings.embedding_mode != "local":
        parser.error("This frozen baseline requires the configured local BGE encoder")
    cases_path = ROOT / "evaluations/retrieval_challenge.json"
    policies_path = ROOT / "data/policies.json"
    cases = json.loads(cases_path.read_text())
    baseline = json.loads(
        (ROOT / "evaluations/results/jev_stage1/offline_profile.json").read_text()
    )
    if digest(cases_path) != baseline["cases_sha256"]:
        parser.error("Frozen cases changed")
    if digest(policies_path) != baseline["policies_sha256"]:
        parser.error("Frozen policies changed")
    for name, expected in baseline["code_sha256"].items():
        if digest(ROOT / "src/servicepilot" / name) != expected:
            parser.error(f"Baseline source changed: {name}")
    if args.resume:
        report = json.loads(args.output.read_text())
        if report["status"] != "budget_exhausted":
            parser.error("Only a clean budget stop can be resumed")
        for key, value in [
            ("cases_sha256", digest(cases_path)),
            ("policies_sha256", digest(policies_path)),
            ("model", settings.intent_model),
            ("base_url", settings.base_url),
        ]:
            if report[key] != value:
                parser.error(f"Resume configuration mismatch: {key}")
    else:
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "status": "initializing",
            "python": platform.python_version(),
            "model": settings.intent_model,
            "base_url": settings.base_url,
            "embedding_model": settings.embedding_model,
            "cases_sha256": digest(cases_path),
            "policies_sha256": digest(policies_path),
            "source_sha256": baseline["code_sha256"],
            "parameters": baseline["parameters"],
            "attempts": 0,
            "cases": [],
            "retries": 0,
            "scope": "cold-cache search latency; excludes initialization and trace persistence",
            "preflight_not_included": True,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Refuse accidental overwrites that could silently repeat paid requests.
        with args.output.open("x") as handle:
            json.dump(report, handle)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)

    # We already downloaded this model separately. Never trigger hidden downloads here.
    os.environ["HF_HUB_OFFLINE"] = "1"
    encoder = FastEmbedTextEncoder(settings.embedding_model)
    chunks = load_policy_chunks(policies_path)
    lexical = BM25PolicyRetriever(chunks)
    vector = VectorPolicyRetriever(chunks, encoder)
    transport = UrllibJsonTransport(settings)
    service = TraceService.create()

    class Metered:
        def post(self, path, payload):
            if report["attempts"] >= args.max_calls:
                raise RuntimeError("Request budget exhausted")
            report["attempts"] += 1
            save()  # Reserve before sending; a crash cannot silently erase the attempt.
            response = transport.post(path, payload)
            raw = response.get("usage") or {}
            report["cases"][-1]["usage"] = {
                k: raw.get(k)
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "cost",
                    "input_tokens_details",
                    "output_tokens_details",
                )
            }
            report["cases"][-1]["returned_model"] = response.get("model")
            save()
            return response

    reranker = ResponsesPolicyReranker(Metered(), settings.intent_model)

    class Capture:
        def rerank(self, query, candidates, *, limit):
            row = report["cases"][-1]
            row["candidates"] = [
                {
                    "policy_id": hit.chunk.policy_id,
                    "score": hit.score,
                    "content_sha256": hashlib.sha256(hit.chunk.content.encode()).hexdigest(),
                }
                for hit in candidates
            ]
            row["heuristic_ids"] = [hit.chunk.policy_id for hit in candidates[:limit]]
            start = time.perf_counter()
            try:
                return reranker.rerank(query, candidates, limit=limit)
            except Exception as exc:
                row.update(
                    status="error",
                    error_type=type(exc).__name__,
                    http_status=getattr(exc.__cause__, "code", None),
                )
                raise
            finally:
                row["rerank_ms"] = (time.perf_counter() - start) * 1000

    retriever = AdvancedHybridPolicyRetriever(lexical, vector, reranker=Capture())
    done = {row["case_id"] for row in report["cases"]}
    report["status"] = "running"
    for case in cases:
        if case["case_id"] in done:
            continue
        if report["attempts"] >= args.max_calls:
            report["status"] = "budget_exhausted"
            break
        row = {"case_id": case["case_id"], "status": "running", "usage": {}}
        report["cases"].append(row)
        save()
        with service.trace("baseline.online", user_id="benchmark") as collector:
            start = time.perf_counter()
            result = retriever.search(case["query"], category=case.get("category"), limit=5)
            row["search_ms"] = (time.perf_counter() - start) * 1000
            row["ranked_ids"] = [hit.chunk.policy_id for hit in result]
            row["trace_id"] = collector.record["trace_id"]
            row["spans"] = collector.spans
        if row["status"] == "error" or retriever.last_diagnostics.reranker_degraded:
            report["status"] = "failed_stopped"
            break
        if not row.get("candidates"):
            row["status"] = "empty_candidates"
            report["status"] = "failed_stopped"
            break
        row["status"] = "ok"
        save()
        print(case["case_id"], "ok", flush=True)
    else:
        report["status"] = "completed"
    summarize(report, cases)
    save()
    print(
        json.dumps(
            {
                "status": report["status"],
                "coverage": report["coverage"],
                "usage": report["usage_totals"],
                "latency_ms": report["latency_ms"],
            }
        )
    )


if __name__ == "__main__":
    main()
