"""Bounded live ablation with shared, recorded prefixes; never retries model calls."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from servicepilot.corrective_rag import CorrectivePolicyRetriever
from servicepilot.evaluators import (
    DeepSeekEvidenceEvaluator,
    EvaluatorConfig,
    EvaluatorRanker,
    ProviderSufficiencyEvaluator,
)
from servicepilot.evidence import EvidenceContext
from servicepilot.models import PolicyChunk, RetrievalHit
from servicepilot.providers import OnlineModelSettings, ProviderError, UrllibJsonTransport
from servicepilot.rag_evaluation import evaluate
from servicepilot.retrieval import (
    AdvancedHybridPolicyRetriever,
    BM25PolicyRetriever,
    FastEmbedTextEncoder,
    VectorPolicyRetriever,
    load_policy_chunks,
)

ROOT = Path(__file__).resolve().parents[1]


class BudgetTransport:
    def __init__(self, inner, report, save, cap):
        self.inner, self.report, self.save, self.cap = inner, report, save, cap

    def post(self, path, payload):
        if self.report["attempts"] >= self.cap:
            raise RuntimeError("Live request budget exhausted")
        self.report["attempts"] += 1
        self.save()
        started = time.perf_counter()
        try:
            result = self.inner.post(path, payload)
        except Exception as exc:
            self.report["failed_request"] = type(exc).__name__
            self.save()
            raise
        self.report["physical_calls"].append(
            {
                "model": result.get("model"),
                "response_status": result.get("status"),
                "incomplete_details": result.get("incomplete_details"),
                "task_schema": payload["text"]["format"]["name"],
                "max_output_tokens": payload.get("max_output_tokens"),
                "usage": result.get("usage"),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
        )
        self.save()
        return result


class RecordedProvider:
    """Cache experiment prefixes, not production decisions. Report logical usage explicitly."""

    def __init__(self, inner):
        self.inner, self.cache, self.used = inner, {}, []
        self.failure = None
        self.failure_details = None

    async def call(self, method, query, documents, context):
        key = json.dumps(
            [method, query, [d.model_dump() for d in documents], context.model_dump()],
            sort_keys=True,
            ensure_ascii=False,
        )
        if key not in self.cache:
            try:
                self.cache[key] = await getattr(self.inner, method)(query, documents, context)
            except Exception as exc:
                self.failure = type(exc).__name__
                if isinstance(exc, ValidationError):
                    self.failure_details = [
                        {"type": error["type"], "loc": list(error["loc"])}
                        for error in exc.errors(include_input=False, include_url=False)
                    ]
                elif isinstance(exc, ProviderError) and str(exc) in {
                    "Invalid evidence references",
                    "Incomplete evidence scoring",
                    "Inconsistent sufficient decision",
                    "Evidence response did not complete",
                }:
                    self.failure_details = {"reason": str(exc)}
                raise
        result = self.cache[key]
        self.used.append(result)
        return result

    async def score_documents(self, query, documents, context=None):
        return await self.call("score_documents", query, documents, context)

    async def evaluate_sufficiency(self, query, documents, context=None):
        return await self.call("evaluate_sufficiency", query, documents, context)


def fixture(case, chunks):
    if case["case_id"] == "crag_wrong_version":
        common = dict(source="fixture-official", effective_from="2025-01-01")
        return [
            RetrievalHit(
                PolicyChunk("CONFLICT", "运费", content, version=version, **common), 1.0, ()
            )
            for version, content in [("v1", "商家承担"), ("v2", "客户承担")]
        ]
    if case["case_id"] == "crag_expired":
        original = next(c for c in chunks if c.policy_id == "POLICY-RETURN-7D")
        return [
            RetrievalHit(
                replace(
                    original,
                    source="fixture-official",
                    version="historical",
                    effective_from="2020-01-01",
                    effective_to="2021-01-01",
                ),
                1.0,
                (),
            )
        ]
    return None


async def run(args):
    if not args.allow_network or not 1 <= args.max_calls <= 96:
        raise ValueError("Explicit network opt-in and a 1..96 request budget are required")
    if args.output.exists():
        raise ValueError("Refusing to overwrite existing observations")
    load_dotenv(ROOT / ".env", override=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    settings = OnlineModelSettings.from_env()
    frozen = json.loads((ROOT / "evaluations/retrieval_challenge.json").read_text())
    extra = json.loads((ROOT / "evaluations/corrective_rag_cases.json").read_text())
    cases = frozen + extra
    excluded = set(args.exclude_case)
    if not excluded <= {case["case_id"] for case in cases}:
        raise ValueError("Unknown excluded case")
    baseline = json.loads((ROOT / "evaluations/results/jev_stage1/b0_online.json").read_text())
    chunks = load_policy_chunks(ROOT / "data/policies.json")
    lookup = {c.policy_id: c for c in chunks}
    for name, expected in [
        ("evaluations/retrieval_challenge.json", baseline["cases_sha256"]),
        ("data/policies.json", baseline["policies_sha256"]),
    ]:
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen baseline inputs changed")
    report = {
        "status": "running",
        "attempts": 0,
        "max_calls": args.max_calls,
        "method": "Shared recorded initial scoring and judgments; logical path latency/usage "
        "reuses actual call measurements. Not independently timed chat requests.",
        "cases_sha256": baseline["cases_sha256"],
        "policies_sha256": baseline["policies_sha256"],
        "excluded_cases": sorted(excluded),
        "evaluator_config": {
            "max_output_tokens": 8192,
            "timeout_seconds": 65.0,
            "prompt_version": DeepSeekEvidenceEvaluator.CONFIG_VERSION,
        },
        "physical_calls": [],
        "runs": [],
        "decisions": [],
    }
    report["source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "src/servicepilot/evaluators.py",
            "src/servicepilot/corrective_rag.py",
            "src/servicepilot/evidence.py",
            "scripts/pluggable_rag_eval.py",
            "evaluations/corrective_rag_cases.json",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temp = args.output.with_suffix(".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temp.replace(args.output)

    transport = BudgetTransport(
        UrllibJsonTransport(replace(settings, timeout_seconds=60)), report, save, args.max_calls
    )
    provider = RecordedProvider(
        DeepSeekEvidenceEvaluator(
            transport,
            settings.intent_model,
            EvaluatorConfig(timeout_seconds=65.0, max_output_tokens=8192),
        )
    )
    ranker, judge = EvaluatorRanker(provider), ProviderSufficiencyEvaluator(provider)
    vector = VectorPolicyRetriever(chunks, FastEmbedTextEncoder(settings.embedding_model))
    search = AdvancedHybridPolicyRetriever(BM25PolicyRetriever(chunks), vector, cache_ttl_seconds=0)
    frozen_rows = {c["case_id"]: c for c in baseline["cases"]}
    for case in cases:
        if case["case_id"] in excluded:
            continue
        conditions = case.get("conditions", {})
        context = EvidenceContext(
            question=case["query"],
            category=case.get("category") or conditions.get("category"),
            days_since_delivery=conditions.get("days_since_delivery"),
            as_of="2026-09-24",
            trusted_sources=("fixture-official",),
            safe_empty=True,
        )
        initial = fixture(case, chunks)
        retrieval_started = time.perf_counter()
        if initial is None and case["case_id"] in frozen_rows:
            row = frozen_rows[case["case_id"]]
            initial = [
                RetrievalHit(lookup[c["policy_id"]], c["score"], ()) for c in row["candidates"]
            ]
        elif initial is None:
            initial = search.search(case["query"], category=context.category, limit=48)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        fixture_value = fixture(case, chunks)

        class Base:
            calls = 0
            elapsed = 0.0

            def search(self, query, _initial=initial, _fixture=fixture_value, **kwargs):
                started = time.perf_counter()
                self.calls += 1
                result = (
                    _initial
                    if self.calls == 1 or _fixture is not None
                    else search.search(query, **kwargs)
                )
                self.elapsed += (time.perf_counter() - started) * 1000
                return result

        for group in ("B1", "B2", "B3"):
            if group == "B1" and fixture_value is not None:
                continue  # Metadata fixtures measure rule gates, not the original scorer.
            provider.used = []
            base = Base()
            if group == "B1":
                ranked = await ranker.arank(case["query"], initial, context=context)
                hits, rounds, status, filtered, stop = ranked.hits[:5], 1, None, [], ranked.reason
                candidate_ids = [h.chunk.policy_id for h in initial]
            else:
                rag = CorrectivePolicyRetriever(
                    base,
                    ranker,
                    judge,
                    max_corrections=2 if group == "B3" else 0,
                    trusted_chunks=chunks,
                    trusted_sources=("fixture-official",),
                )
                result = await rag.retrieve(case["query"], context=context, limit=5)
                hits, rounds, status = result.hits, result.rounds, result.judgment.status
                filtered, stop = list(result.filtered_ids), result.stop_reason
                candidate_ids = list(result.candidate_ids)
            if provider.failure:
                report["status"] = "failed"
                report["failure"] = {
                    "case_id": case["case_id"],
                    "group": group,
                    "error_type": provider.failure,
                    "validation_details": provider.failure_details,
                }
                save()
                return False
            stats = [r.stats for r in provider.used]

            def total(key, stats=stats):
                values = [getattr(s, key) for s in stats]
                return sum(values) if all(v is not None for v in values) else None

            report["runs"].append(
                dict(
                    group=group,
                    case_id=case["case_id"],
                    source="deepseek_live_shared_prefix",
                    retrieved_ids=[h.chunk.policy_id for h in hits],
                    candidate_ids=candidate_ids,
                    filtered_ids=filtered,
                    sufficiency=status,
                    rounds=rounds,
                    deepseek_calls=len(stats),
                    latency_ms=retrieval_ms + base.elapsed + total("elapsed_ms"),
                    input_tokens=total("input_tokens"),
                    output_tokens=total("output_tokens"),
                    api_cost=None,
                    stop_reason=stop,
                    answer_correct=None,
                    unsupported_promises=None,
                )
            )
            report["decisions"].append(
                {
                    "case_id": case["case_id"],
                    "group": group,
                    "results": [r.model_dump() for r in provider.used],
                }
            )
            save()
        print(case["case_id"], report["attempts"], flush=True)
    old = json.loads((ROOT / "evaluations/results/jev_corrective/observed_runs.json").read_text())
    report["runs"] = old + report["runs"]
    report["metrics"] = evaluate(cases, report["runs"])
    report["status"] = "completed"
    save()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-case", action="append", default=[])
    if not asyncio.run(run(parser.parse_args())):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
