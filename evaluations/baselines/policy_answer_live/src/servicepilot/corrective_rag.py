"""Bounded corrective retrieval; observations never grant authority to commerce tools."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, replace

from .evidence import (
    REWRITES,
    EvidenceContext,
    EvidenceJudgment,
    EvidenceSufficiencyEvaluator,
    Sufficiency,
    corpus_fingerprint,
    current_evidence_context,
    eligible_evidence,
)
from .evidence_calibration import load_calibration
from .evidence_runtime import run_sync
from .models import RetrievalHit
from .observability import Span, event
from .retrieval import AdvancedHybridPolicyRetriever, BM25PolicyRetriever


@dataclass(frozen=True)
class CorrectiveResult:
    hits: tuple
    judgment: EvidenceJudgment
    rounds: int
    candidate_ids: tuple[str, ...]
    filtered_ids: tuple[str, ...]
    stop_reason: str


class CorrectivePolicyRetriever:
    def __init__(
        self, base, ranker, evaluator, *, max_corrections=2, trusted_sources=(), trusted_chunks=()
    ):
        if type(max_corrections) is not int or not 0 <= max_corrections <= 2:
            raise ValueError("At most two correction rounds are permitted")
        self.base, self.ranker, self.evaluator = base, ranker, evaluator
        self.fallback = None
        self.trusted_chunks = tuple(trusted_chunks)
        self.max_corrections, self.trusted_sources = max_corrections, tuple(trusted_sources)

    async def retrieve(self, query, *, context=None, category=None, limit=3):
        context = context or current_evidence_context()
        context = replace(
            context,
            category=category or context.category,
            trusted_sources=self.trusted_sources or context.trusted_sources,
        )
        with Span(
            "rag.corrective",
            "rag",
            inputs={"query": context.question or query},
            attributes={"max_corrections": self.max_corrections},
        ) as span:
            if limit <= 0:
                return CorrectiveResult(
                    (), EvidenceJudgment(Sufficiency.UNCERTAIN), 0, (), (), "zero_limit"
                )
            candidates = {}
            prior_effective = None
            queries = {query}
            current_query = query
            filtered = set()
            judgment = EvidenceJudgment(Sufficiency.UNCERTAIN, (), "not_evaluated")
            rounds, stop = 0, "budget_exhausted"
            for round_index in range(self.max_corrections + 1):
                rounds += 1
                with Span(
                    "rag.hybrid_retrieve" if round_index == 0 else "rag.corrective_retrieve",
                    "rag",
                    inputs={"query": current_query},
                    attributes={"round": round_index},
                ) as retrieval_span:
                    try:
                        fresh = await asyncio.to_thread(
                            self.base.search, current_query, category=context.category, limit=48
                        )
                    except Exception as exc:
                        retrieval_span.failed(type(exc).__name__)
                        stop = "retrieval_failed"
                        break
                # Check conflicting active versions before merging IDs; never overwrite silently.
                merged = [*candidates.values(), *fresh]
                eligible, conflict = eligible_evidence(
                    merged, context, trusted_chunks=self.trusted_chunks
                )
                if conflict:
                    judgment, stop = conflict, conflict.reason
                    break
                filtered.update(
                    {h.chunk.policy_id for h in merged} - {h.chunk.policy_id for h in eligible}
                )
                if round_index and not ({h.chunk.policy_id for h in eligible} - candidates.keys()):
                    stop = "no_new_effective_evidence"
                    break
                candidates.update({h.chunk.policy_id: h for h in eligible})
                ranked = await self.ranker.arank(
                    context.question or query, candidates.values(), context=context
                )
                if ranked.degraded:
                    judgment = EvidenceJudgment(Sufficiency.UNCERTAIN, (), ranked.reason)
                    stop = ranked.reason
                    break
                filtered.update(d.policy_id for d in ranked.scores if not d.keep)
                # Evaluate only evidence that will actually be returned to the caller.
                selected = ranked.hits[:limit]
                effective = frozenset(h.chunk.policy_id for h in selected)
                if prior_effective is not None and effective <= prior_effective:
                    stop = "no_new_effective_evidence"
                    break
                judgment = await self.evaluator.evaluate(context, selected)
                if judgment.status is Sufficiency.SUFFICIENT:
                    span.output({"policy_ids": list(effective), "rounds": rounds})
                    event(
                        "EVIDENCE_FINALIZED",
                        policy_ids=list(effective),
                        rounds=rounds,
                        verdict=judgment.status,
                    )
                    return CorrectiveResult(
                        tuple(selected),
                        judgment,
                        rounds,
                        tuple(candidates),
                        tuple(sorted(filtered)),
                        "sufficient",
                    )
                if judgment.status is Sufficiency.CONFLICTING:
                    stop = "unresolved_conflict"
                    break
                if judgment.status is Sufficiency.UNCERTAIN:
                    stop = judgment.reason
                    break
                prior_effective = effective
                if round_index >= self.max_corrections:
                    break
                terms = [REWRITES[key] for key in judgment.missing if key in REWRITES]
                if not terms:
                    stop = "no_targeted_rewrite"
                    break
                current_query = f"{context.question or query} {context.category or ''} " + " ".join(
                    terms
                )
                if current_query in queries:
                    stop = "repeated_query"
                    break
                queries.add(current_query)
                event(
                    "CORRECTIVE_RETRIEVAL_TRIGGERED",
                    reason_code=judgment.reason,
                    missing=list(judgment.missing),
                    query=current_query,
                    round=round_index + 1,
                )
            event("RAG_DEGRADED", reason_code=stop, verdict=judgment.status, rounds=rounds)
            span.output({"policy_ids": [], "rounds": rounds, "reason_code": stop})
            return CorrectiveResult(
                (), judgment, rounds, tuple(candidates), tuple(sorted(filtered)), stop
            )

    async def asearch(self, query, *, category=None, limit=3, context=None):
        return list(
            (await self.retrieve(query, context=context, category=category, limit=limit)).hits
        )

    def search(self, query, *, category=None, limit=3):
        if self.fallback is not None and not current_evidence_context().safe_empty:
            event("RAG_DEGRADED", reason_code="unsupported_safe_empty_path")
            return self.fallback.search(query, category=category, limit=limit)
        return run_sync(lambda: self.asearch(query, category=category, limit=limit))


def configure_rag(original, chunks, vector, *, frozen_ids=()):
    """Default is a literal identity operation: no SDK import, key read or behavior change."""
    mode = os.getenv("SERVICEPILOT_RAG_MODE", "original")
    if mode == "original":
        return original
    if mode in {"evidence_score", "evidence_check", "corrective"}:
        from .evaluators import (
            EvaluatorConfig,
            EvaluatorFactory,
            EvaluatorRanker,
            ProviderSufficiencyEvaluator,
        )
        from .providers import OnlineModelSettings, UrllibJsonTransport

        prior = original.reranker
        if prior is not None and hasattr(prior, "transport"):
            transport, model = prior.transport, prior.model
        else:
            settings = OnlineModelSettings.from_env()
            transport, model = UrllibJsonTransport(settings), settings.intent_model
        provider = EvaluatorFactory.create(
            EvaluatorConfig.from_env(), transport=transport, model=model
        )
        ranker = EvaluatorRanker(provider)
        if mode == "evidence_score":
            return AdvancedHybridPolicyRetriever(
                BM25PolicyRetriever(chunks), vector, reranker=ranker, cache_ttl_seconds=0
            )
        result = CorrectivePolicyRetriever(
            AdvancedHybridPolicyRetriever(BM25PolicyRetriever(chunks), vector, cache_ttl_seconds=0),
            ranker,
            ProviderSufficiencyEvaluator(provider),
            max_corrections=(
                int(os.getenv("SERVICEPILOT_RAG_MAX_CORRECTIONS", "2"))
                if mode == "corrective"
                else 0
            ),
            trusted_chunks=chunks,
            trusted_sources=tuple(
                s.strip()
                for s in os.getenv("SERVICEPILOT_POLICY_SOURCES", "").split(",")
                if s.strip()
            ),
        )
        result.fallback = original
        return result
    if mode not in {"jev_rerank", "jev_filter", "jev_corrective"}:
        raise ValueError("Unsupported SERVICEPILOT_RAG_MODE")
    from .jev import JevBackend, JevEvidenceRanker

    calibration = None
    model = os.getenv("SERVICEPILOT_JEV_MODEL", "jev-latest")
    if mode != "jev_rerank":
        artifact = os.getenv("SERVICEPILOT_JEV_CALIBRATION")
        if not artifact:
            raise ValueError("Filtering/correction is disabled until calibration is available")
        calibration = load_calibration(
            artifact,
            corpus_hash=corpus_fingerprint(chunks),
            model=model,
            ranker_version=JevEvidenceRanker.CONFIG_VERSION,
            frozen_ids=frozen_ids
            or tuple(
                f"{prefix}_{i:02}"
                for prefix in ("para", "multi", "negative", "noise")
                for i in range(1, 7)
            ),
            corrective=mode == "jev_corrective",
            scoring_mode=os.getenv("SERVICEPILOT_JEV_SCORING_MODE", "listwise"),
        )
    backend = JevBackend(
        os.getenv("TYPESAFE_API_KEY", ""),
        model=model,
        timeout=float(os.getenv("SERVICEPILOT_JEV_TIMEOUT", "8")),
        mode=os.getenv("SERVICEPILOT_JEV_SCORING_MODE", "listwise"),
    )
    ranker = JevEvidenceRanker(backend, filtering=mode != "jev_rerank", calibration=calibration)
    if mode != "jev_corrective":
        ranker.fallback = original.reranker
        return AdvancedHybridPolicyRetriever(
            BM25PolicyRetriever(chunks), vector, reranker=ranker, cache_ttl_seconds=0
        )
    trusted = tuple(
        s.strip() for s in os.getenv("SERVICEPILOT_POLICY_SOURCES", "").split(",") if s.strip()
    )
    if not trusted:
        raise ValueError("Corrective RAG requires an explicit trusted policy source list")
    # Existing corpus has no verified dates/versions; never silently mark it authoritative.
    eligible, conflict = eligible_evidence(
        [RetrievalHit(c, 0, ()) for c in chunks], EvidenceContext(trusted_sources=trusted)
    )
    if conflict or not eligible:
        raise ValueError("No authoritative, versioned and effective policies for corrective RAG")
    result = CorrectivePolicyRetriever(
        AdvancedHybridPolicyRetriever(BM25PolicyRetriever(chunks), vector, cache_ttl_seconds=0),
        ranker,
        EvidenceSufficiencyEvaluator(backend, confidence_floor=calibration.confidence_floor),
        max_corrections=int(os.getenv("SERVICEPILOT_RAG_MAX_CORRECTIONS", "2")),
        trusted_sources=trusted,
    )
    result.fallback = original
    return result
