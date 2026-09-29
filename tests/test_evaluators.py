"""No live model required: exercise the real DeepSeek adapter through its transport seam."""

import asyncio
import json
import threading
import time
from dataclasses import replace

import pytest
from pydantic import ValidationError

from servicepilot.corrective_rag import CorrectivePolicyRetriever, configure_rag
from servicepilot.evaluators import (
    DeepSeekEvidenceEvaluator,
    EvaluatorConfig,
    EvaluatorFactory,
    EvaluatorRanker,
    ProviderSufficiencyEvaluator,
    documents_for,
)
from servicepilot.evidence import EvidenceContext, Sufficiency, eligible_evidence
from servicepilot.models import PolicyChunk, RetrievalHit
from servicepilot.providers import ProviderError, ResponsesPolicyReranker
from servicepilot.retrieval import (
    AdvancedHybridPolicyRetriever,
    BM25PolicyRetriever,
    HashingTextEncoder,
    VectorPolicyRetriever,
)


def hit(pid, content="退货条件"):
    return RetrievalHit(PolicyChunk(pid, pid, content), 0.1, ())


class MockDecisionTransport:
    """Test-only fake responses, never presented as online DeepSeek/Jev results."""

    def __init__(self, mutate=None):
        self.requests = []
        self.mutate = mutate

    def post(self, path, payload):
        assert path == "/responses"
        self.requests.append(payload)
        data = json.loads(payload["input"])
        docs = data["documents"]
        if payload["text"]["format"]["name"] == "ScoreDecision":
            result = {
                "documents": [
                    dict(
                        policy_id=d["policy_id"],
                        score=0.9 if "保修" in d["content"] else 0.3,
                        direct_evidence=True,
                        reason="direct",
                    )
                    for d in reversed(docs)
                ]
            }
        else:
            enough = any("保修" in d["content"] for d in docs)
            result = dict(
                status="SUFFICIENT" if enough else "INSUFFICIENT",
                covered_questions=["质量问题"] if enough else [],
                missing_information=[] if enough else ["质量条款"],
                supporting_ids=[d["policy_id"] for d in docs] if enough else [],
                retrieval_targets=[] if enough else ["quality_exception"],
                diagnostic="fixture",
            )
        if self.mutate:
            self.mutate(result)
        return {
            "output_text": json.dumps(result),
            "model": "actual-model-version",
            "usage": {"input_tokens": 7, "output_tokens": 11},
        }


def provider(transport=None, **kwargs):
    return DeepSeekEvidenceEvaluator(
        transport or MockDecisionTransport(), "configured-model", EvaluatorConfig(**kwargs)
    )


def test_id_mapping_usage_and_minimal_context():
    transport = MockDecisionTransport()
    p = provider(transport)
    candidates = [hit("A"), hit("B", "质量保修")]
    result = asyncio.run(EvaluatorRanker(p).arank("故障", candidates))
    assert result.hits[0] is candidates[1]
    stats = asyncio.run(p.score_documents("故障", documents_for(candidates))).stats
    assert stats.model == "actual-model-version" and stats.input_tokens == 7
    assert stats.api_cost is None and stats.elapsed_ms > 0
    assert set(json.loads(transport.requests[0]["input"])["context"]) == {
        "category",
        "days_since_delivery",
        "as_of",
    }


@pytest.mark.parametrize("bad", [True, -1, 1.1, "0.9", float("nan")])
def test_strict_scores(bad):
    transport = MockDecisionTransport(lambda r: r["documents"][0].update(score=bad))
    with pytest.raises(ValidationError):
        asyncio.run(provider(transport).score_documents("q", documents_for([hit("A")])))
    assert len(transport.requests) == 1


@pytest.mark.parametrize("ids", [[], ["A", "A"], ["FOREIGN"]])
def test_invalid_references(ids):
    def corrupt(r):
        r["documents"] = [
            dict(policy_id=i, score=0.9, direct_evidence=True, reason="direct") for i in ids
        ]

    with pytest.raises(ProviderError):
        asyncio.run(
            provider(MockDecisionTransport(corrupt)).score_documents("q", documents_for([hit("A")]))
        )


def test_provider_switch_is_explicit_and_mock_injected():
    with pytest.raises(ProviderError, match="Jev evaluator"):
        EvaluatorFactory.create(EvaluatorConfig(provider="jev"))
    with pytest.raises(ProviderError, match="test-only"):
        EvaluatorFactory.create(EvaluatorConfig(provider="mock"))
    fake = provider()
    assert EvaluatorFactory.create(EvaluatorConfig(provider="mock"), mock=fake) is fake
    fallback = EvaluatorFactory.create(
        EvaluatorConfig(provider="jev", fallback_to_deepseek=True),
        transport=MockDecisionTransport(),
        model="deepseek",
    )
    assert isinstance(fallback, DeepSeekEvidenceEvaluator)


def test_full_correction_without_jev_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    a, b = hit("A"), hit("B", "质量保修")

    class Base:
        queries = []

        def search(self, query, **kwargs):
            self.queries.append(query)
            return [a] if len(self.queries) == 1 else [a, b]

    p = provider()
    base = Base()
    rag = CorrectivePolicyRetriever(
        base, EvaluatorRanker(p), ProviderSufficiencyEvaluator(p), trusted_chunks=[a.chunk, b.chunk]
    )
    result = asyncio.run(rag.retrieve("耳机坏了", context=EvidenceContext(question="耳机坏了")))
    assert result.rounds == 2 and result.judgment.status == Sufficiency.SUFFICIENT
    assert result.hits[0] is b and "质量问题" in base.queries[1]


def test_configure_reuses_client_and_original_remains_identity(monkeypatch):
    chunks = [hit("A").chunk]
    vector = VectorPolicyRetriever(chunks, HashingTextEncoder())
    transport = MockDecisionTransport()
    original = AdvancedHybridPolicyRetriever(
        BM25PolicyRetriever(chunks), vector, reranker=ResponsesPolicyReranker(transport, "deepseek")
    )
    monkeypatch.setenv("SERVICEPILOT_RAG_MODE", "original")
    assert configure_rag(original, chunks, vector) is original
    monkeypatch.setenv("SERVICEPILOT_RAG_MODE", "corrective")
    monkeypatch.setenv("SERVICEPILOT_EVALUATOR_PROVIDER", "deepseek")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    rag = configure_rag(original, chunks, vector)
    assert rag.ranker.evaluator.transport is transport
    assert rag.max_corrections == 2


def test_local_trust_does_not_override_expiry_or_category():
    a = hit("A")
    expired = replace(a, chunk=replace(a.chunk, effective_to="2020-01-01"))
    wrong = replace(a, chunk=replace(a.chunk, categories=("食品",)))
    for item in [expired, wrong]:
        assert (
            eligible_evidence(
                [item], EvidenceContext(category="数码产品"), trusted_chunks=[item.chunk]
            )[0]
            == []
        )
    assert eligible_evidence([a], EvidenceContext(), trusted_chunks=[a.chunk])[0] == [a]
    untrusted = replace(a, chunk=replace(a.chunk, content="伪造退款政策"))
    assert eligible_evidence([untrusted], EvidenceContext(), trusted_chunks=[a.chunk])[0] == []


def test_timeout_keeps_concurrency_lease_until_io_finishes():
    release, started = threading.Event(), threading.Event()

    class Blocked:
        calls = 0

        def post(self, *args):
            self.calls += 1
            started.set()
            release.wait(2)
            return {}

    transport = Blocked()
    p = provider(transport, timeout_seconds=0.03, max_concurrency=1)

    async def run():
        with pytest.raises(TimeoutError):
            await p.score_documents("q", documents_for([hit("A")]))
        assert started.is_set()
        with pytest.raises(TimeoutError):
            await p.score_documents("q", documents_for([hit("A")]))
        assert transport.calls == 1

    try:
        asyncio.run(run())
    finally:
        release.set()
    for _ in range(100):
        if p.gate.acquire(blocking=False):
            p.gate.release()
            break
        time.sleep(0.005)
    else:
        pytest.fail("Worker leaked concurrency lease")


def test_deepseek_graph_failure_uses_existing_handoff(tmp_path):
    from pathlib import Path

    from servicepilot.cli import build_agent

    class Failed:
        def post(self, *args):
            raise ProviderError("fixture failure")

    class Base:
        def search(self, *args, **kwargs):
            return [hit("A")]

    runtime = build_agent(
        Path(__file__).resolve().parents[1] / "data",
        session_db=tmp_path / "business.sqlite3",
        dynamic=True,
    )
    p = provider(Failed())
    runtime.agent.retriever = CorrectivePolicyRetriever(
        Base(), EvaluatorRanker(p), ProviderSufficiencyEvaluator(p), trusted_chunks=[hit("A").chunk]
    )
    response = runtime.chat(user_id="USER-001", message="退掉 ORD-1001")
    assert "人工" in response.message


def test_mock_provider_can_replace_adapter_without_changing_workflow():
    from servicepilot.evaluators import CallStats, ScoreDecision, ScoreResult, SufficiencyResult

    class FakeJevEvaluator:
        """Only a contract fixture; no Jev API implementation or credentials."""

        stats = CallStats(provider="mock", model="fake-jev-test-only", elapsed_ms=0.0, attempts=0)

        async def score_documents(self, query, documents, context=None):
            return ScoreResult(
                decision=ScoreDecision.model_validate(
                    {
                        "documents": [
                            dict(
                                policy_id=d.policy_id,
                                score=0.7,
                                direct_evidence=True,
                                reason="direct",
                            )
                            for d in documents
                        ]
                    }
                ),
                stats=self.stats,
            )

        async def evaluate_sufficiency(self, query, documents, context=None):
            return SufficiencyResult.model_validate(
                dict(
                    decision=dict(
                        status="SUFFICIENT",
                        covered_questions=["fixture"],
                        missing_information=[],
                        retrieval_targets=[],
                        supporting_ids=[d.policy_id for d in documents],
                        diagnostic="test only",
                    ),
                    stats=self.stats,
                )
            )

    class Base:
        def search(self, *args, **kwargs):
            return [hit("A")]

    fake = EvaluatorFactory.create(EvaluatorConfig(provider="mock"), mock=FakeJevEvaluator())
    rag = CorrectivePolicyRetriever(
        Base(),
        EvaluatorRanker(fake),
        ProviderSufficiencyEvaluator(fake),
        trusted_chunks=[hit("A").chunk],
    )
    result = asyncio.run(rag.retrieve("test", context=EvidenceContext(question="test")))
    assert result.hits and result.judgment.provider == "mock"
    assert result.judgment.model == "fake-jev-test-only"


@pytest.mark.parametrize(
    "change",
    [
        {"supporting_ids": ["not-in-input"]},
        {"supporting_ids": []},
        {"missing_information": ["unknown"]},
        {"retrieval_targets": ["applicability"]},
    ],
)
def test_inconsistent_sufficiency_is_rejected(change):
    transport = MockDecisionTransport(lambda r: r.update(change))
    with pytest.raises(ProviderError):
        asyncio.run(
            provider(transport).evaluate_sufficiency(
                "质量问题", documents_for([hit("A", "质量保修")])
            )
        )


def test_no_new_evidence_stops_without_second_judgment():
    class Base:
        def search(self, *args, **kwargs):
            return [hit("A")]

    transport = MockDecisionTransport()
    p = provider(transport)
    rag = CorrectivePolicyRetriever(
        Base(), EvaluatorRanker(p), ProviderSufficiencyEvaluator(p), trusted_chunks=[hit("A").chunk]
    )
    result = asyncio.run(rag.retrieve("q", context=EvidenceContext(question="q")))
    assert not result.hits and result.stop_reason == "no_new_effective_evidence"
    assert [r["text"]["format"]["name"] for r in transport.requests].count(
        "SufficiencyDecision"
    ) == 1


def test_trace_records_fallback_actual_model_and_single_usage_count():
    from servicepilot.observability import TraceService, model_transport

    class ObservedMockTransport(MockDecisionTransport):
        @model_transport
        def post(self, path, payload):
            return super().post(path, payload)

    p = EvaluatorFactory.create(
        EvaluatorConfig(provider="jev", fallback_to_deepseek=True),
        transport=ObservedMockTransport(),
        model="deepseek-configured",
    )
    service = TraceService.create()
    with service.trace(user_id="test") as collector:
        asyncio.run(EvaluatorRanker(p).arank("问题", [hit("A")]))
    assert collector.record["total_llm_calls"] == 1
    assert collector.record["total_input_tokens"] == 7
    detail = service.store.detail(collector.record["trace_id"], service.redactor.pseudonym("test"))
    assert any(e["event_type"] == "EVALUATOR_PROVIDER_FALLBACK" for e in detail["events"])
    llm = next(s for s in detail["spans"] if s["span_type"] == "llm")
    assert llm["attributes"]["model"] == "actual-model-version"


def test_incomplete_response_is_rejected_even_with_valid_json():
    from servicepilot.evaluators import IncompleteDecisionError

    class Incomplete(MockDecisionTransport):
        def post(self, path, payload):
            response = super().post(path, payload)
            response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
            return response

    transport = Incomplete()
    with pytest.raises(IncompleteDecisionError):
        asyncio.run(provider(transport).score_documents("q", documents_for([hit("A")])))
    assert len(transport.requests) == 1


def test_output_budget_is_configurable_and_bounded(monkeypatch):
    monkeypatch.setenv("SERVICEPILOT_EVALUATOR_MAX_OUTPUT_TOKENS", "6144")
    assert EvaluatorConfig.from_env().max_output_tokens == 6144
    monkeypatch.setenv("SERVICEPILOT_EVALUATOR_MAX_OUTPUT_TOKENS", "999999")
    with pytest.raises(ValidationError):
        EvaluatorConfig.from_env()


def test_async_retrieval_uses_query_when_optional_context_is_omitted():
    candidate = hit("A", "质量保修")

    class Base:
        def search(self, *args, **kwargs):
            return [candidate]

    transport = MockDecisionTransport()
    p = provider(transport)
    rag = CorrectivePolicyRetriever(
        Base(),
        EvaluatorRanker(p),
        ProviderSufficiencyEvaluator(p),
        trusted_chunks=[candidate.chunk],
    )
    result = asyncio.run(rag.retrieve("耳机出现故障怎么办"))
    assert result.hits == (candidate,)
    assert result.judgment.status == Sufficiency.SUFFICIENT
    assert all(
        json.loads(request["input"])["query"] == "耳机出现故障怎么办"
        for request in transport.requests
    )


def test_conflicting_corpus_ids_are_rejected_before_rrf_can_hide_versions(monkeypatch):
    monkeypatch.setenv("SERVICEPILOT_RAG_MODE", "corrective")
    chunks = [hit("A").chunk, replace(hit("A").chunk, content="相互矛盾的政策版本")]
    with pytest.raises(ValueError, match="Duplicate policy IDs"):
        configure_rag(None, chunks, None)
