import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from servicepilot.corrective_rag import CorrectivePolicyRetriever, configure_rag
from servicepilot.evidence import (
    EvidenceContext,
    EvidenceSufficiencyEvaluator,
    Sufficiency,
    current_evidence_context,
    eligible_evidence,
    evidence_context,
)
from servicepilot.evidence_calibration import Calibration, load_calibration, threshold_sweep
from servicepilot.jev import JevBackend, JevEvidenceRanker
from servicepilot.models import PolicyChunk, RetrievalHit
from servicepilot.observability import TraceService


def hit(pid, content="普通退款", **kwargs):
    defaults = dict(source="official-test", version="v1", effective_from="2024-01-01")
    defaults.update(kwargs)
    return RetrievalHit(PolicyChunk(pid, pid, content, **defaults), 1.0, ())


def context(question="耳机超过七天有质量问题能退货吗"):
    return EvidenceContext(question, "数码产品", 10, "2026-09-24", ("official-test",))


class StubBackend:
    model = "mock-jev"

    def __init__(self, scores=None, failure=None):
        self.scores = scores
        self.failure = failure
        self.calls = 0

    async def score(self, query, documents, *, relevance=False):
        self.calls += 1
        if self.failure:
            raise self.failure
        scores = self.scores or [0.9] * len(documents)
        return {
            "results": [
                {"document_index": i, "score": score}
                for i, score in reversed(list(enumerate(scores)))
            ]
        }

    async def decide(self, state, dimensions):
        enough = any("保修" in c["content"] for c in state["policies"])
        status = "SUFFICIENT" if enough else "INSUFFICIENT"
        return {
            "answers": {
                "sufficiency": {
                    "type": "choice",
                    "choice": status,
                    "confidence": 0.99,
                    "probabilities": {s.value: float(s.value == status) for s in Sufficiency},
                },
                **{
                    key: {
                        "type": "noul",
                        "noul": 0 if key == "quality_exception" and not enough else 1,
                    }
                    for key in dimensions
                },
            }
        }


class Base:
    def __init__(self, first, later=None):
        self.first, self.later, self.queries = first, later, []

    def search(self, query, *, category=None, limit=3):
        self.queries.append(query)
        return (self.later if len(self.queries) > 1 and self.later is not None else self.first)[
            :limit
        ]


def test_ranking_preserves_original_object_ids_and_does_not_filter():
    candidates = [hit("A"), hit("B", source="other")]
    result = asyncio.run(JevEvidenceRanker(StubBackend([0.1, 0.9])).arank("query", candidates))
    assert [h.chunk.policy_id for h in result.hits] == ["B", "A"]
    assert result.hits[0].chunk is candidates[1].chunk
    assert all(s.keep for s in result.scores)


@pytest.mark.parametrize("scores", [[0.9], [float("nan"), 0.2], [True, 0.2], [1.1, 0.1]])
def test_malformed_scores_fallback_preserves_candidates(scores):
    candidates = [hit("A"), hit("B")]
    result = asyncio.run(JevEvidenceRanker(StubBackend(scores)).arank("query", candidates))
    assert result.degraded and result.hits == tuple(candidates)


def test_filter_requires_calibration_and_records_every_score():
    with pytest.raises(ValueError):
        JevEvidenceRanker(StubBackend(), filtering=True)
    ranker = JevEvidenceRanker(
        StubBackend([0.1, 0.9]), filtering=True, calibration=Calibration(0.5, 0.9, "test-only")
    )
    result = asyncio.run(ranker.arank("q", [hit("rare"), hit("ordinary")], protected_ids=["rare"]))
    assert len(result.hits) == 2
    result = asyncio.run(ranker.arank("q", [hit("rare"), hit("ordinary")]))
    assert len(result.hits) == 1 and len(result.scores) == 2


def test_empty_and_zero_limit_make_no_requests():
    backend = StubBackend()
    ranker = JevEvidenceRanker(backend)
    assert asyncio.run(ranker.arank("q", [])).hits == ()
    assert asyncio.run(ranker.arerank("q", [hit("A")], limit=0)) == []
    assert backend.calls == 0


def test_real_library_with_mock_http_records_actual_request_usage_and_tree():
    calls = []

    async def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(
            200,
            json={
                "model": "mock-jev",
                "usage": {"input_tokens": 31, "output_tokens": 2},
                "answers": {k: {"type": "noul", "noul": 0.8} for k in body["questions"]},
            },
        )

    backend = JevBackend(
        "mock-secret-never-stored", transport_factory=lambda: httpx.MockTransport(handle)
    )
    service = TraceService.create()
    with service.trace(user_id="test") as collector:
        result = asyncio.run(JevEvidenceRanker(backend).arank("退款", [hit("A"), hit("B")]))
    assert len(result.hits) == 2 and len(calls) == 1
    assert collector.record["total_input_tokens"] == 31
    assert collector.record["total_llm_calls"] == 1
    detail = service.store.detail(collector.record["trace_id"], service.redactor.pseudonym("test"))
    assert "mock-secret-never-stored" not in json.dumps(detail)
    llm = next(s for s in detail["spans"] if s["span_type"] == "llm")
    assert llm["parent_span_id"] == detail["tree"][0]["span_id"]
    assert llm["duration_ms"] >= 0


def test_http_failure_has_no_retries_and_corrective_does_not_release_evidence():
    calls = []

    async def handle(request):
        calls.append(1)
        return httpx.Response(429)

    backend = JevBackend("mock-key", transport_factory=lambda: httpx.MockTransport(handle))
    retriever = CorrectivePolicyRetriever(
        Base([hit("A")]), JevEvidenceRanker(backend), EvidenceSufficiencyEvaluator(backend)
    )
    result = asyncio.run(retriever.retrieve("q", context=context()))
    assert calls == [1] and not result.hits
    assert result.judgment.status is Sufficiency.UNCERTAIN


def test_timeout_and_cancellation_close_call_without_retry():
    closed = []

    async def handle(request):
        try:
            await asyncio.sleep(5)
        finally:
            closed.append(True)

    backend = JevBackend(
        "mock-key", timeout=0.02, transport_factory=lambda: httpx.MockTransport(handle)
    )
    result = asyncio.run(JevEvidenceRanker(backend).arank("q", [hit("A")]))
    assert result.degraded and closed == [True]

    async def cancel():
        backend.timeout = 10
        task = asyncio.create_task(JevEvidenceRanker(backend).arank("q", [hit("A")]))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    assert closed == [True, True]


def test_date_category_source_and_conflict_are_deterministic():
    ctx = context()
    candidates = [
        hit("good"),
        hit("old", effective_to="2025-01-01"),
        hit("future", effective_from="2030-01-01"),
        hit("unknown", source=None),
        hit("wrong-category", categories=("食品",)),
        hit("bad-date", effective_to="bad"),
    ]
    eligible, conflict = eligible_evidence(candidates, ctx)
    assert [h.chunk.policy_id for h in eligible] == ["good"] and conflict is None
    _, conflict = eligible_evidence([hit("A"), hit("A", "不一样", version="v2")], ctx)
    assert conflict.status is Sufficiency.CONFLICTING


def test_missing_order_condition_does_not_let_model_guess():
    backend = StubBackend()
    judgment = asyncio.run(
        EvidenceSufficiencyEvaluator(backend).evaluate(
            replace(context(), days_since_delivery=None), [hit("quality", "保修")]
        )
    )
    assert judgment.status is Sufficiency.UNCERTAIN
    assert judgment.reason == "missing_order_date_condition"


def test_targeted_correction_adds_missing_quality_clause():
    backend = StubBackend()
    base = Base([hit("seven")], [hit("quality", "质量问题保修")])
    retriever = CorrectivePolicyRetriever(
        base, JevEvidenceRanker(backend), EvidenceSufficiencyEvaluator(backend)
    )
    result = asyncio.run(retriever.retrieve("七天无理由", context=context()))
    assert result.judgment.status is Sufficiency.SUFFICIENT and result.rounds == 2
    assert {h.chunk.policy_id for h in result.hits} == {"seven", "quality"}
    assert "质量问题" in base.queries[1]
    assert context().question in base.queries[1]


def test_no_new_evidence_and_unresolved_conflict_stop():
    backend = StubBackend()
    base = Base([hit("seven")])
    retriever = CorrectivePolicyRetriever(
        base, JevEvidenceRanker(backend), EvidenceSufficiencyEvaluator(backend)
    )
    result = asyncio.run(retriever.retrieve("q", context=context()))
    assert not result.hits and result.rounds == 2
    assert result.stop_reason == "no_new_effective_evidence"
    base = Base([hit("A"), hit("A", "conflict", version="v2")])
    result = asyncio.run(
        CorrectivePolicyRetriever(
            base, JevEvidenceRanker(backend), EvidenceSufficiencyEvaluator(backend)
        ).retrieve("q", context=context())
    )
    assert result.rounds == 1 and not result.hits
    assert result.judgment.status is Sufficiency.CONFLICTING


def test_evaluator_checks_only_returned_top_k_evidence():
    backend = StubBackend([0.9, 0.1])
    retriever = CorrectivePolicyRetriever(
        Base([hit("seven"), hit("quality", "保修")]),
        JevEvidenceRanker(backend),
        EvidenceSufficiencyEvaluator(backend),
        max_corrections=0,
    )
    result = asyncio.run(retriever.retrieve("q", context=context(), limit=1))
    assert not result.hits  # Relevant evidence exists below K, but cannot justify the returned set.


def test_concurrent_contexts_are_isolated():
    async def run():
        async def request(question):
            with evidence_context(context(question)):
                await asyncio.sleep(0.01)
                return current_evidence_context().question

        assert await asyncio.gather(request("A"), request("B")) == ["A", "B"]

    asyncio.run(run())
    assert current_evidence_context().question == ""


def test_original_configuration_is_identity_and_uncalibrated_mode_is_rejected(monkeypatch):
    original = object()
    monkeypatch.setenv("SERVICEPILOT_RAG_MODE", "original")
    assert configure_rag(original, [], None) is original
    monkeypatch.setenv("SERVICEPILOT_RAG_MODE", "jev_corrective")
    monkeypatch.delenv("SERVICEPILOT_JEV_CALIBRATION", raising=False)
    with pytest.raises(ValueError, match="calibration"):
        configure_rag(original, [], None)


def test_threshold_sweep_exposes_rare_evidence_loss_and_mock_cannot_enable_live(tmp_path):
    rows = [
        {"policy_id": "rare", "score": 0.1, "relevant": True, "critical": True},
        {"policy_id": "normal", "score": 0.9, "relevant": True, "critical": False},
        {"policy_id": "noise", "score": 0.2, "relevant": False, "critical": False},
    ]
    report = threshold_sweep(rows)
    assert any(r["critical_retention"] == 0 and "rare" in r["false_filtered_ids"] for r in report)
    path = tmp_path / "mock.json"
    path.write_text(json.dumps({"source": "mock", "corpus_sha256": "test"}))
    with pytest.raises(ValueError, match="live Jev"):
        load_calibration(path, corpus_hash="test", model="m", ranker_version="v")


def test_node_context_carries_original_question_and_observed_conditions():
    from types import SimpleNamespace

    from servicepilot.evidence import retrieval_node_context

    @retrieval_node_context
    def node(state):
        return current_evidence_context()

    with evidence_context(EvidenceContext(question="耳机质量问题")):
        result = node(
            {
                "task": SimpleNamespace(
                    observed_state={
                        "category": "数码产品",
                        "days_since_delivery": 60,
                        "amount": 999,
                        "user_id": "private-user",
                    }
                )
            }
        )
    assert result.question == "耳机质量问题"
    assert result.days_since_delivery == 60 and result.safe_empty
    assert not hasattr(result, "amount")


def test_legacy_path_uses_original_retrieval_when_empty_results_are_not_safe():
    backend = StubBackend()
    original = Base([hit("original")])
    retriever = CorrectivePolicyRetriever(
        Base([]), JevEvidenceRanker(backend), EvidenceSufficiencyEvaluator(backend)
    )
    retriever.fallback = original
    assert retriever.search("q")[0].chunk.policy_id == "original"
    assert backend.calls == 0


def test_sufficiency_low_confidence_is_not_released():
    class LowConfidence(StubBackend):
        async def decide(self, state, dimensions):
            response = await super().decide(state, dimensions)
            response["answers"]["sufficiency"]["confidence"] = 0.3
            return response

    backend = LowConfidence()
    result = asyncio.run(
        EvidenceSufficiencyEvaluator(backend).evaluate(context(), [hit("A", "保修")])
    )
    assert result.status is Sufficiency.UNCERTAIN


def test_ablation_unknown_metrics_and_wrong_filtering_are_not_hidden():
    from servicepilot.rag_evaluation import evaluate

    cases = [
        {
            "case_id": "one",
            "relevant_policy_ids": ["rare"],
            "critical_policy_ids": ["rare"],
            "irrelevant_policy_ids": ["noise"],
            "expected_sufficiency": "INSUFFICIENT",
        }
    ]
    report = evaluate(
        cases,
        [
            {
                "group": "B2",
                "case_id": "one",
                "source": "mock",
                "retrieved_ids": ["noise"],
                "candidate_ids": ["rare", "noise"],
                "filtered_ids": ["rare"],
                "sufficiency": "SUFFICIENT",
                "rounds": 1,
                "latency_ms": 2,
            }
        ],
    )
    assert report["B2"]["critical_retention"] == 0
    assert report["B2"]["sufficiency_accuracy"] == 0
    assert report["B2"]["answer_accuracy"] is None
    assert report["B2"]["usage"]["input_tokens"] is None
    assert report["B2"]["false_filtered_cases"] == [{"case_id": "one", "policy_ids": ["rare"]}]
    assert report["B3"]["status"] == "not_run" and report["B3"]["recall@5"] is None


def test_baseline_sources_remain_reproducible_and_new_labels_reference_real_policies():
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "evaluations/results/jev_stage1/offline_profile.json").read_text()
    )
    for name, digest in manifest["code_sha256"].items():
        assert (
            hashlib.sha256(
                (root / "evaluations/baselines/jev_stage1" / name).read_bytes()
            ).hexdigest()
            == digest
        )
    assert (
        hashlib.sha256((root / "evaluations/retrieval_challenge.json").read_bytes()).hexdigest()
        == manifest["cases_sha256"]
    )
    policy_ids = {p["policy_id"] for p in json.loads((root / "data/policies.json").read_text())}
    for case in json.loads((root / "evaluations/corrective_rag_cases.json").read_text()):
        assert set(case["relevant_policy_ids"] + case["irrelevant_policy_ids"]) <= policy_ids


def test_official_decision_payload_and_unknown_usage_are_preserved():
    async def handle(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/systemone"
        assert body["questions"]["sufficiency"]["type"] == "choice"
        assert "amount" not in body["state"]["conditions"]
        return httpx.Response(
            200,
            json={
                "model": "mock-jev",
                "answers": {
                    "sufficiency": {
                        "type": "choice",
                        "choice": "SUFFICIENT",
                        "confidence": 0.98,
                        "probabilities": {
                            s.value: float(s is Sufficiency.SUFFICIENT) for s in Sufficiency
                        },
                    },
                    **{
                        k: {"type": "noul", "noul": 0.99}
                        for k in body["questions"]
                        if k != "sufficiency"
                    },
                },
            },
        )

    backend = JevBackend("mock-key", transport_factory=lambda: httpx.MockTransport(handle))
    service = TraceService.create()
    with service.trace(user_id="A") as collector:
        judgment = asyncio.run(
            EvidenceSufficiencyEvaluator(backend).evaluate(context(), [hit("quality", "保修")])
        )
    assert judgment.status is Sufficiency.SUFFICIENT
    assert collector.record["total_llm_calls"] == 1
    assert collector.record["total_input_tokens"] is None


def test_new_round_cannot_accept_same_evidence_just_because_judge_changes_mind():
    class Unstable(StubBackend):
        decisions = 0

        async def decide(self, state, dimensions):
            self.decisions += 1
            if self.decisions > 1:
                state["policies"][0]["content"] = "保修"
            return await super().decide(state, dimensions)

    backend = Unstable()
    result = asyncio.run(
        CorrectivePolicyRetriever(
            Base([hit("seven")]), JevEvidenceRanker(backend), EvidenceSufficiencyEvaluator(backend)
        ).retrieve("q", context=context())
    )
    assert not result.hits and backend.decisions == 1
    assert result.stop_reason == "no_new_effective_evidence"


def test_maximum_two_corrections_even_if_new_evidence_keeps_arriving():
    from servicepilot.evidence import EvidenceJudgment

    class MoreEvidence:
        calls = 0

        def search(self, query, *, category=None, limit=3):
            self.calls += 1
            return [hit(str(self.calls))]

    class StillInsufficient:
        calls = 0

        async def evaluate(self, context, hits):
            self.calls += 1
            missing = ("quality_exception",) if self.calls == 1 else ("return_conditions",)
            return EvidenceJudgment(Sufficiency.INSUFFICIENT, missing)

    base = MoreEvidence()
    result = asyncio.run(
        CorrectivePolicyRetriever(
            base, JevEvidenceRanker(StubBackend()), StillInsufficient()
        ).retrieve("q", context=context())
    )
    assert result.rounds == base.calls == 3
    assert not result.hits


def test_two_order_conditions_do_not_share_sufficiency_results():
    backend = StubBackend()
    retriever = CorrectivePolicyRetriever(
        Base([hit("quality", "保修")]),
        JevEvidenceRanker(backend),
        EvidenceSufficiencyEvaluator(backend),
    )

    async def run():
        return await asyncio.gather(
            retriever.retrieve("same", context=context()),
            retriever.retrieve("same", context=replace(context(), days_since_delivery=None)),
        )

    complete, missing_date = asyncio.run(run())
    assert complete.hits and not missing_date.hits
    assert missing_date.judgment.status is Sufficiency.UNCERTAIN


def test_shared_http_concurrency_limit_across_requests_and_pointwise_batches():
    active = peak = 0

    async def handle(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            data = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "mock",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "answers": {key: {"type": "noul", "noul": 0.7} for key in data["questions"]},
                },
            )
        finally:
            active -= 1

    backend = JevBackend(
        "mock",
        mode="pointwise",
        max_concurrency=2,
        transport_factory=lambda: httpx.MockTransport(handle),
    )

    async def run():
        ranker = JevEvidenceRanker(backend)
        return await asyncio.gather(*[ranker.arank("q", [hit("A"), hit("B")]) for _ in range(4)])

    assert all(not result.degraded for result in asyncio.run(run()))
    assert peak == 2


def test_calibration_recomputed_and_bound_to_model_corpus_mode_and_holdout(tmp_path):
    # Synthetic fixture tests artifact validation only; it is not a release calibration.
    rows = []
    for i in range(20):
        rows.extend(
            [
                {
                    "case_id": f"cal-{i}",
                    "policy_id": f"critical-{i % 3}",
                    "score": 0.2,
                    "relevant": True,
                    "critical": True,
                },
                {
                    "case_id": f"cal-{i}",
                    "policy_id": "noise",
                    "score": 0.1,
                    "relevant": False,
                    "critical": False,
                },
            ]
        )
    artifact = {
        "artifact_id": "synthetic-validator-test",
        "source": "jev_live",
        "corpus_sha256": "corpus",
        "model": "mock",
        "config_version": "v1",
        "scoring_mode": "listwise",
        "library_version": "0.1.2",
        "confidence_floor": 0.9,
        "sufficiency_config_version": "sufficiency-v1",
        "observations": rows,
        "sufficiency_validation": [
            {
                "case_id": f"validation-{i}",
                "expected": list(Sufficiency)[i % 4].value,
                "predicted": list(Sufficiency)[i % 4].value,
            }
            for i in range(20)
        ],
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact))
    kwargs = dict(corpus_hash="corpus", model="mock", ranker_version="v1", corrective=True)
    calibration = load_calibration(path, **kwargs)
    assert calibration.threshold == 0.2 and len(calibration.protected_ids) == 3
    with pytest.raises(ValueError, match="disjoint"):
        load_calibration(path, frozen_ids=["cal-0"], **kwargs)
    with pytest.raises(ValueError, match="mode/library"):
        load_calibration(path, scoring_mode="pointwise", **kwargs)
    artifact["sufficiency_validation"][1]["predicted"] = "SUFFICIENT"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="validation failed"):
        load_calibration(path, **kwargs)


def test_dynamic_graph_uses_empty_evidence_safety_without_changing_tool_rules(tmp_path):
    from pathlib import Path

    from servicepilot.cli import build_agent

    root = Path(__file__).resolve().parents[1]
    runtime = build_agent(root / "data", session_db=tmp_path / "business.sqlite3", dynamic=True)
    backend = StubBackend()
    retriever = CorrectivePolicyRetriever(
        Base([]), JevEvidenceRanker(backend), EvidenceSufficiencyEvaluator(backend)
    )
    original = Base([hit("original")])
    retriever.fallback = original
    runtime.agent.retriever = retriever
    response = runtime.chat(user_id="USER-001", message="退掉 ORD-1001")
    assert "人工" in response.message
    assert not original.queries  # Dynamic node is known to handle empty evidence safely.
    assert backend.calls == 0
