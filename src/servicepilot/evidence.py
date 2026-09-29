"""Evidence conditions and conservative sufficiency, separate from commerce authorization."""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import date
from enum import StrEnum

from .jev import JevValidationError, probability
from .observability import Span, event


class Sufficiency(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    CONFLICTING = "CONFLICTING"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class EvidenceContext:
    question: str = ""
    category: str | None = None
    days_since_delivery: int | None = None
    as_of: str | None = None
    trusted_sources: tuple[str, ...] = ()
    safe_empty: bool = False


_REQUEST = contextvars.ContextVar("rag_request_context", default=None)


@contextmanager
def evidence_context(context):
    token = _REQUEST.set(context)
    try:
        yield context
    finally:
        _REQUEST.reset(token)


def current_evidence_context():
    return _REQUEST.get() or EvidenceContext()


def retrieval_request(fn):
    @functools.wraps(fn)
    def call(self, *args, **kwargs):
        from inspect import signature

        arguments = signature(fn).bind(self, *args, **kwargs).arguments
        question = arguments.get("message") or current_evidence_context().question
        with evidence_context(EvidenceContext(question=question)):
            return fn(self, *args, **kwargs)

    return call


def corpus_fingerprint(chunks):
    values = sorted((asdict(c) for c in chunks), key=lambda c: c["policy_id"])
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class EvidenceJudgment:
    status: Sufficiency
    missing: tuple[str, ...] = ()
    reason: str = ""
    confidence: float | None = None
    covered_questions: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    supporting_ids: tuple[str, ...] = ()
    provider: str | None = None
    model: str | None = None


DIMENSIONS = {
    "quality_exception": "Are quality defects covered after opening or return expiry?",
    "return_conditions": "Do the policies cover applicable return conditions and exclusions?",
    "refund_timing": "Are refund arrival time and original payment handling both covered?",
    "applicability": "Can applicability be established for the supplied category and conditions?",
}
REWRITES = {
    "quality_exception": "质量问题 保修 拆封 超过七天 特殊退货条款",
    "return_conditions": "退货条件 无理由 排除商品 包装 吊牌",
    "refund_timing": "退款到账时效 原支付账户 注销 银行卡",
    "applicability": "政策适用商品类别 生效日期 政策版本",
}


def eligible_evidence(hits, context, *, trusted_chunks=()):
    """Unknown provenance is not silently converted into authoritative evidence."""
    eligible, rejected = [], []
    trusted = {c.policy_id: c for c in trusted_chunks}
    seen = {}
    try:
        today = date.fromisoformat(context.as_of) if context.as_of else date.today()
    except (TypeError, ValueError):
        return [], EvidenceJudgment(Sufficiency.UNCERTAIN, ("applicability",), "invalid_date")
    for hit in hits:
        c = hit.chunk
        if c.categories and (not context.category or context.category not in c.categories):
            rejected.append(c.policy_id)
            continue
        local_snapshot = trusted.get(c.policy_id) == c
        if not local_snapshot and (
            not c.source or c.source not in context.trusted_sources or not c.version
        ):
            rejected.append(c.policy_id)
            continue
        try:
            start = date.fromisoformat(c.effective_from) if c.effective_from else None
            end = date.fromisoformat(c.effective_to) if c.effective_to else None
        except (TypeError, ValueError):
            rejected.append(c.policy_id)
            continue
        if (start is None and not local_snapshot) or (
            start is not None and end is not None and end < start
        ):
            rejected.append(c.policy_id)
            continue
        if (start is not None and today < start) or (end is not None and today > end):
            rejected.append(c.policy_id)
            continue
        old = seen.get(c.policy_id)
        if old and (old.version != c.version or old.content != c.content):
            return [], EvidenceJudgment(
                Sufficiency.CONFLICTING, ("applicability",), "overlapping_policy_versions"
            )
        seen[c.policy_id] = c
        eligible.append(hit)
    if rejected:
        event("EVIDENCE_METADATA_REJECTED", policy_ids=rejected)
    return eligible, None


class EvidenceSufficiencyEvaluator:
    CONFIG_VERSION = "sufficiency-v1"

    def __init__(self, backend, *, confidence_floor=0.9):
        self.backend = backend
        self.confidence_floor = probability(confidence_floor)

    async def evaluate(self, context, hits):
        with Span("rag.evidence_check", "rag", inputs={"query": context.question}) as span:
            if not hits:
                result = EvidenceJudgment(
                    Sufficiency.INSUFFICIENT, ("applicability",), "no_authoritative_evidence"
                )
            elif not context.question.strip():
                result = EvidenceJudgment(Sufficiency.UNCERTAIN, (), "missing_original_question")
            elif any(word in context.question for word in ("超过", "七天", "过期", "两个月")) and (
                type(context.days_since_delivery) is not int or context.days_since_delivery < 0
            ):
                result = EvidenceJudgment(
                    Sufficiency.UNCERTAIN, ("applicability",), "missing_order_date_condition"
                )
            else:
                dimensions = {"applicability": DIMENSIONS["applicability"]}
                for key, words in {
                    "quality_exception": ("质量", "故障", "坏", "破损"),
                    "return_conditions": ("退货", "退款", "拆封", "七天"),
                    "refund_timing": ("到账", "注销", "银行卡", "钱没到"),
                }.items():
                    if any(word in context.question for word in words):
                        dimensions[key] = DIMENSIONS[key]
                try:
                    response = await self.backend.decide(
                        {
                            "question": context.question,
                            "conditions": {
                                k: v
                                for k, v in asdict(context).items()
                                if k not in {"question", "trusted_sources", "safe_empty"}
                            },
                            "policies": [asdict(h.chunk) for h in hits],
                        },
                        dimensions,
                    )
                    answers = response["answers"]
                    if set(answers) != {"sufficiency", *dimensions}:
                        raise JevValidationError("Decision answers do not match requested keys")
                    answer = answers["sufficiency"]
                    if answer["type"] != "choice":
                        raise JevValidationError("Expected Choice answer")
                    status = Sufficiency(answer["choice"])
                    confidence = probability(answer["confidence"])
                    probabilities = answer["probabilities"]
                    if set(probabilities) != {s.value for s in Sufficiency}:
                        raise JevValidationError("Incomplete Choice probabilities")
                    values = {k: probability(v) for k, v in probabilities.items()}
                    if abs(sum(values.values()) - 1) > 0.02:
                        raise JevValidationError("Invalid probability distribution")
                    if values[status] < max(values.values()):
                        raise JevValidationError("Choice does not match distribution")
                    missing = []
                    for key in dimensions:
                        dimension = answers[key]
                        if dimension["type"] != "noul":
                            raise JevValidationError("Expected Noul answer")
                        if probability(dimension["noul"]) < self.confidence_floor:
                            missing.append(key)
                    if confidence < self.confidence_floor:
                        status = Sufficiency.UNCERTAIN
                    elif status is Sufficiency.SUFFICIENT and missing:
                        status = Sufficiency.INSUFFICIENT
                    result = EvidenceJudgment(
                        status, tuple(missing), "model_evaluation", confidence
                    )
                except Exception as exc:
                    span.failed(type(exc).__name__)
                    result = EvidenceJudgment(Sufficiency.UNCERTAIN, (), "decision_failed")
            span.output(result)
            event(
                "EVIDENCE_JUDGED",
                verdict=result.status,
                missing=list(result.missing),
                reason_code=result.reason,
                confidence=result.confidence,
            )
            return result


def retrieval_node_context(fn):
    @functools.wraps(fn)
    def call(state):
        context = current_evidence_context()
        task = state.get("task")
        order = getattr(task, "observed_state", None) or state.get("order") or {}
        updated = replace(
            context,
            safe_empty=bool(task),
            category=order.get("category", context.category),
            days_since_delivery=order.get("days_since_delivery", context.days_since_delivery),
        )
        with evidence_context(updated):
            return fn(state)

    return call
