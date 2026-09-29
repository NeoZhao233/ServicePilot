from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from typing import Any, Protocol

from .models import Intent, RoutingDecision, UrgencyLevel
from .retrieval import HashingTextEncoder


class IntentRouter(Protocol):
    def route(self, message: str) -> RoutingDecision: ...


class StructuredIntentModel(Protocol):
    """Provider-neutral interface for a future structured-output LLM client."""

    def classify_intent(self, message: str) -> Mapping[str, Any]: ...


class RuleIntentRouter:
    """Auditable routing baseline used for fallback and offline evaluation."""

    SIGNALS: dict[Intent, tuple[tuple[str, float], ...]] = {
        Intent.CANCEL: (
            ("算了", 1.0),
            ("取消", 1.0),
            ("不用了", 1.0),
            ("不退了", 1.0),
        ),
        Intent.HUMAN_HANDOFF: (
            ("人工", 1.0),
            ("客服人员", 1.0),
            ("转接", 0.9),
            ("投诉", 0.7),
        ),
        Intent.DAMAGED_ITEM: (
            ("破损", 1.0),
            ("坏了", 1.0),
            ("故障", 1.0),
            ("不能用", 0.9),
            ("质量问题", 1.0),
            ("碎了", 1.0),
        ),
        Intent.REFUND: (
            ("退款", 1.0),
            ("退货", 1.0),
            ("不要了", 0.9),
            ("退掉", 0.9),
            ("买错", 0.7),
        ),
        Intent.LOGISTICS: (
            ("物流", 1.0),
            ("快递", 1.0),
            ("到哪", 0.9),
            ("发货", 0.8),
            ("送到", 0.8),
            ("签收", 0.6),
        ),
        Intent.EXCHANGE: (("换货", 1.0), ("换一个", 0.9), ("换新", 0.9)),
        Intent.INVOICE: (("发票", 1.0), ("开票", 1.0), ("抬头", 0.8)),
        Intent.PAYMENT: (("支付", 1.0), ("扣款", 1.0), ("付款", 0.9)),
        Intent.MISSING_ITEM: (("少件", 1.0), ("漏发", 1.0), ("东西少了", 1.0)),
        Intent.ADDRESS_CHANGE: (("改地址", 1.0), ("收货地址", 0.9), ("地址错", 1.0)),
        Intent.PRICE_PROTECTION: (("保价", 1.0), ("降价", 0.9), ("退差价", 1.0)),
    }

    def route(self, message: str) -> RoutingDecision:
        scores: dict[Intent, float] = defaultdict(float)
        matches: dict[Intent, list[str]] = defaultdict(list)
        lowered = message.lower()
        for intent, signals in self.SIGNALS.items():
            for signal, weight in signals:
                if signal in lowered:
                    scores[intent] += weight
                    matches[intent].append(signal)

        if not scores:
            return RoutingDecision(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                urgency=detect_urgency(message),
            )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_intent, best_score = ranked[0]
        total = sum(scores.values())
        dominance = best_score / total if total else 0.0
        coverage_bonus = min(0.15, 0.05 * (len(matches[best_intent]) - 1))
        confidence = min(0.99, 0.55 + 0.4 * dominance + coverage_bonus)
        alternatives = tuple((intent, round(score / total, 3)) for intent, score in ranked[1:])
        return RoutingDecision(
            intent=best_intent,
            confidence=round(confidence, 3),
            matched_signals=tuple(matches[best_intent]),
            alternatives=alternatives,
            urgency=detect_urgency(message),
        )


def detect_urgency(message: str) -> UrgencyLevel:
    if any(signal in message for signal in ("人身安全", "起火", "爆炸", "触电", "受伤")):
        return UrgencyLevel.CRITICAL
    if any(signal in message for signal in ("欺诈", "盗刷", "马上投诉", "非常紧急")):
        return UrgencyLevel.HIGH
    if any(signal in message for signal in ("尽快", "着急", "今天必须")):
        return UrgencyLevel.NORMAL
    return UrgencyLevel.LOW


class EmbeddingIntentRouter:
    """Similarity route over curated intent exemplars."""

    EXEMPLARS: dict[Intent, tuple[str, ...]] = {
        intent: tuple(signal for signal, _ in signals)
        for intent, signals in RuleIntentRouter.SIGNALS.items()
        if intent is not Intent.CANCEL
    }

    def __init__(self, encoder=None) -> None:
        self.encoder = encoder or HashingTextEncoder()
        self._labels = list(self.EXEMPLARS)
        texts = [" ".join(self.EXEMPLARS[label]) for label in self._labels]
        self._vectors = self.encoder.encode(texts)

    def route(self, message: str) -> RoutingDecision:
        query = self.encoder.encode([message])[0]
        scores = [
            sum(left * right for left, right in zip(query, vector, strict=True))
            for vector in self._vectors
        ]
        ranked = sorted(
            zip(self._labels, scores, strict=True), key=lambda item: item[1], reverse=True
        )
        best_intent, best_score = ranked[0]
        if best_score <= 0:
            return RoutingDecision(Intent.UNKNOWN, 0.0, urgency=detect_urgency(message))
        return RoutingDecision(
            best_intent,
            min(0.99, best_score),
            matched_signals=("embedding_similarity",),
            alternatives=tuple(ranked[1:4]),
            urgency=detect_urgency(message),
        )


class CachedVotingIntentRouter:
    """Weighted voting over independent routes with a bounded LRU cache."""

    def __init__(
        self,
        routes: tuple[tuple[str, IntentRouter, float], ...],
        *,
        cache_capacity: int = 512,
        minimum_confidence: float = 0.35,
    ) -> None:
        if not routes or sum(weight for _, _, weight in routes) <= 0:
            raise ValueError("at least one positive-weight route is required")
        self.routes = routes
        self.cache_capacity = cache_capacity
        self.minimum_confidence = minimum_confidence
        self._cache: OrderedDict[str, RoutingDecision] = OrderedDict()

    def route(self, message: str) -> RoutingDecision:
        cache_key = re.sub(r"\s+", " ", message.strip().lower())
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            cached = self._cache[cache_key]
            return RoutingDecision(
                cached.intent,
                cached.confidence,
                (*cached.matched_signals, "lru_cache_hit"),
                cached.alternatives,
                cached.urgency,
            )
        scores: dict[Intent, float] = defaultdict(float)
        signals: list[str] = []
        urgency = UrgencyLevel.LOW
        active_weight = 0.0
        for name, route, weight in self.routes:
            try:
                decision = route.route(message)
            except Exception:
                signals.append(f"{name}:error")
                continue
            signals.append(f"{name}:{decision.intent.value}")
            if decision.intent is not Intent.UNKNOWN:
                scores[decision.intent] += weight * decision.confidence
                active_weight += weight
            if list(UrgencyLevel).index(decision.urgency) > list(UrgencyLevel).index(urgency):
                urgency = decision.urgency
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        normalized_best = ranked[0][1] / active_weight if ranked and active_weight else 0.0
        if active_weight < 0.3 or not ranked or normalized_best < self.minimum_confidence:
            result = RoutingDecision(Intent.UNKNOWN, 0.0, tuple(signals), urgency=urgency)
        else:
            best_intent, _ = ranked[0]
            result = RoutingDecision(
                best_intent,
                round(min(0.99, normalized_best), 3),
                tuple(signals),
                tuple((intent, round(score / active_weight, 3)) for intent, score in ranked[1:]),
                urgency,
            )
        self._cache[cache_key] = result
        self._cache.move_to_end(cache_key)
        if len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)
        return result


class LLMIntentRouter:
    """Validate model output before it is allowed into the business workflow."""

    def __init__(self, model: StructuredIntentModel, *, minimum_confidence: float = 0.6) -> None:
        self.model = model
        self.minimum_confidence = minimum_confidence

    def route(self, message: str) -> RoutingDecision:
        raw = self.model.classify_intent(message)
        try:
            intent = Intent(str(raw["intent"]))
            confidence = float(raw["confidence"])
        except (KeyError, TypeError, ValueError):
            return RoutingDecision(intent=Intent.UNKNOWN, confidence=0.0)
        confidence = max(0.0, min(1.0, confidence))
        if confidence < self.minimum_confidence:
            return RoutingDecision(
                intent=Intent.UNKNOWN,
                confidence=confidence,
                matched_signals=("llm_below_threshold",),
            )
        return RoutingDecision(
            intent=intent,
            confidence=confidence,
            matched_signals=("llm_structured_output",),
            urgency=detect_urgency(message),
        )


class FallbackIntentRouter:
    """Use a primary router, then fall back when it cannot decide confidently."""

    def __init__(
        self,
        primary: IntentRouter,
        fallback: IntentRouter,
        *,
        threshold: float = 0.6,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.threshold = threshold

    def route(self, message: str) -> RoutingDecision:
        try:
            primary = self.primary.route(message)
        except Exception:
            primary = RoutingDecision(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                matched_signals=("primary_error",),
            )
        if primary.intent is not Intent.UNKNOWN and primary.confidence >= self.threshold:
            return primary
        fallback = self.fallback.route(message)
        return RoutingDecision(
            intent=fallback.intent,
            confidence=fallback.confidence,
            matched_signals=("fallback", *primary.matched_signals, *fallback.matched_signals),
            alternatives=fallback.alternatives,
            urgency=fallback.urgency,
        )
