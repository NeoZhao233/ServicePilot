import unittest

from servicepilot.models import Intent, UrgencyLevel
from servicepilot.routing import (
    CachedVotingIntentRouter,
    EmbeddingIntentRouter,
    FallbackIntentRouter,
    LLMIntentRouter,
    RuleIntentRouter,
)


class FakeModel:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def classify_intent(self, message: str) -> dict[str, object]:
        return self.payload


class BrokenModel:
    def classify_intent(self, message: str):
        raise ConnectionError("model unavailable")


class FixedRouter:
    def __init__(self, decision):
        self.decision = decision

    def route(self, message):
        return self.decision


class RoutingTests(unittest.TestCase):
    def test_rule_router_returns_evidence(self) -> None:
        result = RuleIntentRouter().route("我的耳机坏了，完全不能用")
        self.assertEqual(result.intent, Intent.DAMAGED_ITEM)
        self.assertGreater(result.confidence, 0.8)
        self.assertIn("坏了", result.matched_signals)

    def test_cancel_has_priority_over_pending_business_words(self) -> None:
        result = RuleIntentRouter().route("算了，我不退款了")
        self.assertEqual(result.intent, Intent.CANCEL)

    def test_llm_router_rejects_invalid_structured_output(self) -> None:
        result = LLMIntentRouter(FakeModel({"label": "refund"})).route("我要退款")
        self.assertEqual(result.intent, Intent.UNKNOWN)

    def test_llm_router_applies_confidence_threshold(self) -> None:
        model = FakeModel({"intent": "refund", "confidence": 0.2})
        result = LLMIntentRouter(model, minimum_confidence=0.6).route("可能想退")
        self.assertEqual(result.intent, Intent.UNKNOWN)

    def test_fallback_router_uses_rule_router(self) -> None:
        primary = LLMIntentRouter(FakeModel({"intent": "unknown", "confidence": 0.1}))
        result = FallbackIntentRouter(primary, RuleIntentRouter()).route("查一下快递")
        self.assertEqual(result.intent, Intent.LOGISTICS)
        self.assertEqual(result.matched_signals[0], "fallback")

    def test_fallback_router_contains_provider_failure(self) -> None:
        primary = LLMIntentRouter(BrokenModel())
        result = FallbackIntentRouter(primary, RuleIntentRouter()).route("查一下快递")
        self.assertEqual(result.intent, Intent.LOGISTICS)
        self.assertIn("primary_error", result.matched_signals)

    def test_ten_business_intents_are_covered(self) -> None:
        business_intents = set(RuleIntentRouter.SIGNALS) - {Intent.CANCEL}
        self.assertEqual(len(business_intents), 10)

    def test_urgency_has_four_levels_and_detects_safety_incident(self) -> None:
        result = RuleIntentRouter().route("充电器起火了，我要退款")
        self.assertEqual(len(UrgencyLevel), 4)
        self.assertEqual(result.urgency, UrgencyLevel.CRITICAL)

    def test_weighted_router_uses_lru_cache(self) -> None:
        router = CachedVotingIntentRouter(
            (
                ("keyword", RuleIntentRouter(), 0.8),
                ("embedding", EmbeddingIntentRouter(), 0.2),
            ),
            cache_capacity=2,
        )

        first = router.route("帮我查快递")
        second = router.route("帮我查快递")

        self.assertEqual(first.intent, Intent.LOGISTICS)
        self.assertIn("lru_cache_hit", second.matched_signals)

    def test_abstaining_primary_does_not_suppress_keyword_fallback(self) -> None:
        from servicepilot.models import RoutingDecision

        router = CachedVotingIntentRouter(
            (
                ("llm", FixedRouter(RoutingDecision(Intent.UNKNOWN, 0.9)), 0.5),
                ("embedding", FixedRouter(RoutingDecision(Intent.PAYMENT, 0.2)), 0.2),
                ("keyword", FixedRouter(RoutingDecision(Intent.DAMAGED_ITEM, 0.95)), 0.3),
            )
        )

        result = router.route("完全不能用")

        self.assertEqual(result.intent, Intent.DAMAGED_ITEM)

    def test_embedding_route_cannot_decide_without_another_route(self) -> None:
        from servicepilot.models import RoutingDecision

        router = CachedVotingIntentRouter(
            (
                ("llm", FixedRouter(RoutingDecision(Intent.UNKNOWN, 0.9)), 0.5),
                ("embedding", FixedRouter(RoutingDecision(Intent.REFUND, 0.9)), 0.2),
                ("keyword", FixedRouter(RoutingDecision(Intent.UNKNOWN, 0.0)), 0.3),
            )
        )

        self.assertEqual(router.route("你好").intent, Intent.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
