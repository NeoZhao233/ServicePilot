import os
import unittest
from unittest.mock import patch

from servicepilot.models import PolicyChunk, RetrievalHit
from servicepilot.providers import (
    APIEmbeddingEncoder,
    OnlineModelSettings,
    ProviderError,
    ResponsesDialogueJudge,
    ResponsesIntentModel,
    ResponsesPolicyReranker,
)


class FakeTransport:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post(self, path: str, payload):
        self.calls.append((path, dict(payload)))
        return self.response


class FlakyTransport(FakeTransport):
    def __init__(self, response: dict[str, object]) -> None:
        super().__init__(response)
        self.attempts = 0

    def post(self, path: str, payload):
        from servicepilot.providers import ProviderError

        self.attempts += 1
        if self.attempts == 1:
            raise ProviderError("无法连接模型服务：timeout")
        return super().post(path, payload)


class ProviderTests(unittest.TestCase):
    def test_dialogue_judge_uses_disclosed_strict_rubric(self) -> None:
        transport = FakeTransport(
            {
                "output_text": (
                    '{"correctness":5,"groundedness":4,"safety":5,"coherence":5,"reason":"ok"}'
                )
            }
        )

        result = ResponsesDialogueJudge(transport, "judge-model").judge(
            user_messages=["查物流"],
            assistant_messages=["运输中"],
            called_tools=["get_order"],
        )

        self.assertEqual(result["safety"], 5)
        self.assertTrue(transport.calls[0][1]["text"]["format"]["strict"])

    def test_dialogue_judge_retries_transient_provider_failure(self) -> None:
        transport = FlakyTransport(
            {
                "output_text": (
                    '{"correctness":5,"groundedness":5,"safety":5,"coherence":5,"reason":"ok"}'
                )
            }
        )

        result = ResponsesDialogueJudge(transport, "judge-model").judge(
            user_messages=["查物流"],
            assistant_messages=["运输中"],
            called_tools=["get_order"],
        )

        self.assertEqual(result["correctness"], 5)
        self.assertEqual(transport.attempts, 2)

    def test_responses_model_requests_strict_structured_output(self) -> None:
        transport = FakeTransport(
            {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"intent":"refund","confidence":0.91}',
                            }
                        ]
                    }
                ]
            }
        )
        result = ResponsesIntentModel(transport, "intent-model").classify_intent("想退货")
        self.assertEqual(result["intent"], "refund")
        path, payload = transport.calls[0]
        self.assertEqual(path, "/responses")
        self.assertFalse(payload["store"])
        self.assertTrue(payload["text"]["format"]["strict"])

    def test_embedding_vectors_are_sorted_by_index(self) -> None:
        transport = FakeTransport(
            {
                "data": [
                    {"index": 1, "embedding": [0, 1]},
                    {"index": 0, "embedding": [1, 0]},
                ]
            }
        )
        vectors = APIEmbeddingEncoder(transport, "embedding-model").encode(["a", "b"])
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(transport.calls[0][0], "/embeddings")

    def test_embedding_count_must_match_input_count(self) -> None:
        transport = FakeTransport({"data": []})
        with self.assertRaises(ProviderError):
            APIEmbeddingEncoder(transport, "embedding-model").encode(["text"])

    def test_online_settings_do_not_require_openai_specific_variable_names(self) -> None:
        environment = {
            "SERVICEPILOT_API_KEY": "secret",
            "SERVICEPILOT_INTENT_MODEL": "intent-model",
            "SERVICEPILOT_EMBEDDING_MODEL": "embedding-model",
            "SERVICEPILOT_BASE_URL": "https://example.invalid/v1",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = OnlineModelSettings.from_env()
        self.assertEqual(settings.base_url, "https://example.invalid/v1")
        self.assertEqual(settings.api_key, "secret")

    def test_deepseek_style_settings_allow_local_embedding_mode(self) -> None:
        environment = {
            "SERVICEPILOT_API_KEY": "secret",
            "SERVICEPILOT_BASE_URL": "https://api.deepseek.com",
            "SERVICEPILOT_INTENT_MODEL": "deepseek-flash",
            "SERVICEPILOT_EMBEDDING_MODE": "local",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = OnlineModelSettings.from_env()

        self.assertEqual(settings.embedding_mode, "local")
        self.assertEqual(settings.embedding_model, "BAAI/bge-small-zh-v1.5")

    def test_policy_reranker_only_uses_candidate_ids(self) -> None:
        transport = FakeTransport(
            {"output_text": '{"ordered_policy_ids":["P2","P1","INVENTED"]}'}
        )
        candidates = [
            RetrievalHit(PolicyChunk("P1", "退货", "七天退货"), 0.8, ()),
            RetrievalHit(PolicyChunk("P2", "退款", "到账时效"), 0.7, ()),
        ]

        result = ResponsesPolicyReranker(transport, "deepseek").rerank(
            "退款多久到账", candidates, limit=2
        )

        self.assertEqual([hit.chunk.policy_id for hit in result], ["P2", "P1"])
        self.assertTrue(transport.calls[0][1]["text"]["format"]["strict"])


if __name__ == "__main__":
    unittest.main()
