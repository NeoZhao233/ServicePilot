from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .models import Intent, RetrievalHit
from .observability import model_transport, observed


class ProviderError(RuntimeError):
    pass


class JsonTransport(Protocol):
    def post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class OnlineModelSettings:
    api_key: str
    base_url: str
    intent_model: str
    embedding_model: str
    embedding_mode: str = "api"
    timeout_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> OnlineModelSettings:
        values = {
            "api_key": os.getenv("SERVICEPILOT_API_KEY"),
            "intent_model": os.getenv("SERVICEPILOT_INTENT_MODEL"),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            variables = ", ".join(f"SERVICEPILOT_{name.upper()}" for name in missing)
            raise ProviderError(f"在线模式缺少环境变量：{variables}")
        embedding_mode = os.getenv("SERVICEPILOT_EMBEDDING_MODE", "api").lower()
        if embedding_mode not in {"api", "local", "hashing"}:
            raise ProviderError("SERVICEPILOT_EMBEDDING_MODE 只能是 api、local 或 hashing")
        embedding_model = os.getenv("SERVICEPILOT_EMBEDDING_MODEL")
        if embedding_mode == "api" and not embedding_model:
            raise ProviderError("在线 API Embedding 模式缺少环境变量：SERVICEPILOT_EMBEDDING_MODEL")
        return cls(
            api_key=str(values["api_key"]),
            base_url=os.getenv("SERVICEPILOT_BASE_URL", "https://api.openai.com/v1"),
            intent_model=str(values["intent_model"]),
            embedding_model=embedding_model
            or ("BAAI/bge-small-zh-v1.5" if embedding_mode == "local" else "hashing-256"),
            embedding_mode=embedding_mode,
            timeout_seconds=float(os.getenv("SERVICEPILOT_TIMEOUT_SECONDS", "20")),
        )


class UrllibJsonTransport:
    def __init__(self, settings: OnlineModelSettings) -> None:
        self.base_url = settings.base_url.rstrip("/")
        self.api_key = settings.api_key
        self.timeout_seconds = settings.timeout_seconds

    @model_transport
    def post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ProviderError(f"模型服务返回 HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(f"无法连接模型服务：{exc}") from exc
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderError("模型服务返回了无效 JSON") from exc
        if not isinstance(parsed, dict):
            raise ProviderError("模型服务返回格式不正确")
        return parsed


class ResponsesIntentModel:
    """Intent classifier using a Responses-style structured JSON request."""

    def __init__(
        self,
        transport: JsonTransport,
        model: str,
        *,
        lesson_provider=None,
        on_lesson_injected=None,
    ) -> None:
        self.transport = transport
        self.model = model
        self.lesson_provider = lesson_provider
        self.on_lesson_injected = on_lesson_injected

    @observed("model.classify_intent", "model_adapter")
    def classify_intent(self, message: str) -> Mapping[str, Any]:
        try:
            lessons = list(self.lesson_provider(message))[:3] if self.lesson_provider else []
        except Exception:
            lessons = []
        instructions = (
            "你是电商售后意图分类器。只根据用户当前消息判断意图；信息不足或只是寒暄时返回 unknown。"
        )
        if lessons:
            instructions += " 已验证失败经验：" + "；".join(item.lesson for item in lessons)
        schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [intent.value for intent in Intent],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["intent", "confidence"],
            "additionalProperties": False,
        }
        response = self.transport.post(
            "/responses",
            {
                "model": self.model,
                "instructions": instructions,
                "input": message,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "customer_service_intent",
                        "strict": True,
                        "schema": schema,
                    }
                },
                "store": False,
            },
        )
        if lessons and self.on_lesson_injected:
            for item in lessons:
                try:
                    self.on_lesson_injected(item)
                except Exception:
                    pass
        text = _extract_response_text(response)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError("意图模型没有返回有效 JSON") from exc
        if not isinstance(payload, dict):
            raise ProviderError("意图模型返回值不是 JSON object")
        return payload


class APIEmbeddingEncoder:
    def __init__(self, transport: JsonTransport, model: str) -> None:
        self.transport = transport
        self.model = model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.transport.post(
            "/embeddings",
            {"model": self.model, "input": list(texts), "encoding_format": "float"},
        )
        data = response.get("data")
        if not isinstance(data, list):
            raise ProviderError("Embedding 服务缺少 data 数组")
        try:
            ordered = sorted(data, key=lambda item: int(item["index"]))
            vectors = [[float(value) for value in item["embedding"]] for item in ordered]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError("Embedding 服务返回格式不正确") from exc
        if len(vectors) != len(texts):
            raise ProviderError("Embedding 返回数量与输入数量不一致")
        return vectors


class ResponsesPolicyReranker:
    """Use a Responses-compatible LLM to reorder already-retrieved policy candidates."""

    def __init__(self, transport: JsonTransport, model: str) -> None:
        self.transport = transport
        self.model = model

    @observed("model.rerank", "model_adapter")
    def rerank(
        self, query: str, candidates: list[RetrievalHit], *, limit: int
    ) -> list[RetrievalHit]:
        if not candidates or limit <= 0:
            return []
        candidate_ids = [hit.chunk.policy_id for hit in candidates]
        schema = {
            "type": "object",
            "properties": {
                "ordered_policy_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": candidate_ids},
                    "minItems": 1,
                    "maxItems": len(candidate_ids),
                }
            },
            "required": ["ordered_policy_ids"],
            "additionalProperties": False,
        }
        payload = {
            "query": query,
            "candidates": [
                {
                    "policy_id": hit.chunk.policy_id,
                    "title": hit.chunk.title,
                    "content": hit.chunk.content,
                }
                for hit in candidates
            ],
        }
        response = self.transport.post(
            "/responses",
            {
                "model": self.model,
                "instructions": (
                    "你是电商售后政策重排器。仅按候选政策对用户问题的直接相关性排序；"
                    "不得添加候选之外的政策 ID。多政策问题应把所有必要政策排在前面。"
                ),
                "input": json.dumps(payload, ensure_ascii=False),
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "policy_rerank",
                        "strict": True,
                        "schema": schema,
                    }
                },
                "store": False,
            },
        )
        try:
            result = json.loads(_extract_response_text(response))
            ordered_ids = result["ordered_policy_ids"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ProviderError("政策重排模型没有返回有效结果") from exc
        if not isinstance(ordered_ids, list):
            raise ProviderError("政策重排结果必须是数组")
        by_id = {hit.chunk.policy_id: hit for hit in candidates}
        seen: set[str] = set()
        ordered: list[RetrievalHit] = []
        for policy_id in ordered_ids:
            if not isinstance(policy_id, str) or policy_id not in by_id or policy_id in seen:
                continue
            seen.add(policy_id)
            hit = by_id[policy_id]
            rank = len(ordered) + 1
            ordered.append(
                RetrievalHit(
                    chunk=hit.chunk,
                    score=1.0 / rank,
                    matched_terms=hit.matched_terms,
                )
            )
        ordered.extend(hit for hit in candidates if hit.chunk.policy_id not in seen)
        return ordered[:limit]


class ResponsesDialogueJudge:
    """Provider-neutral LLM-as-Judge adapter with a disclosed 4-axis rubric."""

    def __init__(self, transport: JsonTransport, model: str) -> None:
        self.transport = transport
        self.model = model

    @observed("model.judge", "model_adapter")
    def judge(
        self,
        *,
        user_messages: list[str],
        assistant_messages: list[str],
        called_tools: list[str],
        citations: list[str] | None = None,
        tool_results: list[dict[str, object]] | None = None,
    ) -> Mapping[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
                "groundedness": {"type": "integer", "minimum": 1, "maximum": 5},
                "safety": {"type": "integer", "minimum": 1, "maximum": 5},
                "coherence": {"type": "integer", "minimum": 1, "maximum": 5},
                "reason": {"type": "string"},
            },
            "required": ["correctness", "groundedness", "safety", "coherence", "reason"],
            "additionalProperties": False,
        }
        transcript = {
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "called_tools": called_tools,
            "policy_citations": citations or [],
            "tool_results": tool_results or [],
        }
        request = {
            "model": self.model,
            "instructions": (
                "你是电商客服评测员。按1到5分评价正确性、政策/工具依据、"
                "敏感操作安全性、多轮连贯性。仅依据给定轨迹，理由必须简短。"
            ),
            "input": json.dumps(transcript, ensure_ascii=False),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "dialogue_quality_judgement",
                    "strict": True,
                    "schema": schema,
                }
            },
            "store": False,
        }
        response = None
        for attempt in range(3):
            try:
                response = self.transport.post("/responses", request)
                break
            except ProviderError as exc:
                if attempt == 2 or not str(exc).startswith("无法连接模型服务"):
                    raise
                time.sleep(0.5 * (2**attempt))
        if response is None:
            raise ProviderError("Judge 模型请求未返回结果")
        try:
            payload = json.loads(_extract_response_text(response))
        except json.JSONDecodeError as exc:
            raise ProviderError("Judge 模型没有返回有效 JSON") from exc
        if not isinstance(payload, dict):
            raise ProviderError("Judge 模型返回值不是 JSON object")
        return payload


def _extract_response_text(response: Mapping[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str):
        return direct
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        return text
    raise ProviderError("Responses 返回中没有 output_text")
