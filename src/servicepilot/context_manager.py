"""Bounded conversation history; business state remains in SessionState and TaskState."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import deque
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from .models import SessionState
from .observability import event


class ContextSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_goal: str = Field(max_length=160)
    confirmed_facts: list[str] = Field(max_length=6)
    completed_steps: list[str] = Field(max_length=6)
    unresolved_issues: list[str] = Field(max_length=6)
    important_constraints: list[str] = Field(max_length=6)
    relevant_policy_refs: list[str] = Field(max_length=6)
    important_tool_results: list[str] = Field(max_length=6)


class SummaryModel(Protocol):
    def summarize(self, old_summary: dict | None, messages: list[dict[str, str]]) -> dict: ...


class DeepSeekContextSummaryModel:
    def __init__(self, transport, model: str) -> None:
        self.transport, self.model = transport, model
        self.usage: dict[str, int] = {}
        self.last_usage: dict[str, int] = {}
        self.usage_records: deque[dict[str, int]] = deque(maxlen=64)
        self.calls = 0

    def summarize(self, old_summary: dict | None, messages: list[dict[str, str]]) -> dict:
        response = self.transport.post(
            "/chat/completions",
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "将旧摘要与新增对话合并为 JSON，字段为 current_goal, confirmed_facts, "
                            "completed_steps, unresolved_issues, important_constraints, "
                            "relevant_policy_refs, important_tool_results。"
                            "只能逐字摘录来源中的短句；"
                            "不推断新业务事实，不记录推理。订单、金额、状态、授权和版本只保留引用，"
                            "不复制值。relevant_policy_refs 必须为空数组，当前政策由 RAG 单独提供。"
                            "每个列表最多 6 条，不重复抄写同一工具结果或步骤。"
                            "忽略旧计划、失效政策、调试日志、Trace 和 Failure Diary。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"old_summary": old_summary, "new_messages": messages},
                            ensure_ascii=False,
                        ),
                    },
                ],
            },
        )
        self.calls += 1
        self.last_usage = {
            key: value for key, value in response.get("usage", {}).items() if isinstance(value, int)
        }
        self.usage_records.append(dict(self.last_usage))
        for key, value in self.last_usage.items():
            self.usage[key] = self.usage.get(key, 0) + value
        return json.loads(response["choices"][0]["message"]["content"])


_MECHANICAL = re.compile(
    r"\b(?:ORD-\d+|(?:order|user|operation|refund|execution|tool_execution)_id"
    r"|(?:refund_)?amount|payment_status|refund_status|order_status"
    r"|authorization|plan_version|goal_version)\b|(?:¥|￥|\$)\s*\d+"
    r"|(?:退款金额|支付状态|退款状态|订单状态|授权状态|计划版本|目标版本)\s*[:：]?\s*\S+",
    re.IGNORECASE,
)


class ContextCompactor:
    def __init__(
        self,
        model: SummaryModel | None,
        *,
        max_messages: int = 30,
        max_tokens: int = 4000,
        keep_recent_messages: int = 8,
    ) -> None:
        if min(max_messages, max_tokens, keep_recent_messages) < 1:
            raise ValueError("context limits must be positive")
        self.model = model
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.keep_recent_messages = keep_recent_messages

    @classmethod
    def from_env(cls, model: SummaryModel | None) -> ContextCompactor:
        return cls(
            model,
            max_messages=int(os.getenv("CONTEXT_MAX_MESSAGES", "30")),
            max_tokens=int(os.getenv("CONTEXT_MAX_TOKENS", "4000")),
            keep_recent_messages=int(os.getenv("CONTEXT_KEEP_RECENT_MESSAGES", "8")),
        )

    @staticmethod
    def tokens(messages: list[dict[str, str]] | tuple[dict[str, str], ...]) -> int:
        return sum(max(1, math.ceil(len(item.get("content", "")) / 3)) for item in messages)

    @staticmethod
    def summary_tokens(summary: dict | None) -> int:
        content = (summary or {}).get("content")
        return math.ceil(len(json.dumps(content, ensure_ascii=False)) / 3) if content else 0

    @staticmethod
    def _select(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        selected = []
        seen = set()
        for item in messages:
            content = item.get("content", "").strip()
            if (
                not content
                or len(content) > 1500
                or content.startswith(("TRACE:", "DEBUG:", "Failure Diary:", "RAG_CHUNK:"))
            ):
                continue
            key = (item.get("role"), content)
            if key in seen:
                continue
            seen.add(key)
            selected.append({"role": item.get("role", "user"), "content": content})
        return selected

    @staticmethod
    def _validate(raw: dict, source: str) -> ContextSummary:
        if not isinstance(raw, dict):
            raise ValueError("summary must be an object")
        bounded = dict(raw)
        for key in ContextSummary.model_fields:
            if key == "current_goal" or key not in bounded:
                continue
            value = bounded[key]
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                bounded[key] = list(dict.fromkeys(value))[-6:]
        summary = ContextSummary.model_validate(bounded)
        if len(json.dumps(summary.model_dump(), ensure_ascii=False)) > 1500:
            raise ValueError("summary exceeds its size limit")

        def safe(item: str) -> bool:
            return (
                len(item) <= 240
                and not _MECHANICAL.search(item)
                and "FM-" not in item
                and "trace:" not in item.lower()
                and not re.search(r"第\s*\d+\s*轮|补充说明", item)
                and (not item or item in source)
            )

        cleaned = summary.model_dump()
        if not safe(cleaned["current_goal"]):
            cleaned["current_goal"] = ""
        for key in cleaned:
            if key == "current_goal":
                continue
            # Current policy evidence is supplied by RAG, never historical summary.
            cleaned[key] = (
                []
                if key == "relevant_policy_refs"
                else [item for item in dict.fromkeys(cleaned[key]) if safe(item)]
            )
        return ContextSummary.model_validate(cleaned)

    def compact(self, state: SessionState) -> dict | None:
        before = len(state.turns)
        tokens_before = self.tokens(state.turns) + self.summary_tokens(state.context_summary)
        if before < state.context_retry_after_messages:
            return None
        if before <= self.max_messages and tokens_before <= self.max_tokens:
            return None
        if not self.model or before <= self.keep_recent_messages:
            return None
        previous = state.context_summary
        older = state.turns[: -self.keep_recent_messages]
        selected = self._select(older)
        start = time.monotonic()
        next_version = (previous or {}).get("summary_version", 0) + 1
        event(
            "CONTEXT_COMPACTION_TRIGGERED",
            before_messages=before,
            before_tokens=tokens_before,
            summary_version=next_version,
        )
        try:
            if selected:
                old_content = (previous or {}).get("content")
                raw = self.model.summarize(old_content, selected)
                source = "\n".join(item["content"] for item in selected)
                if old_content:
                    source += "\n" + "\n".join(
                        str(value) for value in old_content.values() if isinstance(value, str)
                    )
                    source += "\n" + "\n".join(
                        text
                        for value in old_content.values()
                        if isinstance(value, list)
                        for text in value
                        if isinstance(text, str)
                    )
                summary = self._validate(raw, source)
            else:
                summary = ContextSummary.model_validate((previous or {})["content"])
        except Exception as exc:
            state.context_retry_after_messages = before + self.keep_recent_messages
            known_reasons = {
                "summary exceeds its size limit",
                "historical policy references cannot be authoritative",
                "summary contains mechanical state or oversized text",
                "summary contains an unsupported assertion",
            }
            reason = str(exc) if str(exc) in known_reasons else type(exc).__name__
            event(
                "CONTEXT_COMPACTION_FAILED",
                before_messages=before,
                summary_version=next_version,
                error_type=type(exc).__name__,
                reason_code=reason,
            )
            return {"status": "failed", "error_type": type(exc).__name__, "reason_code": reason}
        state.context_summary = {
            "content": summary.model_dump(),
            "summary_version": next_version,
            "source_message_range": [
                (previous or {}).get("source_message_range", [0, 0])[0],
                (previous or {}).get("source_message_range", [0, 0])[1] + len(older),
            ],
            "created_at": datetime.now(UTC).isoformat(),
        }
        state.context_retry_after_messages = 0
        state.turns = state.turns[-self.keep_recent_messages :]
        result = {
            "status": "completed",
            "before_messages": before,
            "after_messages": len(state.turns),
            "before_tokens": tokens_before,
            "after_tokens": self.tokens(state.turns) + self.summary_tokens(state.context_summary),
            "summary_version": next_version,
            "duration_ms": round((time.monotonic() - start) * 1000, 2),
        }
        if isinstance(getattr(self.model, "last_usage", None), dict):
            result["summary_model_usage"] = dict(self.model.last_usage)
        event(
            "CONTEXT_COMPACTION_COMPLETED",
            **{key: value for key, value in result.items() if key != "status"},
        )
        return result
