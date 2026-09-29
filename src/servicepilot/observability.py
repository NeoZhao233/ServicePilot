from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import hmac
import inspect
import logging
import os
import random
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from uuid import uuid4

from .trace_store import TraceStore

_CURRENT = contextvars.ContextVar("servicepilot_trace", default=None)
_PARENT = contextvars.ContextVar("servicepilot_span", default=None)
_SUPPRESSED = contextvars.ContextVar("servicepilot_trace_suppressed", default=False)
LOG = logging.getLogger("servicepilot.observability")
EVENT_TYPES = frozenset(
    {
        "GOAL_CHANGED",
        "PLAN_CREATED",
        "PLAN_INVALIDATED",
        "REPLAN_TRIGGERED",
        "PLAN_STEP_STARTED",
        "PLAN_STEP_COMPLETED",
        "TOOL_FAILED",
        "EXECUTION_UNCERTAIN",
        "RECOVERY_STARTED",
        "RECOVERY_COMPLETED",
        "POLICY_REJECTED",
        "HUMAN_HANDOFF",
        "GOAL_EVALUATED",
        "CACHE_HIT",
        "RETRIEVAL_FAILED",
        "PLAN_VALIDATION_FAILED",
        "EVIDENCE_SCORED",
        "EVALUATOR_PROVIDER_FALLBACK",
        "EVIDENCE_JUDGED",
        "EVIDENCE_METADATA_REJECTED",
        "EVIDENCE_FINALIZED",
        "CORRECTIVE_RETRIEVAL_TRIGGERED",
        "RAG_DEGRADED",
        "PDF_PARSE_STARTED",
        "PDF_PARSE_COMPLETED",
        "PDF_EVIDENCE_RETRIEVED",
        "PDF_VISION_STARTED",
        "PDF_VISION_COMPLETED",
        "PDF_VISION_FAILED",
        "PDF_ANSWERED",
    }
)


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = True
    sample_rate: float = 1.0
    details: bool = False
    retention_days: int = 14
    max_spans: int = 1000
    max_events: int = 2000

    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.getenv("SERVICEPILOT_TRACE_ENABLED", "true").lower() in {"true", "1", "yes"},
            sample_rate=max(0, min(1, float(os.getenv("SERVICEPILOT_TRACE_SAMPLE_RATE", "1")))),
            details=os.getenv("SERVICEPILOT_TRACE_DETAILS", "false").lower()
            in {"true", "1", "yes"},
            retention_days=max(1, int(os.getenv("SERVICEPILOT_TRACE_RETENTION_DAYS", "14"))),
        )


class Redactor:
    BLOCK = re.compile(
        r"(authorization|api.?key|password|secret|credential|access.?token|"
        r"refresh.?token|^token$|reasoning|chain.of.thought|hidden|"
        r"phone|email|address|amount|payment|card|^embedding$|^embeddings$|^vector$|姓名|地址|手机|金额)",
        re.I,
    )
    TEXT = {
        "message",
        "content",
        "input",
        "output_text",
        "prompt",
        "instructions",
        "query",
        "description",
        "reason",
        "error",
        "error_message",
        "reply",
        "refund_reason",
        "product_name",
        "turns",
    }
    STRUCTURAL = {
        "trace_id",
        "span_id",
        "parent_span_id",
        "thread_id",
        "task_id",
        "request_id",
        "session_id",
        "operation_id",
        "step_id",
        "plan_id",
        "policy_id",
        "policy_ids",
        "document_id",
        "document_ids",
        "page_number",
        "pages",
        "filtered_policy_ids",
        "config_version",
        "missing",
        "step_ids",
        "dependencies",
        "completed_steps",
        "pending_steps",
        "cancelled_steps",
        "retained_steps",
        "retry_steps",
        "actions",
        "name",
        "span_type",
        "status",
        "state",
        "execution_status",
        "event_type",
        "action",
        "intent",
        "strategy",
        "category",
        "code",
        "outcome",
        "verdict",
        "reason_code",
        "expected_effect",
        "preconditions",
        "model",
        "model_name",
        "provider",
        "requested_provider",
        "source",
        "transport",
        "path",
        "method",
        "next",
        "node",
        "error_type",
        "cause_type",
        "validation",
        "trigger",
        "reason_type",
        "prompt_profile",
        "currency",
    }

    def __init__(self, key: str, details=False):
        self.key, self.details = key.encode(), details

    def pseudonym(self, value):
        return "anon_" + hmac.new(self.key, str(value).encode(), hashlib.sha256).hexdigest()[:24]

    def text(self, value):
        text = str(value)
        result = {"length": len(text), "digest": self.pseudonym(text)}
        if self.details:
            # Even detail mode exposes only a lexical preview of known domain words.
            words = re.findall(r"退款|换货|物流|确认|取消|政策|订单|超时|失败|成功|人工|库存", text)
            result["safe_preview"] = " ".join(dict.fromkeys(words))[:160]
        return result

    def clean(self, value, key="", depth=0):
        if depth > 8:
            return "[depth-limited]"
        if self.BLOCK.search(key):
            return "[REDACTED]"
        if key in {"user_id", "order_id", "request_id_external"}:
            return self.pseudonym(value) if value is not None else None
        if key in self.TEXT:
            if key == "reason" and isinstance(value, str) and re.fullmatch(r"[a-z_]{1,64}", value):
                return value
            return self.text(value)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if key in self.STRUCTURAL and re.fullmatch(r"[a-zA-Z0-9_./: -]{1,160}", value):
                if re.search(r"(?i)(sk-|bearer |token[=:]|password[=:])", value):
                    return "[REDACTED]"
                return value
            return self.text(value)
        if isinstance(value, (list, tuple, set)):
            return [self.clean(v, key, depth + 1) for v in list(value)[:100]]
        if is_dataclass(value):
            kind = type(value).__name__
            if kind == "RetrievalHit":
                return {
                    "policy_id": value.chunk.policy_id,
                    "score": value.score,
                    "version": hashlib.sha256(value.chunk.content.encode()).hexdigest()[:16],
                }
            allowed = {
                f.name: getattr(value, f.name)
                for f in fields(value)
                if f.name not in {"turns", "authorization", "events", "history", "chunk"}
            }
            return self.clean(allowed, depth=depth + 1)
        if isinstance(value, dict):
            return {
                (
                    str(k)
                    if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]{0,79}", str(k))
                    else self.pseudonym(k)
                ): self.clean(v, str(k), depth + 1)
                for k, v in list(value.items())[:80]
                if not self.BLOCK.search(str(k))
                and str(k) not in {"self", "cls", "operation", "pre_events", "events", "turns"}
            }
        return {"type": type(value).__name__}


class TraceService:
    def __init__(self, store: TraceStore | None, config: TraceConfig | None = None):
        self.store, self.config = store, config or TraceConfig()
        self.redactor = Redactor(store.key if store else uuid4().hex, self.config.details)
        self.write_failures = 0

    @classmethod
    def create(cls, path=":memory:", config=None):
        config = config or TraceConfig.from_env()
        try:
            store = TraceStore(path, key=os.getenv("SERVICEPILOT_TRACE_HMAC_KEY"))
        except Exception:
            LOG.warning("trace_storage_initialization_failed")
            store = None
        return cls(store, config)

    def safe_write(self, method, *args):
        try:
            if self.store:
                getattr(self.store, method)(*args)
        except Exception:
            self.write_failures += 1
            LOG.warning("trace_write_failed")  # Do not log raw exceptions or credentials.

    @contextmanager
    def trace(self, name="request", *, user_id=None, thread_id=None, attributes=None):
        if _CURRENT.get() is not None:
            yield _CURRENT.get()
            return
        if _SUPPRESSED.get():
            yield None
            return
        if not self.config.enabled or random.random() >= self.config.sample_rate:
            suppression = _SUPPRESSED.set(True)
            try:
                yield None
            finally:
                _SUPPRESSED.reset(suppression)
            return
        collector = Collector(self, name, user_id, thread_id, attributes)
        token = _CURRENT.set(collector)
        try:
            yield collector
        except BaseException as exc:
            collector.status = exception_status(exc)
            raise
        finally:
            try:
                collector.finish()
            except Exception:
                LOG.warning("trace_finalization_failed")
            finally:
                _CURRENT.reset(token)


class Collector:
    def __init__(self, service, name, user_id, thread_id, attributes):
        self.service = service
        self.lock = threading.RLock()
        self.started = time.perf_counter()
        self.status = "ok"
        self.closed = False
        self.spans, self.events = [], []
        self.seen_events = set()
        self.record = {
            "trace_id": uuid4().hex,
            "request_id": uuid4().hex,
            "name": name,
            "thread_id": thread_id,
            "task_id": None,
            "user_id": service.redactor.pseudonym(user_id) if user_id else None,
            "start_time": time.time(),
            "end_time": None,
            "status": "running",
            "attributes": service.redactor.clean(attributes or {}),
            "total_latency_ms": None,
            "total_input_tokens": None,
            "total_output_tokens": None,
            "total_llm_calls": None,
            "total_tool_calls": None,
            "replan_count": None,
            "error_count": None,
            "incomplete": False,
        }
        service.safe_write("start", self.record)

    def bind(self, *, user_id=None, thread_id=None, task_id=None, outcome=None):
        with self.lock:
            if user_id:
                owner = self.service.redactor.pseudonym(user_id)
                if self.record["user_id"] and self.record["user_id"] != owner:
                    return  # A trace's authenticated principal cannot be overwritten downstream.
                self.record["user_id"] = owner
            if thread_id:
                self.record["thread_id"] = thread_id
            if task_id:
                self.record["task_id"] = task_id
            if outcome:
                self.record["outcome"] = self.service.redactor.clean(outcome, "outcome")

    def _has_tool_ancestor(self, span):
        by_id = {s["span_id"]: s for s in self.spans}
        parent = by_id.get(span["parent_span_id"])
        while parent:
            if parent["span_type"] in {"tool_attempt", "mcp", "tool_query"}:
                return True
            parent = by_id.get(parent["parent_span_id"])
        return False

    def finish(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for span in self.spans:
                if span["status"] == "running":
                    span.update(status="interrupted", end_time=time.time(), duration_ms=None)
                    self.record["incomplete"] = True
            llms = [s for s in self.spans if s["span_type"] == "llm"]

            def tokens(key):
                values = [s["attributes"].get(key) for s in llms]
                values = [
                    v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None
                    for v in values
                ]
                return None if any(v is None for v in values) else sum(values)

            self.record.update(
                status=self.status,
                end_time=time.time(),
                total_latency_ms=round((time.perf_counter() - self.started) * 1000, 3),
                total_input_tokens=tokens("input_tokens"),
                total_output_tokens=tokens("output_tokens"),
                total_llm_calls=len(llms),
                total_tool_calls=sum(
                    s["span_type"] in {"tool_attempt", "mcp"}
                    or (s["span_type"] == "tool_query" and not self._has_tool_ancestor(s))
                    for s in self.spans
                ),
                replan_count=sum(e["event_type"] == "REPLAN_TRIGGERED" for e in self.events),
                error_count=sum(s["status"] in {"error", "timeout"} for s in self.spans),
            )
            by_id = {s["span_id"]: s for s in self.spans}
            for plan in self.spans:
                if not plan["attributes"].get("is_replan"):
                    continue
                descendants = []
                for llm in llms:
                    parent = by_id.get(llm["parent_span_id"])
                    while parent:
                        if parent is plan:
                            descendants.append(llm)
                            break
                        parent = by_id.get(parent["parent_span_id"])
                plan["attributes"]["replan_duration_ms"] = plan["duration_ms"]
                for key in ("input_tokens", "output_tokens"):
                    values = [s["attributes"].get(key) for s in descendants]
                    valid = all(type(v) is int and v >= 0 for v in values)
                    plan["attributes"][key] = sum(values) if valid else None
            if self.record["incomplete"]:
                for key in (
                    "total_input_tokens",
                    "total_output_tokens",
                    "total_llm_calls",
                    "total_tool_calls",
                    "replan_count",
                    "error_count",
                ):
                    self.record[key] = None
            self.service.safe_write("finish", self.record, self.spans, self.events)


def current_trace():
    return _CURRENT.get()


def exception_status(exc):
    if isinstance(exc, asyncio.CancelledError):
        return "interrupted"
    return "timeout" if isinstance(exc, TimeoutError) else "error"


class Span:
    def __init__(self, name, span_type, inputs=None, attributes=None):
        self.collector = _CURRENT.get()
        self.data = None
        self.token = None
        self.started = time.perf_counter()
        c = self.collector
        if c and not c.closed:
            with c.lock:
                if len(c.spans) >= c.service.config.max_spans:
                    c.record["incomplete"] = True
                    return
                clean = c.service.redactor.clean
                self.data = {
                    "trace_id": c.record["trace_id"],
                    "span_id": uuid4().hex,
                    "parent_span_id": _PARENT.get(),
                    "name": name,
                    "span_type": span_type,
                    "start_time": time.time(),
                    "end_time": None,
                    "duration_ms": None,
                    "status": "running",
                    "input_summary": clean(inputs),
                    "output_summary": None,
                    "attributes": clean(attributes or {}),
                    "error_type": None,
                    "error_message": None,
                }
                c.spans.append(self.data)

    def __enter__(self):
        if self.data:
            self.token = _PARENT.set(self.data["span_id"])
        return self

    def set(self, **attributes):
        if self.data and not self.collector.closed:
            self.data["attributes"].update(self.collector.service.redactor.clean(attributes))

    def output(self, value):
        if self.data and not self.collector.closed:
            self.data["output_summary"] = self.collector.service.redactor.clean(value)

    def failed(self, error_type, message=None, status="error"):
        if self.data and not self.collector.closed:
            self.data.update(
                status=status,
                error_type=error_type,
                error_message=self.collector.service.redactor.text(message or error_type),
            )

    def __exit__(self, exc_type, exc, tb):
        if self.token is not None:
            _PARENT.reset(self.token)
        if self.data and not self.collector.closed:
            if exc is not None:
                cause = exc.__cause__
                self.failed(type(exc).__name__, str(exc), exception_status(cause or exc))
                if cause:
                    self.set(cause_type=type(cause).__name__)
            elif self.data["status"] == "running":
                self.data["status"] = "ok"
            self.data.update(
                end_time=time.time(),
                duration_ms=round((time.perf_counter() - self.started) * 1000, 3),
            )
        return False


def event(event_type, *, task_id=None, plan_version=None, **attributes):
    c = _CURRENT.get()
    if not c or c.closed:
        return
    with c.lock:
        if len(c.events) >= c.service.config.max_events:
            c.record["incomplete"] = True
            return
        c.events.append(
            {
                "event_id": uuid4().hex,
                "trace_id": c.record["trace_id"],
                "span_id": _PARENT.get(),
                "event_type": event_type,
                "timestamp": time.time(),
                "task_id": task_id or c.record.get("task_id"),
                "plan_version": plan_version,
                "attributes": c.service.redactor.clean(attributes),
            }
        )


def observed(name=None, span_type="internal"):
    """Observe sync/async units without retries or swallowing business errors."""

    def decorate(fn):
        signature = inspect.signature(fn)

        def inputs(args, kwargs):
            try:
                return {
                    k: v
                    for k, v in signature.bind_partial(*args, **kwargs).arguments.items()
                    if k not in {"self", "cls"}
                }
            except TypeError:
                return None

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                with Span(name or fn.__qualname__, span_type, inputs(args, kwargs)) as span:
                    result = await fn(*args, **kwargs)
                    span.output(result)
                    return result

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with Span(name or fn.__qualname__, span_type, inputs(args, kwargs)) as span:
                result = fn(*args, **kwargs)
                span.output(result)
                if span_type == "rag" and isinstance(result, list):
                    span.set(result_count=len(result))
                if span_type == "validation":
                    span.set(validation="accepted")
                if getattr(result, "ok", True) is False:
                    span.failed(str(result.data.get("code", "ToolRejected")), result.error)
                return result

        return wrapper

    return decorate


def traced_chat(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        service = getattr(self, "trace_service", None)
        if service is None:
            return fn(self, *args, **kwargs)
        with service.trace(
            "agent.request", user_id=kwargs.get("user_id"), thread_id=kwargs.get("session_id")
        ):
            c = current_trace()
            if c:
                c.bind(user_id=kwargs.get("user_id"), thread_id=kwargs.get("session_id"))
            with Span("agent.chat", "agent", kwargs) as span:
                response = fn(self, *args, **kwargs)
                if c:
                    c.bind(thread_id=response.session.session_id)
                span.output(response)
                return response

    return wrapper


def submit_with_context(pool, fn, *args, **kwargs):
    # A fresh Context per future is essential: one Context cannot be entered concurrently.
    return pool.submit(contextvars.copy_context().run, fn, *args, **kwargs)


def model_transport(fn):
    @functools.wraps(fn)
    def run(self, path, payload):
        kind = "embedding" if path.rstrip("/").endswith("embeddings") else "llm"
        with Span(
            "model.request",
            kind,
            payload,
            {
                "model": payload.get("model"),
                "path": path,
                "input_tokens": None,
                "output_tokens": None,
                "cache_hit_tokens": None,
                "cache_miss_tokens": None,
                "cost": None,
            },
        ) as span:
            response = fn(self, path, payload)
            usage = response.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            details = usage.get("prompt_tokens_details")
            details = details if isinstance(details, dict) else {}
            span.set(
                model=response.get("model", payload.get("model")),
                input_tokens=usage.get("input_tokens", usage.get("prompt_tokens")),
                output_tokens=usage.get("output_tokens", usage.get("completion_tokens")),
                cache_hit_tokens=usage.get("prompt_cache_hit_tokens", details.get("cached_tokens")),
                cache_miss_tokens=usage.get("prompt_cache_miss_tokens"),
                cost=usage.get("cost"),
                usage=usage,
            )
            span.output(response)
            return response

    return run
