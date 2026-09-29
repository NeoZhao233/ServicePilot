import asyncio
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from servicepilot.cli import build_agent
from servicepilot.observability import (
    Span,
    TraceConfig,
    TraceService,
    current_trace,
    model_transport,
    observed,
    submit_with_context,
)
from servicepilot.trace_http import TraceMiddleware
from servicepilot.trace_store import TraceStore
from servicepilot.webapp import create_app

ROOT = Path(__file__).resolve().parents[1]


def service(tmp_path, **kwargs):
    return TraceService(TraceStore(tmp_path / "trace.sqlite3"), TraceConfig(**kwargs))


def details(s, user="USER-001"):
    owner = s.redactor.pseudonym(user)
    rows = s.store.list(owner)["items"]
    return [s.store.detail(row["trace_id"], owner) for row in rows]


def runtime(tmp_path, s, **kwargs):
    return build_agent(
        ROOT / "data",
        session_db=tmp_path / "business.sqlite3",
        trace_service=s,
        dynamic=True,
        **kwargs,
    )


def test_request_has_persistent_tree_and_distinct_correlation(tmp_path):
    s = service(tmp_path)
    r = runtime(tmp_path, s)
    response = r.chat(user_id="USER-001", message="退掉 ORD-1001")
    d = details(s)[0]
    assert d["trace"]["trace_id"] != response.session.session_id
    assert d["trace"]["thread_id"] == response.session.session_id
    assert d["trace"]["task_id"] == response.session.session_id
    spans = {sp["span_id"]: sp for sp in d["spans"]}
    assert len(d["tree"]) == 1
    plan = next(sp for sp in d["spans"] if sp["name"] == "dynamic.plan")
    assert spans[plan["parent_span_id"]]["name"] == "main.dynamic_task"
    assert all(sp["end_time"] is not None and sp["duration_ms"] >= 0 for sp in d["spans"])
    assert d["trace"]["total_llm_calls"] == 0
    assert d["trace"]["total_input_tokens"] == 0  # observed absence, not missing usage
    assert d["trace"]["status"] == "ok"
    with sqlite3.connect(tmp_path / "trace.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM obs_spans").fetchone()[0] == len(spans)


def test_goal_change_and_replan_across_requests(tmp_path):
    s = service(tmp_path)
    r = runtime(tmp_path, s)
    first = r.chat(user_id="USER-001", message="退掉 ORD-1001")
    r.chat(user_id="USER-001", message="确认改成换货", session_id=first.session.session_id)
    traces = details(s)
    assert len({d["trace"]["trace_id"] for d in traces}) == 2
    assert len({d["trace"]["task_id"] for d in traces}) == 1
    d = traces[0]
    types = {e["event_type"] for e in d["events"]}
    assert {"GOAL_CHANGED", "PLAN_INVALIDATED", "REPLAN_TRIGGERED", "PLAN_CREATED"} <= types
    old = next(e for e in d["events"] if e["event_type"] == "PLAN_INVALIDATED")
    new = next(e for e in d["events"] if e["event_type"] == "PLAN_CREATED")
    assert new["plan_version"] > old["plan_version"]
    assert old["attributes"]["cancelled_steps"] == ["submit"]
    assert old["attributes"]["retained_steps"] == ["order"]
    assert d["trace"]["replan_count"] == 1


def test_parallel_rag_children_share_parent_not_other_requests(tmp_path):
    s = service(tmp_path)
    r = runtime(tmp_path, s)
    barrier = threading.Barrier(2)
    lexical = r.agent.retriever.lexical.search
    vector = r.agent.retriever.vector.search

    def wrap(fn):
        def search(*args, **kwargs):
            barrier.wait(timeout=5)
            return fn(*args, **kwargs)

        return search

    r.agent.retriever.lexical.search = wrap(lexical)
    r.agent.retriever.vector.search = wrap(vector)
    r.chat(user_id="USER-001", message="退掉 ORD-1001")
    d = details(s)[0]
    retrievers = [
        sp
        for sp in d["spans"]
        if sp["name"] in {"rag.BM25PolicyRetriever", "rag.VectorPolicyRetriever"}
    ]
    assert len(retrievers) == 2
    assert len({sp["parent_span_id"] for sp in retrievers}) == 1
    assert all(sp["trace_id"] == d["trace"]["trace_id"] for sp in retrievers)


def test_parallel_async_and_threads_isolate_context(tmp_path):
    s = service(tmp_path)

    @observed("worker", "rag")
    def worker():
        time.sleep(0.002)
        return current_trace().record["trace_id"]

    async def request(user):
        with s.trace(user_id=user) as c:
            with Span("parent", "agent"):
                await asyncio.sleep(0.002)
                with ThreadPoolExecutor(2) as pool:
                    calls = [submit_with_context(pool, worker) for _ in range(2)]
                    assert {f.result() for f in calls} == {c.record["trace_id"]}
                return c.record["trace_id"]

    async def main():
        return await asyncio.gather(request("A"), request("B"))

    identifiers = asyncio.run(main())
    assert len(set(identifiers)) == 2
    for user in ["A", "B"]:
        d = details(s, user)[0]
        assert len(d["spans"]) == 3
        assert all(sp["trace_id"] == d["trace"]["trace_id"] for sp in d["spans"])
    assert current_trace() is None


def test_real_provider_usage_and_unknown_usage(tmp_path):
    s = service(tmp_path)

    class Transport:
        @model_transport
        def post(self, path, payload):
            return {
                "usage": {
                    "prompt_tokens": 37,
                    "completion_tokens": 11,
                    "prompt_cache_hit_tokens": 20,
                },
                "model": "mock-deepseek",
                "choices": [
                    {
                        "message": {
                            "content": '{"intent":"refund"}',
                            "reasoning_content": "HIDDEN_REASONING_SECRET",
                        }
                    }
                ],
            }

    with s.trace(user_id="A"):
        Transport().post(
            "/chat/completions",
            {"model": "mock-deepseek", "messages": [{"role": "user", "content": "secret prompt"}]},
        )
    d = details(s, "A")[0]
    assert d["trace"]["total_input_tokens"] == 37
    assert d["trace"]["total_output_tokens"] == 11
    assert d["spans"][0]["attributes"]["cache_hit_tokens"] == 20
    assert d["spans"][0]["attributes"]["cost"] is None
    assert "HIDDEN_REASONING_SECRET" not in json.dumps(d)

    class Missing:
        @model_transport
        def post(self, path, payload):
            return {}

    with s.trace(user_id="B"):
        Missing().post("/responses", {"model": "mock"})
    missing = details(s, "B")[0]["trace"]
    assert missing["total_llm_calls"] == 1
    assert missing["total_input_tokens"] is None
    assert missing["total_output_tokens"] is None


def test_tool_timeout_preserves_exception_and_successful_business_outcome(tmp_path):
    s = service(tmp_path)
    r = runtime(tmp_path, s)
    first = r.chat(user_id="USER-001", message="退掉 ORD-1001")
    real = r.agent.repository.create_refund_request
    count = 0

    def lost(**kwargs):
        nonlocal count
        count += 1
        real(**kwargs)
        raise TimeoutError("sk-do-not-store 13800138000")

    r.agent.repository.create_refund_request = lost
    result = r.chat(user_id="USER-001", message="确认", session_id=first.session.session_id)
    assert "已提交" in result.message and count == 1
    d = details(s)[0]
    timeout = next(sp for sp in d["spans"] if sp["name"] == "create_refund_request")
    assert timeout["status"] == "timeout" and timeout["error_type"] == "TimeoutError"
    assert d["trace"]["outcome"] == "completed"
    assert {"RECOVERY_STARTED", "RECOVERY_COMPLETED"} <= {e["event_type"] for e in d["events"]}
    assert "sk-do-not-store" not in json.dumps(d)
    assert "13800138000" not in json.dumps(d)


def test_trace_storage_failure_never_changes_refund(tmp_path, monkeypatch):
    s = service(tmp_path)
    r = runtime(tmp_path, s)
    first = r.chat(user_id="USER-001", message="退掉 ORD-1001")

    def broken(*args):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(s.store, "start", broken)
    monkeypatch.setattr(s.store, "finish", broken)
    response = r.chat(user_id="USER-001", message="确认", session_id=first.session.session_id)
    assert "已提交" in response.message
    assert s.write_failures >= 2
    with sqlite3.connect(tmp_path / "business.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM commerce_refunds").fetchone()[0] == 1


def test_privacy_filters_nested_secrets_and_hidden_reasoning(tmp_path):
    s = service(tmp_path, details=True)
    with s.trace(user_id="private-user-123"):
        with Span(
            "test",
            "tool",
            {
                "authorization": "Bearer secret-ABC",
                "nested": {"api_key": "sk-SECRET", "reasoning_content": "DO NOT STORE"},
                "query": "退款 13800138000 alice@example.com 北京市秘密地址 ORD-1001",
                "amount": 499,
                "order_id": "ORD-1001",
                "card_number": "6222000000000000",
            },
        ) as span:
            span.output(
                {
                    "prompt": "MY PRIVATE PROMPT",
                    "tool_calls": [
                        {"arguments": {"password": "passwdSECRET", "address": "my private address"}}
                    ],
                }
            )
    raw = json.dumps(details(s, "private-user-123"), ensure_ascii=False)
    for secret in [
        "private-user-123",
        "secret-ABC",
        "sk-SECRET",
        "DO NOT STORE",
        "13800138000",
        "alice@example.com",
        "秘密地址",
        "ORD-1001",
        "6222000000000000",
        "PRIVATE PROMPT",
        "passwdSECRET",
        "my private address",
    ]:
        assert secret not in raw
    assert "退款" in raw


def authenticated_client(tmp_path, monkeypatch):
    monkeypatch.setenv("SERVICEPILOT_AUTH_TOKENS", '{"token-a":"USER-001","token-b":"USER-002"}')
    s = service(tmp_path)
    app = create_app(
        data_dir=ROOT / "data",
        session_db=tmp_path / "business.sqlite3",
        dynamic=True,
        trace_service=s,
    )
    return TestClient(app), s


def test_http_trace_queries_enforce_authenticated_owner(tmp_path, monkeypatch):
    client, s = authenticated_client(tmp_path, monkeypatch)
    headers = {"Authorization": "Bearer token-a"}
    response = client.post(
        "/api/chat", headers=headers, json={"user_id": "USER-001", "message": "退掉 ORD-1001"}
    )
    assert response.status_code == 200
    tid = response.headers["x-trace-id"]
    assert tid == response.json()["trace_id"]
    d = client.get("/api/traces/" + tid, headers=headers).json()
    assert d["tree"][0]["name"] == "http.request"
    assert d["tree"][0]["children"][0]["name"] == "agent.chat"
    assert client.get("/api/traces/" + tid).status_code == 401
    other = {"Authorization": "Bearer token-b"}
    for suffix in ["", "/metrics", "/export"]:
        assert client.get("/api/traces/" + tid + suffix, headers=other).status_code == 404
    assert client.get("/api/traces", headers=other).json()["total"] == 0
    assert client.get("/api/traces?user_id=USER-001", headers=other).status_code == 403
    task = d["trace"]["task_id"]
    assert client.get(f"/api/tasks/{task}/traces", headers=headers).json()["total"] == 1
    assert client.get(f"/api/tasks/{task}/traces", headers=other).json()["total"] == 0
    assert client.get("/api/traces?node=dynamic.plan", headers=headers).json()["total"] == 1
    assert client.get("/api/traces?limit=101", headers=headers).status_code == 422
    exported = client.get("/api/traces/" + tid + "/export", headers=headers)
    assert "attachment" in exported.headers["content-disposition"]
    assert (
        s.store.list(s.redactor.pseudonym("USER-001"))["total"] == 1
    )  # reads do not trace themselves


def test_unconfigured_demo_cannot_read_traces(tmp_path, monkeypatch):
    monkeypatch.delenv("SERVICEPILOT_AUTH_TOKENS", raising=False)
    client = TestClient(create_app(data_dir=ROOT / "data", trace_service=service(tmp_path)))
    assert client.get("/api/traces?user_id=USER-001").status_code == 401
    assert client.get("/traces").status_code == 200
    assert client.get("/traces.js").status_code == 200


def test_validation_failure_is_completed_trace(tmp_path, monkeypatch):
    client, s = authenticated_client(tmp_path, monkeypatch)
    response = client.post(
        "/api/chat",
        headers={"Authorization": "Bearer token-a"},
        json={"user_id": "USER-001", "message": ""},
    )
    assert response.status_code == 422
    d = details(s)[0]
    assert d["trace"]["status"] == "error"
    assert d["trace"]["end_time"] is not None
    assert d["spans"][0]["attributes"]["http_status"] == 422


def test_stream_is_running_until_last_body_and_cancellation_finalizes(tmp_path):
    s = service(tmp_path)

    async def application(scope, receive, send):
        current_trace().bind(user_id="A")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        assert (
            s.store.connection.execute("SELECT status FROM obs_traces").fetchone()[0] == "running"
        )
        await asyncio.sleep(0.002)
        await send({"type": "http.response.body", "body": b"last", "more_body": False})

    async def receive():
        return {"type": "http.request", "body": b""}

    messages = []

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "method": "GET", "path": "/stream", "headers": []}
    asyncio.run(TraceMiddleware(application, s)(scope, receive, send))
    assert details(s, "A")[0]["trace"]["status"] == "ok"
    assert len(messages) == 3

    async def cancelled(scope, receive, send):
        current_trace().bind(user_id="B")
        with Span("await_data", "internal"):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(TraceMiddleware(cancelled, s)(scope, receive, send))
    d = details(s, "B")[0]
    assert d["trace"]["status"] == "interrupted"
    assert all(sp["end_time"] is not None for sp in d["spans"])


def test_real_streamingresponse_and_disconnect(tmp_path):
    s = service(tmp_path)
    app = FastAPI()
    app.add_middleware(TraceMiddleware, service=s)

    @app.get("/stream")
    async def stream():
        current_trace().bind(user_id="A")

        async def generate():
            with Span("stream.generate", "internal"):
                yield b"hello"
                await asyncio.sleep(0.002)
                yield b"world"

        return StreamingResponse(generate())

    with TestClient(app) as client:
        assert client.get("/stream").content == b"helloworld"
    assert details(s, "A")[0]["trace"]["status"] == "ok"

    async def disconnect_app(scope, receive, send):
        current_trace().bind(user_id="B")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"x", "more_body": True})
        await receive()

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    asyncio.run(
        TraceMiddleware(disconnect_app, s)(
            {"type": "http", "method": "GET", "path": "/stream", "headers": []}, receive, send
        )
    )
    assert details(s, "B")[0]["trace"]["status"] == "interrupted"


def test_query_indices_retention_and_abandoned_are_observation_only(tmp_path):
    s = service(tmp_path)
    with s.trace(user_id="A"):
        with Span("expired", "internal"):
            pass
    trace = details(s, "A")[0]["trace"]
    with s.store.connection:
        s.store.connection.execute("CREATE TABLE important_business(value TEXT)")
        s.store.connection.execute("INSERT INTO important_business VALUES ('keep')")
    assert s.store.cleanup(1, now=trace["end_time"] + 86401) == 1
    assert s.store.connection.execute("SELECT count(*) FROM obs_spans").fetchone()[0] == 0
    assert (
        s.store.connection.execute("SELECT value FROM important_business").fetchone()[0] == "keep"
    )
    plan = s.store.connection.execute(
        "EXPLAIN QUERY PLAN SELECT trace_id FROM obs_traces "
        "WHERE owner=? AND task_id=? ORDER BY start_time",
        ("owner", "task"),
    ).fetchall()
    assert any("obs_trace_task" in row[3] for row in plan)
    abandoned = dict(
        trace, trace_id="abandoned", status="running", end_time=None, start_time=time.time() - 7200
    )
    s.store.start(abandoned)
    assert s.store.mark_abandoned() == 1
    row = s.store.detail("abandoned", s.redactor.pseudonym("A"))["trace"]
    assert row["status"] == "interrupted" and row["incomplete"]
    assert row["total_input_tokens"] is None


def test_sampling_and_disabled_paths_do_not_change_business(tmp_path):
    for index, config in enumerate([TraceConfig(enabled=False), TraceConfig(sample_rate=0)]):
        folder = tmp_path / str(index)
        folder.mkdir()
        s = TraceService(TraceStore(folder / "trace.sqlite3"), config)
        response = runtime(folder, s).chat(user_id="USER-001", message="ORD-1003 物流")
        assert "运输中" in response.message
        assert details(s) == []


def test_exception_span_does_not_replace_original_exception(tmp_path):
    s = service(tmp_path)
    with pytest.raises(ValueError, match="original"):
        with s.trace(user_id="A"):
            with Span("fails", "internal"):
                raise ValueError("original")
    d = details(s, "A")[0]
    assert d["trace"]["status"] == "error"
    assert d["spans"][0]["error_type"] == "ValueError"


def test_mcp_dispatch_persists_real_success_and_failed_spans(tmp_path):
    from mcp.server.fastmcp.exceptions import ToolError

    from servicepilot.mcp_server import create_mcp_server

    database = tmp_path / "mcp.sqlite3"
    server = create_mcp_server(ROOT / "data", database=database, trusted_user_id="USER-001")

    async def run():
        await server.call_tool("get_order", {"order_id": "ORD-1001", "user_id": "USER-001"})
        with pytest.raises(ToolError):
            await server.call_tool("get_order", {"order_id": "ORD-1001", "user_id": "USER-002"})

    asyncio.run(run())
    s = TraceService(TraceStore(database))
    traces = details(s)
    assert len(traces) == 2
    spans = [span for trace in traces for span in trace["spans"] if span["span_type"] == "mcp"]
    assert {span["status"] for span in spans} == {"ok", "error"}
    assert all(span["duration_ms"] >= 0 for span in spans)
    assert any(span["error_type"] == "PermissionError" for span in spans)
    assert all(trace["trace"]["total_tool_calls"] == 1 for trace in traces)


def test_nested_sampling_decision_is_once_and_malformed_usage_is_safe(tmp_path, monkeypatch):
    s = service(tmp_path, sample_rate=0.5)
    calls = []

    def random_value():
        calls.append(1)
        return 0.9

    monkeypatch.setattr("servicepilot.observability.random.random", random_value)
    with s.trace(user_id="A"):
        with s.trace(user_id="A"):
            assert current_trace() is None
    assert len(calls) == 1
    assert details(s, "A") == []
    s = service(tmp_path)

    class Transport:
        @model_transport
        def post(self, path, payload):
            return {"usage": {"prompt_tokens_details": "invalid", "input_tokens": "unknown"}}

    with s.trace(user_id="A"):
        assert Transport().post("/responses", {"model": "mock"})["usage"]
    assert details(s, "A")[0]["trace"]["total_input_tokens"] is None
