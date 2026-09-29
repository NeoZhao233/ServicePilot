from __future__ import annotations

import asyncio
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from .observability import Span


class TraceMiddleware:
    """Pure ASGI: a streaming request ends on its final body, error or disconnect."""

    def __init__(self, app, service):
        self.app, self.service = app, service

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or path.startswith(
            ("/api/traces", "/api/tasks/", "/api/threads/")
        ):
            return await self.app(scope, receive, send)
        if not (path.startswith("/api/") or path in {"/chat", "/stream", "/async"}):
            return await self.app(scope, receive, send)
        with self.service.trace(
            "http.request", attributes={"path": path, "method": scope["method"]}
        ) as collector:
            if not collector:
                return await self.app(scope, receive, send)
            incoming = dict(scope.get("headers", []))
            upstream = incoming.get(b"traceparent", b"").decode("ascii", "ignore")
            if re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}", upstream):
                collector.record["upstream_trace_id"] = upstream.split("-")[1]
            complete = False
            with Span(
                "http.request", "http", attributes={"method": scope["method"], "path": path}
            ) as span:

                async def traced_receive():
                    message = await receive()
                    if message["type"] == "http.disconnect" and not complete:
                        collector.status = "interrupted"
                        span.failed("ClientDisconnect", status="interrupted")
                    return message

                async def traced_send(message):
                    nonlocal complete
                    if message["type"] == "http.response.start":
                        message = dict(message)
                        message["headers"] = [
                            *message.get("headers", []),
                            (b"x-trace-id", collector.record["trace_id"].encode()),
                            (b"x-request-id", collector.record["request_id"].encode()),
                        ]
                        status = message["status"]
                        span.set(http_status=status)
                        if status >= 400:
                            collector.status = "error"
                            span.failed("HTTPError", str(status))
                    await send(message)
                    if message["type"] == "http.response.body" and not message.get(
                        "more_body", False
                    ):
                        complete = True

                try:
                    await self.app(scope, traced_receive, traced_send)
                except asyncio.CancelledError:
                    collector.status = "interrupted"
                    raise
                finally:
                    if not complete and collector.status == "ok":
                        collector.status = "interrupted"
                        span.failed("IncompleteResponse", status="interrupted")


def install_trace_api(app, service, assets_dir: Path):
    app.state.trace_service = service
    app.add_middleware(TraceMiddleware, service=service)

    def private_response(response: Response):
        response.headers["Cache-Control"] = "no-store"

    router = APIRouter(prefix="/api", dependencies=[Depends(private_response)])

    def owner(request):
        actor = getattr(request.state, "user_id", None)
        if not actor:
            raise HTTPException(401, "Trace 查询需要已认证身份；请配置 SERVICEPILOT_AUTH_TOKENS")
        if service.store is None:
            raise HTTPException(503, "Trace 存储暂不可用")
        return service.redactor.pseudonym(actor)

    def list_owned(request, **kwargs):
        identity = owner(request)
        try:
            return service.store.list(identity, **kwargs)
        except Exception as exc:
            raise HTTPException(503, "Trace 存储暂不可用") from exc

    def detail_owned(request, trace_id):
        identity = owner(request)
        try:
            result = service.store.detail(trace_id, identity)
        except Exception as exc:
            raise HTTPException(503, "Trace 存储暂不可用") from exc
        if result is None:
            # Same response for an absent trace and another user's trace.
            raise HTTPException(404, "未找到可访问的 Trace")
        return result

    @router.get("/traces")
    def traces(
        request: Request,
        limit: int = Query(30, ge=1, le=100),
        offset: int = Query(0, ge=0),
        status: str | None = None,
        start: float | None = None,
        end: float | None = None,
        thread_id: str | None = None,
        task_id: str | None = None,
        error_type: str | None = None,
        node: str | None = None,
    ):
        if start is not None and end is not None and start > end:
            raise HTTPException(422, "开始时间不能晚于结束时间")
        return list_owned(
            request,
            limit=limit,
            offset=offset,
            status=status,
            start=start,
            end=end,
            thread_id=thread_id,
            task_id=task_id,
            error_type=error_type,
            node=node,
        )

    @router.get("/traces/{trace_id}")
    def detail(trace_id: str, request: Request):
        return detail_owned(request, trace_id)

    @router.get("/traces/{trace_id}/metrics")
    def metrics(trace_id: str, request: Request):
        trace = detail_owned(request, trace_id)["trace"]
        keys = [
            "total_latency_ms",
            "total_input_tokens",
            "total_output_tokens",
            "total_llm_calls",
            "total_tool_calls",
            "replan_count",
            "error_count",
            "status",
            "incomplete",
        ]
        return {key: trace.get(key) for key in keys}

    @router.get("/traces/{trace_id}/export")
    def export(trace_id: str, request: Request):
        return JSONResponse(
            detail_owned(request, trace_id),
            headers={
                "Content-Disposition": 'attachment; filename="servicepilot-trace.json"',
                "Cache-Control": "no-store",
            },
        )

    @router.get("/tasks/{task_id}/traces")
    def task_traces(
        task_id: str,
        request: Request,
        limit: int = Query(30, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ):
        return list_owned(request, task_id=task_id, limit=limit, offset=offset)

    @router.get("/threads/{thread_id}/traces")
    def thread_traces(
        thread_id: str,
        request: Request,
        limit: int = Query(30, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ):
        return list_owned(request, thread_id=thread_id, limit=limit, offset=offset)

    app.include_router(router)

    @app.get("/traces", include_in_schema=False)
    def viewer():
        return FileResponse(assets_dir / "traces.html", media_type="text/html")

    @app.get("/traces.js", include_in_schema=False)
    def viewer_script():
        return FileResponse(assets_dir / "traces.js", media_type="text/javascript")

    @app.get("/traces.css", include_in_schema=False)
    def viewer_style():
        return FileResponse(assets_dir / "traces.css", media_type="text/css")
