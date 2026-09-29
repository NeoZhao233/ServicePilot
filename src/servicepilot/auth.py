from __future__ import annotations

import hmac
import json
import os

from .observability import current_trace


def install_identity_boundary(app):
    """Optional bearer authentication for the demo HTTP adapters.

    Production gateways may replace this with their existing identity provider.
    An unset token map explicitly retains the original local-demo contract.
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse

    mapping = json.loads(os.getenv("SERVICEPILOT_AUTH_TOKENS", "{}"))
    if not isinstance(mapping, dict) or any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not v
        for k, v in mapping.items()
    ):
        raise ValueError("SERVICEPILOT_AUTH_TOKENS 必须是非空 token 到 user_id 的映射")

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if not mapping or request.url.path in {
            "/",
            "/health",
            "/api/health",
            "/docs",
            "/openapi.json",
        }:
            return await call_next(request)
        if not (request.url.path.startswith("/api/") or request.url.path == "/chat"):
            return await call_next(request)
        header = request.headers.get("authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        actor = next(
            (user for key, user in mapping.items() if hmac.compare_digest(key, token)), None
        )
        if actor is None:
            return JSONResponse({"detail": "需要有效身份凭证"}, status_code=401)
        claimed = request.query_params.get("user_id")
        if request.method == "POST":
            try:
                if request.url.path == "/api/pdf/upload":
                    # The endpoint checks the multipart form identity after authentication.
                    claimed = actor
                else:
                    body = await request.json()
                    claimed = body.get("user_id") if isinstance(body, dict) else None
            except (ValueError, TypeError):
                return JSONResponse({"detail": "请求格式无效"}, status_code=400)
        trace_query = request.method == "GET" and request.url.path.startswith(
            ("/api/traces", "/api/tasks/", "/api/threads/")
        )
        if (claimed is not None or not trace_query) and claimed != actor:
            return JSONResponse({"detail": "请求用户与认证身份不一致"}, status_code=403)
        request.state.user_id = actor
        collector = current_trace()
        if collector:
            collector.bind(user_id=actor)
        return await call_next(request)
