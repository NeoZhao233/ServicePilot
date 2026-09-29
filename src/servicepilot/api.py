from __future__ import annotations

import os
from pathlib import Path

from .auth import install_identity_boundary
from .cli import build_agent
from .observability import current_trace
from .repository import SessionBusyError
from .trace_http import install_trace_api


def create_app(data_dir: Path | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("请先安装 API 依赖：pip install -e '.[api]'") from exc

    class ChatRequest(BaseModel):
        user_id: str
        message: str
        session_id: str | None = None

    app = FastAPI(title="ServicePilot", version="0.1.0")
    root = data_dir or Path(__file__).resolve().parents[2] / "data"
    session_db_value = os.getenv("SERVICEPILOT_SESSION_DB")
    session_db = Path(session_db_value) if session_db_value else None
    online = os.getenv("SERVICEPILOT_ONLINE", "").lower() in {"1", "true", "yes"}
    agent = build_agent(root, session_db=session_db, online=online)

    install_identity_boundary(app)
    install_trace_api(app, agent.trace_service, Path(__file__).parent / "web_assets")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        try:
            response = agent.chat(
                user_id=request.user_id,
                message=request.message,
                session_id=request.session_id,
            )
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, PermissionError) as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return {
            "trace_id": current_trace().record["trace_id"] if current_trace() else None,
            "session_id": response.session.session_id,
            "message": response.message,
            "citations": response.citations,
            "events": [
                {"type": event.event_type, "name": event.name, "detail": event.detail}
                for event in response.events
            ],
        }

    return app


app = create_app()
