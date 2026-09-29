from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .auth import install_identity_boundary
from .cli import build_agent
from .observability import current_trace
from .pdf_service import SessionPdfStore, configured_pdf_answer_model, configured_vision_model
from .repository import SessionBusyError
from .trace_http import install_trace_api


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = Field(default=None, max_length=128)
    document_id: str | None = Field(default=None, max_length=64)


class PdfQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    document_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)


def serialize_response(response) -> dict[str, Any]:
    return {
        "trace_id": current_trace().record["trace_id"] if current_trace() else None,
        "session_id": response.session.session_id,
        "message": response.message,
        "intent": response.session.intent.value,
        "order_id": response.session.order_id,
        "pending_action": response.session.pending_action,
        "citations": list(response.citations),
        "events": [
            {"type": event.event_type, "name": event.name, "detail": event.detail}
            for event in response.events
        ],
    }


def serialize_session(state, events=None) -> dict[str, Any]:
    last_assistant = next(
        (turn["content"] for turn in reversed(state.turns) if turn.get("role") == "assistant"),
        "新会话",
    )
    payload = {
        "session_id": state.session_id,
        "user_id": state.user_id,
        "intent": state.intent.value,
        "order_id": state.order_id,
        "pending_action": state.pending_action,
        "turn_count": state.turn_count or sum(turn.get("role") == "user" for turn in state.turns),
        "last_message": last_assistant,
        "turns": state.turns,
    }
    if events is not None:
        payload["events"] = [
            {"type": event.event_type, "name": event.name, "detail": event.detail}
            for event in events
        ]
    return payload


def create_app(
    *,
    data_dir: Path,
    session_db: Path | None = None,
    online: bool = False,
    chroma_dir: Path | None = None,
    redis_url: str | None = None,
    commerce_db: Path | None = None,
    dynamic: bool | None = None,
    trace_service=None,
    pdf_dir: Path | None = None,
) -> FastAPI:
    agent = build_agent(
        data_dir,
        session_db=session_db,
        online=online,
        chroma_dir=chroma_dir,
        redis_url=redis_url,
        commerce_db=commerce_db,
        dynamic=dynamic,
        trace_service=trace_service,
    )
    assets_dir = Path(__file__).resolve().parent / "web_assets"
    app = FastAPI(
        title="ServicePilot API",
        version="0.1.0",
        description="Policy-grounded e-commerce after-sales Agent API",
    )

    install_identity_boundary(app)
    install_trace_api(app, agent.trace_service, assets_dir)
    pdf_store = SessionPdfStore(
        encoder=agent.agent.pdf_encoder if hasattr(agent, "agent") else agent.pdf_encoder,
        vision=configured_vision_model(),
        answer_model=configured_pdf_answer_model(),
        root=(
            pdf_dir or Path(os.environ["SERVICEPILOT_PDF_DIR"])
            if pdf_dir or os.getenv("SERVICEPILOT_PDF_DIR")
            else session_db.parent / "pdfs"
            if session_db
            else None
        ),
    )
    app.state.pdf_store = pdf_store

    def public_pdf_evidence(result, *, user_id, session_id, document_id):
        evidence = []
        query = urlencode({"user_id": user_id, "session_id": session_id})
        for bundle in result["evidence"]:
            item = {key: value for key, value in bundle.items() if key != "image_paths"}
            item["images"] = [
                f"/api/pdf/{document_id}/images/{Path(path).stem}?{query}"
                for path in bundle.get("image_paths", ())
            ]
            evidence.append(item)
        return {**result, "evidence": evidence}

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        if request.document_id:
            if not request.session_id:
                raise HTTPException(status_code=400, detail="PDF 问答需要已有会话")
            try:
                result = pdf_store.ask(
                    user_id=request.user_id.strip(),
                    session_id=request.session_id,
                    document_id=request.document_id,
                    question=request.message.strip(),
                )
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            result = public_pdf_evidence(
                result,
                user_id=request.user_id.strip(),
                session_id=request.session_id,
                document_id=request.document_id,
            )
            return {
                "trace_id": current_trace().record["trace_id"] if current_trace() else None,
                "session_id": request.session_id,
                "message": result["answer"],
                "intent": "document_qa",
                "order_id": None,
                "pending_action": None,
                "citations": [
                    f"{request.document_id}:p{item['page_number']}" for item in result["evidence"]
                ],
                "events": [],
                "pdf_evidence": result["evidence"],
                "vision_called": result["vision_called"],
                "answer_mode": result["answer_mode"],
            }
        try:
            response = agent.chat(
                user_id=request.user_id.strip(),
                message=request.message.strip(),
                session_id=request.session_id,
            )
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return serialize_response(response)

    @app.post("/api/pdf/upload")
    async def upload_pdf(
        request: Request,
        user_id: Annotated[str, Form()],
        file: Annotated[UploadFile, File()],
        session_id: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        authenticated = getattr(request.state, "user_id", None)
        if authenticated is not None and authenticated != user_id:
            raise HTTPException(status_code=403, detail="请求用户与认证身份不一致")
        if session_id:
            state = agent.get_session(session_id)
            if state is None or state.user_id != user_id:
                raise HTTPException(status_code=403, detail="无权访问该会话")
        else:
            base_agent = agent.agent if hasattr(agent, "agent") else agent
            session_id = base_agent.new_document_session(user_id).session_id
        if file.content_type != "application/pdf" and not (file.filename or "").lower().endswith(
            ".pdf"
        ):
            raise HTTPException(status_code=400, detail="仅支持 PDF 文件")
        content = await file.read(15 * 1024 * 1024 + 1)
        try:
            document_id, elements = pdf_store.upload(
                user_id=user_id,
                session_id=session_id,
                content=content,
                filename=file.filename or "document.pdf",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "document_id": document_id,
            "session_id": session_id,
            "filename": Path(file.filename or "document.pdf").name,
            "counts": {
                kind: sum(e.element_type == kind for e in elements)
                for kind in ("text", "table", "image")
            },
            "elements": [
                {
                    "page_number": e.page_number,
                    "element_id": e.element_id,
                    "element_type": e.element_type,
                    "content": e.content,
                    "has_image": bool(e.image_path),
                    "related_element_ids": e.related_element_ids,
                    "bbox": e.bbox,
                    "label": e.label,
                }
                for e in elements
            ],
        }

    @app.get("/api/pdf/documents")
    def list_pdf_documents(user_id: str, session_id: str) -> dict[str, Any]:
        state = agent.get_session(session_id)
        if state is None or state.user_id != user_id:
            raise HTTPException(status_code=403, detail="无权访问该会话")
        return {"documents": pdf_store.list_documents(user_id=user_id, session_id=session_id)}

    @app.get("/api/pdf/{document_id}/images/{element_id}")
    def pdf_image(document_id: str, element_id: str, user_id: str, session_id: str):
        try:
            path = pdf_store.image_for(
                user_id=user_id,
                session_id=session_id,
                document_id=document_id,
                element_id=element_id,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            path, media_type="image/png", headers={"Cache-Control": "private, no-store"}
        )

    @app.delete("/api/pdf/{document_id}")
    def delete_pdf(document_id: str, user_id: str, session_id: str):
        try:
            pdf_store.delete(user_id=user_id, session_id=session_id, document_id=document_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return {"deleted": document_id}

    @app.post("/api/pdf/ask")
    def ask_pdf(request: PdfQuestion) -> dict[str, Any]:
        try:
            result = pdf_store.ask(**request.model_dump())
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return public_pdf_evidence(
            result,
            user_id=request.user_id,
            session_id=request.session_id,
            document_id=request.document_id,
        )

    @app.get("/api/sessions")
    def list_sessions(
        user_id: str = Query(min_length=1, max_length=128),
    ) -> dict[str, object]:
        return {
            "sessions": [serialize_session(item) for item in agent.list_sessions(user_id.strip())]
        }

    @app.get("/api/sessions/{session_id}")
    def get_session(
        session_id: str,
        user_id: str = Query(min_length=1, max_length=128),
    ) -> dict[str, Any]:
        state = agent.get_session(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        if state.user_id != user_id.strip():
            raise HTTPException(status_code=403, detail="无权访问该会话")
        events = agent.get_events(session_id, user_id=user_id.strip())
        return serialize_session(state, events)

    @app.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        return FileResponse(assets_dir / "index.html", media_type="text/html")

    @app.get("/app.js", response_class=FileResponse)
    def javascript() -> FileResponse:
        return FileResponse(assets_dir / "app.js", media_type="text/javascript")

    @app.get("/styles.css", response_class=FileResponse)
    def stylesheet() -> FileResponse:
        return FileResponse(assets_dir / "styles.css", media_type="text/css")

    return app


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the ServicePilot FastAPI app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument(
        "--session-db",
        type=Path,
        default=project_root / ".servicepilot" / "sessions.sqlite3",
    )
    parser.add_argument("--redis-url", help="Redis URL with automatic SQLite fallback")
    parser.add_argument("--online", action="store_true")
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=project_root / ".servicepilot" / "chroma",
    )
    parser.add_argument("--commerce-db", type=Path)
    parser.add_argument("--dynamic-planning", action="store_true", default=None)
    args = parser.parse_args()
    app = create_app(
        data_dir=args.data_dir,
        session_db=args.session_db,
        online=args.online,
        chroma_dir=args.chroma_dir,
        redis_url=args.redis_url,
        commerce_db=args.commerce_db,
        dynamic=args.dynamic_planning,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
