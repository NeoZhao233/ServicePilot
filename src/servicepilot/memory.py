from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .models import AgentEvent, Intent, SessionState


def serialize_state(state: SessionState) -> str:
    payload = asdict(state)
    payload["intent"] = state.intent.value
    return json.dumps(payload, ensure_ascii=False)


def deserialize_state(serialized: str | bytes) -> SessionState:
    if isinstance(serialized, bytes):
        serialized = serialized.decode("utf-8")
    payload = json.loads(serialized)
    return SessionState(
        session_id=payload["session_id"],
        user_id=payload["user_id"],
        intent=Intent(payload["intent"]),
        order_id=payload.get("order_id"),
        pending_action=payload.get("pending_action"),
        refund_reason=payload.get("refund_reason"),
        turns=list(payload.get("turns", [])),
        context_summary=payload.get("context_summary"),
        turn_count=payload.get("turn_count", 0),
        context_retry_after_messages=payload.get("context_retry_after_messages", 0),
        goal_version=payload.get("goal_version", 1),
        plan_version=payload.get("plan_version", 1),
        authorization=payload.get("authorization"),
    )


class SessionStore(Protocol):
    def get(self, session_id: str) -> SessionState | None: ...

    def save(self, state: SessionState) -> None: ...

    def list_sessions(self, user_id: str, *, limit: int = 20) -> list[SessionState]: ...

    def append_events(self, session_id: str, turn_index: int, events: list[AgentEvent]) -> None: ...

    def get_events(self, session_id: str) -> list[AgentEvent]: ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._events: dict[str, list[AgentEvent]] = {}

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def save(self, state: SessionState) -> None:
        self._sessions[state.session_id] = state

    def list_sessions(self, user_id: str, *, limit: int = 20) -> list[SessionState]:
        matches = [state for state in self._sessions.values() if state.user_id == user_id]
        return list(reversed(matches[-limit:]))

    def append_events(self, session_id: str, turn_index: int, events: list[AgentEvent]) -> None:
        del turn_index
        self._events.setdefault(session_id, []).extend(events)

    def get_events(self, session_id: str) -> list[AgentEvent]:
        return list(self._events.get(session_id, []))


class SQLiteSessionStore:
    """Durable short-term conversation state without an external service."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    event_index INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, turn_index, event_index)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_events_session_id
                ON agent_events(session_id, id)
                """
            )
            connection.execute("PRAGMA optimize")

    def get(self, session_id: str) -> SessionState | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return deserialize_state(row["state_json"])

    def save(self, state: SessionState) -> None:
        serialized = serialize_state(state)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions (session_id, user_id, state_json)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    state_json = excluded.state_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (state.session_id, state.user_id, serialized),
            )

    def list_sessions(self, user_id: str, *, limit: int = 20) -> list[SessionState]:
        safe_limit = max(1, min(limit, 100))
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT session_id FROM sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC, session_id DESC
                LIMIT ?
                """,
                (user_id, safe_limit),
            ).fetchall()
        return [state for row in rows if (state := self.get(row["session_id"])) is not None]

    def append_events(self, session_id: str, turn_index: int, events: list[AgentEvent]) -> None:
        if not events:
            return
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(event_index), -1) AS last_index
                FROM agent_events
                WHERE session_id = ? AND turn_index = ?
                """,
                (session_id, turn_index),
            ).fetchone()
            start_index = int(row["last_index"]) + 1
            rows = [
                (
                    session_id,
                    turn_index,
                    start_index + offset,
                    event.event_type,
                    event.name,
                    json.dumps(event.detail, ensure_ascii=False),
                )
                for offset, event in enumerate(events)
            ]
            connection.executemany(
                """
                INSERT INTO agent_events (
                    session_id, turn_index, event_index, event_type, name, detail_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_events(self, session_id: str) -> list[AgentEvent]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT event_type, name, detail_json
                FROM agent_events
                WHERE session_id = ?
                ORDER BY turn_index, event_index
                """,
                (session_id,),
            ).fetchall()
        return [
            AgentEvent(
                event_type=row["event_type"],
                name=row["name"],
                detail=json.loads(row["detail_json"]),
            )
            for row in rows
        ]


class RedisSessionStore:
    """TTL-backed short-term memory using a redis-py compatible client."""

    def __init__(self, client, *, ttl_seconds: int = 86_400, prefix: str = "servicepilot") -> None:
        self.client = client
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix

    @classmethod
    def from_url(cls, url: str, *, ttl_seconds: int = 86_400) -> RedisSessionStore:
        import redis

        return cls(redis.Redis.from_url(url, decode_responses=True), ttl_seconds=ttl_seconds)

    def _session_key(self, session_id: str) -> str:
        return f"{self.prefix}:session:{session_id}"

    def _events_key(self, session_id: str) -> str:
        return f"{self.prefix}:events:{session_id}"

    def _user_key(self, user_id: str) -> str:
        return f"{self.prefix}:user_sessions:{user_id}"

    def get(self, session_id: str) -> SessionState | None:
        payload = self.client.get(self._session_key(session_id))
        return deserialize_state(payload) if payload is not None else None

    def save(self, state: SessionState) -> None:
        import time

        self.client.setex(
            self._session_key(state.session_id), self.ttl_seconds, serialize_state(state)
        )
        self.client.zadd(self._user_key(state.user_id), {state.session_id: time.time()})
        self.client.expire(self._user_key(state.user_id), self.ttl_seconds)

    def list_sessions(self, user_id: str, *, limit: int = 20) -> list[SessionState]:
        session_ids = self.client.zrevrange(self._user_key(user_id), 0, max(0, limit - 1))
        result = []
        for session_id in session_ids:
            state = self.get(session_id.decode() if isinstance(session_id, bytes) else session_id)
            if state is not None:
                result.append(state)
        return result

    def append_events(self, session_id: str, turn_index: int, events: list[AgentEvent]) -> None:
        if not events:
            return
        rows = [
            json.dumps(
                {
                    "turn_index": turn_index,
                    "event_type": event.event_type,
                    "name": event.name,
                    "detail": event.detail,
                },
                ensure_ascii=False,
            )
            for event in events
        ]
        self.client.rpush(self._events_key(session_id), *rows)
        self.client.expire(self._events_key(session_id), self.ttl_seconds)

    def get_events(self, session_id: str) -> list[AgentEvent]:
        rows = self.client.lrange(self._events_key(session_id), 0, -1)
        result = []
        for row in rows:
            if isinstance(row, bytes):
                row = row.decode("utf-8")
            payload = json.loads(row)
            result.append(AgentEvent(payload["event_type"], payload["name"], payload["detail"]))
        return result


class FallbackSessionStore:
    """Use Redis primarily and transparently degrade to a durable local store."""

    def __init__(self, primary: SessionStore, fallback: SessionStore) -> None:
        self.primary = primary
        self.fallback = fallback
        self.degraded = False

    def _read(self, method: str, *args, **kwargs):
        if not self.degraded:
            try:
                return getattr(self.primary, method)(*args, **kwargs)
            except Exception:
                self.degraded = True
        return getattr(self.fallback, method)(*args, **kwargs)

    def _write(self, method: str, *args, **kwargs) -> None:
        getattr(self.fallback, method)(*args, **kwargs)
        if not self.degraded:
            try:
                getattr(self.primary, method)(*args, **kwargs)
            except Exception:
                self.degraded = True

    def get(self, session_id: str) -> SessionState | None:
        return self._read("get", session_id)

    def save(self, state: SessionState) -> None:
        self._write("save", state)

    def list_sessions(self, user_id: str, *, limit: int = 20) -> list[SessionState]:
        return self._read("list_sessions", user_id, limit=limit)

    def append_events(self, session_id: str, turn_index: int, events: list[AgentEvent]) -> None:
        self._write("append_events", session_id, turn_index, events)

    def get_events(self, session_id: str) -> list[AgentEvent]:
        return self._read("get_events", session_id)
