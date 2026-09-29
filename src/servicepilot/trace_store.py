from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path


class TraceStore:
    """Independent observation tables; never enlisted in a commerce transaction."""

    def __init__(self, path: str | Path = ":memory:", *, key: str | None = None):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, timeout=0.05, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        with self.connection:
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS obs_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS obs_traces (
                    trace_id TEXT PRIMARY KEY, owner TEXT, thread_id TEXT, task_id TEXT,
                    request_id TEXT NOT NULL, start_time REAL NOT NULL, end_time REAL,
                    status TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS obs_spans (
                    span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL REFERENCES obs_traces(trace_id)
                    ON DELETE CASCADE, parent_span_id TEXT, name TEXT NOT NULL,
                    span_type TEXT NOT NULL, status TEXT NOT NULL, error_type TEXT,
                    start_time REAL NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS obs_events (
                    event_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL REFERENCES obs_traces(trace_id)
                    ON DELETE CASCADE, span_id TEXT, event_type TEXT NOT NULL,
                    task_id TEXT, plan_version INTEGER, timestamp REAL NOT NULL,
                    payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS obs_trace_owner_time
                    ON obs_traces(owner,start_time DESC);
                CREATE INDEX IF NOT EXISTS obs_trace_thread
                    ON obs_traces(owner,thread_id,start_time);
                CREATE INDEX IF NOT EXISTS obs_trace_task
                    ON obs_traces(owner,task_id,start_time);
                CREATE INDEX IF NOT EXISTS obs_trace_status
                    ON obs_traces(owner,status,start_time);
                CREATE INDEX IF NOT EXISTS obs_span_trace_parent
                    ON obs_spans(trace_id,parent_span_id);
                CREATE INDEX IF NOT EXISTS obs_span_error
                    ON obs_spans(error_type,trace_id);
                CREATE INDEX IF NOT EXISTS obs_span_name
                    ON obs_spans(name,trace_id);
                CREATE INDEX IF NOT EXISTS obs_event_trace
                    ON obs_events(trace_id,timestamp);
            """)
            self.connection.execute(
                "INSERT OR IGNORE INTO obs_meta VALUES ('hmac_key',?)",
                (key or secrets.token_hex(32),),
            )
            stored = self.connection.execute(
                "SELECT value FROM obs_meta WHERE key='hmac_key'"
            ).fetchone()
        self.key = key or stored[0]

    def start(self, trace):
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO obs_traces VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    trace["trace_id"],
                    trace.get("user_id"),
                    trace.get("thread_id"),
                    trace.get("task_id"),
                    trace["request_id"],
                    trace["start_time"],
                    trace.get("end_time"),
                    trace["status"],
                    json.dumps(trace, ensure_ascii=False),
                ),
            )

    def finish(self, trace, spans, events):
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO obs_traces VALUES (?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(trace_id) DO UPDATE SET owner=excluded.owner,"
                "thread_id=excluded.thread_id,task_id=excluded.task_id,end_time=excluded.end_time,"
                "status=excluded.status,payload=excluded.payload",
                (
                    trace["trace_id"],
                    trace.get("user_id"),
                    trace.get("thread_id"),
                    trace.get("task_id"),
                    trace["request_id"],
                    trace["start_time"],
                    trace.get("end_time"),
                    trace["status"],
                    json.dumps(trace, ensure_ascii=False),
                ),
            )
            self.connection.executemany(
                "INSERT OR REPLACE INTO obs_spans VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        s["span_id"],
                        s["trace_id"],
                        s["parent_span_id"],
                        s["name"],
                        s["span_type"],
                        s["status"],
                        s.get("error_type"),
                        s["start_time"],
                        json.dumps(s, ensure_ascii=False),
                    )
                    for s in spans
                ],
            )
            self.connection.executemany(
                "INSERT OR REPLACE INTO obs_events VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        e["event_id"],
                        e["trace_id"],
                        e.get("span_id"),
                        e["event_type"],
                        e.get("task_id"),
                        e.get("plan_version"),
                        e["timestamp"],
                        json.dumps(e, ensure_ascii=False),
                    )
                    for e in events
                ],
            )

    def list(
        self,
        owner,
        *,
        thread_id=None,
        task_id=None,
        status=None,
        start=None,
        end=None,
        error_type=None,
        node=None,
        limit=30,
        offset=0,
    ):
        terms, values = ["owner=?"], [owner]
        for column, value in [("thread_id", thread_id), ("task_id", task_id), ("status", status)]:
            if value is not None:
                terms.append(f"{column}=?")
                values.append(value)
        for operator, value in [(">=", start), ("<=", end)]:
            if value is not None:
                terms.append(f"start_time{operator}?")
                values.append(value)
        for column, value in [("error_type", error_type), ("name", node)]:
            if value:
                terms.append(
                    "EXISTS (SELECT 1 FROM obs_spans s WHERE s.trace_id=obs_traces.trace_id "
                    f"AND s.{column}=?)"
                )
                values.append(value)
        where = " AND ".join(terms)
        with self.lock:
            total = self.connection.execute(
                "SELECT count(*) FROM obs_traces WHERE " + where, values
            ).fetchone()[0]
            rows = self.connection.execute(
                "SELECT payload FROM obs_traces WHERE "
                + where
                + " ORDER BY start_time DESC,trace_id LIMIT ? OFFSET ?",
                [*values, max(1, min(limit, 100)), max(0, offset)],
            ).fetchall()
        return {
            "items": [json.loads(r[0]) for r in rows],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    def detail(self, trace_id, owner):
        with self.lock:
            trace = self.connection.execute(
                "SELECT payload FROM obs_traces WHERE trace_id=? AND owner=?", (trace_id, owner)
            ).fetchone()
            if not trace:
                return None
            spans = [
                json.loads(r[0])
                for r in self.connection.execute(
                    "SELECT payload FROM obs_spans WHERE trace_id=? ORDER BY start_time,span_id",
                    (trace_id,),
                )
            ]
            events = [
                json.loads(r[0])
                for r in self.connection.execute(
                    "SELECT payload FROM obs_events WHERE trace_id=? ORDER BY timestamp,event_id",
                    (trace_id,),
                )
            ]
        nodes = {s["span_id"]: {**s, "children": []} for s in spans}
        tree = []
        for s in spans:
            node = nodes[s["span_id"]]
            parent = nodes.get(s["parent_span_id"])
            (parent["children"] if parent else tree).append(node)
        return {"trace": json.loads(trace[0]), "spans": spans, "tree": tree, "events": events}

    def cleanup(self, retention_days=14, *, now=None):
        if retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        cutoff = (time.time() if now is None else now) - retention_days * 86400
        with self.lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM obs_traces WHERE end_time<? AND status!='running'", (cutoff,)
            )
            return cursor.rowcount

    def mark_abandoned(self, older_than_seconds=3600):
        """Operator operation: call only after confirming the owning workers stopped."""
        cutoff = time.time() - older_than_seconds
        with self.lock, self.connection:
            rows = self.connection.execute(
                "SELECT trace_id,payload FROM obs_traces WHERE status='running' AND start_time<?",
                (cutoff,),
            ).fetchall()
            for row in rows:
                trace = json.loads(row["payload"])
                trace.update(
                    status="interrupted",
                    end_time=time.time(),
                    total_latency_ms=None,
                    total_input_tokens=None,
                    total_output_tokens=None,
                    total_llm_calls=None,
                    total_tool_calls=None,
                    replan_count=None,
                    error_count=None,
                    incomplete=True,
                )
                self.connection.execute(
                    "UPDATE obs_traces SET status=?,end_time=?,payload=? WHERE trace_id=?",
                    ("interrupted", trace["end_time"], json.dumps(trace), row["trace_id"]),
                )
            return len(rows)
