from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from .models import Order, ToolResult
from .observability import observed
from .policy_guard import PolicyGuard
from .rules import evaluate_refund


def fingerprint(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def money_minor(amount) -> int:
    value = Decimal(str(amount)) * 100
    if not value.is_finite() or value < 0 or value != value.to_integral_value():
        raise ValueError("金额必须是非负整数分")
    return int(value)


def failure(code: str, message: str, outcome: str = "not_submitted", **details) -> ToolResult:
    return ToolResult(False, {"code": code, "outcome": outcome, **details}, message)


class SessionBusyError(RuntimeError):
    pass


class SQLiteCommerceRepository:
    """Transactional commerce simulator shared by local and MCP adapters.

    This is a local business service, not a claim of exactly-once delivery to a
    remote payment provider. External adapters must implement the same contract.
    """

    def __init__(self, path: str | Path, orders: list[Order] | None = None) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None, timeout=10
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout=10000")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS commerce_orders (
                order_id TEXT PRIMARY KEY, payload TEXT NOT NULL, version INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS commerce_contexts (
                context_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                goal_version INTEGER NOT NULL, plan_version INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS commerce_grants (
                token TEXT PRIMARY KEY, operation_id TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL, order_id TEXT NOT NULL, snapshot TEXT NOT NULL,
                context_id TEXT NOT NULL, goal_version INTEGER NOT NULL,
                plan_version INTEGER NOT NULL, expires REAL NOT NULL,
                confirmed INTEGER NOT NULL DEFAULT 0, invalidated INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS commerce_operations (
                operation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, order_id TEXT NOT NULL,
                action TEXT NOT NULL, state TEXT NOT NULL, result TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '');
            CREATE TABLE IF NOT EXISTS commerce_refunds (
                id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT UNIQUE NOT NULL,
                operation_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS commerce_tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT, dedupe_key TEXT UNIQUE NOT NULL,
                payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS commerce_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, operation_id TEXT NOT NULL,
                action TEXT NOT NULL, outcome TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS commerce_tasks (
                task_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS commerce_escalations (
                escalation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                reason TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'open');
            CREATE TABLE IF NOT EXISTS commerce_leases (
                context_id TEXT PRIMARY KEY, owner TEXT NOT NULL);
        """)
        with self.transaction() as db:
            for order in orders or []:
                data = asdict(order)
                data["amount_minor"] = money_minor(data["amount"])
                db.execute(
                    "INSERT OR IGNORE INTO commerce_orders VALUES (?, ?, 1)",
                    (order.order_id, json.dumps(data, ensure_ascii=False)),
                )

    @classmethod
    def from_json(cls, path: str | Path, *, database: str | Path = ":memory:"):
        orders = [Order(**item) for item in json.loads(Path(path).read_text(encoding="utf-8"))]
        return cls(database, orders)

    @contextmanager
    def transaction(self):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    @contextmanager
    def session_guard(self, context_id: str):
        # No automatic lease expiry: a paused writer must never regain authority
        # after another process takes over. A crashed lease requires explicit recovery.
        owner = uuid4().hex
        with self.transaction() as db:
            try:
                db.execute("INSERT INTO commerce_leases VALUES (?, ?)", (context_id, owner))
            except sqlite3.IntegrityError as exc:
                raise SessionBusyError("会话正在处理中或需要恢复，请勿重复提交") from exc
        try:
            yield
        finally:
            with self.transaction() as db:
                db.execute(
                    "DELETE FROM commerce_leases WHERE context_id=? AND owner=?",
                    (context_id, owner),
                )

    def close(self):
        self._db.close()

    @staticmethod
    def _lookup(db, order_id, user_id):
        row = db.execute("SELECT * FROM commerce_orders WHERE order_id=?", (order_id,)).fetchone()
        if row is None:
            return failure("not_found", "订单不存在")
        data = json.loads(row["payload"])
        if data.pop("user_id") != user_id:
            return failure("permission", "无权查看该订单")
        data["version"] = row["version"]
        return ToolResult(True, data)

    @observed("commerce.get_order", "business")
    def get_order(self, *, order_id: str, user_id: str) -> ToolResult:
        with self.transaction() as db:
            return self._lookup(db, order_id, user_id)

    def update_order(self, order_id: str, **changes) -> None:
        """Trusted simulator event API; deliberately not exposed as an agent tool."""
        with self.transaction() as db:
            row = db.execute(
                "SELECT payload FROM commerce_orders WHERE order_id=?", (order_id,)
            ).fetchone()
            if row is None:
                raise KeyError(order_id)
            data = json.loads(row[0])
            data.update(changes)
            data["amount_minor"] = money_minor(data["amount"])
            db.execute(
                "UPDATE commerce_orders SET payload=?, version=version+1 WHERE order_id=?",
                (json.dumps(data, ensure_ascii=False), order_id),
            )

    def set_context(self, context_id: str, user_id: str, goal_version: int, plan_version: int):
        with self.transaction() as db:
            old = db.execute(
                "SELECT * FROM commerce_contexts WHERE context_id=?", (context_id,)
            ).fetchone()
            if old and (
                old["user_id"] != user_id
                or goal_version < old["goal_version"]
                or plan_version < old["plan_version"]
            ):
                raise PermissionError("过期的任务版本或身份")
            db.execute(
                "INSERT INTO commerce_contexts VALUES (?, ?, ?, ?) ON CONFLICT(context_id) "
                "DO UPDATE SET goal_version=excluded.goal_version, "
                "plan_version=excluded.plan_version",
                (context_id, user_id, goal_version, plan_version),
            )

    def invalidate_authorization(self, token: str | None):
        if token:
            with self.transaction() as db:
                db.execute("UPDATE commerce_grants SET invalidated=1 WHERE token=?", (token,))

    @observed("commerce.prepare_refund", "business")
    def prepare_refund(
        self, *, order_id: str, user_id: str, context_id: str, goal_version: int, plan_version: int
    ) -> ToolResult:
        with self.transaction() as db:
            lookup = self._lookup(db, order_id, user_id)
            if not lookup.ok:
                return lookup
            decision = evaluate_refund(lookup.data, damaged=False)
            if not decision.eligible or decision.requires_human:
                return failure("precondition", decision.reason)
            token, operation_id = uuid4().hex, uuid4().hex
            snapshot = fingerprint(lookup.data)
            db.execute(
                "INSERT INTO commerce_grants (token,operation_id,user_id,order_id,snapshot,"
                "context_id,goal_version,plan_version,expires) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    token,
                    operation_id,
                    user_id,
                    order_id,
                    snapshot,
                    context_id,
                    goal_version,
                    plan_version,
                    time.time() + 900,
                ),
            )
            db.execute(
                "INSERT INTO commerce_operations VALUES (?,?,?,?,?,?,?)",
                (operation_id, user_id, order_id, "create_refund_request", "prepared", "{}", ""),
            )
            return ToolResult(
                True,
                {
                    "token": token,
                    "operation_id": operation_id,
                    "snapshot": snapshot,
                    "amount_minor": lookup.data["amount_minor"],
                    "order_version": lookup.data["version"],
                },
            )

    @observed("commerce.confirm_authorization", "business")
    def confirm_authorization(self, *, token: str, user_id: str) -> ToolResult:
        """Trusted UI/agent confirmation boundary. Never available to the planner."""
        with self.transaction() as db:
            row = db.execute("SELECT * FROM commerce_grants WHERE token=?", (token,)).fetchone()
            if not row or row["user_id"] != user_id or row["invalidated"]:
                return failure("authorization", "授权不存在或已失效")
            db.execute("UPDATE commerce_grants SET confirmed=1 WHERE token=?", (token,))
            return ToolResult(True)

    @observed("commerce.create_refund_request", "business")
    def create_refund_request(
        self, *, order_id: str, user_id: str, reason: str, authorization: str | None = None
    ) -> ToolResult:
        with self.transaction() as db:
            grant = db.execute(
                "SELECT * FROM commerce_grants WHERE token=?", (authorization,)
            ).fetchone()
            if not grant or grant["user_id"] != user_id or grant["order_id"] != order_id:
                return failure(
                    "authorization", "缺少绑定订单的有效授权", guard_reason="missing_or_wrong_grant"
                )
            op = grant["operation_id"]
            lookup = self._lookup(db, order_id, user_id)
            if not lookup.ok:
                return lookup
            existing = db.execute(
                "SELECT * FROM commerce_operations WHERE operation_id=?", (op,)
            ).fetchone()
            if existing["state"] == "succeeded":
                db.execute(
                    "INSERT INTO commerce_attempts(operation_id,action,outcome) VALUES (?,?,?)",
                    (op, "create_refund_request", "deduplicated"),
                )
                return ToolResult(True, json.loads(existing["result"]))
            if existing["state"] == "unknown":
                return failure("outcome_unknown", "操作结果待核对，已冻结提交", "unknown")
            context = db.execute(
                "SELECT * FROM commerce_contexts WHERE context_id=?", (grant["context_id"],)
            ).fetchone()
            valid = PolicyGuard.authorization_valid(
                grant, context, user_id=user_id, snapshot=fingerprint(lookup.data)
            )
            if not valid:
                return failure(
                    "authorization",
                    "订单、金额或任务已变化，请重新确认",
                    guard_reason=PolicyGuard.rejection_reason(
                        grant, context, user_id=user_id, snapshot=fingerprint(lookup.data)
                    ),
                )
            decision = evaluate_refund(lookup.data, damaged=False)
            if not decision.eligible or decision.requires_human:
                return failure("precondition", decision.reason)
            if db.execute(
                "SELECT 1 FROM commerce_refunds WHERE order_id=?", (order_id,)
            ).fetchone():
                return failure("precondition", "订单已有退款申请")
            cursor = db.execute(
                "INSERT INTO commerce_refunds(order_id,operation_id,payload) VALUES (?,?,?)",
                (order_id, op, "{}"),
            )
            record = {
                "request_id": f"REF-{cursor.lastrowid:04d}",
                "order_id": order_id,
                "operation_id": op,
                "reason": reason,
                "amount_minor": lookup.data["amount_minor"],
                "outcome": "succeeded",
            }
            payload = json.dumps(record, ensure_ascii=False)
            db.execute("UPDATE commerce_refunds SET payload=? WHERE operation_id=?", (payload, op))
            data = dict(lookup.data, user_id=user_id, status="refunding")
            data.pop("version")
            db.execute(
                "UPDATE commerce_orders SET payload=?,version=version+1 WHERE order_id=?",
                (json.dumps(data, ensure_ascii=False), order_id),
            )
            db.execute(
                "UPDATE commerce_operations SET state='succeeded',result=? WHERE operation_id=?",
                (payload, op),
            )
            db.execute(
                "INSERT INTO commerce_attempts(operation_id,action,outcome) VALUES (?,?,?)",
                (op, "create_refund_request", "succeeded"),
            )
            return ToolResult(True, record)

    @observed("commerce.get_operation", "tool_query")
    def get_operation(self, *, operation_id: str, user_id: str) -> ToolResult:
        with self.transaction() as db:
            row = db.execute(
                "SELECT * FROM commerce_operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
            if not row or row["user_id"] != user_id:
                return failure("permission", "无法查询该业务操作", "unknown")
            result = json.loads(row["result"])
            return ToolResult(
                True,
                {
                    **result,
                    "operation_id": operation_id,
                    "state": row["state"],
                    "reason": row["reason"],
                },
            )

    @observed("commerce.freeze_operation", "business")
    def freeze_operation(self, operation_id: str, user_id: str, reason: str):
        with self.transaction() as db:
            db.execute(
                "UPDATE commerce_operations SET state='unknown',reason=? "
                "WHERE operation_id=? AND user_id=? AND state!='succeeded'",
                (reason, operation_id, user_id),
            )

        self.escalate(operation_id, user_id, reason)

    def escalate(self, escalation_id: str, user_id: str, reason: str):
        with self.transaction() as db:
            db.execute(
                "INSERT INTO commerce_escalations(escalation_id,user_id,reason) "
                "VALUES (?,?,?) ON CONFLICT(escalation_id) DO UPDATE SET reason=excluded.reason",
                (escalation_id, user_id, reason),
            )

    @observed("commerce.create_ticket", "business")
    def create_ticket(self, *, order_id: str, user_id: str, description: str) -> ToolResult:
        with self.transaction() as db:
            lookup = self._lookup(db, order_id, user_id)
            if not lookup.ok:
                return lookup
            key = fingerprint(
                {"order_id": order_id, "user_id": user_id, "description": description}
            )
            old = db.execute(
                "SELECT payload FROM commerce_tickets WHERE dedupe_key=?", (key,)
            ).fetchone()
            if old:
                return ToolResult(True, json.loads(old[0]))
            cursor = db.execute(
                "INSERT INTO commerce_tickets(dedupe_key,payload) VALUES (?,?)", (key, "{}")
            )
            record = {
                "ticket_id": f"TICKET-{cursor.lastrowid:04d}",
                "order_id": order_id,
                "description": description,
            }
            db.execute(
                "UPDATE commerce_tickets SET payload=? WHERE dedupe_key=?",
                (json.dumps(record, ensure_ascii=False), key),
            )
            return ToolResult(True, record)

    @observed("commerce.get_ticket", "tool_query")
    def get_ticket(self, *, ticket_id: str, user_id: str) -> ToolResult:
        with self.transaction() as db:
            try:
                numeric = int(ticket_id.removeprefix("TICKET-"))
            except ValueError:
                return failure("schema", "工单编号无效")
            row = db.execute(
                "SELECT payload FROM commerce_tickets WHERE id=?", (numeric,)
            ).fetchone()
            if row is None:
                return failure("not_found", "工单不存在")
            data = json.loads(row[0])
            owned = self._lookup(db, data["order_id"], user_id)
            return ToolResult(True, data) if owned.ok else owned

    def save_task(self, task: dict):
        with self.transaction() as db:
            db.execute(
                "INSERT INTO commerce_tasks VALUES (?,?) ON CONFLICT(task_id) "
                "DO UPDATE SET payload=excluded.payload",
                (task["task_id"], json.dumps(task, ensure_ascii=False)),
            )

    def load_task(self, task_id: str) -> dict | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT payload FROM commerce_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            return json.loads(row[0]) if row else None


class InMemoryCommerceRepository(SQLiteCommerceRepository):
    """Isolated simulator retaining the public constructor used by existing tests."""

    def __init__(self, orders: list[Order]) -> None:
        super().__init__(":memory:", orders)

    @classmethod
    def from_json(cls, path: str | Path):
        return cls([Order(**item) for item in json.loads(Path(path).read_text(encoding="utf-8"))])
