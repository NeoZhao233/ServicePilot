import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from servicepilot.cli import build_agent
from servicepilot.models import ToolResult
from servicepilot.repository import SQLiteCommerceRepository

ROOT = Path(__file__).resolve().parents[1]


def counts(path):
    with sqlite3.connect(path) as db:
        return {
            name: db.execute(f"SELECT count(*) FROM commerce_{name}").fetchone()[0]
            for name in ("refunds", "tickets", "attempts")
        }


def runtime(tmp_path, dynamic=True):
    db = tmp_path / "business.sqlite3"
    return build_agent(ROOT / "data", session_db=db, commerce_db=db, dynamic=dynamic), db


def request(agent, message, sid=None):
    return agent.chat(user_id="USER-001", message=message, session_id=sid)


@pytest.mark.parametrize("dynamic", [False, True])
def test_explicit_goal_switch_wins_confirmation(tmp_path, dynamic):
    agent, db = runtime(tmp_path, dynamic)
    first = request(agent, "退掉 ORD-1001")
    second = request(agent, "确认改成换货", first.session.session_id)
    assert second.session.intent.value == "exchange"
    assert counts(db)["refunds"] == 0
    assert counts(db)["tickets"] == 1


@pytest.mark.parametrize("dynamic", [False, True])
def test_amount_change_invalidates_confirmation(tmp_path, dynamic):
    agent, db = runtime(tmp_path, dynamic)
    first = request(agent, "退掉 ORD-1001")
    agent.agent.repository.update_order("ORD-1001", amount=599)
    second = request(agent, "确认", first.session.session_id)
    assert counts(db)["refunds"] == 0
    assert "599.00" in second.message
    third = request(agent, "确认", first.session.session_id)
    assert counts(db)["refunds"] == 1
    assert any(text in third.message for text in ("已提交", "没有等待确认", "存在退款申请"))


@pytest.mark.parametrize("dynamic", [False, True])
def test_response_loss_queries_without_second_write(tmp_path, dynamic):
    agent, db = runtime(tmp_path, dynamic)
    first = request(agent, "退掉 ORD-1001")
    repo = agent.agent.repository
    original = repo.create_refund_request
    calls = []

    def lost(**kwargs):
        calls.append(kwargs)
        result = original(**kwargs)
        assert result.ok
        raise TimeoutError("lost after commit")

    repo.create_refund_request = lost
    second = request(agent, "确认", first.session.session_id)
    assert len(calls) == 1
    assert counts(db)["refunds"] == 1
    assert "已提交" in second.message
    assert any(e.name == "response_lost_reconciled" for e in second.events)
    request(agent, "确认", first.session.session_id)
    assert len(calls) == 1


@pytest.mark.parametrize("dynamic", [False, True])
def test_unknown_freezes_and_survives_restart(tmp_path, dynamic):
    agent, db = runtime(tmp_path, dynamic)
    first = request(agent, "退掉 ORD-1001")
    repo = agent.agent.repository
    calls = []

    def lost(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("unknown at transport")

    repo.create_refund_request = lost
    second = request(agent, "确认", first.session.session_id)
    assert "冻结" in second.message or "人工" in second.message
    assert len(calls) == 1
    restarted, _ = runtime(tmp_path, dynamic)
    third = request(restarted, "确认", first.session.session_id)
    assert "人工" in third.message
    assert counts(db)["refunds"] == 0
    op = restarted.agent.repository.get_operation(
        operation_id=first.session.authorization["operation_id"], user_id="USER-001"
    )
    assert op.data["state"] == "unknown"


@pytest.mark.parametrize("dynamic", [False, True])
def test_state_change_at_write_is_rejected(tmp_path, dynamic):
    agent, db = runtime(tmp_path, dynamic)
    first = request(agent, "退掉 ORD-1001")
    repo = agent.agent.repository
    original = repo.create_refund_request

    def race(**kwargs):
        repo.update_order("ORD-1001", status="shipped")
        return original(**kwargs)

    repo.create_refund_request = race
    second = request(agent, "确认", first.session.session_id)
    assert counts(db)["refunds"] == 0
    assert "已提交" not in second.message


@pytest.mark.parametrize("dynamic", [False, True])
def test_confirmation_and_ledger_survive_restart(tmp_path, dynamic):
    first_agent, db = runtime(tmp_path, dynamic)
    first = request(first_agent, "退掉 ORD-1001")
    second_agent, _ = runtime(tmp_path, dynamic)
    second = request(second_agent, "确认", first.session.session_id)
    assert "已提交" in second.message
    third_agent, _ = runtime(tmp_path, dynamic)
    third = request(third_agent, "确认", first.session.session_id)
    assert counts(db)["refunds"] == 1
    assert any(text in third.message for text in ("已提交", "没有等待确认", "存在退款申请"))


def authorized(repo, context="test"):
    repo.set_context(context, "USER-001", 1, 1)
    prepared = repo.prepare_refund(
        order_id="ORD-1001", user_id="USER-001", context_id=context, goal_version=1, plan_version=1
    )
    assert prepared.ok
    repo.confirm_authorization(token=prepared.data["token"], user_id="USER-001")
    return prepared.data


def test_two_connections_cannot_double_refund(tmp_path):
    db = tmp_path / "business.sqlite3"
    repo = SQLiteCommerceRepository.from_json(ROOT / "data/orders.json", database=db)
    grant = authorized(repo)
    other = SQLiteCommerceRepository.from_json(ROOT / "data/orders.json", database=db)

    def submit(r):
        return r.create_refund_request(
            order_id="ORD-1001", user_id="USER-001", reason="test", authorization=grant["token"]
        )

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(submit, [repo, other]))
    assert all(r.ok for r in results)
    assert len({r.data["request_id"] for r in results}) == 1
    assert counts(db)["refunds"] == 1


def test_atomic_rollback_on_database_failure(tmp_path):
    db = tmp_path / "business.sqlite3"
    repo = SQLiteCommerceRepository.from_json(ROOT / "data/orders.json", database=db)
    grant = authorized(repo)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TRIGGER simulate_crash BEFORE UPDATE ON commerce_orders "
            "BEGIN SELECT RAISE(ABORT, 'injected storage failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_refund_request(
            order_id="ORD-1001", user_id="USER-001", reason="test", authorization=grant["token"]
        )
    assert counts(db)["refunds"] == 0
    assert repo.get_order(order_id="ORD-1001", user_id="USER-001").data["status"] == "delivered"
    assert (
        repo.get_operation(operation_id=grant["operation_id"], user_id="USER-001").data["state"]
        == "prepared"
    )


@pytest.mark.parametrize("change", ["plan", "goal", "owner", "expiry"])
def test_guard_rejects_stale_or_foreign_authority(tmp_path, change):
    db = tmp_path / "business.sqlite3"
    repo = SQLiteCommerceRepository.from_json(ROOT / "data/orders.json", database=db)
    grant = authorized(repo)
    if change == "plan":
        repo.set_context("test", "USER-001", 1, 2)
    elif change == "goal":
        repo.set_context("test", "USER-001", 2, 2)
    elif change == "owner":
        repo.update_order("ORD-1001", user_id="USER-002")
    else:
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE commerce_grants SET expires=0")
    result = repo.create_refund_request(
        order_id="ORD-1001", user_id="USER-001", reason="test", authorization=grant["token"]
    )
    assert not result.ok
    assert counts(db)["refunds"] == 0


def test_direct_call_cannot_bypass_confirmation(tmp_path):
    agent, db = runtime(tmp_path)
    assert not agent.agent.repository.create_refund_request(
        order_id="ORD-1001", user_id="USER-001", reason="test"
    ).ok
    first = request(agent, "退掉 ORD-1001")
    assert not agent.agent.repository.create_refund_request(
        order_id="ORD-1001",
        user_id="USER-001",
        reason="test",
        authorization=first.session.authorization["token"],
    ).ok
    assert counts(db)["refunds"] == 0


def test_dynamic_cancellation_and_negation(tmp_path):
    agent, db = runtime(tmp_path)
    first = request(agent, "退掉 ORD-1001")
    request(agent, "不确认", first.session.session_id)
    response = request(agent, "算了，不退了", first.session.session_id)
    assert "取消" in response.message
    request(agent, "确认", first.session.session_id)
    assert counts(db)["refunds"] == 0
    task = agent.agent.repository.load_task(first.session.session_id)
    assert task["execution_status"] == "cancelled"


def test_switch_after_success_does_not_undo_refund(tmp_path):
    agent, db = runtime(tmp_path)
    first = request(agent, "退掉 ORD-1001")
    request(agent, "确认", first.session.session_id)
    response = request(agent, "改成换货", first.session.session_id)
    assert "不能" in response.message
    assert counts(db)["refunds"] == 1
    assert counts(db)["tickets"] == 0


def test_task_history_reuses_verified_read(tmp_path):
    agent, db = runtime(tmp_path)
    first = request(agent, "退掉 ORD-1001")
    response = request(agent, "确认改成换货", first.session.session_id)
    task = agent.agent.repository.load_task(first.session.session_id)
    assert task["history"][-1]["reason"] == "goal_changed"
    assert task["history"][-1]["plan"]["steps"][-1]["execution_status"] == "cancelled"
    assert task["current_plan"]["steps"][0]["execution_status"] == "succeeded"
    assert sum(e.name == "get_order" for e in response.events) == 1


def test_missing_policy_and_budget_fail_closed(tmp_path):
    agent, db = runtime(tmp_path)
    agent.agent.retriever.search = lambda *a, **kw: []
    response = request(agent, "退掉 ORD-1001")
    assert "政策" in response.message
    assert counts(db)["refunds"] == 0
    other = tmp_path / "other"
    other.mkdir()
    agent, db = runtime(other)
    agent.dynamic.max_tools = 0
    response = request(agent, "退掉 ORD-1001")
    assert "预算" in response.message
    assert counts(db)["refunds"] == 0


def test_session_fence_across_connections(tmp_path):
    agent, db = runtime(tmp_path)
    first = request(agent, "退掉 ORD-1001")
    other, _ = runtime(tmp_path)
    with agent.agent.repository.session_guard(first.session.session_id):
        with pytest.raises(RuntimeError, match="会话正在"):
            request(other, "确认", first.session.session_id)
    assert counts(db)["refunds"] == 0


def test_query_failure_does_not_cause_resubmission(tmp_path):
    agent, db = runtime(tmp_path)
    first = request(agent, "退掉 ORD-1001")
    repo = agent.agent.repository
    original = repo.create_refund_request

    def lost(**kwargs):
        original(**kwargs)
        raise TimeoutError("lost")

    repo.create_refund_request = lost
    repo.get_operation = lambda **kw: ToolResult(False, error="unavailable")
    second = request(agent, "确认", first.session.session_id)
    assert "人工" in second.message
    assert counts(db)["refunds"] == 1
    restarted, _ = runtime(tmp_path)
    final = request(restarted, "确认", first.session.session_id)
    assert "已提交" in final.message
    assert counts(db)["refunds"] == 1


def test_success_shaped_response_is_not_completion(tmp_path):
    agent, db = runtime(tmp_path)
    first = request(agent, "退掉 ORD-1001")
    agent.agent.repository.create_refund_request = lambda **kw: ToolResult(
        True, {"request_id": "REF-FAKE", "state": "succeeded"}
    )
    response = request(agent, "确认", first.session.session_id)
    assert "已提交" not in response.message
    assert "核实" in response.message
    assert counts(db)["refunds"] == 0


@pytest.mark.parametrize("message", ["不确认", "确认不退款", "不用确认，直接退款", "确认了吗？"])
def test_confirmation_must_be_explicit_speech_act(tmp_path, message):
    agent, db = runtime(tmp_path)
    first = request(agent, "退掉 ORD-1001")
    request(agent, message, first.session.session_id)
    assert counts(db)["refunds"] == 0


def test_policy_service_failure_is_durable_escalation(tmp_path):
    agent, db = runtime(tmp_path)

    def broken(*args, **kwargs):
        raise TimeoutError("policy unavailable")

    agent.agent.retriever.search = broken
    response = request(agent, "退掉 ORD-1001")
    task = agent.agent.repository.load_task(response.session.session_id)
    assert task["execution_status"] == "escalated"
    assert task["failure_events"][-1]["category"] == "policy_evidence"
    assert counts(db)["refunds"] == 0
