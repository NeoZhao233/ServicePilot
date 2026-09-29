import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from servicepilot.agent import ServiceAgent
from servicepilot.memory import SQLiteSessionStore
from servicepilot.repository import InMemoryCommerceRepository
from servicepilot.retrieval import HybridPolicyRetriever

ROOT = Path(__file__).resolve().parents[1]


def dependencies():
    repository = InMemoryCommerceRepository.from_json(ROOT / "data" / "orders.json")
    retriever = HybridPolicyRetriever.from_json(ROOT / "data" / "policies.json")
    return repository, retriever


class SQLiteSessionStoreTests(unittest.TestCase):
    def test_agent_can_resume_pending_action_after_restart(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "sessions.sqlite3"
            repository, retriever = dependencies()
            first_agent = ServiceAgent(
                repository, retriever, session_store=SQLiteSessionStore(database)
            )
            first = first_agent.chat(user_id="USER-001", message="我要退掉 ORD-1001")

            second_agent = ServiceAgent(
                repository, retriever, session_store=SQLiteSessionStore(database)
            )
            second = second_agent.chat(
                user_id="USER-001",
                message="确认",
                session_id=first.session.session_id,
            )
            self.assertIn("退款申请已提交", second.message)
            self.assertEqual(len(second.session.turns), 4)
            events = second_agent.get_events(first.session.session_id, user_id="USER-001")
            self.assertTrue(any(event.name == "create_refund_request" for event in events))
            sessions = second_agent.list_sessions("USER-001")
            self.assertEqual([item.session_id for item in sessions], [first.session.session_id])

    def test_persisted_session_still_checks_owner(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "sessions.sqlite3"
            repository, retriever = dependencies()
            agent = ServiceAgent(repository, retriever, session_store=SQLiteSessionStore(database))
            first = agent.chat(user_id="USER-001", message="查询 ORD-1001 的物流")
            with self.assertRaises(PermissionError):
                agent.chat(
                    user_id="USER-002",
                    message="继续",
                    session_id=first.session.session_id,
                )
            with self.assertRaises(PermissionError):
                agent.get_events(first.session.session_id, user_id="USER-002")

    def test_event_replay_query_uses_session_index(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "sessions.sqlite3"
            repository, retriever = dependencies()
            agent = ServiceAgent(repository, retriever, session_store=SQLiteSessionStore(database))
            response = agent.chat(user_id="USER-001", message="查询 ORD-1003 的物流")

            with closing(sqlite3.connect(database)) as connection:
                indexes = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_schema WHERE type = 'index'"
                    )
                }
                plan = connection.execute(
                    """
                    EXPLAIN QUERY PLAN
                    SELECT event_type, name, detail_json
                    FROM agent_events
                    WHERE session_id = ?
                    ORDER BY turn_index, event_index
                    """,
                    (response.session.session_id,),
                ).fetchall()
            self.assertIn("idx_agent_events_session_id", indexes)
            self.assertTrue(any("INDEX" in row[3] for row in plan))


if __name__ == "__main__":
    unittest.main()
