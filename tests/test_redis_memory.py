import unittest

from servicepilot.memory import FallbackSessionStore, InMemorySessionStore, RedisSessionStore
from servicepilot.models import AgentEvent, SessionState


class FakeRedis:
    def __init__(self) -> None:
        self.values = {}
        self.sorted_sets = {}
        self.lists = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value

    def zadd(self, key, mapping):
        self.sorted_sets.setdefault(key, {}).update(mapping)

    def expire(self, key, ttl):
        return True

    def zrevrange(self, key, start, end):
        items = sorted(
            self.sorted_sets.get(key, {}).items(), key=lambda item: item[1], reverse=True
        )
        return [item[0] for item in items[start : end + 1]]

    def rpush(self, key, *values):
        self.lists.setdefault(key, []).extend(values)

    def lrange(self, key, start, end):
        return self.lists.get(key, []) if end == -1 else self.lists.get(key, [])[start : end + 1]


class BrokenStore(InMemorySessionStore):
    def save(self, state):
        raise ConnectionError("redis down")


class RedisMemoryTests(unittest.TestCase):
    def test_redis_store_round_trips_session_and_events(self) -> None:
        store = RedisSessionStore(FakeRedis(), ttl_seconds=60)
        state = SessionState("S-1", "USER-001")
        store.save(state)
        store.append_events("S-1", 1, [AgentEvent("tool", "get_order", {"ok": True})])

        self.assertEqual(store.get("S-1"), state)
        self.assertEqual(store.list_sessions("USER-001"), [state])
        self.assertEqual(store.get_events("S-1")[0].name, "get_order")

    def test_primary_failure_degrades_to_fallback(self) -> None:
        fallback = InMemorySessionStore()
        store = FallbackSessionStore(BrokenStore(), fallback)
        state = SessionState("S-1", "USER-001")

        store.save(state)

        self.assertTrue(store.degraded)
        self.assertEqual(store.get("S-1"), state)


if __name__ == "__main__":
    unittest.main()
