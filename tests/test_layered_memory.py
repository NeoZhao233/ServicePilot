import unittest

from servicepilot.layered_memory import LayeredMemoryManager
from servicepilot.models import SessionState


class LayeredMemoryTests(unittest.TestCase):
    def test_profile_is_reused_across_sessions_for_same_user_only(self) -> None:
        memory = LayeredMemoryManager()
        memory.profiles.update("USER-001", {"shoe_size": "42", "preferred_contact": "email"})

        same_user = memory.assemble(SessionState("S-2", "USER-001"), "换鞋")
        other_user = memory.assemble(SessionState("S-3", "USER-002"), "换鞋")

        self.assertEqual(same_user.profile.shoe_size, "42")
        self.assertIsNone(other_user.profile.shoe_size)

    def test_profile_rejects_unapproved_sensitive_fields(self) -> None:
        memory = LayeredMemoryManager()

        with self.assertRaises(ValueError):
            memory.profiles.update("USER-001", {"id_card": "secret"})

    def test_context_selection_stays_within_token_budget(self) -> None:
        memory = LayeredMemoryManager(token_budget=20)
        state = SessionState(
            "S-1",
            "USER-001",
            turns=[
                {"role": "user", "content": "很长的历史消息" * 20},
                {"role": "assistant", "content": "旧回复" * 20},
                {"role": "user", "content": "当前问题"},
            ],
        )

        context = memory.assemble(state, "当前问题")

        self.assertLessEqual(context.estimated_tokens, memory.token_budget)
        self.assertGreater(context.dropped_turns, 0)

    def test_episode_search_never_crosses_user_boundary(self) -> None:
        memory = LayeredMemoryManager()
        memory.episodes.add(user_id="USER-001", session_id="A", summary="鞋码换货 42码")
        memory.episodes.add(user_id="USER-002", session_id="B", summary="鞋码换货 39码")

        episodes = memory.episodes.search(user_id="USER-001", query="鞋码换货")

        self.assertEqual({item.user_id for item in episodes}, {"USER-001"})


if __name__ == "__main__":
    unittest.main()
