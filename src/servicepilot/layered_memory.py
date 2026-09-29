from __future__ import annotations

import math
from dataclasses import dataclass, field

from .models import SessionState
from .retrieval import HashingTextEncoder, TextEncoder


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    locale: str = "zh-CN"
    preferred_contact: str | None = None
    shoe_size: str | None = None


class InMemoryProfileStore:
    WRITABLE_FIELDS = frozenset({"locale", "preferred_contact", "shoe_size"})

    def __init__(self) -> None:
        self._profiles: dict[str, UserProfile] = {}

    def get(self, user_id: str) -> UserProfile:
        return self._profiles.get(user_id, UserProfile(user_id=user_id))

    def update(self, user_id: str, changes: dict[str, str | None]) -> UserProfile:
        forbidden = set(changes) - self.WRITABLE_FIELDS
        if forbidden:
            raise ValueError(f"profile fields are not writable: {sorted(forbidden)}")
        current = self.get(user_id)
        payload = {
            "user_id": user_id,
            "locale": current.locale,
            "preferred_contact": current.preferred_contact,
            "shoe_size": current.shoe_size,
            **changes,
        }
        profile = UserProfile(**payload)
        self._profiles[user_id] = profile
        return profile


@dataclass(frozen=True)
class Episode:
    user_id: str
    session_id: str
    summary: str
    vector: tuple[float, ...]


class VectorEpisodeStore:
    def __init__(self, encoder: TextEncoder | None = None) -> None:
        self.encoder = encoder or HashingTextEncoder()
        self._episodes: list[Episode] = []

    def add(self, *, user_id: str, session_id: str, summary: str) -> None:
        vector = tuple(self.encoder.encode([summary])[0])
        self._episodes.append(Episode(user_id, session_id, summary, vector))

    def search(self, *, user_id: str, query: str, limit: int = 3) -> list[Episode]:
        query_vector = self.encoder.encode([query])[0]
        scored = []
        for episode in self._episodes:
            if episode.user_id != user_id:
                continue
            score = sum(
                left * right for left, right in zip(query_vector, episode.vector, strict=True)
            )
            if not math.isclose(score, 0.0):
                scored.append((score, episode))
        return [
            episode for _, episode in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
        ]


@dataclass(frozen=True)
class ContextBundle:
    profile: UserProfile
    history: tuple[dict[str, str], ...]
    episodes: tuple[Episode, ...]
    estimated_tokens: int
    dropped_turns: int
    summary: dict | None = None


@dataclass
class LayeredMemoryManager:
    profiles: InMemoryProfileStore = field(default_factory=InMemoryProfileStore)
    episodes: VectorEpisodeStore = field(default_factory=VectorEpisodeStore)
    token_budget: int = 800

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, math.ceil(len(text) / 3))

    def assemble(self, state: SessionState, query: str) -> ContextBundle:
        profile = self.profiles.get(state.user_id)
        relevant = tuple(self.episodes.search(user_id=state.user_id, query=query, limit=3))
        fixed_text = " ".join(
            filter(
                None,
                (
                    profile.locale,
                    profile.preferred_contact,
                    profile.shoe_size,
                    *(episode.summary for episode in relevant),
                ),
            )
        )
        used = self.estimate_tokens(fixed_text)
        summary = (state.context_summary or {}).get("content")
        if summary:
            import json

            summary_tokens = self.estimate_tokens(json.dumps(summary, ensure_ascii=False))
            if used + summary_tokens <= self.token_budget:
                used += summary_tokens
            else:
                summary = None
        selected: list[dict[str, str]] = []
        for turn in reversed(state.turns):
            turn_tokens = self.estimate_tokens(turn["content"])
            if used + turn_tokens > self.token_budget:
                continue
            selected.append(turn)
            used += turn_tokens
        selected.reverse()
        return ContextBundle(
            profile=profile,
            history=tuple(selected),
            episodes=relevant,
            estimated_tokens=used,
            dropped_turns=len(state.turns) - len(selected),
            summary=summary,
        )

    def remember_turn(self, state: SessionState) -> None:
        if len(state.turns) < 2:
            return
        user_turn = next(
            (turn["content"] for turn in reversed(state.turns) if turn["role"] == "user"),
            "",
        )
        assistant_turn = next(
            (turn["content"] for turn in reversed(state.turns) if turn["role"] == "assistant"),
            "",
        )
        summary = f"用户：{user_turn[:120]}；客服：{assistant_turn[:160]}"
        self.episodes.add(user_id=state.user_id, session_id=state.session_id, summary=summary)
