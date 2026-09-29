"""Shared retrieval results and synchronous compatibility, independent of model provider."""

import asyncio
from dataclasses import dataclass

from .models import RetrievalHit


def run_sync(factory):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    raise RuntimeError("Use the async RAG interface inside an event loop")


@dataclass(frozen=True)
class EvidenceScore:
    policy_id: str
    score: float
    position: int
    keep: bool


@dataclass(frozen=True)
class RankedEvidence:
    hits: tuple[RetrievalHit, ...]
    scores: tuple[EvidenceScore, ...]
    degraded: bool = False
    reason: str | None = None
