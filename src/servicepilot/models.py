from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Intent(StrEnum):
    LOGISTICS = "logistics"
    REFUND = "refund"
    DAMAGED_ITEM = "damaged_item"
    HUMAN_HANDOFF = "human_handoff"
    EXCHANGE = "exchange"
    INVOICE = "invoice"
    PAYMENT = "payment"
    MISSING_ITEM = "missing_item"
    ADDRESS_CHANGE = "address_change"
    PRICE_PROTECTION = "price_protection"
    CANCEL = "cancel"
    UNKNOWN = "unknown"


class UrgencyLevel(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class OrderStatus(StrEnum):
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    REFUNDING = "refunding"
    REFUNDED = "refunded"


@dataclass(frozen=True)
class Order:
    order_id: str
    user_id: str
    product_name: str
    category: str
    amount: float
    status: OrderStatus
    days_since_delivery: int | None = None
    logistics_status: str | None = None


@dataclass(frozen=True)
class PolicyChunk:
    policy_id: str
    title: str
    content: str
    categories: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    source: str | None = None
    version: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None


@dataclass(frozen=True)
class RetrievalHit:
    chunk: PolicyChunk
    score: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class RoutingDecision:
    intent: Intent
    confidence: float
    matched_signals: tuple[str, ...] = ()
    alternatives: tuple[tuple[Intent, float], ...] = ()
    urgency: UrgencyLevel = UrgencyLevel.NORMAL


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class AgentEvent:
    event_type: str
    name: str
    detail: dict[str, Any]


@dataclass
class SessionState:
    session_id: str
    user_id: str
    intent: Intent = Intent.UNKNOWN
    order_id: str | None = None
    pending_action: str | None = None
    refund_reason: str | None = None
    turns: list[dict[str, str]] = field(default_factory=list)
    context_summary: dict[str, Any] | None = None
    turn_count: int = 0
    context_retry_after_messages: int = 0
    goal_version: int = 1
    plan_version: int = 1
    authorization: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentResponse:
    message: str
    session: SessionState
    events: tuple[AgentEvent, ...]
    citations: tuple[str, ...] = ()
