from __future__ import annotations

from dataclasses import dataclass, field

from .models import Intent

DEFAULT_TOOL_ALLOWLIST: dict[Intent, frozenset[str]] = {
    Intent.LOGISTICS: frozenset({"get_order"}),
    Intent.REFUND: frozenset({"get_order", "create_refund_request", "create_ticket"}),
    Intent.DAMAGED_ITEM: frozenset({"get_order", "create_ticket"}),
    Intent.HUMAN_HANDOFF: frozenset({"get_order", "create_ticket"}),
    Intent.EXCHANGE: frozenset({"get_order", "create_ticket"}),
    Intent.INVOICE: frozenset({"get_order", "create_ticket"}),
    Intent.PAYMENT: frozenset({"get_order", "create_ticket"}),
    Intent.MISSING_ITEM: frozenset({"get_order", "create_ticket"}),
    Intent.ADDRESS_CHANGE: frozenset({"get_order", "create_ticket"}),
    Intent.PRICE_PROTECTION: frozenset({"get_order", "create_ticket"}),
    Intent.CANCEL: frozenset(),
    Intent.UNKNOWN: frozenset(),
}

PROMPT_PROFILES: dict[Intent, str] = {
    Intent.LOGISTICS: "logistics_concise",
    Intent.REFUND: "refund_confirmation_first",
    Intent.DAMAGED_ITEM: "damage_evidence_first",
    Intent.HUMAN_HANDOFF: "handoff_empathy",
    Intent.EXCHANGE: "exchange_inventory_check",
    Intent.INVOICE: "invoice_compliance",
    Intent.PAYMENT: "payment_security",
    Intent.MISSING_ITEM: "missing_evidence",
    Intent.ADDRESS_CHANGE: "address_risk_control",
    Intent.PRICE_PROTECTION: "price_policy_grounded",
    Intent.CANCEL: "cancel_confirmation",
    Intent.UNKNOWN: "clarification",
}


@dataclass(frozen=True)
class ToolAccessPolicy:
    allowlist: dict[Intent, frozenset[str]] = field(
        default_factory=lambda: dict(DEFAULT_TOOL_ALLOWLIST)
    )

    def allowed_tools(self, intent: Intent) -> frozenset[str]:
        return self.allowlist.get(intent, frozenset())

    def is_allowed(self, intent: Intent, tool_name: str) -> bool:
        return tool_name in self.allowed_tools(intent)

    def prompt_profile(self, intent: Intent) -> str:
        return PROMPT_PROFILES[intent]
