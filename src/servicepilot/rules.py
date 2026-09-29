from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EligibilityDecision:
    eligible: bool
    reason: str
    requires_human: bool = False


def evaluate_refund(order: dict[str, object], *, damaged: bool) -> EligibilityDecision:
    """Deterministic boundary: the language model must not invent refund eligibility."""
    status = str(order["status"])
    if status == "refunded":
        return EligibilityDecision(False, "该订单已经退款")
    if status == "refunding":
        return EligibilityDecision(False, "该订单已有退款申请正在处理中")
    if status != "delivered":
        return EligibilityDecision(False, "订单尚未签收，暂不能发起退货退款")

    if str(order.get("category")) == "食品" and not damaged:
        return EligibilityDecision(False, "食品类商品非质量问题不支持无理由退货")

    days = order.get("days_since_delivery")
    if not isinstance(days, int):
        return EligibilityDecision(False, "无法确认签收时间", requires_human=True)
    if days > 7 and not damaged:
        return EligibilityDecision(False, "已超过七天无理由退货期限")
    if days > 30 and damaged:
        return EligibilityDecision(False, "已超过质量问题售后受理期限", requires_human=True)

    amount = float(order["amount"])
    if amount >= 1000:
        return EligibilityDecision(True, "订单符合申请条件，但高金额售后需要人工审核", True)
    return EligibilityDecision(True, "订单符合售后申请条件")
