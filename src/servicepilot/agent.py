from __future__ import annotations

import re
from collections.abc import Callable
from uuid import uuid4

from .context_manager import ContextCompactor
from .evidence import retrieval_request
from .layered_memory import LayeredMemoryManager
from .memory import InMemorySessionStore, SessionStore
from .models import AgentEvent, AgentResponse, Intent, RoutingDecision, SessionState, ToolResult
from .observability import current_trace, traced_chat
from .recovery import RecoveryPolicy
from .repository import SQLiteCommerceRepository, fingerprint
from .retrieval import PolicySearch
from .routing import IntentRouter, RuleIntentRouter
from .rules import evaluate_refund
from .tool_policy import ToolAccessPolicy
from .tool_runtime import ToolExecutor
from .trace_instrumentation import bridge_events

ORDER_PATTERN = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
CONFIRM_WORDS = ("确认", "确定", "同意", "可以提交", "帮我提交")
EVIDENCE_WORDS = ("已上传", "上传了", "有照片", "发了照片")


def is_confirmation(message: str) -> bool:
    # Confirmation is a dedicated speech act, not a substring in an arbitrary instruction.
    normalized = re.sub(r"[\s，。！!,.]", "", message)
    return normalized in {
        "确认",
        "确定",
        "同意",
        "确认退款",
        "确认退货",
        "确认退货退款",
        "确认提交",
        "确认提交申请",
        "确认提交退款申请",
        "同意退款",
        "可以提交",
        "帮我提交",
        "好的确认",
        "我确认",
        "我确认退款",
    }


class ServiceAgent:
    """A deterministic, auditable customer-service runtime.

    The current runtime deliberately keeps intent routing deterministic. A future
    LLM router can implement the same interface and be measured against this baseline.
    """

    def __init__(
        self,
        repository: SQLiteCommerceRepository,
        retriever: PolicySearch,
        router: IntentRouter | None = None,
        session_store: SessionStore | None = None,
        tool_executor: ToolExecutor | None = None,
        tool_policy: ToolAccessPolicy | None = None,
        layered_memory: LayeredMemoryManager | None = None,
        context_compactor: ContextCompactor | None = None,
    ) -> None:
        self.repository = repository
        self.retriever = retriever
        self.router = router or RuleIntentRouter()
        self.session_store = session_store or InMemorySessionStore()
        self.tool_executor = tool_executor or ToolExecutor()
        self.tool_policy = tool_policy or ToolAccessPolicy()
        self.layered_memory = layered_memory or LayeredMemoryManager()
        self.context_compactor = context_compactor

    def new_document_session(self, user_id: str) -> SessionState:
        """Create a session for PDF questions without invoking commerce tools."""
        return self._get_or_create_session(user_id=user_id, session_id=None)

    @retrieval_request
    @traced_chat
    def chat(
        self,
        *,
        user_id: str,
        message: str,
        session_id: str | None = None,
        routing_decision: RoutingDecision | None = None,
        pre_events: tuple[AgentEvent, ...] = (),
    ) -> AgentResponse:
        session = self._get_or_create_session(user_id=user_id, session_id=session_id)
        with self.repository.session_guard(session.session_id):
            return self._chat(
                user_id=user_id,
                message=message,
                session_id=session.session_id,
                routing_decision=routing_decision,
                pre_events=pre_events,
            )

    def invalidate_goal(self, state: SessionState) -> None:
        self.repository.invalidate_authorization((state.authorization or {}).get("token"))
        state.authorization = None
        state.pending_action = None
        state.refund_reason = None
        state.context_summary = None
        state.context_retry_after_messages = 0
        state.turns = state.turns[-1:] if state.turns and state.turns[-1]["role"] == "user" else []
        state.goal_version += 1
        state.plan_version += 1
        self.repository.set_context(
            state.session_id, state.user_id, state.goal_version, state.plan_version
        )

    def _chat(
        self,
        *,
        user_id: str,
        message: str,
        session_id: str | None = None,
        routing_decision: RoutingDecision | None = None,
        pre_events: tuple[AgentEvent, ...] = (),
    ) -> AgentResponse:
        state = self._get_or_create_session(user_id=user_id, session_id=session_id)
        state.turns.append({"role": "user", "content": message})
        state.turn_count = max(
            state.turn_count + 1, sum(turn["role"] == "user" for turn in state.turns)
        )
        events: list[AgentEvent] = list(pre_events)
        context = self.layered_memory.assemble(state, message)
        events.append(
            AgentEvent(
                event_type="memory",
                name="assemble_context",
                detail={
                    "history_turns": len(context.history),
                    "episodic_memories": len(context.episodes),
                    "estimated_tokens": context.estimated_tokens,
                    "dropped_turns": context.dropped_turns,
                    "profile_fields": {
                        "locale": context.profile.locale,
                        "preferred_contact": context.profile.preferred_contact,
                        "shoe_size": context.profile.shoe_size,
                    },
                },
            )
        )

        order_match = ORDER_PATTERN.search(message)
        if order_match:
            next_order_id = order_match.group(0).upper()
            if state.order_id and state.order_id != next_order_id:
                self.invalidate_goal(state)
                events.append(
                    AgentEvent(
                        event_type="guardrail",
                        name="clear_pending_action_on_order_change",
                        detail={
                            "previous_order_id": state.order_id,
                            "next_order_id": next_order_id,
                        },
                    )
                )
            state.order_id = next_order_id

        routing = routing_decision or self.router.route(message)
        routing = self.resolve_routing_context(state, message, routing)
        if routing.intent is Intent.CANCEL:
            cancelled_action = state.pending_action
            self.invalidate_goal(state)
            state.intent = Intent.UNKNOWN
            events.append(
                AgentEvent(
                    event_type="guardrail",
                    name="cancel_pending_action",
                    detail={"cancelled_action": cancelled_action},
                )
            )
            message_text = "已取消当前操作。" if cancelled_action else "当前没有等待确认的操作。"
            return self._respond(state, message_text, events)
        if (
            routing.intent is Intent.UNKNOWN
            and not state.pending_action
            and is_confirmation(message)
        ):
            events.append(
                AgentEvent(
                    event_type="guardrail",
                    name="reject_orphan_confirmation",
                    detail={"state_changed": False},
                )
            )
            return self._respond(
                state, "当前没有等待确认的操作，请重新说明需要办理的事项。", events
            )
        if routing.intent is not Intent.UNKNOWN:
            if state.intent not in (Intent.UNKNOWN, routing.intent):
                self.invalidate_goal(state)
            state.intent = routing.intent
        self.repository.set_context(
            state.session_id, user_id, state.goal_version, state.plan_version
        )
        events.append(
            AgentEvent(
                event_type="decision",
                name="intent_routing",
                detail={
                    "intent": state.intent.value,
                    "detected_intent": routing.intent.value,
                    "confidence": routing.confidence,
                    "urgency": routing.urgency.value,
                    "matched_signals": routing.matched_signals,
                    "alternatives": [
                        {"intent": intent.value, "score": score}
                        for intent, score in routing.alternatives
                    ],
                    "order_id": state.order_id,
                },
            )
        )

        if state.intent is Intent.UNKNOWN:
            return self._respond(
                state,
                "我目前可以处理物流查询、退货退款、商品破损和人工客服。请告诉我你要处理哪一类问题。",
                events,
            )

        if not state.order_id:
            return self._respond(state, "请提供需要处理的订单号，例如 ORD-1001。", events)

        lookup = self._call_tool(
            "get_order",
            events,
            lambda: self.repository.get_order(order_id=state.order_id or "", user_id=user_id),
            {"order_id": state.order_id},
            intent=state.intent,
        )
        if not lookup.ok:
            return self._respond(state, f"暂时无法处理这个订单：{lookup.error}。", events)

        order = lookup.data
        if state.intent is Intent.REFUND and state.authorization:
            op = self.repository.get_operation(
                operation_id=state.authorization["operation_id"], user_id=user_id
            )
            if op.ok and op.data.get("state") in {"succeeded", "unknown"}:
                recovered = RecoveryPolicy.reconcile_refund(
                    self.repository,
                    operation_id=state.authorization["operation_id"],
                    user_id=user_id,
                    events=events,
                )
                if recovered.ok:
                    state.pending_action = None
                    return self._respond(
                        state,
                        f"已核实存在退款申请，编号为 {recovered.data['request_id']}，"
                        "无需重复提交。",
                        events,
                    )
                return self._respond(state, recovered.error or "需人工核对", events)
        if state.intent is Intent.LOGISTICS:
            return self._handle_logistics(state, order, events)
        if state.intent is Intent.HUMAN_HANDOFF:
            return self._handle_handoff(state, events)
        if state.intent in (Intent.REFUND, Intent.DAMAGED_ITEM):
            return self._handle_after_sales(state, message, order, events)
        if state.intent in {
            Intent.EXCHANGE,
            Intent.INVOICE,
            Intent.PAYMENT,
            Intent.MISSING_ITEM,
            Intent.ADDRESS_CHANGE,
            Intent.PRICE_PROTECTION,
        }:
            return self._handle_specialist_request(state, message, order, events)

        return self._respond(state, "我还不能处理这个问题，可以为你转接人工客服。", events)

    def get_session(self, session_id: str) -> SessionState | None:
        return self.session_store.get(session_id)

    @staticmethod
    def resolve_routing_context(
        state: SessionState, message: str, routing: RoutingDecision
    ) -> RoutingDecision:
        # Explicit changes and cancellation take precedence over confirmation context.
        change = re.search(r"(?:改成|改为|改办|换成)(.+)", message)
        if change:
            target = RuleIntentRouter().route(change.group(1))
            if target.intent is not Intent.UNKNOWN:
                return target
        if routing.intent not in (Intent.UNKNOWN, state.intent):
            return routing
        if state.pending_action and (
            is_confirmation(message)
            or any(word in message for word in EVIDENCE_WORDS)
            or state.pending_action == "address_change"
        ):
            return RoutingDecision(
                intent=state.intent,
                confidence=1.0,
                matched_signals=("pending_action_context",),
                urgency=routing.urgency,
            )
        return routing

    def list_sessions(self, user_id: str, *, limit: int = 20) -> list[SessionState]:
        return self.session_store.list_sessions(user_id, limit=limit)

    def get_events(self, session_id: str, *, user_id: str) -> list[AgentEvent]:
        state = self.session_store.get(session_id)
        if state is None:
            raise ValueError("会话不存在")
        if state.user_id != user_id:
            raise PermissionError("无权访问该会话")
        return self.session_store.get_events(session_id)

    def record_events(self, response: AgentResponse, events: list[AgentEvent]) -> AgentResponse:
        """Attach post-execution validation events to the response and audit log."""
        if not events:
            return response
        turn_index = response.session.turn_count or sum(
            turn["role"] == "user" for turn in response.session.turns
        )
        self.session_store.append_events(response.session.session_id, turn_index, events)
        return AgentResponse(
            message=response.message,
            session=response.session,
            events=(*response.events, *events),
            citations=response.citations,
        )

    def safe_fallback(
        self,
        *,
        user_id: str,
        message: str,
        session_id: str | None,
        reason: str,
        pre_events: tuple[AgentEvent, ...] = (),
    ) -> AgentResponse:
        """Return a stable handoff message when the workflow cannot finish safely."""
        state = self._get_or_create_session(user_id=user_id, session_id=session_id)
        if (
            len(state.turns) >= 2
            and state.turns[-1].get("role") == "assistant"
            and state.turns[-2] == {"role": "user", "content": message}
        ):
            state.turns.pop()
        if not state.turns or state.turns[-1] != {"role": "user", "content": message}:
            state.turns.append({"role": "user", "content": message})
        events = [
            *pre_events,
            AgentEvent(
                event_type="runtime",
                name="safe_fallback",
                detail={"reason": reason, "state_changed": False},
            ),
        ]
        return self._respond(
            state,
            (
                "请求为空，订单状态未被更改。"
                if reason == "empty_message"
                else "系统暂时无法安全完成该操作；请联系人工核对业务状态，勿重复提交。"
            ),
            events,
        )

    def _get_or_create_session(self, *, user_id: str, session_id: str | None) -> SessionState:
        if session_id:
            state = self.session_store.get(session_id)
            if state is None:
                raise ValueError("会话不存在")
            if state.user_id != user_id:
                raise PermissionError("无权访问该会话")
            return state
        state = SessionState(session_id=f"SESSION-{uuid4().hex[:8]}", user_id=user_id)
        self.session_store.save(state)
        return state

    def _handle_logistics(
        self, state: SessionState, order: dict[str, object], events: list[AgentEvent]
    ) -> AgentResponse:
        status = order.get("logistics_status") or "暂时没有新的物流信息"
        switched = any(event.name == "clear_pending_action_on_order_change" for event in events)
        prefix = "已切换订单，原订单等待确认的操作已取消。" if switched else ""
        return self._respond(
            state,
            f"{prefix}订单 {state.order_id}（{order['product_name']}）的物流状态：{status}。",
            events,
        )

    def _handle_handoff(self, state: SessionState, events: list[AgentEvent]) -> AgentResponse:
        result = self._call_tool(
            "create_ticket",
            events,
            lambda: self.repository.create_ticket(
                order_id=state.order_id or "",
                user_id=state.user_id,
                description="用户主动申请人工客服",
            ),
            intent=state.intent,
        )
        if not result.ok:
            return self._respond(state, f"人工工单创建失败：{result.error}。", events)
        return self._respond(
            state,
            f"已为你创建人工客服工单 {result.data['ticket_id']}，客服人员会继续处理。",
            events,
        )

    def _handle_specialist_request(
        self,
        state: SessionState,
        message: str,
        order: dict[str, object],
        events: list[AgentEvent],
    ) -> AgentResponse:
        labels = {
            Intent.EXCHANGE: "换货",
            Intent.INVOICE: "发票",
            Intent.PAYMENT: "支付异常",
            Intent.MISSING_ITEM: "少件漏发",
            Intent.ADDRESS_CHANGE: "修改地址",
            Intent.PRICE_PROTECTION: "价格保护",
        }
        label = labels[state.intent]
        if state.intent is Intent.INVOICE:
            hits = self.retriever.search("电子发票 开票 抬头", limit=2)
            citations = tuple(hit.chunk.policy_id for hit in hits)
            events.append(
                AgentEvent(
                    event_type="retrieval",
                    name="search_policy",
                    detail={
                        "query": "电子发票 开票 抬头",
                        "hits": [
                            {"policy_id": hit.chunk.policy_id, "score": round(hit.score, 3)}
                            for hit in hits
                        ],
                    },
                )
            )
            return self._respond(
                state,
                "该订单已完成，可以申请电子发票；开具前可填写或修改抬头与税号，"
                "审核通过后会发送到账户票夹或预留邮箱。",
                events,
                citations,
            )
        if state.intent is Intent.ADDRESS_CHANGE and state.pending_action != "address_change":
            state.pending_action = "address_change"
            shipped = str(order.get("status")) == "shipped"
            hits = self.retriever.search("发货后更改地址 承运商转寄 费用 延误", limit=2)
            citations = tuple(hit.chunk.policy_id for hit in hits)
            events.append(
                AgentEvent(
                    event_type="retrieval",
                    name="search_policy",
                    detail={
                        "query": "发货后更改地址 承运商转寄 费用 延误",
                        "hits": [
                            {"policy_id": hit.chunk.policy_id, "score": round(hit.score, 3)}
                            for hit in hits
                        ],
                    },
                )
            )
            return self._respond(
                state,
                (
                    "该订单已发货且正在运输，平台无法直接修改收货地址；"
                    if shipped
                    else "该订单尚未确认出库，"
                )
                + "如仍需尝试承运商转寄，请提供完整的新地址。转寄不能保证成功，"
                "并可能产生额外费用和配送延误。",
                events,
                citations,
            )
        result = self._call_tool(
            "create_ticket",
            events,
            lambda: self.repository.create_ticket(
                order_id=state.order_id or "",
                user_id=state.user_id,
                description=f"{label}：{message}",
            ),
            intent=state.intent,
        )
        if not result.ok:
            return self._respond(state, f"{label}工单创建失败：{result.error}。", events)
        state.pending_action = None
        guidance = ""
        citations: tuple[str, ...] = ()
        policy_queries = {
            Intent.EXCHANGE: "鞋服尺码换货 吊牌 鞋底 库存",
            Intent.PRICE_PROTECTION: "价格保护 七天 降价 价保标识",
        }
        if query := policy_queries.get(state.intent):
            hits = self.retriever.search(query, category=str(order["category"]), limit=2)
            citations = tuple(hit.chunk.policy_id for hit in hits)
            events.append(
                AgentEvent(
                    event_type="retrieval",
                    name="search_policy",
                    detail={
                        "query": query,
                        "hits": [
                            {"policy_id": hit.chunk.policy_id, "score": round(hit.score, 3)}
                            for hit in hits
                        ],
                    },
                )
            )
        if state.intent is Intent.EXCHANGE:
            guidance = "换货需保持吊牌、包装和商品完好，且目标规格有库存。"
        elif state.intent is Intent.PRICE_PROTECTION:
            guidance = "价保通常要求支付后七天内发生直接降价且商品带有价保标识。"
        return self._respond(
            state,
            f"{guidance}已针对{label}问题创建核验工单 {result.data['ticket_id']}。",
            events,
            citations,
        )

    def _handle_after_sales(
        self,
        state: SessionState,
        message: str,
        order: dict[str, object],
        events: list[AgentEvent],
    ) -> AgentResponse:
        damaged = state.intent is Intent.DAMAGED_ITEM
        safety_incident = any(word in message for word in ("起火", "爆炸", "触电", "受伤"))
        query = f"{order['category']} {'质量问题 破损' if damaged else '七天无理由 退货退款'}"
        hits = self.retriever.search(query, category=str(order["category"]), limit=2)
        citations = tuple(hit.chunk.policy_id for hit in hits)
        events.append(
            AgentEvent(
                event_type="retrieval",
                name="search_policy",
                detail={
                    "query": query,
                    "hits": [
                        {"policy_id": hit.chunk.policy_id, "score": round(hit.score, 3)}
                        for hit in hits
                    ],
                },
            )
        )

        decision = evaluate_refund(order, damaged=damaged)
        events.append(
            AgentEvent(
                event_type="rule",
                name="evaluate_refund",
                detail={
                    "eligible": decision.eligible,
                    "reason": decision.reason,
                    "requires_human": decision.requires_human,
                },
            )
        )
        if not decision.eligible:
            suffix = "我可以为你转接人工复核。" if decision.requires_human else ""
            return self._respond(
                state,
                f"暂不能直接申请退款：{decision.reason}。{suffix}",
                events,
                citations,
            )

        if decision.requires_human:
            result = self._call_tool(
                "create_ticket",
                events,
                lambda: self.repository.create_ticket(
                    order_id=state.order_id or "",
                    user_id=state.user_id,
                    description=f"高金额售后人工审核：{state.intent.value}",
                ),
                intent=state.intent,
            )
            if not result.ok:
                return self._respond(
                    state,
                    f"人工审核工单暂时创建失败：{result.error}。订单尚未发生变更，请稍后重试。",
                    events,
                    citations,
                )
            return self._respond(
                state,
                ("请立即停止使用商品、断开电源并远离危险区域；" if safety_incident else "")
                + f"{decision.reason}，已创建审核工单 {result.data['ticket_id']}。",
                events,
                citations,
            )

        if damaged:
            if not any(word in message for word in EVIDENCE_WORDS):
                state.pending_action = "damage_ticket"
                return self._respond(
                    state,
                    "订单符合质量问题售后条件。请上传能够显示问题的照片，然后回复“已上传照片”。",
                    events,
                    citations,
                )
            result = self._call_tool(
                "create_ticket",
                events,
                lambda: self.repository.create_ticket(
                    order_id=state.order_id or "",
                    user_id=state.user_id,
                    description="商品破损或质量问题，用户已提供图片证据",
                ),
                intent=state.intent,
            )
            state.pending_action = None
            if not result.ok:
                return self._respond(
                    state,
                    f"售后工单暂时创建失败：{result.error}。订单尚未发生变更，请稍后重试。",
                    events,
                    citations,
                )
            return self._respond(
                state,
                f"证据已记录，并创建售后审核工单 {result.data['ticket_id']}。",
                events,
                citations,
            )

        if state.pending_action == "refund" and is_confirmation(message):
            if state.authorization and state.authorization["snapshot"] == fingerprint(order):
                self.repository.confirm_authorization(
                    token=state.authorization["token"], user_id=state.user_id
                )
                result = self.submit_refund(state, events)
                if result.ok:
                    state.pending_action = None
                    return self._respond(
                        state,
                        f"退款申请已提交，申请编号为 {result.data['request_id']}。",
                        events,
                        citations,
                    )
                if result.data.get("code") != "authorization":
                    return self._respond(state, result.error or "退款提交失败", events, citations)
            self.repository.invalidate_authorization((state.authorization or {}).get("token"))
            state.authorization = None
            state.plan_version += 1
            events.append(
                AgentEvent(
                    "guardrail", "authorization_invalidated", {"reason": "order_or_amount_changed"}
                )
            )

        self.repository.set_context(
            state.session_id, state.user_id, state.goal_version, state.plan_version
        )
        prepared = self.repository.prepare_refund(
            order_id=state.order_id or "",
            user_id=state.user_id,
            context_id=state.session_id,
            goal_version=state.goal_version,
            plan_version=state.plan_version,
        )
        if not prepared.ok:
            return self._respond(state, prepared.error or "暂不能办理退款", events, citations)
        state.authorization = prepared.data
        state.pending_action = "refund"
        state.refund_reason = message
        return self._respond(
            state,
            f"{decision.reason}。退款金额 {prepared.data['amount_minor'] / 100:.2f} 元，"
            "退款属于敏感操作，请回复“确认”后提交申请。",
            events,
            citations,
        )

    def submit_refund(self, state: SessionState, events: list[AgentEvent]) -> ToolResult:
        authorization = state.authorization or {}
        result = self._call_tool(
            "create_refund_request",
            events,
            lambda: self.repository.create_refund_request(
                order_id=state.order_id or "",
                user_id=state.user_id,
                reason=state.refund_reason or "用户申请退货退款",
                authorization=authorization.get("token"),
            ),
            intent=Intent.REFUND,
        )
        if result.data.get("outcome") == "unknown" and authorization:
            return RecoveryPolicy.reconcile_refund(
                self.repository,
                operation_id=authorization["operation_id"],
                user_id=state.user_id,
                events=events,
            )
        return result

    def _call_tool(
        self,
        name: str,
        events: list[AgentEvent],
        operation: Callable[[], ToolResult],
        context: dict[str, object] | None = None,
        *,
        intent: Intent,
    ) -> ToolResult:
        if not self.tool_policy.is_allowed(intent, name):
            result = ToolResult(ok=False, error=f"工具 {name} 不在 {intent.value} 意图白名单中")
            events.append(
                AgentEvent(
                    event_type="guardrail",
                    name="tool_blocked",
                    detail={
                        "tool": name,
                        "intent": intent.value,
                        "allowed_tools": sorted(self.tool_policy.allowed_tools(intent)),
                    },
                )
            )
            return result
        execution = self.tool_executor.execute(name, operation)
        result = execution.result
        events.append(
            AgentEvent(
                event_type="tool",
                name=name,
                detail={
                    **(context or {}),
                    "ok": result.ok,
                    "error": result.error,
                    "attempts": execution.attempts,
                    "circuit_open": execution.circuit_open,
                    **result.data,
                },
            )
        )
        return result

    def _respond(
        self,
        state: SessionState,
        message: str,
        events: list[AgentEvent],
        citations: tuple[str, ...] = (),
    ) -> AgentResponse:
        collector = current_trace()
        if collector:
            collector.bind(user_id=state.user_id, thread_id=state.session_id)
        bridge_events(events, plan_version=state.plan_version)
        state.turns.append({"role": "assistant", "content": message})
        self.layered_memory.remember_turn(state)
        if not state.turn_count:
            state.turn_count = sum(turn["role"] == "user" for turn in state.turns)
        compaction = self.context_compactor.compact(state) if self.context_compactor else None
        if compaction:
            events.append(AgentEvent("context", "context_compaction", compaction))
        self.session_store.save(state)
        turn_index = state.turn_count
        self.session_store.append_events(state.session_id, turn_index, events)
        return AgentResponse(
            message=message,
            session=state,
            events=tuple(events),
            citations=citations,
        )
