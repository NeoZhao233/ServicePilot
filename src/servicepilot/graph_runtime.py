from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import ServiceAgent
from .evidence import retrieval_request
from .models import AgentEvent, AgentResponse, Intent, RoutingDecision
from .observability import traced_chat
from .repository import SessionBusyError
from .trace_instrumentation import node


class WorkflowState(TypedDict, total=False):
    user_id: str
    message: str
    session_id: str | None
    routing: RoutingDecision
    pre_events: tuple[AgentEvent, ...]
    response: AgentResponse
    error: str


class LangGraphServiceRuntime:
    """Controlled graph around the deterministic customer-service business engine."""

    def __init__(
        self,
        agent: ServiceAgent,
        *,
        dynamic: bool = False,
        planner=None,
        failure_memory=None,
        experience_extractor=None,
        naive_reflections=None,
    ) -> None:
        self.agent = agent
        self.trace_service = getattr(agent, "trace_service", None)
        self.dynamic = None
        if dynamic:
            from .dynamic_runtime import DynamicTaskRuntime

            self.dynamic = DynamicTaskRuntime(
                agent,
                planner,
                failure_memory=failure_memory,
                experience_extractor=experience_extractor,
                naive_reflections=naive_reflections,
            )
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        builder.add_node("validate_request", node("validate_request", self._validate_request))
        builder.add_node("route_intent", node("route_intent", self._route_intent))
        builder.add_node("execute_business", node("execute_business", self._execute_business))
        builder.add_node("validate_result", node("validate_result", self._validate_result))
        builder.add_node("safe_fallback", node("safe_fallback", self._safe_fallback))
        if self.dynamic:
            builder.add_node("dynamic_task", node("dynamic_task", self._execute_dynamic))
            builder.add_edge("dynamic_task", "validate_result")
        builder.add_edge(START, "validate_request")
        builder.add_conditional_edges(
            "validate_request",
            lambda state: "safe_fallback" if state.get("error") else "route_intent",
        )
        builder.add_conditional_edges(
            "route_intent",
            self._business_route,
        )
        builder.add_conditional_edges(
            "execute_business",
            lambda state: "safe_fallback" if state.get("error") else "validate_result",
        )
        builder.add_conditional_edges(
            "validate_result",
            lambda state: "safe_fallback" if state.get("error") else END,
        )
        builder.add_edge("safe_fallback", END)
        return builder.compile()

    def _business_route(self, state):
        if state.get("error"):
            return "safe_fallback"
        intent = state["routing"].intent
        if self.dynamic and intent is Intent.UNKNOWN and state.get("session_id"):
            session = self.agent.get_session(state["session_id"])
            if session:
                intent = session.intent
        if self.dynamic and intent in {Intent.REFUND, Intent.EXCHANGE, Intent.CANCEL}:
            return "dynamic_task"
        return "execute_business"

    def _execute_dynamic(self, state):
        response = self.dynamic.run(
            user_id=state["user_id"],
            message=state["message"],
            session_id=state.get("session_id"),
            routing=state["routing"],
            pre_events=state.get("pre_events", ()),
        )
        return {"response": response, "error": ""}

    @retrieval_request
    @traced_chat
    def chat(self, *, user_id: str, message: str, session_id: str | None = None) -> AgentResponse:
        result = self.graph.invoke(
            {"user_id": user_id, "message": message, "session_id": session_id}
        )
        return result["response"]

    def get_session(self, session_id: str):
        return self.agent.get_session(session_id)

    def list_sessions(self, user_id: str, *, limit: int = 20):
        return self.agent.list_sessions(user_id, limit=limit)

    def get_events(self, session_id: str, *, user_id: str):
        return self.agent.get_events(session_id, user_id=user_id)

    def graph_nodes(self) -> tuple[str, ...]:
        return tuple(
            node for node in self.graph.get_graph().nodes if node not in {"__start__", "__end__"}
        )

    def _validate_request(self, state: WorkflowState) -> WorkflowState:
        message = state.get("message", "")
        error = "empty_message" if not message.strip() else ""
        event = AgentEvent(
            event_type="workflow",
            name="validate_request",
            detail={"ok": not error, "message_length": len(message)},
        )
        return {"error": error, "pre_events": (event,)}

    def _route_intent(self, state: WorkflowState) -> WorkflowState:
        events = state.get("pre_events", ())
        try:
            routing = self.agent.router.route(state["message"])
            if session_id := state.get("session_id"):
                session = self.agent.get_session(session_id)
                if session is not None:
                    if session.user_id != state["user_id"]:
                        raise PermissionError("session owner mismatch")
                    routing = self.agent.resolve_routing_context(session, state["message"], routing)
        except Exception as exc:
            return {
                "error": f"routing_error:{type(exc).__name__}",
                "pre_events": events,
            }
        allowed_tools = sorted(self.agent.tool_policy.allowed_tools(routing.intent))
        event = AgentEvent(
            event_type="workflow",
            name="route_intent",
            detail={
                "intent": routing.intent.value,
                "confidence": routing.confidence,
                "urgency": routing.urgency.value,
                "allowed_tools": allowed_tools,
                "prompt_profile": self.agent.tool_policy.prompt_profile(routing.intent),
            },
        )
        return {"routing": routing, "pre_events": (*events, event), "error": ""}

    def _execute_business(self, state: WorkflowState) -> WorkflowState:
        events = state.get("pre_events", ())
        execute_event = AgentEvent(
            event_type="workflow",
            name="execute_business",
            detail={"controlled_tool_access": True},
        )
        events = (*events, execute_event)
        try:
            response = self.agent.chat(
                user_id=state["user_id"],
                message=state["message"],
                session_id=state.get("session_id"),
                routing_decision=state["routing"],
                pre_events=events,
            )
        except SessionBusyError:
            raise
        except Exception as exc:
            return {
                "error": f"execution_error:{type(exc).__name__}",
                "pre_events": events,
            }
        return {"response": response, "error": "", "pre_events": events}

    def _validate_result(self, state: WorkflowState) -> WorkflowState:
        response = state["response"]
        error = self._result_error(response)
        event = AgentEvent(
            event_type="workflow",
            name="validate_result",
            detail={"ok": error is None, "error": error},
        )
        updated = self.agent.record_events(response, [event])
        if error is not None:
            return {
                "response": updated,
                "error": f"result_validation_error:{error}",
            }
        return {"response": updated, "error": ""}

    @staticmethod
    def _result_error(response: AgentResponse) -> str | None:
        if not response.message.strip():
            return "empty_response"
        successful_tools = {
            event.name
            for event in response.events
            if event.event_type == "tool" and event.detail.get("ok") is True
        }
        if "退款申请已提交" in response.message and not (
            {"create_refund_request", "get_operation"} & successful_tools
        ):
            return "refund_claim_without_successful_tool"
        ticket_claims = ("已创建审核工单", "创建售后审核工单", "创建人工客服工单")
        if any(claim in response.message for claim in ticket_claims):
            if "create_ticket" not in successful_tools:
                return "ticket_claim_without_successful_tool"
        if any(event.name == "tool_blocked" for event in response.events):
            if "已提交" in response.message or "已创建" in response.message:
                return "success_claim_after_blocked_tool"
        return None

    def _safe_fallback(self, state: WorkflowState) -> WorkflowState:
        existing = state.get("response")
        session_id = existing.session.session_id if existing else state.get("session_id")
        response = self.agent.safe_fallback(
            user_id=state["user_id"],
            message=state["message"],
            session_id=session_id,
            reason=state.get("error", "unknown_runtime_error"),
            pre_events=() if existing else state.get("pre_events", ()),
        )
        return {"response": response}
