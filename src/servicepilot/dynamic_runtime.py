from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import ORDER_PATTERN, is_confirmation
from .failure_memory import (
    ExperienceExtractor,
    FailureCandidate,
    FailureMemoryRetriever,
    FailureMemoryStore,
    NaiveReflectionStore,
    record_verified_failure,
)
from .failure_memory import (
    FailureAnalyzer as MemoryFailureAnalyzer,
)
from .models import AgentEvent, AgentResponse, Intent, ToolResult
from .observability import current_trace, event
from .planning import (
    GoalStateEvaluator,
    PlanValidationError,
    StateReconciler,
    TaskPlanner,
    TaskState,
)
from .recovery import FailureAnalyzer, RecoveryPolicy
from .repository import fingerprint
from .rules import evaluate_refund
from .trace_instrumentation import node


class ExecutionState(TypedDict, total=False):
    task: TaskState
    session: Any
    goal: dict
    message: str
    events: list[AgentEvent]
    reply: str
    citations: tuple[str, ...]
    next: str
    step_id: str
    allow_confirmation: bool
    result: Any


class DynamicTaskRuntime:
    """A bounded plan/execute/observe/reconcile subgraph for refund and exchange intake."""

    def __init__(
        self,
        agent,
        planner: TaskPlanner | None = None,
        *,
        max_plans=6,
        max_tools=40,
        max_failures=3,
        failure_memory: FailureMemoryStore | None = None,
        experience_extractor: ExperienceExtractor | None = None,
        naive_reflections: NaiveReflectionStore | None = None,
    ):
        if failure_memory is not None and naive_reflections is not None:
            raise ValueError("B1 and B2 memory modes are mutually exclusive")
        self.agent, self.repository = agent, agent.repository
        self.planner = planner or TaskPlanner()
        self.failure_memory = failure_memory
        self.memory_retriever = FailureMemoryRetriever(failure_memory) if failure_memory else None
        self.experience_extractor = experience_extractor or ExperienceExtractor()
        self.naive_reflections = naive_reflections
        self.max_plans, self.max_tools, self.max_failures = max_plans, max_tools, max_failures
        graph = StateGraph(ExecutionState)
        for name in (
            "observe",
            "reconcile",
            "plan",
            "select_step",
            "execute_step",
            "evaluate",
            "recover",
            "finish",
        ):
            graph.add_node(name, node(name, getattr(self, "_" + name), graph="dynamic"))
        graph.add_edge(START, "observe")
        for name in (
            "observe",
            "reconcile",
            "plan",
            "select_step",
            "execute_step",
            "evaluate",
            "recover",
        ):
            graph.add_conditional_edges(name, lambda state: state["next"])
        graph.add_edge("finish", END)
        self.graph = graph.compile()

    def run(self, *, user_id, message, session_id, routing, pre_events) -> AgentResponse:
        session = self.agent._get_or_create_session(user_id=user_id, session_id=session_id)
        with self.repository.session_guard(session.session_id):
            # Reload under the serialization fence.
            session = self.agent.get_session(session.session_id)
            saved = self.repository.load_task(session.session_id)
            task = (
                TaskState.from_dict(saved)
                if saved
                else TaskState(
                    session.session_id,
                    user_id,
                    {},
                    goal_version=session.goal_version,
                    plan_version=session.plan_version,
                )
            )
            if saved and (
                session.goal_version > task.goal_version or session.plan_version > task.plan_version
            ):
                task.history.append({"reason": "legacy_context_changed"})
                task.goal_version = max(task.goal_version, session.goal_version)
                task.plan_version = max(task.plan_version, session.plan_version)
                task.current_plan = None
                task.last_plan_input = ""
            if task.user_id != user_id:
                raise PermissionError("任务归属不匹配")
            collector = current_trace()
            if collector:
                collector.bind(user_id=user_id, thread_id=session.session_id, task_id=task.task_id)
            match = ORDER_PATTERN.search(message)
            order_id = match.group(0).upper() if match else session.order_id
            intent = routing.intent if routing.intent is not Intent.UNKNOWN else session.intent
            session.turns.append({"role": "user", "content": message})
            session.turn_count = max(
                session.turn_count + 1,
                sum(turn["role"] == "user" for turn in session.turns),
            )
            state = ExecutionState(
                task=task,
                session=session,
                goal={"intent": intent.value, "order_id": order_id or ""},
                message=message,
                events=list(pre_events),
                citations=(),
                reply="",
                allow_confirmation=bool(session.authorization) and is_confirmation(message),
            )
            if not order_id:
                session.intent = intent
                return self.agent._respond(
                    session, "请提供需要处理的订单号，例如 ORD-1001。", state["events"]
                )
            result = self.graph.invoke(state, {"recursion_limit": 100})
            return self.agent._respond(
                result["session"], result["reply"], result["events"], result.get("citations", ())
            )

    def _save(self, state):
        self.repository.save_task(state["task"].to_dict())
        self.agent.session_store.save(state["session"])

    def _stop(self, state, reply, status="escalated"):
        state["task"].execution_status = status
        if self.failure_memory and status in {
            "completed",
            "waiting_confirmation",
            "escalated",
            "blocked",
            "cancelled",
        }:
            try:
                self._verify_failure_candidates(state)
            except Exception:
                # Experience bookkeeping must not prevent a safe pause or handoff.
                pass
        if status == "completed" and self.failure_memory:
            try:
                memory_ids = self.failure_memory.finalize_task_outcome(
                    state["task"].task_id, task_success=True
                )
                for memory_id in memory_ids:
                    event("FAILURE_MEMORY_OUTCOME", memory_id=memory_id, task_success=True)
            except Exception:
                pass
        if status == "escalated":
            event(
                "HUMAN_HANDOFF",
                task_id=state["task"].task_id,
                plan_version=state["task"].plan_version,
                reason=reply,
            )
            self.repository.escalate(state["task"].task_id, state["session"].user_id, reply)
        state["reply"], state["next"] = reply, "finish"
        self._save(state)
        return state

    def _queue_failure_candidate(
        self, state, *, trigger, tool_name=None, assertion=None, tool_call=None, high_risk=False
    ):
        if not self.failure_memory:
            return
        collector = current_trace()
        candidate = FailureCandidate(
            task_id=state["task"].task_id,
            trace_id=collector.record["trace_id"] if collector else "untraced",
            trigger=trigger,
            tool_name=tool_name,
            assertion=assertion or {},
            tool_call=tool_call or {},
            high_risk=high_risk,
        )
        state["task"].failure_candidates.append(candidate.model_dump())
        event(
            "FAILURE_CANDIDATE_CREATED",
            candidate_id=candidate.candidate_id,
            task_id=candidate.task_id,
            trigger=trigger,
        )
        self.repository.save_task(state["task"].to_dict())

    def queue_independent_rag_failure(
        self, state, *, rule, evidence_ref, observed, expected, actual
    ):
        """Accept a verdict from an external oracle, never from RAG self-assessment."""
        if rule not in {"policy_grounding", "policy_version"} or not evidence_ref.startswith(
            "oracle:"
        ):
            raise ValueError("independent RAG verdict requires a supported rule and evidence")
        self._queue_failure_candidate(
            state,
            trigger="rag_judgment",
            tool_name="search_policy",
            assertion={
                "rule": rule,
                "source": "test_oracle",
                "passed": False,
                "step": "rag_judgment",
                "evidence_ref": evidence_ref,
                "observed": observed,
                "expected": expected,
                "actual": actual,
            },
        )

    def _verify_failure_candidates(self, state):
        task = state["task"]
        pending = list(task.failure_candidates)
        collector = current_trace()
        trace = {**collector.record, "events": collector.events} if collector else None
        for data in pending:
            candidate = FailureCandidate.model_validate(data)
            assertion = candidate.assertion
            if candidate.trigger == "invalid_plan" and not trace:
                task.failure_candidates.remove(data)
                continue
            business_state = {}
            calls = [candidate.tool_call] if candidate.tool_call else []
            if candidate.trigger == "guard_blocked":
                business_state = {
                    "guard_reason": candidate.tool_call.get("guard_reason"),
                    "guard_evidence_ref": candidate.tool_call.get("evidence_ref"),
                }
            elif candidate.trigger == "duplicate_recovery":
                original = candidate.tool_call.get("original_operation_id")
                new = candidate.tool_call.get("operation_id")
                if original and new and original != new:
                    try:
                        ledger = self.repository.get_operation(
                            operation_id=original, user_id=state["session"].user_id
                        )
                        business_state = {
                            "original_operation_state": ledger.data.get("state")
                            if ledger.ok
                            else None,
                            "same_operation": False,
                            "ledger_evidence_ref": f"ledger:operation:{original}",
                        }
                    except Exception:
                        pass
            elif candidate.trigger == "rag_judgment":
                # Only an independent oracle may assert an incorrect RAG conclusion.
                if assertion.get("source") != "test_oracle":
                    assertion = {}
            analysis = MemoryFailureAnalyzer.analyze(
                trace=trace,
                task_state=task,
                plan=task.current_plan,
                tool_calls=calls,
                business_state=business_state,
                assertions=[assertion] if assertion else [],
            )
            if not analysis.worth_storing:
                if candidate.high_risk:
                    self.failure_memory.queue_review(candidate, analysis.reason)
                task.failure_candidates.remove(data)
                continue
            event(
                "FAILURE_VERIFIED",
                candidate_id=candidate.candidate_id,
                failure_type=analysis.failure_type.value,
            )
            try:
                record_verified_failure(
                    self.failure_memory,
                    self.experience_extractor,
                    analysis,
                    task_id=task.task_id,
                    trace_id=candidate.trace_id,
                    task_type=task.current_goal.get("intent", "unknown"),
                    tool_name=candidate.tool_name,
                )
            except Exception:
                if candidate.high_risk:
                    self.failure_memory.queue_review(candidate, "experience extraction failed")
                    task.failure_candidates.remove(data)
                continue
            task.failure_candidates.remove(data)

    def _observe(self, state):
        task, session = state["task"], state["session"]
        if task.tool_calls >= self.max_tools:
            return self._stop(state, "任务工具调用预算已用尽，需人工接管。")
        # A completed or uncertain irreversible action is reconciled before planning anything new.
        if task.operation_id:
            task.tool_calls += 1
            result = RecoveryPolicy.reconcile_refund(
                self.repository,
                operation_id=task.operation_id,
                user_id=session.user_id,
                events=state["events"],
            )
            if result.ok:
                session.pending_action = None
                if state["goal"] != task.current_goal:
                    self.agent.invalidate_goal(session)
                    return self._stop(
                        state, "原退款申请已提交，不能通过改计划撤销；改办换货需人工核对。"
                    )
                return self._stop(
                    state,
                    f"已核实退款申请已提交，申请编号为 {result.data['request_id']}。",
                    "completed",
                )
            return self._stop(state, result.error or "业务结果尚未确认，需人工对账。")
        if state["goal"]["intent"] == "cancel":
            self.agent.invalidate_goal(session)
            session.intent = Intent.UNKNOWN
            if task.current_plan:
                for step in task.current_plan.steps:
                    if step.execution_status != "succeeded":
                        step.execution_status = "cancelled"
            state["events"].append(AgentEvent("guardrail", "cancel_pending_action", {}))
            return self._stop(state, "已取消当前操作。", "cancelled")
        if task.execution_status == "escalated":
            return self._stop(state, "任务已转人工处理，请等待核对，勿重复提交。")
        if task.tool_calls + self.agent.tool_executor.max_retries + 1 > self.max_tools:
            return self._stop(state, "任务工具调用预算已用尽，需人工接管。")
        task.tool_calls += 1
        result = self.agent._call_tool(
            "get_order",
            state["events"],
            lambda: self.repository.get_order(
                order_id=state["goal"]["order_id"], user_id=session.user_id
            ),
            intent=Intent(state["goal"]["intent"]),
        )
        task.tool_calls += max(0, state["events"][-1].detail.get("attempts", 1) - 1)
        if not result.ok:
            state["result"] = result
            state["next"] = "recover"
        else:
            state["result"] = result
            state["next"] = "reconcile"
        self._save(state)
        return state

    def _reconcile(self, state):
        task, session = state["task"], state["session"]
        reason = StateReconciler.reconcile(task, state["goal"], state["result"].data)
        if reason:
            self.repository.invalidate_authorization((session.authorization or {}).get("token"))
            session.authorization, session.pending_action = None, None
            if reason == "goal_changed":
                session.context_summary = None
                session.context_retry_after_messages = 0
                session.turns = (
                    session.turns[-1:]
                    if session.turns and session.turns[-1]["role"] == "user"
                    else []
                )
            state["allow_confirmation"] = False
            state["events"].append(AgentEvent("planning", "replan_trigger", {"reason": reason}))
        session.intent = Intent(task.current_goal["intent"])
        session.order_id = task.current_goal["order_id"]
        session.goal_version = task.goal_version
        state["next"] = "plan" if task.current_plan is None else "select_step"
        self._save(state)
        return state

    def _plan(self, state):
        task = state["task"]
        inputs = fingerprint({"goal": task.current_goal, "order": task.observed_state})
        if task.planning_count >= self.max_plans or task.last_plan_input == inputs:
            return self._stop(state, "没有新的业务事实或规划预算已用尽，需人工接管。")
        task.last_plan_input = inputs
        task.planning_count += 1
        decision = evaluate_refund(task.observed_state, damaged=False)
        if task.current_goal["intent"] == "refund" and not decision.eligible:
            return self._stop(state, f"暂不能直接申请退款：{decision.reason}。", "blocked")
        lessons = (
            self._memory_context(state, "planning")
            if self.planner.model and not self.naive_reflections
            else []
        )
        naive_lessons = (
            self.naive_reflections.all_lessons()
            if self.naive_reflections and self.planner.model
            else ()
        )

        def mark_injected():
            for item in lessons:
                try:
                    self._mark_memory_applied(state, item, "planning")
                except Exception:
                    pass

        try:
            session = state["session"]
            prompt_context = self.agent.layered_memory.assemble(session, state["message"])
            task.current_plan = self.planner.plan(
                task,
                human_review=decision.requires_human,
                lessons=naive_lessons or tuple(item.lesson for item in lessons),
                on_lessons_injected=mark_injected if lessons else None,
                context={
                    "summary": prompt_context.summary,
                    "recent_messages": list(prompt_context.history),
                }
                if self.planner.model and self.agent.context_compactor
                else None,
            )
            if self.planner.model:
                for key, value in getattr(self.planner.model, "usage", {}).items():
                    if isinstance(value, (int, float)):
                        task.llm_usage[key] = task.llm_usage.get(key, 0) + value
        except Exception as exc:
            if self.naive_reflections:
                self.naive_reflections.add(
                    f"候选计划失败：{type(exc).__name__}；以后规划应检查步骤和前置条件。"
                )
            event(
                "PLAN_VALIDATION_FAILED",
                task_id=task.task_id,
                plan_version=task.plan_version,
                error_type=type(exc).__name__,
            )
            if self.failure_memory and isinstance(exc, PlanValidationError):
                collector = current_trace()
                if collector:
                    trace_id = collector.record["trace_id"]
                    assertion = {
                        "rule": "plan_schema",
                        "passed": False,
                        "source": "validator",
                        "step": "plan",
                        "actual": type(exc).__name__,
                        "evidence_ref": f"trace:{trace_id}:PLAN_VALIDATION_FAILED",
                    }
                    self._queue_failure_candidate(
                        state, trigger="invalid_plan", assertion=assertion
                    )
            # An invalid model plan does not get executed or recursively repaired.
            task.failure_events.append(
                {
                    "category": "schema",
                    "error": type(exc).__name__,
                    "strategy": "deterministic_template",
                }
            )
            task.current_plan = TaskPlanner().plan(task, human_review=decision.requires_human)
        task.plan_version = task.current_plan.plan_version
        task.execution_status = "active"
        session = state["session"]
        session.plan_version = task.plan_version
        self.repository.set_context(
            session.session_id, session.user_id, session.goal_version, session.plan_version
        )
        # Reuse only the authoritative order observation obtained in this turn.
        for step in task.current_plan.steps:
            if step.action == "get_order":
                step.execution_status, step.result = "succeeded", task.observed_state
        state["events"].append(
            AgentEvent(
                "planning",
                "plan_validated",
                {
                    "plan_version": task.plan_version,
                    "goal_version": task.goal_version,
                    "steps": [s.action for s in task.current_plan.steps],
                },
            )
        )
        state["next"] = "select_step"
        self._save(state)
        return state

    def _select_step(self, state):
        task = state["task"]
        for step in task.current_plan.steps:
            if step.execution_status == "waiting_confirmation":
                step.execution_status = "pending"
            if step.execution_status == "pending" and set(step.dependencies).issubset(
                task.completed_steps
            ):
                reserved = {
                    "create_refund_request": 2,
                    "create_ticket": self.agent.tool_executor.max_retries + 2,
                }.get(step.action, 1)
                if task.tool_calls + reserved > self.max_tools:
                    return self._stop(state, "工具调用预算已用尽，需人工接管。")
                state["step_id"], state["next"] = step.step_id, "execute_step"
                return state
        if GoalStateEvaluator.evaluate(task) == "completed":
            return self._stop(state, state["reply"] or "该售后受理任务已完成。", "completed")
        return self._stop(state, "步骤依赖无法满足，需人工接管。")

    def _execute_step(self, state):
        task, session = state["task"], state["session"]
        step = next(s for s in task.current_plan.steps if s.step_id == state["step_id"])
        if task.current_plan.goal_version != task.goal_version or (
            step.parameters.get("order_id")
            and step.parameters["order_id"] != task.current_goal.get("order_id")
        ):
            collector = current_trace()
            trace_id = collector.record["trace_id"] if collector else "untraced"
            rule = (
                "goal_version"
                if task.current_plan.goal_version != task.goal_version
                else "order_snapshot"
            )
            self._queue_failure_candidate(
                state,
                trigger="stale_plan",
                tool_name=step.action,
                assertion={
                    "rule": rule,
                    "source": "validator",
                    "passed": False,
                    "step": step.action,
                    "observed": "旧计划仍待执行",
                    "expected": "当前目标与订单对应的新计划",
                    "actual": "执行前被拦截",
                    "evidence_ref": f"trace:{trace_id}:STALE_PLAN_BLOCKED",
                },
            )
            event("STALE_PLAN_BLOCKED", task_id=task.task_id, action=step.action)
            return self._stop(state, "计划已过期，需重新核对后执行。", "escalated")
        self._memory_context(state, "tool", tool_name=step.action)
        if step.action == "search_policy":
            query = "七天无理由 退货退款" if session.intent is Intent.REFUND else "换货 库存 吊牌"
            task.tool_calls += 1
            try:
                hits = self.agent.retriever.search(
                    query, category=task.observed_state["category"], limit=3
                )
            except Exception:
                state["result"] = ToolResult(
                    False, {"code": "policy_evidence"}, "政策检索不可用，需人工核对。"
                )
                state["next"] = "recover"
                return state
            state["citations"] = tuple(hit.chunk.policy_id for hit in hits)
            state["events"].append(
                AgentEvent(
                    "retrieval",
                    "search_policy",
                    {"hits": [{"policy_id": h.chunk.policy_id, "score": h.score} for h in hits]},
                )
            )
            if not hits:
                task.failure_events.append({"category": "policy_evidence", "strategy": "escalate"})
                if self.naive_reflections:
                    self.naive_reflections.add("政策检索没有结果；以后遇到类似任务应小心。")
                return self._stop(state, "缺少可信政策依据，需人工核对。")
            step.result = {"policy_ids": list(state["citations"])}
            step.execution_status = "succeeded"
        elif step.action == "create_refund_request":
            if not session.authorization:
                prepared = self.repository.prepare_refund(
                    order_id=session.order_id,
                    user_id=session.user_id,
                    context_id=session.session_id,
                    goal_version=session.goal_version,
                    plan_version=session.plan_version,
                )
                if not prepared.ok:
                    state["result"], state["next"] = prepared, "recover"
                    return state
                session.authorization = prepared.data
                session.refund_reason = state["message"]
                session.pending_action = "refund"
                state["allow_confirmation"] = False
            if not state["allow_confirmation"]:
                step.execution_status = "waiting_confirmation"
                return self._stop(
                    state,
                    f"退款金额 {session.authorization['amount_minor'] / 100:.2f} 元，"
                    "请回复“确认”后提交申请。",
                    "waiting_confirmation",
                )
            self.repository.confirm_authorization(
                token=session.authorization["token"], user_id=session.user_id
            )
            # Persist operation identity before the irreversible call for crash recovery.
            task.operation_id = session.authorization["operation_id"]
            task.tool_calls += 1
            self._save(state)
            event_start = len(state["events"])
            result = self.agent.submit_refund(session, state["events"])
            task.tool_calls += sum(e.name == "get_operation" for e in state["events"][event_start:])
            if not result.ok:
                state["result"], state["next"] = result, "recover"
                return state
            step.result, step.execution_status = result.data, "succeeded"
            session.pending_action = None
            state["reply"] = f"退款申请已提交，申请编号为 {result.data['request_id']}。"
        elif step.action == "create_ticket":
            task.tool_calls += 1
            result = self.agent._call_tool(
                "create_ticket",
                state["events"],
                lambda: self.repository.create_ticket(
                    order_id=session.order_id,
                    user_id=session.user_id,
                    description=f"任务 {task.task_id}：{session.intent.value} 核验",
                ),
                intent=session.intent,
            )
            if not result.ok:
                state["result"], state["next"] = result, "recover"
                return state
            step.result, step.execution_status = result.data, "succeeded"
            session.pending_action = None
            state["reply"] = f"已创建核验工单 {result.data['ticket_id']}，实际售后处理需等待审核。"
        else:
            return self._stop(state, "执行器拒绝未知计划操作。")
        task.consecutive_failures = 0
        state["next"] = "evaluate"
        self._save(state)
        return state

    def _evaluate(self, state):
        task = state["task"]
        step = next(s for s in task.current_plan.steps if s.step_id == state["step_id"])
        already_reconciled = any(
            e.event_type == "recovery"
            and e.name == "response_lost_reconciled"
            and e.detail.get("operation_id") == task.operation_id
            for e in state["events"]
        )
        if step.action in {"create_refund_request", "create_ticket"} and not already_reconciled:
            # Verify effects against the authoritative ledger, not a success-shaped reply.
            task.tool_calls += 1
            try:
                if step.action == "create_refund_request":
                    result = self.repository.get_operation(
                        operation_id=task.operation_id, user_id=state["session"].user_id
                    )
                    verified = result.ok and result.data.get("state") == "succeeded"
                    name = "get_operation"
                else:
                    result = self.repository.get_ticket(
                        ticket_id=step.result.get("ticket_id", ""), user_id=state["session"].user_id
                    )
                    verified, name = result.ok, "get_ticket"
            except Exception:
                result = ToolResult(False, error="执行结果核对不可用")
                verified, name = False, "verify_effect"
            state["events"].append(
                AgentEvent("tool", name, {"ok": result.ok, **result.data, "error": result.error})
            )
            if not verified:
                step.execution_status = "unknown"
                if task.operation_id:
                    self.repository.freeze_operation(
                        task.operation_id, state["session"].user_id, "执行结果与账本不符"
                    )
                return self._stop(state, "业务执行结果尚未核实，已冻结并转人工核对。")
        verdict = GoalStateEvaluator.evaluate(task)
        state["events"].append(
            AgentEvent("planning", "goal_state_evaluation", {"verdict": verdict})
        )
        if verdict == "completed":
            return self._stop(state, state["reply"], "completed")
        state["next"] = "select_step"
        return state

    def _recover(self, state):
        task, result = state["task"], state["result"]
        step = (
            next((s for s in task.current_plan.steps if s.step_id == state.get("step_id")), None)
            if task.current_plan
            else None
        )
        tool_name = step.action if step else None
        self._memory_context(state, "recovery", tool_name=tool_name)
        analysis = FailureAnalyzer.analyze(result)
        if self.failure_memory:
            collector = current_trace()
            trace_id = collector.record["trace_id"] if collector else "untraced"
            if result.data.get("code") == "authorization" and result.data.get("guard_reason"):
                event(
                    "POLICY_GUARD_BLOCKED",
                    task_id=task.task_id,
                    guard_reason=result.data["guard_reason"],
                    tool_name=tool_name,
                )
                self._queue_failure_candidate(
                    state,
                    trigger="guard_blocked",
                    tool_name=tool_name,
                    high_risk=True,
                    tool_call={
                        "action": tool_name,
                        "outcome": "guard_blocked",
                        "guard_reason": result.data["guard_reason"],
                        "evidence_ref": f"trace:{trace_id}:POLICY_GUARD_BLOCKED",
                    },
                )
            if result.data.get("outcome") == "duplicate":
                self._queue_failure_candidate(
                    state,
                    trigger="duplicate_recovery",
                    tool_name=tool_name,
                    high_risk=True,
                    tool_call={
                        "action": tool_name,
                        "outcome": "duplicate",
                        "original_operation_id": result.data.get("original_operation_id"),
                        "operation_id": result.data.get("operation_id"),
                    },
                )
        if self.naive_reflections:
            self.naive_reflections.add(
                f"{analysis.category}失败；以后遇到类似工具结果应采用{analysis.strategy}。"
            )
        task.consecutive_failures += 1
        task.failure_events.append(
            {
                "category": analysis.category,
                "strategy": analysis.strategy,
                "result": {"ok": result.ok, "data": result.data, "error": result.error},
            }
        )
        state["events"].append(AgentEvent("recovery", "failure_analyzed", task.failure_events[-1]))
        if analysis.strategy in {"reconcile", "reconfirm"} and (
            task.consecutive_failures < self.max_failures
        ):
            # The repository has explicitly rejected the write without effects.
            task.operation_id = None
            state["allow_confirmation"] = False
            self.repository.invalidate_authorization(
                (state["session"].authorization or {}).get("token")
            )
            state["session"].authorization = None
            state["next"] = "observe"
            self._save(state)
            return state
        if task.current_plan and state.get("step_id"):
            step = next(s for s in task.current_plan.steps if s.step_id == state["step_id"])
            step.execution_status = (
                "unknown" if analysis.category == "outcome_unknown" else "failed"
            )
        return self._stop(state, result.error or "任务无法安全恢复，需人工处理。")

    def _memory_context(self, state, stage, *, tool_name=None):
        if not self.memory_retriever:
            return []
        task = state["task"]
        collector = current_trace()
        try:
            return self.memory_retriever.retrieve(
                goal=task.current_goal.get("intent", ""),
                task_type=task.current_goal.get("intent", ""),
                stage=stage,
                tool_name=tool_name,
                conditions={task.current_goal.get("intent", ""), tool_name or ""},
                trace_id=collector.record["trace_id"] if collector else None,
                task_id=task.task_id,
            )
        except Exception:
            return []

    def _mark_memory_applied(self, state, item, stage):
        collector = current_trace()
        trace_id = collector.record["trace_id"] if collector else None
        self.failure_memory.record_usage(
            item.memory_id,
            trace_id=trace_id,
            task_id=state["task"].task_id,
            stage=stage,
            score=item.score,
            applied=True,
        )
        event(
            "FAILURE_MEMORY_APPLIED",
            memory_id=item.memory_id,
            score=item.score,
            applicable_reason=item.why_relevant,
            stage=stage,
        )

    def _finish(self, state):
        if not state["citations"] and state["task"].current_plan:
            for step in state["task"].current_plan.steps:
                if step.action == "search_policy":
                    state["citations"] = tuple(step.result.get("policy_ids", []))
        self._save(state)
        return state
