from __future__ import annotations

import functools

from .evidence import retrieval_node_context
from .observability import Span, current_trace, event


def node(name, function, *, graph="main"):
    """Wrap registration once, preserving each graph node's original return value."""

    @retrieval_node_context
    @functools.wraps(function)
    def run(state):
        task = state.get("task")
        old_plan = getattr(task, "plan_version", None)
        old_goal = dict(getattr(task, "current_goal", {}))
        old_observation = dict(getattr(task, "observed_state", {}))
        plan = getattr(task, "current_plan", None)
        old_steps = {s.step_id: s.execution_status for s in plan.steps} if plan else {}
        history_count = len(task.history) if task else 0
        c = current_trace()
        if c and task:
            c.bind(
                user_id=task.user_id, thread_id=state["session"].session_id, task_id=task.task_id
            )
        with Span(
            f"{graph}.{name}",
            "langgraph",
            state,
            {"node": name, "source": graph, "plan_version": old_plan},
        ) as span:
            if name == "execute_step" and task:
                event(
                    "PLAN_STEP_STARTED",
                    task_id=task.task_id,
                    plan_version=old_plan,
                    step_id=state.get("step_id"),
                )
            result = function(state)
            span.output(result)
            if result.get("error"):
                span.failed("NodeResultError", result["error"])
            if task:
                version = task.plan_version
                span.set(
                    plan_version=version,
                    goal_version=task.goal_version,
                    outcome=task.execution_status,
                    is_replan=name == "plan" and task.planning_count > 1,
                )
                if c:
                    c.bind(outcome=task.execution_status)
                if old_goal and task.current_goal != old_goal:
                    event(
                        "GOAL_CHANGED",
                        task_id=task.task_id,
                        plan_version=version,
                        old_goal=old_goal,
                        current_goal=task.current_goal,
                    )
                if len(task.history) > history_count:
                    history = task.history[-1]
                    cancelled = [
                        s["step_id"]
                        for s in history.get("plan", {}).get("steps", [])
                        if s["execution_status"] == "cancelled"
                    ]
                    completed_previous = [
                        s["step_id"]
                        for s in history.get("plan", {}).get("steps", [])
                        if s["execution_status"] == "succeeded"
                    ]
                    retained = [
                        s["step_id"]
                        for s in history.get("plan", {}).get("steps", [])
                        if s["action"] == "get_order"
                        and s["execution_status"] == "succeeded"
                        and old_observation == task.observed_state
                    ]
                    attributes = {
                        "reason": history["reason"],
                        "old_plan_version": old_plan,
                        "cancelled_steps": cancelled,
                        "retained_steps": retained,
                        "retry_steps": cancelled,
                        "completed_previous_steps": completed_previous,
                    }
                    event(
                        "PLAN_INVALIDATED",
                        task_id=task.task_id,
                        plan_version=old_plan,
                        **attributes,
                    )
                    event(
                        "REPLAN_TRIGGERED",
                        task_id=task.task_id,
                        plan_version=old_plan,
                        **attributes,
                    )
                if name == "plan" and task.current_plan and version != old_plan:
                    event(
                        "PLAN_CREATED",
                        task_id=task.task_id,
                        plan_version=version,
                        old_plan_version=old_plan,
                        new_plan_version=version,
                        goal_version=task.goal_version,
                        step_count=len(task.current_plan.steps),
                        validation="accepted",
                        steps=task.current_plan.steps,
                    )
                    span.set(
                        input_tokens=None, output_tokens=None
                    )  # Actual usage is in LLM children.
                if name == "execute_step" and task.current_plan:
                    for step in task.current_plan.steps:
                        if (
                            step.execution_status == "succeeded"
                            and old_steps.get(step.step_id) != "succeeded"
                        ):
                            event(
                                "PLAN_STEP_COMPLETED",
                                task_id=task.task_id,
                                plan_version=version,
                                step_id=step.step_id,
                                expected_effect=step.expected_effect,
                                result=step.result,
                            )
                if name == "evaluate":
                    event(
                        "GOAL_EVALUATED",
                        task_id=task.task_id,
                        plan_version=version,
                        current_goal=task.current_goal,
                        observed_state=task.observed_state,
                        verdict=task.execution_status,
                        completed_steps=task.completed_steps,
                    )
            return result

    return run


def bridge_events(events, *, task_id=None, plan_version=None):
    c = current_trace()
    if not c:
        return
    mapping = {
        "tool_blocked": "POLICY_REJECTED",
        "authorization_invalidated": "POLICY_REJECTED",
        "failure_analyzed": "RECOVERY_STARTED",
        "cancel_pending_action": "GOAL_CHANGED",
    }
    for item in events:
        identity = id(item)
        with c.lock:
            if identity in c.seen_events:
                continue
            c.seen_events.add(identity)
        kind = mapping.get(item.name)
        if kind:
            event(kind, task_id=task_id, plan_version=plan_version, **item.detail)
        if item.event_type == "tool" and item.detail.get("ok") is False:
            event(
                "TOOL_FAILED",
                task_id=task_id,
                plan_version=plan_version,
                name=item.name,
                **item.detail,
            )
            if item.detail.get("outcome") == "unknown":
                event(
                    "EXECUTION_UNCERTAIN",
                    task_id=task_id,
                    plan_version=plan_version,
                    name=item.name,
                    **item.detail,
                )
