from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from .models import ToolResult
from .observability import Span, observed


@dataclass(frozen=True)
class ToolExecution:
    result: ToolResult
    attempts: int
    circuit_open: bool = False


class ToolExecutor:
    """Centralized retries and a small per-tool circuit breaker."""

    def __init__(
        self,
        *,
        max_retries: int = 1,
        failure_threshold: int = 3,
        recovery_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.max_retries = max_retries
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.clock = clock
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    @observed("tool_dispatch", "tool")
    def execute(self, name: str, operation: Callable[[], ToolResult]) -> ToolExecution:
        if self._is_open(name):
            return ToolExecution(
                result=ToolResult(ok=False, error="服务熔断中，请稍后重试"),
                attempts=0,
                circuit_open=True,
            )

        last_result = ToolResult(ok=False, error="工具执行失败")
        sensitive = name in {"create_refund_request", "create_ticket"}
        attempts_limit = 1 if name == "create_refund_request" else self.max_retries + 1
        for attempt in range(1, attempts_limit + 1):
            try:
                with Span(name, "tool_attempt", attributes={"attempt": attempt,
                          "retry_count": attempt - 1, "transport": "local_tool"}) as span:
                    last_result = operation()
                    span.output(last_result)
                    span.set(outcome=last_result.data.get("outcome", "unknown"))
                    if not last_result.ok:
                        span.failed(str(last_result.data.get("code", "ToolRejected")),
                                    last_result.error)
            except (ConnectionError, TimeoutError) as exc:
                last_result = ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
                if sensitive:
                    return ToolExecution(
                        ToolResult(
                            False,
                            {"code": "outcome_unknown", "outcome": "unknown"},
                            "工具执行结果不确定",
                        ),
                        attempt,
                    )
            except Exception as exc:  # the boundary must not crash the conversation runtime
                last_result = ToolResult(ok=False, error=f"unexpected tool error: {exc}")
                self._record_failure(name)
                if sensitive:
                    last_result = ToolResult(
                        False,
                        {"code": "outcome_unknown", "outcome": "unknown"},
                        "工具执行结果不确定",
                    )
                return ToolExecution(result=last_result, attempts=attempt)

            if last_result.ok:
                self._failures[name] = 0
                self._opened_at.pop(name, None)
                return ToolExecution(result=last_result, attempts=attempt)
            if attempt <= self.max_retries:
                continue

        self._record_failure(name)
        return ToolExecution(result=last_result, attempts=attempts_limit)

    def _record_failure(self, name: str) -> None:
        failures = self._failures.get(name, 0) + 1
        self._failures[name] = failures
        if failures >= self.failure_threshold:
            self._opened_at[name] = self.clock()

    def _is_open(self, name: str) -> bool:
        opened_at = self._opened_at.get(name)
        if opened_at is None:
            return False
        if self.clock() - opened_at >= self.recovery_seconds:
            self._opened_at.pop(name, None)
            self._failures[name] = 0
            return False
        return True
