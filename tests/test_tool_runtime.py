import unittest

from servicepilot.models import ToolResult
from servicepilot.tool_runtime import ToolExecutor


class ToolExecutorTests(unittest.TestCase):
    def test_transient_failure_is_retried(self) -> None:
        calls = 0

        def flaky_operation() -> ToolResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("upstream timeout")
            return ToolResult(ok=True, data={"value": 42})

        execution = ToolExecutor(max_retries=1).execute("flaky", flaky_operation)
        self.assertTrue(execution.result.ok)
        self.assertEqual(execution.attempts, 2)

    def test_circuit_opens_after_repeated_failures_and_recovers(self) -> None:
        now = 0.0

        def clock() -> float:
            return now

        executor = ToolExecutor(
            max_retries=0,
            failure_threshold=2,
            recovery_seconds=10,
            clock=clock,
        )

        def failure() -> ToolResult:
            return ToolResult(ok=False, error="down")

        self.assertFalse(executor.execute("orders", failure).circuit_open)
        self.assertFalse(executor.execute("orders", failure).circuit_open)

        blocked = executor.execute("orders", failure)
        self.assertTrue(blocked.circuit_open)
        self.assertEqual(blocked.attempts, 0)

        now = 11.0
        recovered = executor.execute("orders", lambda: ToolResult(ok=True))
        self.assertTrue(recovered.result.ok)
        self.assertFalse(recovered.circuit_open)

    def test_unexpected_exception_is_contained(self) -> None:
        def broken() -> ToolResult:
            raise RuntimeError("bad payload")

        execution = ToolExecutor().execute("broken", broken)
        self.assertFalse(execution.result.ok)
        self.assertIn("unexpected tool error", execution.result.error or "")


if __name__ == "__main__":
    unittest.main()
