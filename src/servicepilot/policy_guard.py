from __future__ import annotations

import time


class PolicyGuard:
    """Pure predicates called inside the business repository's write transaction."""

    @staticmethod
    def authorization_valid(grant, context, *, user_id: str, snapshot: str) -> bool:
        return bool(
            grant["confirmed"]
            and not grant["invalidated"]
            and grant["expires"] > time.time()
            and context
            and context["user_id"] == user_id
            and context["goal_version"] == grant["goal_version"]
            and context["plan_version"] == grant["plan_version"]
            and snapshot == grant["snapshot"]
        )

    @staticmethod
    def rejection_reason(grant, context, *, user_id: str, snapshot: str) -> str | None:
        if not grant["confirmed"]:
            return "unconfirmed"
        if grant["invalidated"] or grant["expires"] <= time.time():
            return "stale_authorization"
        if not context or context["user_id"] != user_id:
            return "context_owner"
        if (
            context["goal_version"] != grant["goal_version"]
            or context["plan_version"] != grant["plan_version"]
        ):
            return "stale_plan"
        if snapshot != grant["snapshot"]:
            return "stale_snapshot"
        return None
