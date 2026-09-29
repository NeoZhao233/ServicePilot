"""Offline maintenance of observability tables; never touches commerce records."""

import argparse
import json
import os

from .observability import TraceConfig
from .trace_store import TraceStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=os.getenv("SERVICEPILOT_TRACE_DB"), required=False)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--retention-days", type=int, default=TraceConfig.from_env().retention_days)
    parser.add_argument("--mark-abandoned", action="store_true")
    parser.add_argument("--workers-stopped", action="store_true")
    args = parser.parse_args()
    if not args.database:
        parser.error("--database or SERVICEPILOT_TRACE_DB is required")
    if args.retention_days < 1:
        parser.error("--retention-days must be positive")
    if args.mark_abandoned and not args.workers_stopped:
        parser.error("Stop every worker first, then pass --workers-stopped")
    store = TraceStore(args.database)
    result = {}
    if args.cleanup:
        result["deleted_traces"] = store.cleanup(args.retention_days)
    if args.mark_abandoned:
        result["interrupted_traces"] = store.mark_abandoned(older_than_seconds=0)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
