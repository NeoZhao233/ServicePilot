"""Local operator commands. Never registered as LLM tools."""

from __future__ import annotations

import argparse
import json
import sqlite3


def main():
    parser = argparse.ArgumentParser(description="Inspect escalations or recover a stopped session")
    parser.add_argument("--database", required=True)
    parser.add_argument("--release-session")
    parser.add_argument(
        "--process-stopped",
        action="store_true",
        help="Assert every worker using this session has stopped",
    )
    args = parser.parse_args()
    with sqlite3.connect(args.database) as db:
        db.row_factory = sqlite3.Row
        if args.release_session:
            if not args.process_stopped:
                parser.error("Stop the owning worker and supply --process-stopped before releasing")
            # Do not clear grants, operations or task history. On resume the runtime
            # reconciles the original operation rather than replaying a write.
            db.execute("DELETE FROM commerce_leases WHERE context_id=?", (args.release_session,))
        rows = db.execute("SELECT * FROM commerce_escalations WHERE state='open'").fetchall()
        print(json.dumps([dict(row) for row in rows], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
