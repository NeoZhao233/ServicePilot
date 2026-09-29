"""Run from project root: python scripts/benchmark_trace.py. Offline HTTP benchmark."""

import json
import math
import statistics
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from servicepilot.observability import TraceConfig, TraceService
from servicepilot.trace_store import TraceStore
from servicepilot.webapp import create_app

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as folder:
    clients = {}
    results = {False: [], True: []}
    for enabled in results:
        db = Path(folder) / f"{enabled}.sqlite3"
        service = TraceService(TraceStore(db), TraceConfig(enabled=enabled))
        clients[enabled] = TestClient(
            create_app(
                data_dir=root / "data",
                session_db=db,
                commerce_db=db,
                dynamic=True,
                trace_service=service,
            )
        )
    for iteration in range(55):
        for enabled in [False, True] if iteration % 2 else [True, False]:
            start = time.perf_counter()
            response = clients[enabled].post(
                "/api/chat",
                json={
                    "user_id": "USER-001",
                    "message": "ORD-1003 物流",
                },
            )
            elapsed = (time.perf_counter() - start) * 1000
            assert response.status_code == 200, response.text
            if iteration >= 5:
                results[enabled].append(elapsed)
    print(
        json.dumps(
            {
                str(enabled): {
                    "samples": len(values),
                    "mean_ms": round(statistics.mean(values), 3),
                    "p50_ms": round(statistics.median(values), 3),
                    "p95_ms": round(sorted(values)[math.ceil(len(values) * 0.95) - 1], 3),
                }
                for enabled, values in results.items()
            },
            indent=2,
        )
    )
