"""Budget/exclusion tests for paid experiments; all transport calls are fake."""

import asyncio
import importlib
import json
import shutil
from argparse import Namespace
from pathlib import Path

import pytest

from servicepilot.retrieval import HashingTextEncoder


@pytest.fixture
def runner(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    module = importlib.import_module("pluggable_rag_eval")
    fixture_root = tmp_path / "fixture"
    paths = [
        "data/policies.json",
        "evaluations/retrieval_challenge.json",
        "evaluations/corrective_rag_cases.json",
        "evaluations/results/jev_stage1/b0_online.json",
        "evaluations/results/jev_corrective/observed_runs.json",
        "src/servicepilot/evaluators.py",
        "src/servicepilot/corrective_rag.py",
        "src/servicepilot/evidence.py",
        "scripts/pluggable_rag_eval.py",
    ]
    for name in paths:
        target = fixture_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    monkeypatch.setattr(module, "ROOT", fixture_root)
    monkeypatch.setattr(module, "FastEmbedTextEncoder", lambda model: HashingTextEncoder())
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SERVICEPILOT_API_KEY", "fake-test-key")
    monkeypatch.setenv("SERVICEPILOT_INTENT_MODEL", "fake-model")
    monkeypatch.setenv("SERVICEPILOT_EMBEDDING_MODE", "hashing")
    requests = []

    class Transport:
        def __init__(self, settings):
            pass

        def post(self, path, payload):
            requests.append(payload)
            data = json.loads(payload["input"])
            ids = [d["policy_id"] for d in data["documents"]]
            if payload["text"]["format"]["name"] == "ScoreDecision":
                decision = {
                    "documents": [
                        dict(policy_id=pid, score=0.8, direct_evidence=True, reason="direct")
                        for pid in ids
                    ]
                }
            else:
                decision = dict(
                    status="SUFFICIENT",
                    covered_questions=["test"],
                    missing_information=[],
                    supporting_ids=ids,
                    retrieval_targets=[],
                    diagnostic="test",
                )
            return dict(
                output_text=json.dumps(decision),
                model="fake-model",
                status="completed",
                usage={"input_tokens": 10, "output_tokens": 5},
            )

    monkeypatch.setattr(module, "UrllibJsonTransport", Transport)
    cases = json.loads((fixture_root / "evaluations/retrieval_challenge.json").read_text())
    cases += json.loads((fixture_root / "evaluations/corrective_rag_cases.json").read_text())
    excluded = [c["case_id"] for c in cases if c["case_id"] != "para_02"]
    return module, requests, excluded


def test_budget_is_reserved_and_stops_without_retry(runner, tmp_path):
    module, requests, excluded = runner
    output = tmp_path / "bounded.json"
    ok = asyncio.run(
        module.run(Namespace(allow_network=True, max_calls=1, output=output, exclude_case=excluded))
    )
    report = json.loads(output.read_text())
    assert not ok and report["status"] == "failed"
    assert report["attempts"] == len(requests) == 1
    assert report["physical_calls"][0]["usage"]["input_tokens"] == 10
    assert report["failure"]["error_type"] == "RuntimeError"


def test_exclusion_avoids_retry_and_existing_output_is_never_overwritten(runner, tmp_path):
    module, requests, excluded = runner
    output = tmp_path / "partial.json"
    args = Namespace(allow_network=True, max_calls=10, output=output, exclude_case=excluded)
    assert asyncio.run(module.run(args))
    report = json.loads(output.read_text())
    rows = [r for r in report["runs"] if r["group"] != "B0"]
    assert {r["case_id"] for r in rows} == {"para_02"}
    assert {r["group"] for r in rows} == {"B1", "B2", "B3"}
    assert all("钱退了个寂寞" not in r["input"] for r in requests)
    before = len(requests)
    with pytest.raises(ValueError, match="overwrite"):
        asyncio.run(module.run(args))
    assert len(requests) == before


def test_unknown_exclusion_is_rejected_before_call(runner, tmp_path):
    module, requests, _ = runner
    with pytest.raises(ValueError, match="Unknown excluded"):
        asyncio.run(
            module.run(
                Namespace(
                    allow_network=True,
                    max_calls=10,
                    output=tmp_path / "bad.json",
                    exclude_case=["invalid"],
                )
            )
        )
    assert not requests


def test_summary_uses_common_cases_and_counts_failed_call_usage_once(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    summarize = importlib.import_module("summarize_pluggable_rag").summarize
    cases = [{"case_id": key, "relevant_policy_ids": ["A"]} for key in ("one", "two")]
    rows = [
        {"group": group, "case_id": key, "retrieved_ids": ["A"]}
        for group in ("B0", "B1", "B2", "B3")
        for key in ("one", "two")
        if not (key == "one" and group in ("B2", "B3"))
    ]
    artifact = dict(
        cases_sha256="a",
        policies_sha256="b",
        status="failed",
        runs=rows,
        attempts=2,
        failure={"case_id": "one", "group": "B2"},
        physical_calls=[
            dict(elapsed_ms=1, usage={"input_tokens": 7, "output_tokens": 4}),
            dict(elapsed_ms=2, usage={"input_tokens": 8, "output_tokens": 9}),
        ],
    )
    summary = summarize([artifact], cases, [])
    assert summary["status"] == "partial"
    assert summary["common_frozen_case_ids"] == ["two"]
    assert summary["physical"]["usage"] == {"input_tokens": 15, "output_tokens": 13, "cost": None}
    assert summary["common_frozen"]["B1"]["answer_accuracy"] is None
    artifact["attempts"] += 1  # No response: billed tokens are unknown, not zero.
    summary = summarize([artifact], cases, [])
    assert summary["physical"]["usage"]["input_tokens"] is None
    assert summary["physical"]["reported_usage_subtotal"]["input_tokens"] == 15
