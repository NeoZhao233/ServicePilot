"""Budget and failure accounting tests. No real provider calls."""

import importlib
import json
from pathlib import Path

import pytest

from servicepilot.providers import ProviderError
from servicepilot.retrieval import HashingTextEncoder


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("jev_stage1_online")
    import shutil

    original_root = module.ROOT
    fixture_root = tmp_path / "fixture"
    for relative in [
        "evaluations/retrieval_challenge.json",
        "data/policies.json",
        "evaluations/results/jev_stage1/offline_profile.json",
        *[
            "src/servicepilot/" + name
            for name in ("models.py", "retrieval.py", "providers.py", "retrieval_evaluation.py")
        ],
    ]:
        target = fixture_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / relative, target)
    baseline_path = fixture_root / "evaluations/results/jev_stage1/offline_profile.json"
    baseline = json.loads(baseline_path.read_text())
    baseline["code_sha256"] = {
        name: module.digest(fixture_root / "src/servicepilot" / name)
        for name in baseline["code_sha256"]
    }
    baseline_path.write_text(json.dumps(baseline))
    monkeypatch.setattr(module, "ROOT", fixture_root)
    monkeypatch.setattr(module, "FastEmbedTextEncoder", lambda model: HashingTextEncoder())
    monkeypatch.setenv("SERVICEPILOT_API_KEY", "mock-only-key")
    monkeypatch.setenv("SERVICEPILOT_INTENT_MODEL", "mock-model")
    monkeypatch.setenv("SERVICEPILOT_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("SERVICEPILOT_EMBEDDING_MODE", "local")
    # The runner's local-only model setup must not affect other tests.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return module


def invoke(runner, monkeypatch, output, budget, resume=False):
    argv = ["runner", "--allow-network", "--max-calls", str(budget), "--output", str(output)]
    if resume:
        argv.append("--resume")
    monkeypatch.setattr("sys.argv", argv)
    runner.main()
    return json.loads(output.read_text())


def test_budget_stop_resume_and_no_paid_overwrite(runner, monkeypatch, tmp_path):
    calls = []

    class FakeTransport:
        def __init__(self, settings):
            pass

        def post(self, path, payload):
            candidates = json.loads(payload["input"])["candidates"]
            calls.append(candidates)
            return {
                "model": "mock-model",
                "output_text": json.dumps(
                    {"ordered_policy_ids": [c["policy_id"] for c in candidates]}
                ),
                "usage": {"input_tokens": 20, "output_tokens": 5},
            }

    monkeypatch.setattr(runner, "UrllibJsonTransport", FakeTransport)
    output = tmp_path / "run.json"
    report = invoke(runner, monkeypatch, output, 2)
    assert len(calls) == report["attempts"] == 2
    assert report["status"] == "budget_exhausted"
    assert report["usage_totals"] == {"input_tokens": 40, "output_tokens": 10}
    assert report["coverage"] == {"completed": 2, "total": 24}
    assert report["api_cost"] is None
    report = invoke(runner, monkeypatch, output, 3, resume=True)
    assert len(calls) == report["attempts"] == 3
    assert len({r["case_id"] for r in report["cases"]}) == 3
    with pytest.raises(FileExistsError):
        invoke(runner, monkeypatch, output, 3)
    assert len(calls) == 3


def test_failure_stops_and_does_not_count_fallback_as_model_success(runner, monkeypatch, tmp_path):
    calls = []

    class BrokenTransport:
        def __init__(self, settings):
            pass

        def post(self, path, payload):
            calls.append(1)
            raise ProviderError("mock failure")

    monkeypatch.setattr(runner, "UrllibJsonTransport", BrokenTransport)
    report = invoke(runner, monkeypatch, tmp_path / "failure.json", 4)
    assert len(calls) == report["attempts"] == 1
    assert report["status"] == "failed_stopped"
    assert report["coverage"]["completed"] == 0
    assert report["usage_totals"]["input_tokens"] is None
    assert report["cases"][0]["status"] == "error"
    assert report["cases"][0]["ranked_ids"]  # Original production fallback still happened.
    assert report["metrics"]["ranked_ids_k5"]["queries"] == 0
