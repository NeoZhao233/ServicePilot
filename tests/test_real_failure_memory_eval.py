import importlib.util
import json
from pathlib import Path

import pytest

from servicepilot.failure_memory import FailureMemoryStore

MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluations/run_real_failure_memory_eval.py"
spec = importlib.util.spec_from_file_location("real_failure_memory_eval", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
evaluate_one = module.evaluate_one
seed_synthetic_prior_failure = module.seed_synthetic_prior_failure
summarize = module.summarize


class RecordingTransport:
    def __init__(self):
        self.observations = []

    def post(self, _path, payload):
        request = json.loads(payload["messages"][1]["content"])
        self.observations.append(request["observation"])
        return {
            "choices": [{"message": {"content": json.dumps(request["template"])}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def test_real_eval_changes_only_memory_input_and_requires_pairs():
    store = FailureMemoryStore(":memory:")
    seed_synthetic_prior_failure(store)
    case = {"case_id": "fixture", "untrusted_note": "fictional instruction"}
    transport = RecordingTransport()
    baseline = evaluate_one(transport, "test-model", case, "B0", None)
    memory = evaluate_one(transport, "test-model", case, "B2", store)
    assert baseline["valid_plan"] and memory["valid_plan"]
    assert (
        transport.observations[0]["untrusted_note"] == (transport.observations[1]["untrusted_note"])
    )
    assert "verified_failure_lessons" not in transport.observations[0]
    assert len(transport.observations[1]["verified_failure_lessons"]) == 1
    assert baseline["memories_injected"] == 0
    assert memory["memories_injected"] == 1
    with pytest.raises(AssertionError, match="B0/B2"):
        summarize([{**baseline, "repeat": 0}])
    report = summarize([{**baseline, "repeat": 0}, {**memory, "repeat": 0}])
    assert report["paired_cases"] == 1
    assert report["B2_memories_injected"] == 1
