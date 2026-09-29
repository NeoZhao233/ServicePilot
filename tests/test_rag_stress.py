from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stress_cases_are_reproducible_and_independent() -> None:
    builder = _load_script("build_rag_stress_cases.py")
    frozen = json.loads((ROOT / "evaluations/rag_stress_cases.json").read_text())
    original = json.loads((ROOT / "evaluations/retrieval_challenge.json").read_text())
    assert builder.build() == frozen
    assert len(frozen) == 100
    assert {case["query"] for case in frozen}.isdisjoint({case["query"] for case in original})
    assert sorted(len(case["relevant_policy_ids"]) for case in frozen).count(4) == 7


def test_full_evidence_rate_excludes_structurally_impossible_top3_cases() -> None:
    benchmark = _load_script("evaluate_rag_stress.py")
    cases = [
        {"case_id": "a", "slice": "pair", "relevant_policy_ids": ["p1", "p2"]},
        {"case_id": "b", "slice": "quad", "relevant_policy_ids": ["p1", "p2", "p3", "p4"]},
    ]
    predictions = {"a": ["p1", "p2"], "b": ["p1", "p2", "p3", "p4"]}
    result = benchmark._metrics(cases, predictions, 3)
    assert result["n"] == 2
    assert result["all_evidence_eligible_n"] == 1
    assert result["all_evidence_rate"] == 1
    assert result["macro_recall"] == 0.875


def test_controlled_missing_arm_preserves_natural_failures() -> None:
    benchmark = _load_script("evaluate_rag_stress.py")
    cases = [
        {"case_id": "natural", "query": "q1", "relevant_policy_ids": ["a", "b"]},
        {"case_id": "injected", "query": "q2", "relevant_policy_ids": ["a", "b"]},
    ]
    predictions = {"natural": ["a", "x"], "injected": ["a", "b", "x"]}
    rows = benchmark._controlled_missing(cases, predictions)
    assert rows[0]["arm"] == "natural_miss"
    assert rows[0]["missing_policy_ids"] == ["b"]
    assert rows[1]["arm"] == "controlled_omission"
    assert rows[1]["initial_policy_ids"] == ["a", "x"]
    assert rows[1]["removed_policy_id"] == "b"
