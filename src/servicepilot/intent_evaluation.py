from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .models import Intent
from .providers import (
    APIEmbeddingEncoder,
    OnlineModelSettings,
    ResponsesIntentModel,
    UrllibJsonTransport,
)
from .retrieval import FastEmbedTextEncoder, HashingTextEncoder
from .routing import (
    CachedVotingIntentRouter,
    EmbeddingIntentRouter,
    IntentRouter,
    LLMIntentRouter,
    RuleIntentRouter,
)


def _case_slices(case: dict[str, object]) -> tuple[str, ...]:
    raw = case.get("slice", "unspecified")
    if isinstance(raw, list):
        return tuple(str(item) for item in raw) or ("unspecified",)
    return (str(raw),)


def _slice_metrics(predictions: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in predictions:
        for slice_name in item["slices"]:
            grouped[str(slice_name)].append(item)
    metrics: dict[str, dict[str, object]] = {}
    for name, items in sorted(grouped.items()):
        correct = sum(bool(item["correct"]) for item in items)
        metrics[name] = {
            "count": len(items),
            "correct": correct,
            "accuracy": round(correct / len(items), 4),
            "avg_confidence": round(
                sum(float(item["confidence"]) for item in items) / len(items), 4
            ),
            "abstention_rate": round(
                sum(item["predicted"] == Intent.UNKNOWN.value for item in items) / len(items), 4
            ),
        }
    return metrics


def evaluate_intents(cases_path: Path, router: IntentRouter | None = None) -> dict[str, object]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    active_router = router or CachedVotingIntentRouter(
        (
            ("embedding", EmbeddingIntentRouter(HashingTextEncoder()), 0.2),
            ("keyword", RuleIntentRouter(), 0.8),
        )
    )
    labels = list(Intent)
    counts: dict[Intent, dict[str, int]] = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    predictions: list[dict[str, object]] = []

    for case in cases:
        expected = Intent(case["intent"])
        acceptable = {
            Intent(value)
            for value in case.get("acceptable_intents", [expected.value])
        }
        decision = active_router.route(case["message"])
        predicted = decision.intent
        confusion[expected.value][predicted.value] += 1
        for label in labels:
            if expected is label and predicted is label:
                counts[label]["tp"] += 1
            elif expected is not label and predicted is label:
                counts[label]["fp"] += 1
            elif expected is label and predicted is not label:
                counts[label]["fn"] += 1
        predictions.append(
            {
                "case_id": case["case_id"],
                "expected": expected.value,
                "acceptable": sorted(intent.value for intent in acceptable),
                "predicted": predicted.value,
                "confidence": decision.confidence,
                "signals": decision.matched_signals,
                "correct": predicted in acceptable,
                "slices": _case_slices(case),
                "difficulty": str(case.get("difficulty", "unspecified")),
            }
        )

    per_intent: dict[str, dict[str, float]] = {}
    for label, values in counts.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_intent[label.value] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    total = len(cases)
    correct = sum(bool(item["correct"]) for item in predictions)
    fallback_count = sum("fallback" in item["signals"] for item in predictions)
    macro_f1_all = sum(metrics["f1"] for metrics in per_intent.values()) / len(per_intent)
    supported = {Intent(case["intent"]).value for case in cases}
    macro_f1 = (
        sum(per_intent[label]["f1"] for label in supported) / len(supported)
        if supported
        else 0.0
    )
    return {
        "case_count": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(macro_f1, 4),
        "macro_f1_all_labels": round(macro_f1_all, 4),
        "fallback_count": fallback_count,
        "slice_metrics": _slice_metrics(predictions),
        "per_intent": per_intent,
        "confusion": {key: dict(value) for key, value in confusion.items()},
        "predictions": predictions,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate an intent router")
    parser.add_argument(
        "--cases", type=Path, default=project_root / "evaluations" / "intent_cases.json"
    )
    parser.add_argument("--output", type=Path, help="Write predictions and metrics as JSON")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Evaluate the configured Responses model with rule fallback",
    )
    args = parser.parse_args()
    router: IntentRouter | None = None
    if args.online:
        settings = OnlineModelSettings.from_env()
        transport = UrllibJsonTransport(settings)
        if settings.embedding_mode == "api":
            encoder = APIEmbeddingEncoder(transport, settings.embedding_model)
        elif settings.embedding_mode == "local":
            encoder = FastEmbedTextEncoder(settings.embedding_model)
        else:
            encoder = HashingTextEncoder()
        router = CachedVotingIntentRouter(
            (
                (
                    "llm",
                    LLMIntentRouter(ResponsesIntentModel(transport, settings.intent_model)),
                    0.5,
                ),
                ("embedding", EmbeddingIntentRouter(encoder), 0.2),
                ("keyword", RuleIntentRouter(), 0.3),
            )
        )
    report = evaluate_intents(args.cases, router=router)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{serialized}\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
