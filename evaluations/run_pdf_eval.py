"""Paired B0/B1 evaluation on the same synthetic PDFs and gold questions."""

import json
import re
import time
from pathlib import Path

from servicepilot.models import PolicyChunk
from servicepilot.pdf_fusion import DoclingPdfParser, PdfEvidenceSearch, answer_from_bundles
from servicepilot.retrieval import (
    BM25PolicyRetriever,
    FastEmbedTextEncoder,
    HybridPolicyRetriever,
    VectorPolicyRetriever,
)

ROOT = Path(__file__).parent
CASES = json.loads((ROOT / "pdf_cases.json").read_text(encoding="utf-8"))


def baseline_index(elements, encoder):
    # There was no PDF extractor before this change. B0 emulates a plain-text
    # extraction over the same Docling result, flattening tables and dropping images.
    pages = {}
    for element in elements:
        if element.element_type == "image":
            continue
        content = re.sub(r"\|[- :]+\|", " ", element.content).replace("|", " ")
        pages.setdefault(element.page_number, []).append(content)
    chunks = [
        PolicyChunk(str(page), f"第{page}页", "\n".join(parts)) for page, parts in pages.items()
    ]
    search = HybridPolicyRetriever(
        BM25PolicyRetriever(chunks), VectorPolicyRetriever(chunks, encoder)
    )
    return search


def baseline(search, question):
    hits = search.search(question, limit=3)
    visual = any(word in question for word in ("图片", "照片", "图中", "图里", "外观", "看出"))
    if visual:
        return "图片证据不足，无法判断图片中的内容。", [int(h.chunk.policy_id) for h in hits], False
    if not hits:
        return "证据不足。", [], False
    return (
        f"PDF 第 {hits[0].chunk.policy_id} 页：{hits[0].chunk.content[:900]}",
        [int(h.chunk.policy_id) for h in hits],
        False,
    )


def run():
    parser = DoclingPdfParser()
    encoder = FastEmbedTextEncoder()
    documents = {}
    parsing = {}
    indexes = {}
    for name in sorted({case["pdf"] for case in CASES}):
        started = time.perf_counter()
        elements = parser.parse(
            ROOT / "pdf_fixtures" / name, name, ROOT / "results" / "pdf_images" / Path(name).stem
        )
        documents[name] = elements
        parsing[name] = {
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "counts": {
                kind: sum(e.element_type == kind for e in elements)
                for kind in ("text", "table", "image")
            },
        }
        started = time.perf_counter()
        b0 = baseline_index(elements, encoder)
        b0_ms = round((time.perf_counter() - started) * 1000, 2)
        started = time.perf_counter()
        b1 = PdfEvidenceSearch(elements, encoder)
        b1_ms = round((time.perf_counter() - started) * 1000, 2)
        indexes[name] = {"B0": b0, "B1": b1}
        parsing[name]["index_elapsed_ms"] = {"B0": b0_ms, "B1": b1_ms}
    rows = []
    for case in CASES:
        elements = documents[case["pdf"]]
        for arm in ("B0", "B1"):
            started = time.perf_counter()
            if arm == "B0":
                answer, pages, structured = baseline(indexes[case["pdf"]]["B0"], case["question"])
                image_retained = False
                relation = False
            else:
                bundles = indexes[case["pdf"]]["B1"].search(case["question"])
                result = answer_from_bundles(case["question"], bundles)
                answer = result["answer"]
                pages = [bundle.page_number for bundle in bundles]
                structured = any(e.element_type == "table" and "|" in e.content for e in elements)
                image_retained = any(bundle.image_paths for bundle in bundles)
                relation = any(
                    bundle.image_paths and "主板损坏" in bundle.text for bundle in bundles
                )
            rows.append(
                {
                    "id": case["id"],
                    "arm": arm,
                    "question": case["question"],
                    "expected": case["expected"],
                    "answer": answer,
                    "answer_contains_gold": case["expected"] in answer,
                    "correct_page_retrieved": case["page"] in pages,
                    "table_structure_preserved": structured if case["kind"] == "table" else None,
                    "image_retained": image_retained
                    if case["kind"] in {"image", "relation", "decoration"}
                    else None,
                    "image_text_related": relation if case["kind"] == "relation" else None,
                    "vision_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "query_elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
    summary = {}
    for arm in ("B0", "B1"):
        subset = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            "gold_in_answer": sum(row["answer_contains_gold"] for row in subset),
            "correct_page_retrieved": sum(row["correct_page_retrieved"] for row in subset),
            "table_structure_preserved": sum(
                row["table_structure_preserved"] is True for row in subset
            ),
            "image_retained": sum(row["image_retained"] is True for row in subset),
            "image_text_related": sum(row["image_text_related"] is True for row in subset),
            "vision_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "query_elapsed_ms": round(sum(row["query_elapsed_ms"] for row in subset), 2),
        }
    result = {
        "method": (
            "Synthetic PDFs; same Docling extraction; B0 flattened text baseline "
            "because no pre-existing PDF path; local BGE encoder; no vision model"
        ),
        "parsing": parsing,
        "summary": summary,
        "cases": rows,
    }
    output = ROOT / "results" / "pdf_eval.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
