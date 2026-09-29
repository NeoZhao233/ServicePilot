"""Reproducible audit of HP's public Deskjet quick-reference PDF.

Download https://www.hp.com/ctg/Manual/bpd09054.pdf separately, then pass its path.
The third-party PDF is not committed to the repository.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from servicepilot.pdf_fusion import DoclingPdfParser, PdfEvidenceSearch, answer_from_bundles
from servicepilot.retrieval import FastEmbedTextEncoder

CASES = [
    {
        "question": "What is the part number for the black print cartridge?",
        "expected": "HP 15",
        "page": 6,
    },
    {
        "question": "What is the part number for the tri-color print cartridge?",
        "expected": "HP 17",
        "page": 6,
    },
    {
        "question": "Which guide covers warranty information?",
        "expected": "quick reference guide",
        "page": 7,
    },
    {
        "question": "According to the numbered list, what is item 8?",
        "expected": "power module",
        "page": 5,
    },
    {
        "question": "What does the diagram show at item 8?",
        "expected": "证据不足",
        "page": 5,
        "needs_image_reference": True,
    },
    {
        "question": "Is a printer cable included in the box?",
        "expected": "not included",
        "page": 6,
    },
]


def run(pdf_path: Path, output: Path) -> None:
    started = time.perf_counter()
    image_dir = output.parent / "manual_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for old_image in image_dir.glob("*.png"):
        old_image.unlink()
    elements = DoclingPdfParser().parse(pdf_path, "hp-deskjet", image_dir)
    search = PdfEvidenceSearch(elements, FastEmbedTextEncoder())
    parse_and_index_ms = round((time.perf_counter() - started) * 1000, 2)
    rows = []
    for case in CASES:
        started = time.perf_counter()
        bundles = search.search(case["question"])
        result = answer_from_bundles(case["question"], bundles)
        cited_pages = [item["page_number"] for item in result["evidence"]]
        has_image_reference = any(item["image_paths"] for item in result["evidence"])
        rows.append(
            {
                **case,
                "answer": result["answer"],
                "answer_contains_gold": case["expected"].lower() in result["answer"].lower(),
                "correct_page_cited": case["page"] in cited_pages,
                "image_reference_present": has_image_reference,
                "vision_called": result["vision_called"],
                "query_elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "source_url": "https://www.hp.com/ctg/Manual/bpd09054.pdf",
                "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                "pages": sorted({element.page_number for element in elements}),
                "counts": {
                    kind: sum(e.element_type == kind for e in elements)
                    for kind in ("text", "table", "image")
                },
                "parse_and_index_ms": parse_and_index_ms,
                "cases": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"{sum(row['answer_contains_gold'] and row['correct_page_cited'] for row in rows)}"
        f"/{len(rows)} gold answers with cited page"
    )


if __name__ == "__main__":
    command = argparse.ArgumentParser()
    command.add_argument("--manual", type=Path, required=True)
    command.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "manual_pdf_audit.json",
    )
    args = command.parse_args()
    run(args.manual, args.output)
