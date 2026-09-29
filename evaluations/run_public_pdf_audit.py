"""Run a small, separately reported audit on two public real-world PDFs.

Supply downloaded IRS Form W-9 and CFPB sample credit-card statement paths.
No downloaded documents are committed to this repository.
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
        "pdf": "w9",
        "question": "What is line 1 for?",
        "expected": ["Name of entity/individual"],
        "page": 1,
    },
    {
        "pdf": "w9",
        "question": "Which line asks for the business name?",
        "expected": ["Business name/disregarded entity"],
        "page": 1,
    },
    {
        "pdf": "statement",
        "question": "What is the minimum payment due?",
        "expected": ["$25"],
        "page": 1,
    },
    {
        "pdf": "statement",
        "question": "What is the APR for purchases?",
        "expected": ["19.80%"],
        "page": 1,
    },
    {
        "pdf": "statement",
        "question": "What is the new balance?",
        "expected": ["$1,392.71", "$1392.71"],
        "page": 1,
    },
]


def run(paths: dict[str, Path], output: Path):
    parser = DoclingPdfParser()
    encoder = FastEmbedTextEncoder()
    documents = {}
    parsing = {}
    for name, path in paths.items():
        started = time.perf_counter()
        elements = parser.parse(path, name, output.parent / "public_pdf_images" / name)
        documents[name] = PdfEvidenceSearch(elements, encoder)
        parsing[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pages": sorted({element.page_number for element in elements}),
            "counts": {
                kind: sum(e.element_type == kind for e in elements)
                for kind in ("text", "table", "image")
            },
            "parse_and_index_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    rows = []
    for case in CASES:
        started = time.perf_counter()
        bundles = documents[case["pdf"]].search(case["question"])
        result = answer_from_bundles(case["question"], bundles)
        rows.append(
            {
                **case,
                "answer": result["answer"],
                "answer_contains_gold": any(item in result["answer"] for item in case["expected"]),
                "cited_pages": [item["page_number"] for item in result["evidence"]],
                "query_elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"parsing": parsing, "cases": rows}, ensure_ascii=False, indent=2))
    print(f"{sum(row['answer_contains_gold'] for row in rows)}/{len(rows)} gold snippets")


if __name__ == "__main__":
    command = argparse.ArgumentParser()
    command.add_argument("--w9", type=Path, required=True)
    command.add_argument("--statement", type=Path, required=True)
    command.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results" / "public_pdf_audit.json"
    )
    args = command.parse_args()
    run({"w9": args.w9, "statement": args.statement}, args.output)
