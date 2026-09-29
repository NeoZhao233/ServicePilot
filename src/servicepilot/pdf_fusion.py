"""Small, user-scoped PDF evidence adapter. No policy corpus writes."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from .models import PolicyChunk, RetrievalHit
from .observability import Span, event
from .retrieval import (
    BM25PolicyRetriever,
    HashingTextEncoder,
    HybridPolicyRetriever,
    VectorPolicyRetriever,
    tokenize,
)

ElementType = Literal["text", "table", "image"]


@dataclass
class PdfElement:
    document_id: str
    page_number: int
    element_id: str
    element_type: ElementType
    content: str
    image_path: str | None = None
    related_element_ids: list[str] = field(default_factory=list)
    bbox: tuple[float, float, float, float] | None = None
    label: str | None = None


def _page(item) -> int:
    provenance = getattr(item, "prov", None) or []
    return int(provenance[0].page_no) if provenance else 1


def _bbox(item) -> tuple[float, float, float, float] | None:
    provenance = getattr(item, "prov", None) or []
    if not provenance or getattr(provenance[0], "bbox", None) is None:
        return None
    box = provenance[0].bbox
    return (float(box.l), float(box.t), float(box.r), float(box.b))


def _is_corner_decoration(
    caption: str,
    bbox: tuple[float, float, float, float] | None,
    page_width: float,
    page_height: float,
) -> bool:
    """Drop uncaptioned printer marks and page furniture in outer corners."""
    if caption or not bbox:
        return False
    left, top, right, bottom = bbox
    near_side = left < page_width * 0.05 or right > page_width * 0.95
    in_corner_band = bottom > page_height * 0.67 or top < page_height * 0.16
    return near_side and in_corner_band


def _inside_picture(text: PdfElement, picture: PdfElement) -> bool:
    if text.page_number != picture.page_number or not text.bbox or not picture.bbox:
        return False
    x_center = (text.bbox[0] + text.bbox[2]) / 2
    y_center = (text.bbox[1] + text.bbox[3]) / 2
    return (
        picture.bbox[0] <= x_center <= picture.bbox[2]
        and picture.bbox[3] <= y_center <= picture.bbox[1]
    )


def _link(left: PdfElement, right: PdfElement) -> None:
    if left.element_id != right.element_id:
        if right.element_id not in left.related_element_ids:
            left.related_element_ids.append(right.element_id)
        if left.element_id not in right.related_element_ids:
            right.related_element_ids.append(left.element_id)


def _nearby(left: PdfElement, right: PdfElement) -> float | None:
    """Docling boxes use PDF coordinates; return a small same-column gap."""
    if left.page_number != right.page_number or not left.bbox or not right.bbox:
        return None
    l1, t1, r1, b1 = left.bbox
    l2, t2, r2, b2 = right.bbox
    overlap = max(0.0, min(r1, r2) - max(l1, l2))
    if overlap < min(r1 - l1, r2 - l2) * 0.15:
        return None
    vertical_gap = max(0.0, b1 - t2, b2 - t1)
    return vertical_gap if vertical_gap <= 90 else None


class DoclingPdfParser:
    """Docling's own OCR, layout, and table model; no page-wide VLM pass."""

    def __init__(self, converter=None):
        self.converter = converter

    def parse(self, pdf_path: Path, document_id: str, image_dir: Path) -> list[PdfElement]:
        if self.converter is None:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption
            except ImportError as exc:
                raise RuntimeError("PDF 解析需要安装 pdf 依赖：uv sync --extra pdf") from exc
            options = PdfPipelineOptions()
            options.do_ocr = True
            options.do_table_structure = True
            options.generate_picture_images = True
            self.converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        started = time.perf_counter()
        event("PDF_PARSE_STARTED", document_id=document_id)
        with Span("pdf.parse", "document", attributes={"document_id": document_id}) as span:
            doc = self.converter.convert(str(pdf_path)).document
            elements: list[PdfElement] = []
            refs: dict[str, PdfElement] = {}
            captions: list[tuple[PdfElement, str]] = []
            image_dir.mkdir(parents=True, exist_ok=True)
            for item, _level in doc.iterate_items(traverse_pictures=True):
                kind = type(item).__name__
                page = _page(item)
                ref = str(getattr(item, "self_ref", ""))
                element_id = hashlib.sha256(f"{document_id}:{ref}".encode()).hexdigest()[:16]
                label = getattr(item, "label", None)
                label = getattr(label, "value", str(label)) if label is not None else None
                if label in {"page_header", "page_footer"}:
                    continue
                if kind == "TableItem":
                    content = item.export_to_markdown(doc).strip()
                    element_type: ElementType = "table"
                elif kind == "PictureItem":
                    content = item.caption_text(doc).strip()
                    element_type = "image"
                    # A missing crop is unusable evidence. Tiny images and obvious branding
                    # are excluded; no claim is made about visual contents.
                    if any(word in content.lower() for word in ("logo", "徽标", "装饰", "页脚")):
                        continue
                    page_size = doc.pages[page].size
                    if _is_corner_decoration(
                        content, _bbox(item), page_size.width, page_size.height
                    ):
                        continue
                    picture = item.get_image(doc)
                    if picture is None or min(picture.size) < 80:
                        continue
                    output = image_dir / f"{element_id}.png"
                    picture.save(output, format="PNG")
                elif isinstance(getattr(item, "text", None), str):
                    content = str(getattr(item, "text", "")).strip()
                    element_type = "text"
                else:
                    continue
                if not content and element_type != "image":
                    continue
                element = PdfElement(
                    document_id,
                    page,
                    element_id,
                    element_type,
                    content,
                    str(output) if element_type == "image" else None,
                    bbox=_bbox(item),
                    label=label,
                )
                elements.append(element)
                refs[ref] = element
                if element_type in {"table", "image"}:
                    for caption in getattr(item, "captions", []) or []:
                        captions.append((element, str(caption.cref)))
            pictures = [e for e in elements if e.element_type == "image" and e.bbox]
            # OCR fragments centered inside a retained picture are visual evidence,
            # not standalone prose. The picture and outside caption stay available.
            elements = [
                element
                for element in elements
                if not (
                    element.element_type == "text"
                    and element.bbox
                    and any(_inside_picture(element, picture) for picture in pictures)
                )
            ]
            retained = {e.element_id for e in elements}
            refs = {ref: e for ref, e in refs.items() if e.element_id in retained}
            for element, caption_ref in captions:
                caption = refs.get(caption_ref)
                if caption is not None:
                    _link(element, caption)
            # Link nearby text using Docling's own page geometry. This only says
            # the items occur together; it does not verify the picture's contents.
            for element in elements:
                if element.element_type not in {"image", "table"}:
                    continue
                nearby = sorted(
                    (
                        (gap, candidate)
                        for candidate in elements
                        if candidate.element_type == "text"
                        and (gap := _nearby(element, candidate)) is not None
                    ),
                    key=lambda pair: pair[0],
                )
                for _, candidate in nearby[:2]:
                    _link(element, candidate)
            # A heading ending in a colon often introduces short value lines.
            # Keep these lines together for questions such as part numbers.
            for element in elements:
                if element.element_type != "text" or not element.content.endswith(":"):
                    continue
                following = sorted(
                    (
                        (gap, candidate)
                        for candidate in elements
                        if candidate.element_type == "text"
                        and candidate.bbox
                        and element.bbox
                        and candidate.bbox[1] < element.bbox[3]
                        and (gap := _nearby(element, candidate)) is not None
                        and gap <= 45
                    ),
                    key=lambda pair: pair[0],
                )
                for _, candidate in following[:2]:
                    _link(element, candidate)
            if not elements:
                raise ValueError("PDF 没有提取到可用的文字、表格或图片")
            counts = {
                kind: sum(e.element_type == kind for e in elements)
                for kind in ("text", "table", "image")
            }
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            span.output({"counts": counts, "elapsed_ms": elapsed})
            event("PDF_PARSE_COMPLETED", document_id=document_id, counts=counts, elapsed_ms=elapsed)
            return elements


@dataclass(frozen=True)
class EvidenceBundle:
    document_id: str
    page_number: int
    element_ids: tuple[str, ...]
    text: str
    image_paths: tuple[str, ...]


class MultiModalFuser:
    def __init__(self, elements: list[PdfElement]):
        self.elements = {e.element_id: e for e in elements}
        self._position = {e.element_id: index for index, e in enumerate(elements)}
        self._table_rows: dict[str, tuple[str, str]] = {}

    @staticmethod
    def _markdown_rows(content: str) -> list[str]:
        return [line for line in content.splitlines() if line.strip().startswith("|")]

    def index_chunks(self) -> list[PolicyChunk]:
        chunks = []
        for element in self.elements.values():
            related = [
                self.elements[key] for key in element.related_element_ids if key in self.elements
            ]
            searchable = element.content
            if element.element_type == "table":
                rows = self._markdown_rows(element.content)
                if len(rows) >= 3:
                    header = "\n".join(rows[:2])
                    for row_index, row in enumerate(rows[2:], start=1):
                        row_id = f"{element.element_id}:row:{row_index}"
                        excerpt = f"{header}\n{row}"
                        self._table_rows[row_id] = (element.element_id, excerpt)
                        chunks.append(
                            PolicyChunk(
                                policy_id=row_id,
                                title=f"PDF 第 {element.page_number} 页表格行",
                                content=row,
                                source=f"user_pdf:{element.document_id}",
                            )
                        )
                    # Small tables remain useful as a whole for overview questions.
                    if len(rows) > 8:
                        header_id = f"{element.element_id}:header"
                        self._table_rows[header_id] = (element.element_id, header)
                        chunks.append(
                            PolicyChunk(
                                policy_id=header_id,
                                title=f"PDF 第 {element.page_number} 页表格表头",
                                content=header,
                                source=f"user_pdf:{element.document_id}",
                            )
                        )
                        continue
            if element.element_type == "image":
                searchable = " ".join(
                    [searchable, *(e.content for e in related if e.element_type == "text")]
                ).strip()
                if not searchable:
                    searchable = f"第 {element.page_number} 页插图"
            elif element.element_type == "text" and element.content.endswith(":"):
                searchable = " ".join(
                    [searchable, *(e.content for e in related if e.element_type == "text")]
                )
            chunks.append(
                PolicyChunk(
                    policy_id=element.element_id,
                    title=f"PDF 第 {element.page_number} 页 {element.element_type}",
                    content=searchable,
                    source=f"user_pdf:{element.document_id}",
                )
            )
        return chunks

    def bundle(self, element_ids: list[str]) -> list[EvidenceBundle]:
        result = []
        seen = set()
        for element_id in element_ids:
            row = self._table_rows.get(element_id)
            element = self.elements.get(row[0] if row else element_id)
            if element is None:
                continue
            linked_ids = {element.element_id, *element.related_element_ids}
            linked = sorted(
                (self.elements[key] for key in linked_ids if key in self.elements),
                key=lambda item: self._position[item.element_id],
            )
            linked = [e for e in linked if e.page_number == element.page_number]
            key = tuple(
                sorted(
                    [*(e.element_id for e in linked), element_id]
                    if row
                    else [e.element_id for e in linked]
                )
            )
            if key in seen:
                continue
            seen.add(key)
            text = (
                (row[1] if row else element.content)
                if element.element_type == "table"
                else "\n".join(e.content for e in linked if e.element_type != "image")
            )
            result.append(
                EvidenceBundle(
                    element.document_id,
                    element.page_number,
                    tuple(sorted(e.element_id for e in linked)),
                    text,
                    tuple(
                        e.image_path for e in linked if e.element_type == "image" and e.image_path
                    ),
                )
            )
        return result


class VisionModel(Protocol):
    def describe(self, question: str, image_path: Path) -> tuple[str, int | None, int | None]: ...


class PdfEvidenceSearch:
    def __init__(self, elements: list[PdfElement], encoder=None):
        self.fuser = MultiModalFuser(elements)
        chunks = self.fuser.index_chunks()
        self._chunks = chunks
        self._terms = {chunk.policy_id: set(tokenize(chunk.content)) for chunk in chunks}
        self._document_frequency = Counter(
            term for terms in self._terms.values() for term in terms if len(term) > 1
        )
        self._document_count = len(chunks)
        self.retriever = HybridPolicyRetriever(
            BM25PolicyRetriever(chunks),
            VectorPolicyRetriever(chunks, encoder or HashingTextEncoder()),
            candidate_limit=30,
        )

    def search(self, question: str, limit: int = 5) -> list[EvidenceBundle]:
        aliases = {"维修日期": "Repair date", "保修期": "Warranty", "设备编号": "Device"}
        expanded = (
            question
            + " "
            + " ".join(translation for phrase, translation in aliases.items() if phrase in question)
        )
        hits = self.retriever.search(expanded, limit=max(limit * 8, 60))
        requested_item = re.search(r"\b(?:item|number|no\.?|#)\s*(\d+)\b", question, re.I)
        if requested_item:
            label = re.compile(rf"(?:^|\s){re.escape(requested_item[1])}\s*[.:=]")
            existing = {hit.chunk.policy_id for hit in hits}
            hits.extend(
                RetrievalHit(chunk=chunk, score=0.0, matched_terms=())
                for chunk in self._chunks
                if chunk.policy_id not in existing and label.search(chunk.content)
            )
        identifiers = re.findall(r"\b[A-Za-z]+[-]?\d+[A-Za-z0-9-]*\b", question)
        requested_field = re.search(r"\bline\s+(\d+[a-z]?)\b", question, re.I)
        asks_for_field = bool(
            requested_field or re.search(r"\b(?:which|what)\s+line\b", question, re.I)
        )
        stopwords = {
            "what",
            "which",
            "when",
            "where",
            "does",
            "the",
            "for",
            "is",
            "are",
            "of",
            "in",
        }
        terms = list(
            dict.fromkeys(
                term for term in tokenize(expanded) if len(term) > 1 and term not in stopwords
            )
        )

        def rank(hit):
            content = hit.chunk.content.lower()
            item_score = bool(
                requested_item
                and re.search(rf"(?:^|\s){re.escape(requested_item[1])}\s*[.:=]", content)
            )
            leading_field = re.match(r"^\s*(\d+[a-z]?)\s+\S", content)
            field_score = 0
            if leading_field and asks_for_field:
                field_score = (
                    2 if requested_field and leading_field[1] == requested_field[1].lower() else 1
                )
                if requested_field and leading_field[1] != requested_field[1].lower():
                    field_score = 0
            exact_id = sum(identifier.lower() in content for identifier in identifiers)
            present = self._terms.get(hit.chunk.policy_id, set())
            lexical = sum(
                (1 + position / max(1, len(terms)))
                * math.log(
                    1
                    + (self._document_count - self._document_frequency[term] + 0.5)
                    / (self._document_frequency[term] + 0.5)
                )
                for position, term in enumerate(terms)
                if term in present
            )
            coverage = sum(term in present for term in terms) / max(1, len(present))
            return (
                field_score,
                item_score,
                coverage if field_score else 0,
                exact_id,
                lexical,
                hit.score,
            )

        hits.sort(key=rank, reverse=True)
        hits = hits[:limit]
        bundles = self.fuser.bundle([hit.chunk.policy_id for hit in hits])
        event(
            "PDF_EVIDENCE_RETRIEVED",
            pages=[b.page_number for b in bundles],
            document_ids=list(dict.fromkeys(b.document_id for b in bundles)),
        )
        return bundles


def needs_visual_evidence(question: str) -> bool:
    if re.search(r"图|照片|图片|figure|photo|image", question, re.I) and re.search(
        r"对应|关联|哪一页|第几页|图注|标题|引用|所在", question
    ):
        return False
    if re.search(r"图\s*[A-Za-z0-9]+", question):
        return True
    lowered = question.lower()
    return any(
        word in lowered
        for word in (
            "图片",
            "照片",
            "图中",
            "图里",
            "外观",
            "看出",
            "可见",
            "拍到",
            "image",
            "photo",
            "figure",
            "diagram",
            "chart",
        )
    )


def _selected_image(
    question: str, bundles: list[EvidenceBundle]
) -> tuple[EvidenceBundle, str] | None:
    candidates = list(
        {
            (path, bundle.page_number): (bundle, path)
            for bundle in bundles
            for path in bundle.image_paths
        }.values()
    )
    figure = re.search(r"(?:图|figure|fig\.?)\s*([A-Za-z0-9]+)", question, re.I)
    if figure:
        figure_label = re.compile(
            rf"(?:图|figure|fig\.?)\s*{re.escape(figure[1])}(?![A-Za-z0-9])", re.I
        )
        candidates = [
            (bundle, path) for bundle, path in candidates if figure_label.search(bundle.text)
        ]
    return candidates[0] if len(candidates) == 1 else None


def _asks_for_multiple_passages(question: str) -> bool:
    return bool(
        re.search(
            r"分别|对比|比较|总结|概述|列出|哪些|全部|各页|跨页|同时|"
            r"\b(?:compare|summari[sz]e|list all|list the|across pages|both)\b",
            question,
            re.I,
        )
    )


def answer_from_bundles(
    question: str, bundles: list[EvidenceBundle], vision: VisionModel | None = None
) -> dict:
    """Return cited evidence, with no unsupported text-model image claims."""
    if not bundles:
        return {"answer": "未检索到足够的 PDF 证据。", "evidence": [], "vision_called": False}
    visual = needs_visual_evidence(question)
    image_candidate = _selected_image(question, bundles) if visual else None
    candidates = [(bundle, path) for bundle in bundles for path in bundle.image_paths]
    if visual and (image_candidate is None or vision is None):
        reason = (
            "未解析到相关图片"
            if not candidates
            else "无法唯一确定所问图片"
            if image_candidate is None
            else "未配置可用的视觉模型"
        )
        return {
            "answer": f"图片证据不足：{reason}，无法判断图片中的内容。",
            "evidence": [image_candidate[0].__dict__] if image_candidate else [],
            "vision_called": False,
        }
    if visual:
        image_bundle, image_path = image_candidate
        started = time.perf_counter()
        event(
            "PDF_VISION_STARTED",
            document_id=image_bundle.document_id,
            page_number=image_bundle.page_number,
        )
        try:
            description, input_tokens, output_tokens = vision.describe(question, Path(image_path))
        except Exception as exc:
            event(
                "PDF_VISION_FAILED",
                error_type=type(exc).__name__,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return {
                "answer": "视觉模型调用失败，无法判断图片中的内容。",
                "evidence": [image_bundle.__dict__],
                "vision_called": True,
            }
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        event(
            "PDF_VISION_COMPLETED",
            elapsed_ms=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            document_id=image_bundle.document_id,
            page_number=image_bundle.page_number,
        )
        return {
            "answer": (
                f"用户 PDF 第 {image_bundle.page_number} 页文字：{image_bundle.text[:500]}\n"
                f"同页图片观察：{description}"
            ),
            "evidence": [image_bundle.__dict__],
            "vision_called": True,
            "vision_elapsed_ms": elapsed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    # Expose the actual passages to the user instead of inventing a model summary.
    excerpts = [b for b in bundles if b.text]
    if not excerpts:
        return {"answer": "没有足够的文字或表格证据。", "evidence": [], "vision_called": False}
    if _asks_for_multiple_passages(question):
        selected = []
        seen_pages = set()
        for bundle in excerpts:
            if bundle.page_number not in seen_pages:
                selected.append(bundle)
                seen_pages.add(bundle.page_number)
            if len(selected) == 3:
                break
        for bundle in excerpts:
            if len(selected) == 3:
                break
            if bundle not in selected:
                selected.append(bundle)
        return {
            "answer": "检索到以下原文证据：\n"
            + "\n\n".join(
                f"用户 PDF 第 {bundle.page_number} 页：{bundle.text[:900]}" for bundle in selected
            ),
            "evidence": [bundle.__dict__ for bundle in selected],
            "vision_called": False,
        }
    first = excerpts[0]
    return {
        "answer": f"用户 PDF 第 {first.page_number} 页：{first.text[:1400]}",
        "evidence": [first.__dict__],
        "vision_called": False,
    }
