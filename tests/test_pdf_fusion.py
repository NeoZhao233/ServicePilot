import json
from pathlib import Path

from fastapi.testclient import TestClient

from servicepilot.pdf_fusion import (
    EvidenceBundle,
    MultiModalFuser,
    PdfElement,
    PdfEvidenceSearch,
    _inside_picture,
    _is_corner_decoration,
    answer_from_bundles,
    needs_visual_evidence,
)
from servicepilot.pdf_service import ResponsesPdfAnswerModel, ResponsesVisionModel, SessionPdfStore
from servicepilot.webapp import create_app

ROOT = Path(__file__).resolve().parents[1]


def elements():
    return [
        PdfElement("d", 3, "t", "text", "文字检测结论：设备主板损坏。", related_element_ids=["i"]),
        PdfElement("d", 3, "i", "image", "图1：主板检测照片", "/tmp/photo.png", ["t"]),
        PdfElement("d", 2, "tab", "table", "|状态|处理|\n|---|---|\n|故障|换货|"),
    ]


def test_fuser_keeps_table_and_image_reference():
    chunks = MultiModalFuser(elements()).index_chunks()
    assert any("|故障|换货|" in chunk.content for chunk in chunks)
    bundles = PdfEvidenceSearch(elements()).search("主板损坏")
    assert bundles[0].page_number == 3
    assert "/tmp/photo.png" in bundles[0].image_paths
    assert {"t", "i"} == set(bundles[0].element_ids)


def test_visual_answer_requires_actual_vision_model():
    bundles = MultiModalFuser(elements()).bundle(["t"])
    result = answer_from_bundles("照片里有什么损坏？", bundles)
    assert "未配置可用的视觉模型" in result["answer"]
    assert result["vision_called"] is False
    assert "主板损坏" not in result["answer"]


def test_form_field_labels_rank_above_repeated_instructions():
    form = [
        PdfElement(
            "form",
            1,
            "line1",
            "text",
            "1 Name of entity/individual. Enter business name on line 2.",
        ),
        PdfElement("form", 1, "line2", "text", "2 Business name/disregarded entity name."),
        PdfElement(
            "form", 3, "note", "text", "Enter your business name on line 2 in this section."
        ),
    ]
    search = PdfEvidenceSearch(form)
    assert search.search("What is line 1 for?")[0].text.startswith("1 Name")
    assert search.search("Which line asks for the business name?")[0].text.startswith("2 Business")
    assert needs_visual_evidence("What is shown in Figure 1?")


def test_uncaptioned_page_corner_art_is_filtered_but_center_diagram_is_kept():
    assert _is_corner_decoration("", (26, 764, 294, 684), 612, 792)
    assert _is_corner_decoration("", (503, 764, 586, 558), 612, 792)
    assert _is_corner_decoration("", (26, 109, 108, 27), 612, 792)
    assert not _is_corner_decoration("", (137, 485, 467, 266), 612, 792)
    assert not _is_corner_decoration("图1：产品照片", (26, 764, 294, 684), 612, 792)


def test_picture_ocr_fragments_are_not_treated_as_independent_prose():
    picture = PdfElement("d", 5, "image", "image", "图1", bbox=(137, 485, 467, 266))
    fragment = PdfElement("d", 5, "ocr", "text", "Duic iart", bbox=(371, 444, 383, 432))
    wide_picture = PdfElement("d", 3, "photo", "image", "照片", bbox=(80, 662, 501, 401))
    wide_fragment = PdfElement("d", 3, "wide-ocr", "text", "巴", bbox=(72, 671, 511, 391))
    caption = PdfElement("d", 5, "caption", "text", "图1：打印机", bbox=(150, 260, 300, 245))
    assert _inside_picture(fragment, picture)
    assert _inside_picture(wide_fragment, wide_picture)
    assert not _inside_picture(caption, picture)


def test_numbered_figure_legend_can_be_found_without_guessing_visual_content():
    contents = [
        PdfElement(
            "manual", page, f"noise-{page}", "text", f"The list on page {page} has printer advice."
        )
        for page in range(1, 45)
    ]
    contents.extend(
        [
            PdfElement(
                "manual",
                5,
                "legend",
                "text",
                "1. printer 8. power module",
                related_element_ids=["diagram"],
            ),
            PdfElement(
                "manual",
                5,
                "diagram",
                "image",
                "1. printer 8. power module",
                "/tmp/diagram.png",
                ["legend"],
            ),
        ]
    )
    bundles = PdfEvidenceSearch(contents).search("According to the numbered list, what is item 8?")
    assert bundles[0].page_number == 5
    assert "power module" in answer_from_bundles("What is item 8?", bundles)["answer"]
    visual = answer_from_bundles("What does the diagram show at item 8?", bundles)
    assert "证据不足" in visual["answer"]
    assert visual["evidence"][0]["page_number"] == 5


def test_multi_page_question_returns_each_cited_passage():
    bundles = [
        EvidenceBundle("d", 1, ("a",), "第一步：断开电源。", ()),
        EvidenceBundle("d", 2, ("b",), "第二步：长按十秒。", ()),
    ]
    result = answer_from_bundles("分别说明两页的步骤", bundles)
    assert "第 1 页" in result["answer"] and "第 2 页" in result["answer"]
    assert len(result["evidence"]) == 2


def test_multiple_images_need_unambiguous_reference_before_vision_call():
    bundles = [
        EvidenceBundle("d", 1, ("a",), "图A：充电接口。", ("/tmp/a.png",)),
        EvidenceBundle("d", 1, ("b",), "图B：电池仓。", ("/tmp/b.png",)),
    ]

    class Vision:
        def __init__(self):
            self.paths = []

        def describe(self, question, path):
            self.paths.append(path)
            return "可见电池仓", 4, 5

    vision = Vision()
    ambiguous = answer_from_bundles("照片里是什么？", bundles, vision)
    assert "无法唯一确定" in ambiguous["answer"]
    assert not vision.paths
    selected = answer_from_bundles("图B显示什么？", bundles, vision)
    assert selected["vision_called"] is True
    assert vision.paths == [Path("/tmp/b.png")]
    assert selected["evidence"][0]["element_ids"] == ("b",)


def test_grounded_text_answer_requires_exact_quote_and_numbers():
    bundle = EvidenceBundle("d", 2, ("price",), "A12 配件单价 120元。", ())

    class Transport:
        def __init__(self, answer, quote):
            self.answer, self.quote = answer, quote

        def post(self, path, payload):
            assert path == "/responses"
            assert payload["text"]["format"]["type"] == "json_schema"
            assert '"page_number": 2' in payload["input"]
            return {
                "output_text": json.dumps(
                    {
                        "answer": self.answer,
                        "support": [{"evidence_id": "e1", "quote": self.quote}],
                    }
                ),
                "usage": {"input_tokens": 33, "output_tokens": 12},
            }

    valid = ResponsesPdfAnswerModel(Transport("A12 单价为 120元。", "A12 配件单价 120元。"), "text")
    result = valid.answer("A12 单价多少？", [bundle])
    assert result["answer_mode"] == "grounded_model"
    assert result["evidence"][0]["page_number"] == 2
    assert result["input_tokens"] == 33

    invented = ResponsesPdfAnswerModel(
        Transport("A12 单价为 130元。", "A12 配件单价 120元。"), "text"
    )
    try:
        invented.answer("A12 单价多少？", [bundle])
    except ValueError as exc:
        assert "未引用的数字" in str(exc)
    else:
        raise AssertionError("invented amount was accepted")

    fake_quote = ResponsesPdfAnswerModel(
        Transport("A12 单价为 120元。", "A12 配件单价 130元。"), "text"
    )
    try:
        fake_quote.answer("A12 单价多少？", [bundle])
    except ValueError as exc:
        assert "引文无法核对" in str(exc)
    else:
        raise AssertionError("invented quote was accepted")


def test_text_model_failure_falls_back_to_original_evidence(tmp_path):
    class Parser:
        def parse(self, path, document_id, image_dir):
            return [PdfElement(document_id, 1, "e1", "text", "保修期十二个月。")]

    class BrokenModel:
        def answer(self, question, bundles):
            raise ValueError("invalid citation")

    store = SessionPdfStore(parser=Parser(), answer_model=BrokenModel(), root=tmp_path)
    document_id, _ = store.upload(user_id="u", session_id="s", content=b"%PDF-1.4\nfixture")
    result = store.ask(
        user_id="u", session_id="s", document_id=document_id, question="保修期多久？"
    )
    assert "保修期十二个月" in result["answer"]
    assert result["evidence"][0]["page_number"] == 1


def test_vision_adapter_sends_image_bytes_and_reports_usage(tmp_path):
    image = tmp_path / "photo.png"
    image.write_bytes(b"png-data")

    class Transport:
        def post(self, path, payload):
            assert path == "/responses"
            assert payload["input"][0]["content"][1]["type"] == "input_image"
            image_url = payload["input"][0]["content"][1]["image_url"]
            assert image_url.startswith("data:image/png;base64,")
            return {
                "output_text": "可见一块电路板，无法确认是否烧毁。",
                "usage": {"input_tokens": 12, "output_tokens": 10},
            }

    model = ResponsesVisionModel(Transport(), "vision-capable-model")
    sample = elements()
    sample[1].image_path = str(image)
    result = answer_from_bundles("照片里是否烧毁？", MultiModalFuser(sample).bundle(["t"]), model)
    assert result["vision_called"] is True
    assert "文字" in result["answer"] and "图片观察" in result["answer"]
    assert model.describe("照片里是否烧毁？", image)[1:] == (12, 10)


def test_upload_is_scoped_to_authenticated_user_and_session(monkeypatch):
    monkeypatch.setenv("SERVICEPILOT_AUTH_TOKENS", '{"token-a":"USER-001","token-b":"USER-002"}')
    client = TestClient(create_app(data_dir=ROOT / "data"))
    owner = {"Authorization": "Bearer token-a"}
    other = {"Authorization": "Bearer token-b"}
    session_id = client.post(
        "/api/chat", headers=owner, json={"user_id": "USER-001", "message": "ORD-1003 物流"}
    ).json()["session_id"]

    class FakeParser:
        def parse(self, pdf_path, document_id, image_dir):
            return [PdfElement(document_id, 1, "e1", "text", "保修期十二个月")]

    client.app.state.pdf_store.parser = FakeParser()
    payload = {"user_id": "USER-001", "session_id": session_id}
    response = client.post(
        "/api/pdf/upload",
        headers=owner,
        data=payload,
        files={"file": ("receipt.pdf", b"%PDF-1.4\nfixture", "application/pdf")},
    )
    assert response.status_code == 200, response.text
    document_id = response.json()["document_id"]
    question = {**payload, "document_id": document_id, "question": "保修期多久"}
    assert "十二个月" in client.post("/api/pdf/ask", headers=owner, json=question).json()["answer"]
    chat = client.post(
        "/api/chat",
        headers=owner,
        json={
            "user_id": "USER-001",
            "session_id": session_id,
            "document_id": document_id,
            "message": "保修期多久",
        },
    ).json()
    assert chat["intent"] == "document_qa"
    assert chat["citations"] == [f"{document_id}:p1"]
    assert "十二个月" in chat["message"]
    assert client.post("/api/pdf/ask", headers=other, json=question).status_code == 403
    denied = client.post(
        "/api/pdf/upload",
        headers=other,
        data=payload,
        files={"file": ("receipt.pdf", b"%PDF-1.4\nfixture", "application/pdf")},
    )
    assert denied.status_code == 403


def test_pdf_documents_survive_restart_and_images_require_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("SERVICEPILOT_AUTH_TOKENS", '{"token-a":"USER-001","token-b":"USER-002"}')
    database = tmp_path / "sessions.sqlite3"
    pdf_dir = tmp_path / "pdfs"
    owner = {"Authorization": "Bearer token-a"}
    other = {"Authorization": "Bearer token-b"}

    class FakeParser:
        def parse(self, pdf_path, document_id, image_dir):
            image_dir.mkdir()
            image_path = image_dir / "photo.png"
            image_path.write_bytes(b"picture-bytes")
            return [
                PdfElement(
                    document_id,
                    2,
                    "caption",
                    "text",
                    "图1：设备照片",
                    related_element_ids=["photo"],
                ),
                PdfElement(
                    document_id, 2, "photo", "image", "图1：设备照片", str(image_path), ["caption"]
                ),
            ]

    app = create_app(data_dir=ROOT / "data", session_db=database, pdf_dir=pdf_dir)
    app.state.pdf_store.parser = FakeParser()
    client = TestClient(app)
    uploaded = client.post(
        "/api/pdf/upload",
        headers=owner,
        data={"user_id": "USER-001"},
        files={"file": ("daily.pdf", b"%PDF-1.4\nfixture", "application/pdf")},
    )
    assert uploaded.status_code == 200, uploaded.text
    document_id = uploaded.json()["document_id"]
    session_id = uploaded.json()["session_id"]
    assert (pdf_dir / document_id / "metadata.json").is_file()

    restarted = TestClient(create_app(data_dir=ROOT / "data", session_db=database, pdf_dir=pdf_dir))
    question = {
        "user_id": "USER-001",
        "session_id": session_id,
        "document_id": document_id,
        "question": "照片显示什么？",
    }
    answered = restarted.post("/api/pdf/ask", headers=owner, json=question)
    assert answered.status_code == 200
    evidence = answered.json()["evidence"][0]
    assert "image_paths" not in evidence
    image_url = evidence["images"][0]
    assert restarted.get(image_url, headers=owner).content == b"picture-bytes"
    assert restarted.get(image_url, headers=other).status_code == 403
    listing = restarted.get(
        "/api/pdf/documents",
        headers=owner,
        params={"user_id": "USER-001", "session_id": session_id},
    ).json()
    assert listing["documents"][0]["filename"] == "daily.pdf"
    assert (
        restarted.delete(
            f"/api/pdf/{document_id}",
            headers=owner,
            params={"user_id": "USER-001", "session_id": session_id},
        ).status_code
        == 200
    )
    assert restarted.post("/api/pdf/ask", headers=owner, json=question).status_code == 403
