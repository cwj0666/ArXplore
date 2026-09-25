from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.integrations.fulltext_parser import FulltextParser
from src.integrations.pdf_parser.section_roles import is_references_section_title

PDF_URL = "https://arxiv.org/pdf/2401.00001"
REQUESTS_GET = "src.integrations.pdf_parser.extractor.requests.get"

BODY_PARAGRAPH = (
    "We study how preference data shapes the alignment of large language models. "
    "Our approach builds on prior work and extends it with a simple objective that is easy to optimize. "
) * 12

PAPER_TEXT = "\n".join(
    [
        "Aligning Language Models with Preferences",
        "",
        "Abstract",
        BODY_PARAGRAPH,
        "",
        "1 Introduction",
        BODY_PARAGRAPH,
        "",
        "2 Direct Preference Optimization",
        BODY_PARAGRAPH,
        "",
        "3.2 Reference Model",
        BODY_PARAGRAPH,
        "",
        "4 Experiments",
        BODY_PARAGRAPH * 2,
        "",
        "7 References",
        "[1] A. Smith, B. Jones. Learning to align. In Proceedings of NeurIPS, 2023.",
        "[2] C. Lee, D. Kim. Preference models. arXiv preprint arXiv:2301.00001, 2023.",
    ]
)


def _build_text_pdf(lines: list[str]) -> bytes:
    def escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    operators = ["BT", "/F1 10 Tf", "14 TL", "50 780 Td"]
    operators.extend(f"({escape(line)}) Tj T*" for line in lines)
    operators.append("ET")
    stream = "\n".join(operators).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_offset = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        output += f"{offset:010d} 00000 n \n".encode()
    output += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(output)


def _fake_response(content: bytes) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.raise_for_status.return_value = None
    return response


def _fake_layout_client(*, configured: bool, segments=None, error: Exception | None = None) -> MagicMock:
    client = MagicMock()
    client.is_configured.return_value = configured
    if error is not None:
        client.analyze_pdf_bytes.side_effect = error
    else:
        client.analyze_pdf_bytes.return_value = segments or []
    return client


def _segment(text: str, segment_type: str, *, top: float, page: int = 1) -> dict:
    return {
        "left": 50.0,
        "top": top,
        "width": 500.0,
        "height": 12.0,
        "page_number": page,
        "page_width": 612.0,
        "page_height": 792.0,
        "text": text,
        "type": segment_type,
    }


LAYOUT_SEGMENTS = [
    _segment("Aligning Language Models with Preferences", "Title", top=40.0),
    _segment("Page 1 running header", "Page header", top=10.0),
    _segment("1 Introduction", "Section header", top=100.0),
    _segment(BODY_PARAGRAPH, "Text", top=120.0),
    _segment("2 Direct Preference Optimization", "Section header", top=300.0),
    _segment(BODY_PARAGRAPH, "Text", top=320.0),
    _segment("Table 1: Win rates against the reference policy.", "Caption", top=500.0),
    _segment("Model Win rate\nDPO 61.2\nPPO 57.4", "Table", top=520.0),
    _segment("7 References", "Section header", top=60.0, page=2),
    _segment("[1] A. Smith. Learning to align. In Proceedings of NeurIPS, 2023.", "Text", top=80.0, page=2),
]


def test_layout_path_returns_layout_pdf_source():
    parser = FulltextParser(layout_parser_client=_fake_layout_client(configured=True, segments=LAYOUT_SEGMENTS))

    with patch(REQUESTS_GET, return_value=_fake_response(b"%PDF-fake")):
        result = parser.parse_from_pdf_url(PDF_URL, fallback_text="abstract only")

    assert result.source == "layout_pdf"
    assert result.quality_metrics["parse_source"] == "layout_pdf"
    assert result.parser_metadata["segment_count"] == len(LAYOUT_SEGMENTS)
    assert result.parser_metadata["segment_type_counts"]["Section header"] == 3
    assert "Page 1 running header" not in result.text
    titles = [section["title"] for section in result.sections]
    assert "2 Direct Preference Optimization" in titles
    assert "7 References" in titles
    assert result.artifacts["tables"][0]["caption"].startswith("Table 1")


def test_pypdf_path_returns_pdf_source_when_layout_unavailable():
    pdf_bytes = _build_text_pdf(
        [
            "1 Introduction",
            "We propose a method for the alignment of large language models.",
            "2 Direct Preference Optimization",
            "We optimize the policy directly from pairwise preference data.",
        ]
    )
    parser = FulltextParser(layout_parser_client=_fake_layout_client(configured=False))

    with patch(REQUESTS_GET, return_value=_fake_response(pdf_bytes)):
        result = parser.parse_from_pdf_url(PDF_URL, fallback_text="abstract only")

    assert result.source == "pdf"
    assert result.quality_metrics["fallback_used"] is False
    assert "pairwise preference data" in result.text
    assert [section["title"] for section in result.sections] == ["1 Introduction", "2 Direct Preference Optimization"]


def test_pypdf_path_used_when_layout_parser_errors():
    pdf_bytes = _build_text_pdf(["1 Introduction", "We propose a method for the alignment of language models."])
    client = _fake_layout_client(configured=True, error=requests.ConnectionError("layout parser down"))
    parser = FulltextParser(layout_parser_client=client)

    with patch(REQUESTS_GET, return_value=_fake_response(pdf_bytes)):
        result = parser.parse_from_pdf_url(PDF_URL)

    assert result.source == "pdf"


def test_download_failure_returns_fallback_abstract():
    client = _fake_layout_client(configured=True, segments=LAYOUT_SEGMENTS)
    parser = FulltextParser(layout_parser_client=client)

    with patch(REQUESTS_GET, side_effect=requests.RequestException("network down")):
        result = parser.parse_from_pdf_url(PDF_URL, fallback_text="We align models with preferences.")

    assert result.source == "fallback_abstract"
    assert result.quality_metrics["fallback_used"] is True
    assert result.sections == [{"title": "Abstract", "text": "We align models with preferences."}]
    client.analyze_pdf_bytes.assert_not_called()


def test_fallback_abstract_can_be_chunked():
    parser = FulltextParser(layout_parser_client=_fake_layout_client(configured=False))
    with patch(REQUESTS_GET, side_effect=requests.RequestException("network down")):
        result = parser.parse_from_pdf_url(PDF_URL, fallback_text=BODY_PARAGRAPH)

    chunks = parser.build_chunks(result.text, sections=result.sections)

    assert chunks
    assert all(chunk["section_title"] == "Abstract" for chunk in chunks)


def test_extract_sections_and_build_chunks_on_realistic_text():
    sections = FulltextParser._extract_sections(PAPER_TEXT)
    titles = [section["title"] for section in sections]
    assert titles == [
        "Front Matter",
        "Abstract",
        "1 Introduction",
        "2 Direct Preference Optimization",
        "3.2 Reference Model",
        "4 Experiments",
        "7 References",
    ]

    max_chars = 600
    chunks = FulltextParser.build_chunks(PAPER_TEXT, sections=sections, max_chars=max_chars, overlap_chars=100)
    assert len(chunks) > len(sections)
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    # _adjust_chunk_end may extend a chunk by up to 220 chars to reach a sentence boundary.
    assert all(len(chunk["chunk_text"]) <= max_chars + 220 for chunk in chunks)

    roles_by_title: dict[str, set[str]] = {}
    for chunk in chunks:
        roles_by_title.setdefault(chunk["section_title"], set()).add(chunk["metadata"]["content_role"])
    assert roles_by_title["7 References"] == {"references"}
    assert roles_by_title["2 Direct Preference Optimization"] == {"body"}
    assert roles_by_title["3.2 Reference Model"] == {"body"}

    summary = FulltextParser.summarize_chunks(chunks)
    assert summary["chunk_count"] == len(chunks)
    assert summary["reference_chunk_count"] == len([c for c in chunks if c["section_title"] == "7 References"])


def test_build_chunks_without_sections():
    chunks = FulltextParser.build_chunks(PAPER_TEXT)
    assert chunks
    assert chunks[0]["section_title"] == "Full Text"


@pytest.mark.parametrize(
    ("title", "expected_role"),
    [
        ("7 References", "references"),
        ("7. References", "references"),
        ("References", "references"),
        ("REFERENCES", "references"),
        ("VII. References", "references"),
        ("Bibliography", "references"),
        ("Works Cited", "references"),
        ("Direct Preference Optimization", "body"),
        ("2 Direct Preference Optimization", "body"),
        ("3.2 Reference Model", "body"),
        ("Reference Model", "body"),
        ("Cross-Reference Analysis", "body"),
        ("Appendix A References", "appendix"),
        ("Front Matter", "front_matter"),
    ],
)
def test_infer_content_role_uses_word_boundaries(title: str, expected_role: str):
    assert FulltextParser._infer_content_role(title) == expected_role


def test_numbered_heading_helper_restored():
    assert FulltextParser._looks_like_numbered_heading("Direct Preference Optimization")
    assert not FulltextParser._looks_like_numbered_heading("results are shown in the table below for all")
    assert FulltextParser._normalize_section_heading("3.2 Reference Model") == "3.2 Reference Model"


def test_is_references_section_title_ignores_letter_prefixes():
    assert is_references_section_title("7 References")
    assert not is_references_section_title("A References")
    assert not is_references_section_title("Mix References")
