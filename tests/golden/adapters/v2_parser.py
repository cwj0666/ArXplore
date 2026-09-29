from importlib.metadata import version
from typing import Any

from arxplore.parsing import chunker, pdf_text
from arxplore.parsing.fulltext import fallback_result
from arxplore.parsing.layout import parse_layout_segments
from arxplore.parsing.layout_client import normalize_segments
from arxplore.parsing.models import FulltextParseResult
from tests.golden.adapters import ParsedPdf


def _as_parsed(result: FulltextParseResult) -> ParsedPdf:
    return {
        "text": result.text,
        "sections": [dict(section) for section in result.sections],
        "source": result.source,
        "quality_metrics": dict(result.quality_metrics),
        "parser_metadata": dict(result.parser_metadata),
        "artifacts": dict(result.artifacts),
    }


def _from_extracted_text(text: str) -> ParsedPdf:
    return _as_parsed(pdf_text.parse_pdf_text(text) or fallback_result(""))


class V2ParserAdapter:
    def describe(self) -> dict[str, str]:
        return {"implementation": "v2", "pypdf": version("pypdf")}

    def extract_page_texts(self, content: bytes) -> list[str]:
        return pdf_text.extract_page_texts(content)

    def parse_page_texts(self, page_texts: list[str]) -> ParsedPdf:
        return _from_extracted_text(pdf_text.text_from_pages(page_texts))

    def parse_pdf_bytes(self, content: bytes) -> ParsedPdf:
        return _from_extracted_text(pdf_text.extract_pdf_text(content))

    def build_chunks(self, text: str, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(chunk) for chunk in chunker.build_chunks(text, sections)]

    def summarize_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        return dict(chunker.summarize_chunks(chunks))

    def parse_layout_response(self, payload: list[Any]) -> ParsedPdf | None:
        result = parse_layout_segments(normalize_segments(payload))
        return None if result is None else _as_parsed(result)


def parser_adapter() -> V2ParserAdapter:
    return V2ParserAdapter()
