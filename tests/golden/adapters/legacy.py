import io
from importlib.metadata import version
from typing import Any

from pypdf import PdfReader

from src.integrations.fulltext_parser import FulltextParser
from src.integrations.pdf_parser.types import FulltextParseResult
from tests.golden.adapters import ParsedPdf

PYPDF_SOURCE = "pdf"


def _as_parsed(result: FulltextParseResult) -> ParsedPdf:
    return {
        "text": result.text,
        "sections": [dict(section) for section in result.sections],
        "source": result.source,
        "quality_metrics": dict(result.quality_metrics),
        "parser_metadata": dict(result.parser_metadata),
        "artifacts": dict(result.artifacts),
    }


class LegacyParserAdapter:
    def __init__(self) -> None:
        self._parser = FulltextParser()

    def describe(self) -> dict[str, str]:
        return {"implementation": "legacy", "pypdf": version("pypdf")}

    def extract_page_texts(self, content: bytes) -> list[str]:
        return [page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages]

    def parse_page_texts(self, page_texts: list[str]) -> ParsedPdf:
        parser = self._parser
        pages = [parser._normalize_extracted_page_text(text) for text in page_texts]
        return self._from_extracted_text(parser._normalize_text("\n\n".join(page for page in pages if page)))

    def parse_pdf_bytes(self, content: bytes) -> ParsedPdf:
        return self._from_extracted_text(self._parser._extract_pdf_text(content))

    def build_chunks(self, text: str, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._parser.build_chunks(text, sections=sections)

    def summarize_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        return self._parser.summarize_chunks(chunks)

    def _from_extracted_text(self, text: str) -> ParsedPdf:
        parser = self._parser
        if not text:
            return _as_parsed(parser._build_fallback_result(""))
        sections = parser._extract_sections(text) or [{"title": "Full Text", "text": text}]
        metrics = parser._build_fulltext_quality_metrics(text=text, sections=sections, source=PYPDF_SOURCE)
        return _as_parsed(
            FulltextParseResult(text=text, sections=sections, source=PYPDF_SOURCE, quality_metrics=metrics)
        )


def parser_adapter() -> LegacyParserAdapter:
    return LegacyParserAdapter()
