from __future__ import annotations

import re

from src.integrations.layout_parser_client import LayoutParserClient
from src.integrations.pdf_parser.types import FulltextParseResult
from src.integrations.pdf_parser.extractor import PdfExtractorMixin
from src.integrations.pdf_parser.layout_parser import LayoutIntegrationMixin
from src.integrations.pdf_parser.chunker import SemanticChunkerMixin
from src.integrations.pdf_parser.cleaner import TextCleanerMixin

__all__ = ["FulltextParser", "FulltextParseResult"]


class FulltextParser(
    PdfExtractorMixin,
    LayoutIntegrationMixin,
    SemanticChunkerMixin,
    TextCleanerMixin,
):
    """PDF URL에서 텍스트를 추출하고 청크 후보를 생성한다."""

    _INLINE_BODY_STARTERS = (
        "To",
        "While",
        "We",
        "In",
        "Early",
        "This",
        "These",
        "Our",
        "For",
        "As",
        "However",
        "Specifically",
        "Unlike",
        "By",
        "After",
    )
    _KNOWN_SECTION_TITLES = {
        "abstract",
        "introduction",
        "background",
        "related work",
        "preliminaries",
        "method",
        "methods",
        "approach",
        "experiments",
        "experimental setup",
        "results",
        "discussion",
        "limitations",
        "conclusion",
        "conclusions",
        "references",
        "acknowledgements",
        "acknowledgments",
    }
    _KNOWN_UPPERCASE_TITLE_TOKENS = {
        "AI", "API", "ASR", "BERT", "BLEU", "BPE", "CNN", "CPU", "CTC", "GAN",
        "GPU", "GRU", "GPT", "LIDAR", "LLM", "LSTM", "MLP", "NLP", "NMT", "OCR",
        "PDF", "RAG", "RNN", "SOTA", "TTS", "VLM",
    }
    _NUMBERED_SECTION_PATTERN = re.compile(
        r"^(?P<prefix>(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*))(?:[.)])?\s+(?P<title>[A-Z][A-Za-z0-9 ,:/()'&\-\u2013]{1,100})$"
    )
    _LAYOUT_TEXT_TYPES = {"Title", "Section header", "Text", "Table", "Caption", "Formula", "List item", "Footnote"}
    _LAYOUT_IGNORED_TYPES = {"Page header", "Page footer"}
    _LAYOUT_ARTIFACT_TYPES = {"Table", "Picture", "Caption"}

    def __init__(self, *, timeout_seconds: int = 30, layout_parser_client: LayoutParserClient | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.layout_parser_client = layout_parser_client
