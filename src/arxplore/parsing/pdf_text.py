import io
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from pypdf import PdfReader

from arxplore.parsing.cleaner import collapse_whitespace, find_inline_heading_split, looks_like_toc_line, normalize_text
from arxplore.parsing.models import PYPDF_SOURCE, FulltextParseResult, Section
from arxplore.parsing.quality import fulltext_quality_metrics
from arxplore.parsing.sections import FULL_TEXT_TITLE, extract_sections, normalize_section_heading

logger = logging.getLogger(__name__)

_PAGE_NUMBER_LINE = re.compile(r"\d+")
_DATE_LINE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ARXIV_STAMP_LINE = re.compile(r"^arXiv:\d{4}\.\d{4,5}v?\d*")
_EQUAL_CONTRIBUTION_LINE = re.compile(r"^\*?Equal contribution", re.IGNORECASE)
_UNICODE_GLYPH_LINE = re.compile(r"^(?:/uni\d{8})+$")
_UNICODE_GLYPH = re.compile(r"/uni\d{8}")
_SENTENCE_END = (".", "?", "!", ":", ";")
_DIGIT_RUN = re.compile(r"\d+")
_LETTER = re.compile(r"[^\W\d_]")
RUNNING_LINE_EDGE_LINES = 2
RUNNING_LINE_MIN_PAGES = 3
RUNNING_LINE_MIN_PAGE_SHARE = 0.4
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z][A-Za-z-]{2,})")
_SPLIT_MARK = "\x00"
UNSPLIT_BRAND_NAMES = (
    ("ar", "Xiv"),
    ("Li", "DAR"),
    ("Git", "Hub"),
    ("Hugging", "Face"),
    ("Model", "Scope"),
    ("Deep", "Seek"),
)
_UNSPLIT_BRANDS = tuple(
    (re.compile(rf"\b{head}{_SPLIT_MARK}{tail}\b"), f"{head}{tail}") for head, tail in UNSPLIT_BRAND_NAMES
)


def running_line_key(line: str) -> str:
    return _DIGIT_RUN.sub("#", collapse_whitespace(line))


def _page_lines(text: str) -> list[str]:
    return [line.strip() for line in text.replace("\x00", " ").splitlines()]


def _edge_line_indexes(lines: Sequence[str]) -> set[int]:
    filled = [index for index, line in enumerate(lines) if line]
    return {*filled[:RUNNING_LINE_EDGE_LINES], *filled[-RUNNING_LINE_EDGE_LINES:]}


def find_running_lines(page_texts: Sequence[str]) -> frozenset[str]:
    pages_with_line: Counter[str] = Counter()
    for text in page_texts:
        lines = _page_lines(text)
        keys = {running_line_key(lines[index]) for index in _edge_line_indexes(lines)}
        pages_with_line.update(key for key in keys if _LETTER.search(key))
    threshold = max(RUNNING_LINE_MIN_PAGES, math.ceil(RUNNING_LINE_MIN_PAGE_SHARE * len(page_texts)))
    return frozenset(key for key, count in pages_with_line.items() if count >= threshold)


def _is_dropped_page_line(line: str) -> bool:
    return bool(
        _PAGE_NUMBER_LINE.fullmatch(line)
        or _DATE_LINE.fullmatch(line)
        or _ARXIV_STAMP_LINE.match(line)
        or _EQUAL_CONTRIBUTION_LINE.match(line)
        or _UNICODE_GLYPH_LINE.match(line)
        or len(_UNICODE_GLYPH.findall(line)) >= 3
        or looks_like_toc_line(line)
    )


def split_inline_heading_line(line: str) -> list[str]:
    line = line.strip()
    if not line:
        return [""]
    split = find_inline_heading_split(line)
    if split is not None:
        return [split[0], split[1]]
    return [line]


def should_merge_lines(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if previous.endswith("-"):
        return True
    if previous.endswith(_SENTENCE_END):
        return False
    if normalize_section_heading(current) is not None:
        return False
    if current[0].islower() or current[0] in "([\"'":
        return True
    if previous[-1] in ",/":
        return True
    return len(previous) > 40 and current[:1].isupper() and len(current.split()) > 3


def merge_lines(previous: str, current: str) -> str:
    if previous.endswith("-"):
        return previous[:-1] + current.lstrip()
    return previous.rstrip() + " " + current.lstrip()


def normalize_page_text(text: str, running_lines: frozenset[str] = frozenset()) -> str:
    lines = _page_lines(text)
    edges = _edge_line_indexes(lines) if running_lines else set()
    filtered_lines: list[str] = []
    for index, line in enumerate(lines):
        if not line:
            filtered_lines.append("")
            continue
        if index in edges and running_line_key(line) in running_lines:
            continue
        if _is_dropped_page_line(line):
            continue
        filtered_lines.extend(split_inline_heading_line(line))
    stitched_lines: list[str] = []
    for line in filtered_lines:
        if not line:
            if stitched_lines and stitched_lines[-1] != "":
                stitched_lines.append("")
            continue
        if stitched_lines:
            previous = stitched_lines[-1]
            if previous and should_merge_lines(previous, line):
                stitched_lines[-1] = merge_lines(previous, line)
                continue
        stitched_lines.append(line)
    return "\n".join(stitched_lines).strip()


def split_camel_case(text: str) -> str:
    marked = _CAMEL_CASE_BOUNDARY.sub(_SPLIT_MARK, text.replace(_SPLIT_MARK, " "))
    for pattern, brand in _UNSPLIT_BRANDS:
        marked = pattern.sub(brand, marked)
    return marked.replace(_SPLIT_MARK, " ")


def text_from_pages(page_texts: Iterable[str]) -> str:
    raw_pages = list(page_texts)
    running_lines = find_running_lines(raw_pages)
    pages = [normalized for page in raw_pages if (normalized := normalize_page_text(page, running_lines))]
    return split_camel_case(normalize_text("\n\n".join(pages)))


def extract_page_texts(content: bytes) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages]


def extract_pdf_text(content: bytes) -> str:
    try:
        return text_from_pages(extract_page_texts(content))
    except Exception as exc:
        logger.warning("pypdf text extraction failed: %s: %s", type(exc).__name__, exc)
        return ""


def parse_pdf_text(text: str) -> FulltextParseResult | None:
    if not text:
        return None
    sections: list[Section] = extract_sections(text) or [{"title": FULL_TEXT_TITLE, "text": text}]
    return FulltextParseResult(
        text=text,
        sections=sections,
        source=PYPDF_SOURCE,
        quality_metrics=fulltext_quality_metrics(text=text, sections=sections, source=PYPDF_SOURCE),
    )


def parse_pdf_bytes(content: bytes) -> FulltextParseResult | None:
    return parse_pdf_text(extract_pdf_text(content))
