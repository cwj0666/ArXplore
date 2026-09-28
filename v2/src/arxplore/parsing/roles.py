import re

from arxplore.parsing.cleaner import collapse_whitespace, looks_like_toc_line

FRONT_MATTER_ROLE = "front_matter"
REFERENCES_ROLE = "references"
APPENDIX_ROLE = "appendix"
TOC_ROLE = "toc"
BODY_ROLE = "body"
FIGURE_CAPTION_ROLE = "figure_caption"
TABLE_LIKE_ROLE = "table_like"

_SECTION_NUMBER_PREFIX = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?|[IVXLC]+[.)]?|(?i:[ivxlc]+)[.)])\s+")
_REFERENCES_TITLE = re.compile(
    r"(?:references?|bibliography|works\s+cited|literature\s+cited)(?:\s+and\s+notes)?",
    re.IGNORECASE,
)

_BODY_SECTION_KEYWORDS = (
    "introduction",
    "method",
    "approach",
    "experiment",
    "result",
    "discussion",
    "conclusion",
    "abstract",
    "related work",
)
_BODY_PARAGRAPH_OPENING = re.compile(
    r"^(?:As shown|Beyond|In this|In our|We |To address|To evaluate|To further|Notably|While|Since|Early"
    r"|Figure \d+ (?:shows|illustrates|visualizes)|Table \d+ (?:presents|shows|compares|reports|summarizes))\b",
    re.IGNORECASE,
)
_ALPHABETIC_WORD = re.compile(r"[A-Za-z]+")
_NUMBER_TOKEN = re.compile(r"\b\d+(?:\.\d+)?\b")
_CAPITALIZED_OPENING = re.compile(r"^[A-Z][a-z]")
_LONG_ALPHABETIC_WORD = re.compile(r"[A-Za-z]{3,}")
_CITATION_MARKER = re.compile(r"\[\d+\]")
_FIGURE_DESCRIPTION_OPENING = re.compile(
    r"^(?:Figure|Table)\s+\d+\s+(?:presents|shows|compares|illustrates|visualizes|reports|summarizes|demonstrates|plots)\b",
    re.IGNORECASE,
)
_EXPLICIT_CAPTION = re.compile(r"^(?:Figure|Table)\s+\d+[:.]")
_PANEL_CAPTION = re.compile(r"^\d+\.\s*[A-Z][^.]{0,140}\([a-z]\)")
_MULTI_PANEL_OPENINGS = ("a, ", "b, ", "c, ", "d, ")
_MULTI_PANEL_CONTINUATION = re.compile(r"\b[bcd],\s+[A-Z]")

_REFERENCE_ENTRY_OPENING = re.compile(r"^(?:\[\d+\]|\d+\.)\s+[A-Z]")
_NUMBERED_CITATION_OPENING = re.compile(r"^\d+\s+\[\d+\]\s+[A-Z]")
_OPTIONAL_NUMBER_CITATION_OPENING = re.compile(r"^(?:\d+\s*,?\s*)?\[\d+\]\s+[A-Z]")
_VENUE_NAME = re.compile(r"\b(?:Conference|Proceedings|arXiv|Association for Computational Linguistics)\b")
_DIGIT_OPENING = re.compile(r"^\d")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_AUTHOR_LIST_OPENING = re.compile(r"^[A-Z][A-Za-z'`.-]+(?:,\s+[A-Z][A-Za-z'`.-]+){1,}")
_PAGE_NUMBER_BEFORE_CITATION = re.compile(r"(?:^|\s)\d+\s+\[\d+\]")

_HTML_TABLE_TAGS = ("<td", "</td", "<tr", "</tr", "<th", "</th")
_CHECK_MARK = re.compile(r"[✓✔✗✘]")
_PERCENTAGE = re.compile(r"\b\d+(?:\.\d+)?%")
_SHORT_METRIC_CODE = re.compile(r"\b(?:MCQ|OE|A&M|IoU|FVD|PSNR|SSIM|M|A|S|T|R|F1|Top-1|Top-5)\b", re.IGNORECASE)
_NUMERIC_ONLY_LINE = re.compile(r"[\d\s.,:;()%+\-=/]+")
_MATRIX_TOKEN = re.compile(r"\b[a-z]+-\d+-(?:combined|pre|post)\b", re.IGNORECASE)
_NUMERIC_OPENING = re.compile(r"^[\d.,;:()%-]")
_TABLE_OR_FIGURE_MENTION = re.compile(r"\b(?:Table|Figure)\s+\d+\b")
_CAPITALIZED_WORD = re.compile(r"\b[A-Z][A-Za-z-]{2,}\b")


def is_references_section_title(title: str) -> bool:
    normalized = collapse_whitespace(str(title or ""))
    normalized = _SECTION_NUMBER_PREFIX.sub("", normalized, count=1)
    normalized = normalized.strip(" .:")
    return bool(_REFERENCES_TITLE.fullmatch(normalized))


def section_role(section_title: str) -> str:
    lowered = section_title.lower()
    if lowered == "front matter":
        return FRONT_MATTER_ROLE
    if is_references_section_title(section_title):
        return REFERENCES_ROLE
    if "appendix" in lowered:
        return APPENDIX_ROLE
    if "table of contents" in lowered:
        return TOC_ROLE
    return BODY_ROLE


def starts_like_body_paragraph(first_line: str, compact: str) -> bool:
    normalized_first_line = collapse_whitespace(first_line)
    if not normalized_first_line:
        return False
    if _BODY_PARAGRAPH_OPENING.match(normalized_first_line):
        return True
    word_count = len(_ALPHABETIC_WORD.findall(normalized_first_line))
    numeric_cells = len(_NUMBER_TOKEN.findall(normalized_first_line))
    if word_count >= 12 and numeric_cells <= 3 and _CAPITALIZED_OPENING.match(normalized_first_line):
        return True
    compact_prefix = compact[:220]
    return compact_prefix.count(". ") >= 1 and word_count >= 10 and numeric_cells <= 4


def looks_like_body_chunk(section_title: str, compact: str) -> bool:
    if not compact:
        return False
    lowered_title = section_title.lower()
    if any(keyword in lowered_title for keyword in _BODY_SECTION_KEYWORDS) and starts_like_body_paragraph(
        compact[:180], compact
    ):
        return True
    sentence_breaks = compact.count(". ") + compact.count("? ") + compact.count("! ")
    alphabetic_tokens = _LONG_ALPHABETIC_WORD.findall(compact[:260])
    citation_markers = len(_CITATION_MARKER.findall(compact[:260]))
    return (
        sentence_breaks >= 1
        and len(alphabetic_tokens) >= 12
        and citation_markers <= 8
        and bool(_CAPITALIZED_OPENING.match(compact))
    )


def looks_like_reference_chunk(compact: str) -> bool:
    compact_prefix = compact[:260]
    if not compact_prefix:
        return False
    if looks_like_body_chunk("Full Text", compact_prefix):
        return False
    if _REFERENCE_ENTRY_OPENING.match(compact_prefix):
        return True
    if _NUMBERED_CITATION_OPENING.match(compact_prefix):
        return True
    if _OPTIONAL_NUMBER_CITATION_OPENING.match(compact_prefix):
        return True
    if compact_prefix.startswith("In ") and _VENUE_NAME.search(compact_prefix):
        return True
    if (
        _DIGIT_OPENING.match(compact_prefix)
        and _CITATION_MARKER.search(compact_prefix)
        and _YEAR.search(compact_prefix)
    ):
        return True
    if compact_prefix.startswith(("doi:", "https://", "http://")) and _CITATION_MARKER.search(compact_prefix):
        return True
    if _AUTHOR_LIST_OPENING.match(compact_prefix) and _YEAR.search(compact_prefix):
        return True
    return len(_PAGE_NUMBER_BEFORE_CITATION.findall(compact[:220])) >= 1


def looks_like_figure_caption_chunk(compact: str) -> bool:
    if not compact:
        return False
    explicit_caption = _EXPLICIT_CAPTION.match(compact)
    panel_caption = _PANEL_CAPTION.match(compact)
    multi_panel_caption = compact.startswith(_MULTI_PANEL_OPENINGS) and _MULTI_PANEL_CONTINUATION.search(compact[:220])
    if not (explicit_caption or panel_caption or multi_panel_caption):
        return False
    return not _FIGURE_DESCRIPTION_OPENING.match(compact)


def looks_like_table_like_chunk(raw_text: str, compact: str) -> bool:
    if "/uni" in raw_text:
        return True
    lowered = raw_text.lower()
    if any(tag in lowered for tag in _HTML_TABLE_TAGS):
        return True
    digits = sum(character.isdigit() for character in compact)
    numeric_cells = len(_NUMBER_TOKEN.findall(compact))
    symbol_cells = len(_CHECK_MARK.findall(compact))
    percentage_cells = len(_PERCENTAGE.findall(compact))
    short_code_cells = len(_SHORT_METRIC_CODE.findall(compact))
    line_break_count = raw_text.count("\n")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    first_line = lines[0] if lines else compact[:160]
    numeric_heavy_lines = sum(
        1 for line in lines if len(_NUMBER_TOKEN.findall(line)) >= 2 or bool(_NUMERIC_ONLY_LINE.fullmatch(line))
    )
    repeated_matrix_tokens = len(_MATRIX_TOKEN.findall(compact))
    explicit_tabular_opening = bool(
        _NUMERIC_OPENING.match(compact)
        or _EXPLICIT_CAPTION.match(compact)
        or _PANEL_CAPTION.match(compact)
        or repeated_matrix_tokens >= 6
    )
    compact_row_like_block = (
        numeric_cells >= 10
        and (symbol_cells >= 2 or percentage_cells >= 2 or short_code_cells >= 3)
        and len(_CAPITALIZED_WORD.findall(compact)) >= 4
    )
    if starts_like_body_paragraph(first_line, compact) and not explicit_tabular_opening:
        return False
    if compact and digits / len(compact) > 0.28 and line_break_count >= 5:
        return True
    if _NUMERIC_OPENING.match(compact) and numeric_cells >= 10 and line_break_count >= 2:
        return True
    if _TABLE_OR_FIGURE_MENTION.search(compact[:160]) and numeric_heavy_lines >= 3 and line_break_count >= 2:
        return True
    if repeated_matrix_tokens >= 6:
        return True
    if (
        len(lines) >= 8
        and numeric_heavy_lines >= max(6, int(len(lines) * 0.7))
        and digits / max(1, len(compact)) > 0.08
    ):
        return True
    return compact_row_like_block


def chunk_role(section_title: str, chunk_text: str) -> str:
    role = section_role(section_title)
    compact = collapse_whitespace(chunk_text)
    if role == FRONT_MATTER_ROLE:
        if looks_like_table_like_chunk(chunk_text, compact):
            return TABLE_LIKE_ROLE
        return role
    if role != BODY_ROLE:
        return role
    if looks_like_toc_line(compact):
        return TOC_ROLE
    if looks_like_figure_caption_chunk(compact):
        return FIGURE_CAPTION_ROLE
    if _FIGURE_DESCRIPTION_OPENING.match(compact):
        return BODY_ROLE
    if looks_like_body_chunk(section_title, compact):
        return BODY_ROLE
    if looks_like_reference_chunk(compact):
        return REFERENCES_ROLE
    if looks_like_table_like_chunk(chunk_text, compact):
        return TABLE_LIKE_ROLE
    return BODY_ROLE


def refined_chunk_role(section_title: str, chunk_text: str, current_role: str) -> str:
    compact = collapse_whitespace(chunk_text)
    body_like = looks_like_body_chunk(section_title, compact)
    reference_like = looks_like_reference_chunk(compact)
    if section_role(section_title) == REFERENCES_ROLE:
        return REFERENCES_ROLE
    if current_role == REFERENCES_ROLE and body_like:
        return BODY_ROLE
    if current_role in {BODY_ROLE, APPENDIX_ROLE} and reference_like and not body_like:
        return REFERENCES_ROLE
    if (
        current_role in {BODY_ROLE, FRONT_MATTER_ROLE}
        and not body_like
        and looks_like_table_like_chunk(chunk_text, compact)
    ):
        return TABLE_LIKE_ROLE
    return current_role
