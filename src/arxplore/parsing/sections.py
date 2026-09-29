import re
from collections.abc import Sequence

from arxplore.parsing.cleaner import (
    collapse_whitespace,
    looks_like_toc_line,
    normalize_heading_like_text,
    normalize_text,
)
from arxplore.parsing.models import Section
from arxplore.parsing.roles import BODY_ROLE, TOC_ROLE, section_role

FRONT_MATTER_TITLE = "Front Matter"
FULL_TEXT_TITLE = "Full Text"
ABSTRACT_TITLE = "Abstract"

KNOWN_SECTION_TITLES = frozenset(
    {
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
)
KNOWN_UPPERCASE_TITLE_TOKENS = frozenset(
    {
        "AI",
        "API",
        "ASR",
        "BERT",
        "BLEU",
        "BPE",
        "CNN",
        "CPU",
        "CTC",
        "GAN",
        "GPU",
        "GRU",
        "GPT",
        "LIDAR",
        "LLM",
        "LSTM",
        "MLP",
        "NLP",
        "NMT",
        "OCR",
        "PDF",
        "RAG",
        "RNN",
        "SOTA",
        "TTS",
        "VLM",
    }
)
NUMBERED_SECTION_PATTERN = re.compile(
    r"^(?P<prefix>(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*))(?:[.)])?\s+(?P<title>[A-Z][A-Za-z0-9 ,:/()'&\-\u2013]{1,100})$"
)
REFERENCE_TAIL_SECTION_KEYWORDS = (
    "conclusion",
    "discussion",
    "appendix",
    "additional analysis",
    "supplementary",
    "limitations",
    "experimental details",
    "implementation details",
)

_HEADING_WORD = re.compile(r"[A-Za-z0-9-]+")
_UPPERCASE_RUN = re.compile(r"[A-Z]+")
_DASH_WITH_SPACES = re.compile(r"\s*[\u2013\u2014]\s*")
_NON_HEADING_TITLES = frozenset({"figure", "table", "listing"})
_NON_HEADING_OPENINGS = ("figure ", "table ", "listing ", "arxiv:")
_CAPTION_OPENINGS = ("figure ", "table ", "listing ")
_PURE_NUMBER = re.compile(r"\d+")
_APPENDIX_HEADING = re.compile(r"(appendix(?:\s+[A-Z])?)\s+([A-Z][A-Za-z0-9 ,:/()'&-]{1,100})", re.IGNORECASE)
_APPENDIX_CONTENTS_TITLES = frozenset({"table of contents", "contents"})
_TRAILING_DOT_LEADER = re.compile(r"(?:\. ?){2,}\d{1,3}$")
_TRAILING_PAGE_NUMBER = re.compile(r"\s\d{1,3}$")
_BODY_REJECTED_OPENINGS = ("We ", "Our ", "This ", "These ")
_PARAGRAPH_BREAK = re.compile(r"\n{2,}")
_REFERENCE_NUMBER = re.compile(r"\[\d+\]")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_LINK_MARKER = re.compile(r"https?\s*:\s*//|doi:|arxiv:", re.IGNORECASE)
_VENUE_MARKER = re.compile(
    r"\b(?:Proceedings|Conference|CVPR|ICCV|ECCV|NeurIPS|ICLR|ACL|EMNLP|AAAI|Google)\b",
    re.IGNORECASE,
)
_AUTHOR_LIST = re.compile(
    r"^(?:\[\d+\]\s*)?[A-Z][A-Za-z'`.-]+,\s+[A-Z](?:\.[A-Z])?(?:,\s+[A-Z][A-Za-z'`.-]+,\s+[A-Z](?:\.[A-Z])?){1,}"
)
_REFERENCE_TAIL_OPENINGS = ("In: ", "[", "doi:", "https://", "http://")
_LEAD_FRAGMENT = re.compile(
    r"^(?P<lead>[a-z][^.]{0,160}\.)\s+(?P<rest>(?:In this section|Here, we|We |This section|To )\b.+)$"
)
_NUMERIC_SECTION_PREFIX = re.compile(r"^(?P<prefix>\d+(?:\.\d+)*)\b")


def looks_like_numbered_heading(title: str) -> bool:
    words = _HEADING_WORD.findall(title)
    if not words:
        return False
    uppercase_like = sum(1 for word in words if word[0].isupper() or word[0].isdigit() or word.isupper())
    ratio = uppercase_like / len(words)
    if len(words) <= 4:
        return ratio >= 0.5
    return ratio >= 0.6


def _prettify_uppercase_run(match: re.Match[str]) -> str:
    word = match.group(0)
    if word in KNOWN_UPPERCASE_TITLE_TOKENS or len(word) == 1:
        return word
    return word.capitalize()


def prettify_section_title(title: str) -> str:
    compact = collapse_whitespace(title)
    if compact.isupper():
        return _UPPERCASE_RUN.sub(_prettify_uppercase_run, compact)
    return compact


def normalize_section_heading(line: str) -> str | None:
    candidate = normalize_heading_like_text(collapse_whitespace(line))
    candidate = _DASH_WITH_SPACES.sub(" - ", candidate)
    lowered = candidate.lower()
    if lowered in _NON_HEADING_TITLES:
        return None
    if lowered.startswith(_NON_HEADING_OPENINGS):
        return None
    if _PURE_NUMBER.fullmatch(candidate):
        return None
    if lowered in KNOWN_SECTION_TITLES:
        return prettify_section_title(candidate)
    appendix_match = _APPENDIX_HEADING.fullmatch(candidate)
    if appendix_match:
        prefix = appendix_match.group(1)
        title = appendix_match.group(2)
        if title.lower() in _APPENDIX_CONTENTS_TITLES:
            return prettify_section_title(prefix)
        return prettify_section_title(f"{prefix} {title}")
    match = NUMBERED_SECTION_PATTERN.fullmatch(candidate)
    if match is None:
        return None
    prefix = match.group("prefix")
    title = match.group("title")
    prefix_root = prefix.split(".")[0]
    if prefix_root.isdigit():
        if int(prefix_root) >= 20:
            return None
        if prefix.isdigit() and int(prefix) >= 50:
            return None
    if _TRAILING_DOT_LEADER.search(candidate):
        return None
    if _TRAILING_PAGE_NUMBER.search(title):
        return None
    if title.lower().startswith(_CAPTION_OPENINGS):
        return None
    title_words = title.split()
    if len(title_words) > 9:
        return None
    if any(len(word) > 24 for word in title_words):
        return None
    if "," in title and len(title_words) > 6:
        return None
    if title.startswith(_BODY_REJECTED_OPENINGS):
        return None
    if not looks_like_numbered_heading(title):
        return None
    return prettify_section_title(f"{prefix} {title}")


def looks_like_reference_tail_paragraph(text: str) -> bool:
    compact = collapse_whitespace(text)
    if not compact:
        return False
    reference_markers = len(_REFERENCE_NUMBER.findall(compact))
    year_markers = len(_YEAR.findall(compact))
    url_markers = len(_LINK_MARKER.findall(compact))
    sentence_breaks = compact.count(". ") + compact.count("? ") + compact.count("! ")
    venue_markers = len(_VENUE_MARKER.findall(compact))
    author_list_like = bool(_AUTHOR_LIST.match(compact))
    if reference_markers >= 2:
        return True
    if reference_markers >= 1 and year_markers >= 2 and sentence_breaks <= 3:
        return True
    if url_markers >= 1 and year_markers >= 1 and venue_markers >= 1:
        return True
    if author_list_like and year_markers >= 1:
        return True
    return compact.startswith(_REFERENCE_TAIL_OPENINGS) and (year_markers >= 1 or venue_markers >= 1)


def strip_trailing_reference_like_tail(title: str, text: str) -> str:
    lowered_title = title.lower()
    if not any(keyword in lowered_title for keyword in REFERENCE_TAIL_SECTION_KEYWORDS):
        return text
    paragraphs = [paragraph.strip() for paragraph in _PARAGRAPH_BREAK.split(text) if paragraph.strip()]
    if not paragraphs:
        return text
    while paragraphs and looks_like_reference_tail_paragraph(paragraphs[-1]):
        paragraphs.pop()
    trimmed = "\n\n".join(paragraphs).strip()
    return trimmed or text


def normalize_section_text(title: str, text: str) -> str:
    compact = normalize_text(text)
    if not compact:
        return compact
    compact = strip_trailing_reference_like_tail(title, compact)
    if section_role(title) == BODY_ROLE:
        lead_fragment_match = _LEAD_FRAGMENT.match(compact)
        if lead_fragment_match:
            return lead_fragment_match.group("rest").strip()
    return compact


def should_drop_section(title: str, text: str) -> bool:
    if not collapse_whitespace(text):
        return True
    if section_role(title) == TOC_ROLE:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and len(lines) <= 12 and all(looks_like_toc_line(line) for line in lines)


def _flush_section(sections: list[Section], title: str, lines: list[str]) -> None:
    section_text = normalize_text("\n".join(lines))
    section_text = normalize_section_text(title, section_text)
    if section_text and not should_drop_section(title, section_text):
        sections.append({"title": title, "text": section_text})


def extract_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    current_title = FRONT_MATTER_TITLE
    current_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue
        heading = normalize_section_heading(line)
        if heading is not None:
            if current_lines:
                _flush_section(sections, current_title, current_lines)
            current_title = heading
            current_lines = []
            continue
        current_lines.append(line)
    if current_lines:
        _flush_section(sections, current_title, current_lines)
    if len(sections) == 1 and sections[0]["title"] == FRONT_MATTER_TITLE:
        sections[0]["title"] = FULL_TEXT_TITLE
    return reorder_sections(sections)


def numeric_section_sort_key(title: str) -> tuple[int, ...] | None:
    match = _NUMERIC_SECTION_PREFIX.match(title.strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.group("prefix").split("."))


def _reorder_key(item: tuple[int, Section]) -> tuple[int, tuple[int, ...], int]:
    index, section = item
    title = str(section.get("title") or "")
    lowered = title.lower()
    if lowered == "front matter":
        return (0, (), index)
    if lowered == "abstract":
        return (1, (), index)
    numeric_key = numeric_section_sort_key(title)
    if numeric_key is not None:
        return (2, numeric_key, index)
    return (3, (), index)


def reorder_sections(sections: Sequence[Section]) -> list[Section]:
    if len(sections) < 3:
        return list(sections)
    numeric_order = [
        sort_key
        for section in sections
        if (sort_key := numeric_section_sort_key(str(section.get("title") or ""))) is not None
    ]
    if len(numeric_order) < 2 or numeric_order == sorted(numeric_order):
        return list(sections)
    return [section for _, section in sorted(enumerate(sections), key=_reorder_key)]
