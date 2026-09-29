import re

from arxplore.text import collapse_whitespace

REFERENCE_SCAN_CHARS = 1200
OUTLINE_PHRASES = (
    "the remainder of this paper",
    "the rest of this paper",
    "the remainder of the paper",
    "the rest of the paper",
    "this paper is organized as follows",
    "the paper is organized as follows",
    "section 2 presents",
    "section 3 presents",
    "section 4 presents",
    "section 5 presents",
    "section 2 describes",
    "section 3 describes",
    "section 4 describes",
    "section 5 describes",
    "we conclude in section",
)

_CITATION_MARKER = re.compile(r"\[\d+\]")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_VENUE = re.compile(
    r"\b(?:arXiv preprint|Proceedings|Conference|CVPR|ICCV|ECCV|NeurIPS|ICLR|ACL|EMNLP|AAAI)\b",
    re.IGNORECASE,
)
_AUTHOR_LIST_OPENING = re.compile(
    r"^(?:[A-Z][A-Za-z'`.-]+,\s+[A-Z](?:\.[A-Z])?(?:,\s+[A-Z][A-Za-z'`.-]+,\s+[A-Z](?:\.[A-Z])?){1,}"
    r"|(?:\[\d+\]\s*)?[A-Z][A-Za-z'`.-]+,)"
)


def looks_reference_like_text(text: str) -> bool:
    lines: list[str] = []
    total = 0
    for raw_line in text.splitlines():
        line = collapse_whitespace(raw_line)
        if not line:
            continue
        if total >= REFERENCE_SCAN_CHARS:
            break
        lines.append(line)
        total += len(line) + 1
    compact = " ".join(lines)[:REFERENCE_SCAN_CHARS]
    if not compact:
        return False

    entry_markers = sum(1 for line in lines if _CITATION_MARKER.match(line))
    citation_markers = len(_CITATION_MARKER.findall(compact))
    year_markers = len(_YEAR.findall(compact))
    venue_markers = len(_VENUE.findall(compact))
    author_list_opening = bool(_AUTHOR_LIST_OPENING.match(compact))

    if entry_markers >= 3:
        return True
    if citation_markers >= 2 and year_markers >= 2:
        return True
    if venue_markers >= 2 and year_markers >= 2:
        return True
    return author_list_opening and year_markers >= 1 and venue_markers >= 1


def looks_outline_like_text(text: str) -> bool:
    compact = collapse_whitespace(text).lower()
    if not compact:
        return False
    return any(phrase in compact for phrase in OUTLINE_PHRASES)
