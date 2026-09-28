import re

INLINE_BODY_STARTERS = (
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
SHORT_TITLE_REJECTED_OPENINGS = ("This ", "These ", "We ", "Our ", "In ", "To ", "As ", "However ")

_NUL = "\x00"
_HYPHENATED_LINE_BREAK = re.compile(r"(?<=\w)-\n(?=\w)")
_BULLET_PAGE_NUMBER_HEADER = re.compile(r"\n\d+•[A-Z][^\n]{0,100}\n")
_BULLET_PAGE_NUMBER_FOOTER = re.compile(r"\n[A-Z][^\n]{0,100}•\d+\n")
_PAGE_NUMBERED_RUNNING_HEADER = re.compile(r"\n[A-Z][A-Za-z0-9][A-Za-z0-9 :,\-\u2013'.]{3,80}\s+\d{1,3}\n")
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

_SPACED_LETTER_PREFIX = re.compile(r"[A-Z](?:\.\d+)*")
_SINGLE_CAPITAL = re.compile(r"[A-Z]")
_UPPERCASE_WORD = re.compile(r"[A-Z]{2,}")
_SPACED_CAPITAL_BEFORE_UPPERCASE = re.compile(r"\b([A-Z])\s+([A-Z]{2,})\b")
_SPACED_CAPITAL_BEFORE_CAPITALIZED = re.compile(r"\b([A-Z])\s+([A-Z][a-z][A-Za-z-]*)\b")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([:;,.!?])")

_SHORT_TITLE_PREFIX = re.compile(r"^(?P<head>[A-Z0-9][A-Za-z0-9/+&' -]{2,48}\.)\s*(?P<rest>[A-Z].+)$")
_NUMBERED_LINE_OPENING = re.compile(r"^\d+(?:\.\d+)+(?:[.)])?\s+")

_TOC_TITLES = frozenset({"contents", "table of contents"})
_TOC_DOT_LEADER = re.compile(r"(?:\. ?){6,}\d+$")
_TOC_NUMBERED_DOT_LEADER = re.compile(r"(?:\d+(?:\.\d+)*\.?)\s+.+(?:\. ?){3,}\d{1,3}")
_TOC_TITLE_WITH_PAGE = re.compile(r"(?:\d+(?:\.\d+)*|[A-Z])\s+[A-Z][A-Za-z0-9 ,:/()'&-]{1,80}\s+\d{1,3}")
_TOC_NUMBERED_WITH_PAGE = re.compile(r"(?:\d+(?:\.\d+)*)\s+.+\s+\d{1,3}")


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _normalize_once(text: str) -> str:
    normalized = text.replace(_NUL, " ")
    normalized = _HYPHENATED_LINE_BREAK.sub("", normalized)
    normalized = _BULLET_PAGE_NUMBER_HEADER.sub("\n", normalized)
    normalized = _BULLET_PAGE_NUMBER_FOOTER.sub("\n", normalized)
    normalized = _PAGE_NUMBERED_RUNNING_HEADER.sub("\n", normalized)
    normalized = _HORIZONTAL_WHITESPACE.sub(" ", normalized)
    normalized = _EXCESS_BLANK_LINES.sub("\n\n", normalized)
    return normalized.strip()


def normalize_text(text: str) -> str:
    normalized = _normalize_once(text)
    while (again := _normalize_once(normalized)) != normalized:
        normalized = again
    return normalized


def _join_spaced_capital(match: re.Match[str]) -> str:
    return f"{match.group(1)}{match.group(2)}"


def _join_spaced_capital_lowering(match: re.Match[str]) -> str:
    word = match.group(2)
    return f"{match.group(1)}{word[0].lower()}{word[1:]}"


def normalize_heading_like_text(text: str) -> str:
    normalized = collapse_whitespace(text)
    prefix = ""
    tokens = normalized.split()
    if (
        len(tokens) >= 3
        and _SPACED_LETTER_PREFIX.fullmatch(tokens[0])
        and _SINGLE_CAPITAL.fullmatch(tokens[1])
        and _UPPERCASE_WORD.fullmatch(tokens[2])
    ):
        prefix = tokens[0]
        normalized = " ".join(tokens[1:])
    previous = None
    while previous != normalized:
        previous = normalized
        normalized = _SPACED_CAPITAL_BEFORE_UPPERCASE.sub(_join_spaced_capital, normalized)
        normalized = _SPACED_CAPITAL_BEFORE_CAPITALIZED.sub(_join_spaced_capital_lowering, normalized)
    normalized = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", normalized)
    if prefix:
        normalized = f"{prefix} {normalized}"
    return normalized.strip()


def _find_short_title_prefix(text: str) -> tuple[str, str] | None:
    match = _SHORT_TITLE_PREFIX.match(text)
    if match is None:
        return None
    head = match.group("head").strip()
    rest = match.group("rest").strip()
    head_words = head[:-1].split()
    if not 1 <= len(head_words) <= 6:
        return None
    if head.startswith(SHORT_TITLE_REJECTED_OPENINGS):
        return None
    if any(len(word) > 20 for word in head_words):
        return None
    return head, rest


def find_inline_heading_split(text: str) -> tuple[str, str] | None:
    compact = collapse_whitespace(text)
    title_prefix = _find_short_title_prefix(compact)
    if title_prefix is not None:
        return title_prefix
    if not _NUMBERED_LINE_OPENING.match(compact):
        return None
    for starter in INLINE_BODY_STARTERS:
        index = compact.find(f" {starter} ")
        if index == -1:
            continue
        head = compact[:index].strip()
        rest = compact[index + 1 :].strip()
        if len(head.split()) > 16 or len(head) < 8:
            continue
        if not rest:
            continue
        return head, rest
    return None


def looks_like_toc_line(line: str) -> bool:
    compact = collapse_whitespace(line)
    if compact.lower() in _TOC_TITLES:
        return True
    if _TOC_DOT_LEADER.search(compact):
        return True
    if _TOC_NUMBERED_DOT_LEADER.fullmatch(compact):
        return True
    if _TOC_TITLE_WITH_PAGE.fullmatch(compact):
        return True
    return bool(_TOC_NUMBERED_WITH_PAGE.fullmatch(compact) and compact.count(".") >= 4)
