import re
from collections.abc import Sequence
from dataclasses import dataclass

from arxplore.text import collapse_whitespace, strip_control_characters

MIN_CHANNEL_CANDIDATES = 30
CHANNEL_CANDIDATES_PER_RESULT = 5
MIN_HYBRID_BRANCH_RESULTS = 10
HYBRID_BRANCH_RESULTS_PER_RESULT = 3
MIN_QUERY_TOKEN_LENGTH = 3
QUERY_STOPWORDS = frozenset({"the", "and", "for", "with", "from", "that", "this"})

_QUERY_TOKEN = re.compile(r"[a-z0-9]+")
_REFERENCE_INTENT = re.compile(r"\b(?:references?|bibliography|works\s+cited|cited\s+works)\b", re.IGNORECASE)
_CITATION_WORD = re.compile(r"\bcitations?\b")


@dataclass(frozen=True, slots=True)
class SectionIntent:
    query_keywords: tuple[str, ...]
    section_keywords: tuple[str, ...]
    bonus: float


SECTION_INTENTS = (
    SectionIntent(("limitation", "limitations"), ("limitation",), 0.14),
    SectionIntent(("conclusion", "conclusions"), ("conclusion",), 0.12),
    SectionIntent(("future work",), ("future work", "future directions"), 0.12),
    SectionIntent(("discussion",), ("discussion",), 0.1),
)


def normalize_search_query(query: str) -> str:
    return collapse_whitespace(strip_control_characters(query))


def candidate_fetch_limit(limit: int) -> int:
    return max(max(1, int(limit)) * CHANNEL_CANDIDATES_PER_RESULT, MIN_CHANNEL_CANDIDATES)


def hybrid_branch_limit(limit: int) -> int:
    return max(max(1, int(limit)) * HYBRID_BRANCH_RESULTS_PER_RESULT, MIN_HYBRID_BRANCH_RESULTS)


def query_tokens(query: str) -> frozenset[str]:
    return frozenset(
        token
        for token in _QUERY_TOKEN.findall(query.lower())
        if len(token) >= MIN_QUERY_TOKEN_LENGTH and token not in QUERY_STOPWORDS
    )


def reference_intent_requested(query: str) -> bool:
    return bool(_REFERENCE_INTENT.search(query))


def citation_intent_requested(query: str) -> bool:
    return reference_intent_requested(query) or bool(_CITATION_WORD.search(query.lower()))


def requested_section_intents(query: str) -> tuple[SectionIntent, ...]:
    lowered = query.lower()
    return tuple(intent for intent in SECTION_INTENTS if any(keyword in lowered for keyword in intent.query_keywords))


def section_intent_bonus(intents: Sequence[SectionIntent], section_title: str) -> float:
    lowered = section_title.lower()
    for intent in intents:
        if any(keyword in lowered for keyword in intent.section_keywords):
            return intent.bonus
    return 0.0
