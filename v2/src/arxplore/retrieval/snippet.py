import re

from arxplore.text import collapse_whitespace

SNIPPET_MAX_CHARS = 280
MIN_SNIPPET_TERM_LENGTH = 3

_TERM_SEPARATOR = re.compile(r"\W+")


def build_search_snippet(
    query: str,
    chunk_text: str,
    abstract: str,
    title: str,
    *,
    max_chars: int = SNIPPET_MAX_CHARS,
) -> str:
    terms = [term for term in _TERM_SEPARATOR.split(query.lower()) if len(term) >= MIN_SNIPPET_TERM_LENGTH]
    sources = (chunk_text, abstract, title)

    for source in sources:
        if not source:
            continue
        lowered = source.lower()
        for term in terms:
            index = lowered.find(term)
            if index == -1:
                continue
            start = max(0, index - max_chars // 3)
            end = min(len(source), start + max_chars)
            snippet = source[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(source):
                snippet = snippet + "..."
            return snippet

    fallback = next((source for source in sources if source), "")
    compact = collapse_whitespace(fallback)
    return compact[:max_chars] + ("..." if len(compact) > max_chars else "")
