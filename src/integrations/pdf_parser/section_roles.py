from __future__ import annotations

import re

_SECTION_NUMBER_PREFIX = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?|[IVXLC]+[.)]?|(?i:[ivxlc]+)[.)])\s+")
_REFERENCES_TITLE = re.compile(
    r"(?:references?|bibliography|works\s+cited|literature\s+cited)(?:\s+and\s+notes)?",
    re.IGNORECASE,
)


def is_references_section_title(title: str) -> bool:
    """섹션 제목이 참고문헌 섹션 자체인지 단어 경계 기준으로 판별한다."""
    normalized = " ".join(str(title or "").split())
    normalized = _SECTION_NUMBER_PREFIX.sub("", normalized, count=1)
    normalized = normalized.strip(" .:")
    return bool(_REFERENCES_TITLE.fullmatch(normalized))
