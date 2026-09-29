import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from arxplore.parsing.cleaner import normalize_text
from arxplore.parsing.models import Chunk, ChunkMetadata, ChunkSummary
from arxplore.parsing.roles import BODY_ROLE, REFERENCES_ROLE, TABLE_LIKE_ROLE, chunk_role, refined_chunk_role
from arxplore.parsing.sections import FRONT_MATTER_TITLE, FULL_TEXT_TITLE

DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP_CHARS = 200
MIN_MAX_CHARS = 300
MIN_BREAK_CHARS = 220
TINY_CHUNK_CHARS = 160
ABSORB_MAX_CHARS = 360
ABSORB_MAX_WORDS = 12

_END_PREFERRED_BREAK = re.compile(r"(?:\n{2,}|\.\s+|\?\s+|!\s+)")
_END_LINE_BREAK = re.compile(r"\n+")
_LOWERCASE_OR_DIGIT_OPENING = re.compile(r"^[a-z0-9]")
_NUMERIC_ONLY_TEXT = re.compile(r"[\d\s.,;:()[\]{}%-]{1,360}")
_SUSPICIOUS_OPENING = re.compile(r"^[a-z0-9,.;:)\]%-]")
_SENTENCE_END_CHARS = ".?!:\n"
_CONTINUATION_CHARS = ",.;:)]}%-"
_SENTENCE_END_MARKS = ".?!"
_SENTENCE_CLOSERS = "\"')]\u201d\u2019"
_PARAGRAPH_BREAK = "\n\n"


@dataclass(slots=True)
class _ChunkDraft:
    chunk_index: int
    text: str
    section_title: str
    section_index: int
    section_chunk_index: int
    section_char_length: int
    char_start: int
    char_end: int
    starts_mid_sentence: bool
    ends_mid_sentence: bool
    content_role: str = BODY_ROLE


def rough_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def starts_mid_sentence(text: str, start: int) -> bool:
    if start <= 0:
        return False
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        return False
    previous_index = start - 1
    while previous_index >= 0 and text[previous_index].isspace():
        previous_index -= 1
    if previous_index < 0:
        return False
    previous_char = text[previous_index]
    current_char = text[start]
    if previous_char in _SENTENCE_END_CHARS:
        return False
    if current_char in _CONTINUATION_CHARS:
        return True
    if previous_char.isalnum() and current_char.isalnum():
        return True
    return current_char.islower() or current_char.isdigit()


def ends_mid_sentence(text: str, end: int) -> bool:
    if end >= len(text):
        return False
    current_index = max(0, end - 1)
    while current_index >= 0 and text[current_index].isspace():
        current_index -= 1
    if current_index < 0:
        return False
    next_index = end
    while next_index < len(text) and text[next_index].isspace():
        next_index += 1
    if next_index >= len(text):
        return False
    current_char = text[current_index]
    next_char = text[next_index]
    if current_char in _SENTENCE_END_CHARS:
        return False
    if next_char in _CONTINUATION_CHARS:
        return True
    if current_char.isalnum() and next_char.isalnum():
        return True
    return next_char.islower() or next_char.isdigit()


def _chunk_end(text: str, start: int, max_chars: int) -> int:
    tentative_end = min(len(text), start + max_chars)
    if tentative_end >= len(text):
        return len(text)
    window = text[start:tentative_end]
    minimum_break = max(max_chars // 2, MIN_BREAK_CHARS)
    preferred_breaks = [match.end() for match in _END_PREFERRED_BREAK.finditer(window) if match.end() >= minimum_break]
    if preferred_breaks:
        return start + preferred_breaks[-1]
    line_breaks = [match.end() for match in _END_LINE_BREAK.finditer(window) if match.end() >= minimum_break]
    if line_breaks:
        return start + line_breaks[-1]
    last_space = window.rfind(" ")
    if last_space >= minimum_break:
        return start + last_space + 1
    return tentative_end


def _word_starts(text: str, lower: int, upper: int) -> list[int]:
    return [
        position
        for position in range(lower, upper + 1)
        if text[position - 1].isspace() and not text[position].isspace()
    ]


def _is_sentence_start(text: str, position: int) -> bool:
    index = position - 1
    while index >= 0 and text[index].isspace():
        index -= 1
    if _PARAGRAPH_BREAK in text[index + 1 : position]:
        return True
    while index >= 0 and text[index] in _SENTENCE_CLOSERS:
        index -= 1
    return index >= 0 and text[index] in _SENTENCE_END_MARKS and not text[position].islower()


def _next_chunk_start(text: str, *, start: int, end: int, overlap_chars: int) -> int:
    if overlap_chars <= 0:
        return end
    target = end - overlap_chars
    slack = overlap_chars // 2
    lower = max(start + 1, target - slack)
    upper = min(end - 1, target + slack)
    if lower > upper:
        return end
    word_starts = _word_starts(text, lower, upper)
    sentence_starts = [position for position in word_starts if _is_sentence_start(text, position)]
    candidates = sentence_starts or word_starts
    if not candidates:
        return min(upper, max(lower, target))
    return min(candidates, key=lambda position: (abs(position - target), position))


def _should_absorb(drafts: Sequence[_ChunkDraft], new: _ChunkDraft) -> bool:
    if not drafts:
        return False
    if drafts[-1].section_index != new.section_index:
        return False
    new_text = new.text
    if len(new_text) >= TINY_CHUNK_CHARS and not (
        len(new_text) <= ABSORB_MAX_CHARS and _LOWERCASE_OR_DIGIT_OPENING.match(new_text)
    ):
        return False
    if new.starts_mid_sentence:
        return True
    return (
        bool(_NUMERIC_ONLY_TEXT.fullmatch(new_text))
        or len(new_text.split()) <= ABSORB_MAX_WORDS
        or bool(_LOWERCASE_OR_DIGIT_OPENING.match(new_text))
    )


def _merged_text(previous: _ChunkDraft, new: _ChunkDraft, section_text: str) -> str:
    return section_text[previous.char_start : new.char_end].strip()


def _absorb(previous: _ChunkDraft, new: _ChunkDraft, merged_text: str) -> None:
    previous.text = merged_text
    previous.char_end = new.char_end
    previous.ends_mid_sentence = new.ends_mid_sentence


def _chunk_section(
    drafts: list[_ChunkDraft],
    *,
    section_index: int,
    section_title: str,
    section_text: str,
    max_chars: int,
    overlap_chars: int,
) -> None:
    start = 0
    section_chunk_index = 0
    while start < len(section_text):
        end = _chunk_end(section_text, start, max_chars)
        previous_end = drafts[-1].char_end if drafts and drafts[-1].section_index == section_index else start
        new_start = max(start, previous_end)
        new_text = section_text[new_start:end].strip()
        if new_text:
            new_content = _ChunkDraft(
                chunk_index=len(drafts),
                text=new_text,
                section_title=section_title,
                section_index=section_index,
                section_chunk_index=section_chunk_index,
                section_char_length=len(section_text),
                char_start=new_start,
                char_end=end,
                starts_mid_sentence=starts_mid_sentence(section_text, new_start),
                ends_mid_sentence=ends_mid_sentence(section_text, end),
            )
            merged_text = (
                _merged_text(drafts[-1], new_content, section_text) if _should_absorb(drafts, new_content) else None
            )
            if merged_text is not None and len(merged_text) <= max_chars:
                _absorb(drafts[-1], new_content, merged_text)
            else:
                chunk_text = new_text if new_start == start else section_text[start:end].strip()
                drafts.append(
                    _ChunkDraft(
                        chunk_index=len(drafts),
                        text=chunk_text,
                        section_title=section_title,
                        section_index=section_index,
                        section_chunk_index=section_chunk_index,
                        section_char_length=len(section_text),
                        char_start=start,
                        char_end=end,
                        starts_mid_sentence=starts_mid_sentence(section_text, start),
                        ends_mid_sentence=new_content.ends_mid_sentence,
                        content_role=chunk_role(section_title, chunk_text),
                    )
                )
                section_chunk_index += 1
        if end >= len(section_text):
            break
        start = _next_chunk_start(section_text, start=start, end=end, overlap_chars=overlap_chars)


def _finalize(drafts: Sequence[_ChunkDraft], index: int) -> Chunk:
    draft = drafts[index]
    metadata: ChunkMetadata = {
        "section_index": draft.section_index,
        "section_chunk_index": draft.section_chunk_index,
        "section_char_length": draft.section_char_length,
        "char_start": draft.char_start,
        "char_end": draft.char_end,
        "char_length": len(draft.text),
        "starts_mid_sentence": draft.starts_mid_sentence,
        "ends_mid_sentence": draft.ends_mid_sentence,
        "content_role": draft.content_role,
        "prev_chunk_index": drafts[index - 1].chunk_index if index > 0 else None,
        "next_chunk_index": drafts[index + 1].chunk_index if index + 1 < len(drafts) else None,
        "continues_previous": draft.starts_mid_sentence,
        "continues_next": draft.ends_mid_sentence,
    }
    return {
        "chunk_index": draft.chunk_index,
        "chunk_text": draft.text,
        "section_title": draft.section_title,
        "token_count": rough_token_count(draft.text),
        "metadata": metadata,
    }


def build_chunks(
    text: str,
    sections: Sequence[Mapping[str, Any]] | None = None,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    max_chars = max(MIN_MAX_CHARS, max_chars)
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))
    source_sections: Sequence[Mapping[str, Any]] = sections or [{"title": FULL_TEXT_TITLE, "text": normalized}]
    drafts: list[_ChunkDraft] = []
    for section_index, section in enumerate(source_sections):
        section_title = str(section.get("title") or FULL_TEXT_TITLE).strip() or FULL_TEXT_TITLE
        section_text = normalize_text(str(section.get("text") or ""))
        if section_text:
            _chunk_section(
                drafts,
                section_index=section_index,
                section_title=section_title,
                section_text=section_text,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
    for draft in drafts:
        draft.content_role = refined_chunk_role(draft.section_title, draft.text, draft.content_role)
    return [_finalize(drafts, index) for index in range(len(drafts))]


def _role(chunk: Chunk) -> str:
    return chunk["metadata"]["content_role"]


def summarize_chunks(chunks: Sequence[Chunk]) -> ChunkSummary:
    if not chunks:
        return {
            "chunk_count": 0,
            "body_chunk_count": 0,
            "avg_chunk_chars": 0,
            "max_chunk_chars": 0,
            "avg_chunk_tokens": 0,
            "max_chunk_tokens": 0,
            "suspicious_start_count": 0,
            "body_suspicious_start_count": 0,
            "mid_sentence_start_count": 0,
            "mid_sentence_end_count": 0,
            "front_matter_chunk_count": 0,
            "non_body_chunk_count": 0,
            "reference_chunk_count": 0,
            "table_like_chunk_count": 0,
            "tiny_chunk_count": 0,
        }
    char_lengths = [len(chunk["chunk_text"]) for chunk in chunks]
    token_counts = [chunk["token_count"] for chunk in chunks]
    suspicious = [chunk for chunk in chunks if chunk["chunk_text"] and _SUSPICIOUS_OPENING.match(chunk["chunk_text"])]
    return {
        "chunk_count": len(chunks),
        "body_chunk_count": sum(1 for chunk in chunks if _role(chunk) == BODY_ROLE),
        "avg_chunk_chars": round(sum(char_lengths) / len(char_lengths), 2),
        "max_chunk_chars": max(char_lengths),
        "avg_chunk_tokens": round(sum(token_counts) / len(token_counts), 2),
        "max_chunk_tokens": max(token_counts),
        "suspicious_start_count": len(suspicious),
        "body_suspicious_start_count": sum(1 for chunk in suspicious if _role(chunk) == BODY_ROLE),
        "mid_sentence_start_count": sum(1 for chunk in chunks if chunk["metadata"]["starts_mid_sentence"]),
        "mid_sentence_end_count": sum(1 for chunk in chunks if chunk["metadata"]["ends_mid_sentence"]),
        "front_matter_chunk_count": sum(1 for chunk in chunks if chunk["section_title"] == FRONT_MATTER_TITLE),
        "non_body_chunk_count": sum(1 for chunk in chunks if _role(chunk) != BODY_ROLE),
        "reference_chunk_count": sum(1 for chunk in chunks if _role(chunk) == REFERENCES_ROLE),
        "table_like_chunk_count": sum(1 for chunk in chunks if _role(chunk) == TABLE_LIKE_ROLE),
        "tiny_chunk_count": sum(1 for length in char_lengths if length < TINY_CHUNK_CHARS),
    }
