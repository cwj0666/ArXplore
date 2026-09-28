from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass
from typing import Self

from arxplore.llm.citations import Citation


@dataclass(frozen=True, slots=True)
class AnswerChunk:
    text: str


@dataclass(frozen=True, slots=True)
class AnswerCitations:
    citations: tuple[Citation, ...]

    @classmethod
    def of(cls, citations: Sequence[Citation]) -> Self:
        return cls(citations=tuple(citations))


type AnswerEvent = AnswerChunk | AnswerCitations


@dataclass(frozen=True, slots=True)
class CollectedAnswer:
    answer: str
    citations: tuple[Citation, ...]


async def collect_answer(events: AsyncIterable[AnswerEvent]) -> CollectedAnswer:
    parts: list[str] = []
    citations: tuple[Citation, ...] = ()
    async for event in events:
        if isinstance(event, AnswerChunk):
            parts.append(event.text)
        else:
            citations = event.citations
    return CollectedAnswer(answer="".join(parts), citations=citations)
