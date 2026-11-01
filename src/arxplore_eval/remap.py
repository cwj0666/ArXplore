import difflib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from arxplore.text import collapse_whitespace
from arxplore_eval.dataset import EvalQuery

DEFAULT_MIN_OVERLAP = 0.9
MIN_BLOCK_CHARS = 60
MIN_GAIN = 0.05
MIN_GAIN_CHARS = 40
MIN_INSIDE_SHARE = 0.5
REVIEW_SKIPPED_SHARE = 0.1
COMPLETE_COVERAGE = 0.999
SHINGLE_WORDS = 5
MAX_CANDIDATES = 6
READ_ONLY_SESSION_SQL = "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"
OLD_CHUNKS_SQL = """
    SELECT id, arxiv_id, chunk_index, coalesce(section_title, '') AS section_title, chunk_text
    FROM paper_chunks
    WHERE id = ANY(%s)
"""
NEW_CHUNKS_SQL = """
    SELECT id, arxiv_id, chunk_index, coalesce(section_title, '') AS section_title, chunk_text
    FROM paper_chunks
    WHERE arxiv_id = ANY(%s)
    ORDER BY arxiv_id, chunk_index
"""
MISSING_OLD_CHUNK = "옛 DB에 청크 없음"
MISSING_NEW_PAPER = "새 DB에 논문 청크 없음"
LOW_OVERLAP = "겹침 부족"
SKIPPED_OVERLAP = "제외 후보 겹침"

_WORD = re.compile(r"\w+")

type Connection = AsyncConnection[Any]
type Span = tuple[int, int]


@dataclass(frozen=True)
class StoredText:
    chunk_id: int
    arxiv_id: str
    chunk_index: int
    section_title: str
    text: str


@dataclass(frozen=True)
class ChunkShare:
    chunk_id: int
    chunk_index: int
    share: float
    contained: bool
    inside_share: float = 0.0


@dataclass(frozen=True)
class ChunkMapping:
    old_chunk_id: int
    arxiv_id: str
    new_chunks: tuple[ChunkShare, ...]
    overlap: float
    problem: str | None = None
    skipped: tuple[ChunkShare, ...] = ()

    @property
    def new_chunk_ids(self) -> tuple[int, ...]:
        return tuple(share.chunk_id for share in self.new_chunks)


@dataclass(frozen=True)
class QueryRemap:
    query_id: str
    old_chunk_ids: tuple[int, ...]
    mappings: tuple[ChunkMapping, ...]
    min_overlap: float

    @property
    def new_chunk_ids(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(chunk_id for mapping in self.mappings for chunk_id in mapping.new_chunk_ids))

    @property
    def overlap(self) -> float:
        return min((mapping.overlap for mapping in self.mappings), default=0.0)

    @property
    def needs_review(self) -> bool:
        return any(mapping.problem is not None for mapping in self.mappings) or self.overlap < self.min_overlap

    @property
    def changed(self) -> bool:
        return self.new_chunk_ids != self.old_chunk_ids


def normalize(text: str) -> str:
    return collapse_whitespace(text)


def shingles(text: str) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return {tuple(words[index : index + SHINGLE_WORDS]) for index in range(max(0, len(words) - SHINGLE_WORDS + 1))}


def matched_spans(old: str, new: str, *, min_block: int = MIN_BLOCK_CHARS) -> list[Span]:
    if old in new:
        return [(0, len(old))]
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    return [(block.a, block.a + block.size) for block in matcher.get_matching_blocks() if block.size >= min_block]


def covered_length(spans: Iterable[Span]) -> int:
    total = 0
    end = -1
    for start, stop in sorted(spans):
        if stop <= end:
            continue
        total += stop - max(start, end)
        end = stop
    return total


def candidate_chunks(old: str, chunks: Sequence[StoredText], *, limit: int = MAX_CANDIDATES) -> list[StoredText]:
    reference = shingles(old)
    scored = []
    for chunk in chunks:
        text = normalize(chunk.text)
        shared = len(reference & shingles(text)) if reference else int(old in text)
        if shared or old in text:
            scored.append((shared, -chunk.chunk_index, chunk))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [chunk for _, _, chunk in scored[:limit]]


def map_chunk(old: StoredText, chunks: Sequence[StoredText]) -> ChunkMapping:
    text = normalize(old.text)
    if not chunks:
        return ChunkMapping(old.chunk_id, old.arxiv_id, (), 0.0, MISSING_NEW_PAPER)
    if not text:
        return ChunkMapping(old.chunk_id, old.arxiv_id, (), 0.0, LOW_OVERLAP)
    measured: list[tuple[ChunkShare, list[Span]]] = []
    for chunk in candidate_chunks(text, chunks):
        candidate_text = normalize(chunk.text)
        spans = matched_spans(text, candidate_text)
        matched = covered_length(spans)
        measured.append(
            (
                ChunkShare(
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    share=matched / len(text),
                    contained=spans == [(0, len(text))],
                    inside_share=matched / len(candidate_text) if candidate_text else 0.0,
                ),
                spans,
            )
        )
    measured.sort(key=lambda item: (-item[0].share, item[0].chunk_index))
    selected: list[ChunkShare] = []
    skipped: list[ChunkShare] = []
    covered: list[Span] = []
    for candidate, spans in measured:
        gain = covered_length([*covered, *spans]) - covered_length(covered)
        if not adds_enough(candidate, gain, len(text)):
            if gain > 0 and candidate.share >= REVIEW_SKIPPED_SHARE:
                skipped.append(candidate)
            continue
        selected.append(candidate)
        covered.extend(spans)
        if covered_length(covered) / len(text) >= COMPLETE_COVERAGE:
            break
    overlap = covered_length(covered) / len(text)
    ordered = tuple(sorted(selected, key=lambda item: item.chunk_index))
    problem = SKIPPED_OVERLAP if skipped else None
    return ChunkMapping(old.chunk_id, old.arxiv_id, ordered, overlap, problem, tuple(skipped))


def adds_enough(candidate: ChunkShare, gain: int, old_length: int) -> bool:
    if gain / old_length >= MIN_GAIN:
        return True
    return gain >= MIN_GAIN_CHARS and candidate.inside_share >= MIN_INSIDE_SHARE


def remap_queries(
    queries: Iterable[EvalQuery],
    old_chunks: Mapping[int, StoredText],
    new_chunks: Mapping[str, Sequence[StoredText]],
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> list[QueryRemap]:
    remaps = []
    for query in queries:
        if not query.relevant_chunk_ids:
            continue
        mappings = []
        for chunk_id in query.relevant_chunk_ids:
            old = old_chunks.get(chunk_id)
            if old is None:
                arxiv_id = query.relevant_arxiv_ids[0] if query.relevant_arxiv_ids else ""
                mappings.append(ChunkMapping(chunk_id, arxiv_id, (), 0.0, MISSING_OLD_CHUNK))
                continue
            mappings.append(map_chunk(old, new_chunks.get(old.arxiv_id, ())))
        remaps.append(QueryRemap(query.id, query.relevant_chunk_ids, tuple(mappings), min_overlap))
    return remaps


def _stored(row: Mapping[str, Any]) -> StoredText:
    return StoredText(
        chunk_id=int(row["id"]),
        arxiv_id=str(row["arxiv_id"]),
        chunk_index=int(row["chunk_index"]),
        section_title=str(row["section_title"]),
        text=str(row["chunk_text"] or ""),
    )


async def read_only(connection: Connection) -> None:
    await connection.execute(READ_ONLY_SESSION_SQL)


async def fetch_old_chunks(connection: Connection, chunk_ids: Sequence[int]) -> dict[int, StoredText]:
    if not chunk_ids:
        return {}
    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(OLD_CHUNKS_SQL, (list(chunk_ids),))
        return {int(row["id"]): _stored(row) for row in await cursor.fetchall()}


async def fetch_paper_chunks(connection: Connection, arxiv_ids: Sequence[str]) -> dict[str, list[StoredText]]:
    papers: dict[str, list[StoredText]] = {arxiv_id: [] for arxiv_id in arxiv_ids}
    if not arxiv_ids:
        return papers
    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(NEW_CHUNKS_SQL, (list(arxiv_ids),))
        for row in await cursor.fetchall():
            papers.setdefault(str(row["arxiv_id"]), []).append(_stored(row))
    return papers


def describe_mapping(mapping: ChunkMapping) -> str:
    targets = ", ".join(
        f"{share.chunk_id}(#{share.chunk_index} {share.share:.3f}{' 포함' if share.contained else ''})"
        for share in mapping.new_chunks
    )
    skipped = ", ".join(f"{share.chunk_id}(#{share.chunk_index} {share.share:.3f})" for share in mapping.skipped)
    held = f" [제외 후보 {skipped}]" if skipped else ""
    return f"{mapping.old_chunk_id} → {targets or '-'}{held}"


def report_lines(remaps: Sequence[QueryRemap]) -> list[str]:
    lines = []
    for remap in remaps:
        status = "검토 필요" if remap.needs_review else "ok"
        problems = sorted({mapping.problem for mapping in remap.mappings if mapping.problem})
        if remap.needs_review and not problems:
            problems = [LOW_OVERLAP]
        detail = f" ({', '.join(problems)})" if problems else ""
        chunks = "; ".join(describe_mapping(mapping) for mapping in remap.mappings)
        lines.append(f"{remap.query_id}\t{chunks}\toverlap {remap.overlap:.3f}\t{status}{detail}")
    review = sum(1 for remap in remaps if remap.needs_review)
    changed = sum(1 for remap in remaps if remap.changed)
    lines.append(f"청크 정답 질의 {len(remaps)}건: 검토 필요 {review}건, 바뀌는 질의 {changed}건")
    return lines


def rewrite_chunk_ids(path: Path, remaps: Sequence[QueryRemap]) -> int:
    new_ids = {remap.query_id: list(remap.new_chunk_ids) for remap in remaps if remap.changed}
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    rewritten = []
    changed = 0
    for line in lines:
        if not line.strip():
            rewritten.append(line)
            continue
        payload = json.loads(line)
        if payload.get("id") in new_ids:
            payload["relevant_chunk_ids"] = new_ids[payload["id"]]
            rewritten.append(json.dumps(payload, ensure_ascii=False) + "\n")
            changed += 1
        else:
            rewritten.append(line)
    path.write_text("".join(rewritten), encoding="utf-8")
    return changed
