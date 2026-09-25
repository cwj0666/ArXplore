from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import psycopg2
import pytest

from src.integrations import db
from src.pipeline import prepare_papers
from tests.integration.test_retrieval_pg import (
    DsnPaperRepository,
    _chunk_ids,
    _drop_paper_tables,
    _execute,
    _seed_embeddings,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def dsn(test_database_url: str):
    try:
        _execute(test_database_url, "CREATE EXTENSION IF NOT EXISTS vector")
    except psycopg2.Error as exc:
        pytest.skip(f"pgvector extension is not available: {exc}")
    _drop_paper_tables(test_database_url)
    yield test_database_url
    db.close_all_pools()
    _drop_paper_tables(test_database_url)


ARXIV_ID = "2601.00001"
SEEDED_CHUNKS = [
    ("Abstract", "Latent diffusion transformers generate long videos.", "body"),
    ("3 Method", "Patch embeddings are denoised by a transformer.", "body"),
    ("Contents", "1 Introduction 2 Method 3 Experiments", "toc"),
]


def _chunks(texts: list[str], *, role: str = "body") -> list[dict[str, Any]]:
    return [
        {
            "chunk_index": index,
            "chunk_text": text,
            "section_title": SEEDED_CHUNKS[index][0] if index < len(SEEDED_CHUNKS) else None,
            "token_count": len(text.split()),
            "metadata": {"content_role": role},
        }
        for index, text in enumerate(texts)
    ]


def _embedded_chunk_ids(dsn: str, arxiv_id: str) -> set[int]:
    rows = _execute(
        dsn,
        "SELECT e.chunk_id FROM paper_embeddings e JOIN paper_chunks c ON c.id = e.chunk_id WHERE c.arxiv_id = %s",
        (arxiv_id,),
    )
    return {row[0] for row in rows}


def test_identical_chunk_texts_keep_ids_and_embeddings_but_refresh_metadata(dsn):
    ids, _ = _seed_embeddings(dsn)
    repository = DsnPaperRepository(dsn)
    before_ids = {key: value for key, value in ids.items() if key[0] == ARXIV_ID}
    before_embedded = _embedded_chunk_ids(dsn, ARXIV_ID)
    assert before_embedded == set(before_ids.values())

    replaced = repository.save_paper_chunks(ARXIV_ID, _chunks([text for _, text, _ in SEEDED_CHUNKS]))

    assert replaced is False
    assert {key: value for key, value in _chunk_ids(dsn).items() if key[0] == ARXIV_ID} == before_ids
    assert _embedded_chunk_ids(dsn, ARXIV_ID) == before_embedded
    roles = _execute(
        dsn, "SELECT metadata->>'content_role' FROM paper_chunks WHERE arxiv_id = %s ORDER BY chunk_index", (ARXIV_ID,)
    )
    assert [row[0] for row in roles] == ["body", "body", "body"]


def test_changed_chunk_texts_replace_rows_and_drop_embeddings(dsn):
    ids, _ = _seed_embeddings(dsn)
    repository = DsnPaperRepository(dsn)
    texts = [text for _, text, _ in SEEDED_CHUNKS]
    texts[1] = "Patch embeddings are denoised by a bigger transformer."

    assert repository.save_paper_chunks(ARXIV_ID, _chunks(texts)) is True
    assert _embedded_chunk_ids(dsn, ARXIV_ID) == set()
    assert set(_chunk_ids(dsn).values()).isdisjoint({ids[(ARXIV_ID, index)] for index in range(3)})
    assert _embedded_chunk_ids(dsn, "2601.00002")


class _Parser:
    def __init__(self, source: str) -> None:
        self.source = source

    def parse_from_pdf_url(self, pdf_url: str, *, fallback_text: str = ""):
        sections = [{"title": title, "text": text} for title, text, _ in SEEDED_CHUNKS]
        text = "\n\n".join(f"{title}\n{body}" for title, body, _ in SEEDED_CHUNKS)
        return SimpleNamespace(
            text=text,
            sections=sections,
            source=self.source,
            quality_metrics={"fallback_used": False},
            artifacts={},
            parser_metadata={},
        )

    def build_chunks(self, text: str, *, sections: list[dict[str, Any]]):
        return _chunks([body for _, body, _ in SEEDED_CHUNKS])

    def summarize_chunks(self, chunks):
        return {"chunk_count": len(chunks)}


def test_re_prepare_is_idempotent_and_never_downgrades_without_force(dsn):
    ids, _ = _seed_embeddings(dsn)
    repository = DsnPaperRepository(dsn)
    candidate = {"arxiv_id": ARXIV_ID, "prepared": {"arxiv_id": ARXIV_ID, "title": "T", "abstract": "A"}}
    embedded = _embedded_chunk_ids(dsn, ARXIV_ID)

    first = prepare_papers.prepare_single_paper(candidate, parser=_Parser("layout_pdf"), paper_repository=repository)
    assert first["saved_fulltext"] == 1
    assert first["chunks_unchanged"] is True
    state = repository.get_paper_fulltext_state(ARXIV_ID)
    assert state == {"source": "layout_pdf", "content_hash": first["content_hash"], "chunk_count": len(SEEDED_CHUNKS)}

    second = prepare_papers.prepare_single_paper(candidate, parser=_Parser("layout_pdf"), paper_repository=repository)
    assert second["skipped_unchanged"] is True

    downgraded = prepare_papers.prepare_single_paper(candidate, parser=_Parser("pdf"), paper_repository=repository)
    assert downgraded["skipped_lower_rank_overwrite"] is True
    assert repository.get_paper_fulltext_state(ARXIV_ID)["source"] == "layout_pdf"

    forced = prepare_papers.prepare_single_paper(
        candidate, parser=_Parser("pdf"), paper_repository=repository, force=True
    )
    assert forced["saved_fulltext"] == 1
    assert repository.get_paper_fulltext_state(ARXIV_ID)["source"] == "pdf"

    assert _embedded_chunk_ids(dsn, ARXIV_ID) == embedded
    assert {key: value for key, value in _chunk_ids(dsn).items() if key[0] == ARXIV_ID} == {
        key: value for key, value in ids.items() if key[0] == ARXIV_ID
    }


def test_chunk_write_failure_leaves_hash_unset_so_retry_rewrites_chunks(dsn, monkeypatch):
    repository = DsnPaperRepository(dsn)
    repository.ensure_schema()
    arxiv_id = "2601.00009"
    candidate = {"arxiv_id": arxiv_id, "prepared": {"arxiv_id": arxiv_id, "title": "T", "abstract": "A"}}
    original_save_chunks = repository.save_paper_chunks
    calls = {"count": 0}

    def flaky_save_chunks(target_id, chunks):
        calls["count"] += 1
        if calls["count"] == 1:
            raise psycopg2.OperationalError("connection lost")
        return original_save_chunks(target_id, chunks)

    monkeypatch.setattr(repository, "save_paper_chunks", flaky_save_chunks)

    with pytest.raises(psycopg2.OperationalError):
        prepare_papers.prepare_single_paper(candidate, parser=_Parser("layout_pdf"), paper_repository=repository)
    assert repository.get_paper_fulltext_state(arxiv_id) == {
        "source": "layout_pdf",
        "content_hash": None,
        "chunk_count": 0,
    }

    retried = prepare_papers.prepare_single_paper(candidate, parser=_Parser("layout_pdf"), paper_repository=repository)

    assert "skipped_unchanged" not in retried
    assert retried["saved_chunks"] == len(SEEDED_CHUNKS)
    assert repository.get_paper_fulltext_state(arxiv_id) == {
        "source": "layout_pdf",
        "content_hash": retried["content_hash"],
        "chunk_count": len(SEEDED_CHUNKS),
    }
    again = prepare_papers.prepare_single_paper(candidate, parser=_Parser("layout_pdf"), paper_repository=repository)
    assert again["skipped_unchanged"] is True
