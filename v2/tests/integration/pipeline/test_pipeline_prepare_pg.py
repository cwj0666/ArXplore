from datetime import date
from typing import Any

import pytest

from arxplore.db.repositories import fulltexts, raw
from arxplore.pipeline.prepare.mapping import select_candidates
from arxplore.pipeline.prepare.paper import PrepareContext, prepare_paper
from arxplore.pipeline.prepare.policy import compute_content_hash
from arxplore.pipeline.prepare.run import prepare_date

pytestmark = pytest.mark.integration

DAY = date(2026, 9, 15)
ARXIV_ID = "2609.00001"


def _candidate(kit: Any, arxiv_id: str = ARXIV_ID, **item: Any) -> Any:
    return select_candidates(DAY, [kit.hf_item(arxiv_id, **item)]).candidates[0]


def _context(pool: Any, kit: Any, parser: Any = None, pdfs: Any = None) -> PrepareContext:
    return PrepareContext(db=pool, pdfs=pdfs or kit.FakePdfSource(), parser=parser or kit.FakeParser())


async def _state(pool: Any, arxiv_id: str = ARXIV_ID) -> tuple[str, str | None, int] | None:
    state = await fulltexts.get_fulltext_state(pool, arxiv_id)
    return None if state is None else (state.source, state.content_hash, state.chunk_count)


async def _chunk_ids(probe: Any, arxiv_id: str = ARXIV_ID) -> list[int]:
    rows = await probe.rows("SELECT id FROM paper_chunks WHERE arxiv_id = %s ORDER BY chunk_index", (arxiv_id,))
    return [row["id"] for row in rows]


async def test_first_prepare_saves_paper_fulltext_chunks_then_the_hash(pool, probe, kit) -> None:
    outcome = await prepare_paper(_candidate(kit), _context(pool, kit))

    expected_hash = compute_content_hash(
        "\n\n".join(f"{s['title']}\n{s['text']}" for s in kit.body_sections(ARXIV_ID)), kit.body_sections(ARXIV_ID)
    )
    assert (outcome.saved_paper, outcome.saved_fulltext) == (1, 1)
    assert outcome.saved_chunks == outcome.chunk_count > 0
    assert outcome.content_hash == expected_hash
    assert await _state(pool) == ("layout_pdf", expected_hash, outcome.chunk_count)
    paper = await probe.row("SELECT title, source, upvotes, github_url FROM papers WHERE arxiv_id = %s", (ARXIV_ID,))
    assert paper == {
        "title": f"Paper {ARXIV_ID}",
        "source": "hf_daily_papers_raw",
        "upvotes": 3,
        "github_url": f"https://github.com/example/{ARXIV_ID}",
    }
    stored = await probe.row("SELECT quality_metrics FROM paper_fulltexts WHERE arxiv_id = %s", (ARXIV_ID,))
    assert stored is not None
    assert stored["quality_metrics"]["chunk_count"] == outcome.chunk_count
    assert stored["quality_metrics"]["parse_source"] == "layout_pdf"


@pytest.mark.parametrize(
    "existing_source,new_source",
    [("layout_pdf", "fallback_abstract"), ("pdf", "fallback_abstract"), ("layout_pdf", "pdf")],
)
async def test_lower_ranked_source_keeps_the_stored_fulltext(pool, probe, kit, existing_source, new_source) -> None:
    first = await prepare_paper(_candidate(kit), _context(pool, kit, kit.FakeParser(source=existing_source)))
    ids_before = await _chunk_ids(probe)
    lower_parser = kit.FakeParser(source=new_source)
    pdfs = kit.FakePdfSource(missing={ARXIV_ID} if new_source == "fallback_abstract" else set())

    outcome = await prepare_paper(_candidate(kit, upvotes=40), _context(pool, kit, lower_parser, pdfs))

    assert outcome.skipped_lower_rank_overwrite is True
    assert outcome.existing_fulltext_source == existing_source
    assert (outcome.saved_fulltext, outcome.saved_chunks) == (0, 0)
    assert outcome.chunk_count == first.chunk_count
    assert await _state(pool) == (existing_source, first.content_hash, first.chunk_count)
    assert await _chunk_ids(probe) == ids_before
    assert await probe.scalar("SELECT upvotes FROM papers WHERE arxiv_id = %s", (ARXIV_ID,)) == 40


async def test_lower_ranked_source_is_saved_when_no_chunks_are_stored(pool, kit) -> None:
    await prepare_paper(_candidate(kit), _context(pool, kit))
    await fulltexts.save_fulltext(pool, ARXIV_ID, text="old", sections=[], source="layout_pdf")
    await pool_execute(pool, "DELETE FROM paper_chunks WHERE arxiv_id = %s", (ARXIV_ID,))

    outcome = await prepare_paper(_candidate(kit), _context(pool, kit, kit.FakeParser(source="pdf")))

    assert outcome.skipped_lower_rank_overwrite is False
    assert (outcome.saved_fulltext, outcome.saved_chunks > 0) == (1, True)
    assert (await _state(pool) or ("", None, 0))[0] == "pdf"


async def pool_execute(pool: Any, query: str, params: tuple[Any, ...]) -> None:
    async with pool.connection() as connection:
        await connection.execute(query, params)


async def test_force_overwrites_with_a_lower_ranked_source(pool, kit) -> None:
    await prepare_paper(_candidate(kit), _context(pool, kit))

    outcome = await prepare_paper(_candidate(kit), _context(pool, kit, kit.FakeParser(source="pdf")), force=True)

    assert outcome.skipped_lower_rank_overwrite is False
    assert outcome.saved_fulltext == 1
    assert (await _state(pool) or ("", None, 0))[0] == "pdf"


async def test_unchanged_content_skips_fulltext_and_chunk_writes(pool, probe, kit) -> None:
    first = await prepare_paper(_candidate(kit), _context(pool, kit))
    updated_before = await probe.scalar("SELECT updated_at FROM paper_fulltexts WHERE arxiv_id = %s", (ARXIV_ID,))

    second = await prepare_paper(_candidate(kit), _context(pool, kit))

    assert second.skipped_unchanged is True
    assert (second.saved_fulltext, second.saved_chunks) == (0, 0)
    assert second.chunk_count == first.chunk_count
    assert await probe.scalar("SELECT updated_at FROM paper_fulltexts WHERE arxiv_id = %s", (ARXIV_ID,)) == (
        updated_before
    )


async def test_force_rewrites_unchanged_content_but_keeps_identical_chunk_ids(pool, probe, kit) -> None:
    await prepare_paper(_candidate(kit), _context(pool, kit))
    ids_before = await _chunk_ids(probe)

    forced = await prepare_paper(_candidate(kit), _context(pool, kit), force=True)

    assert forced.skipped_unchanged is False
    assert forced.saved_fulltext == 1
    assert forced.chunks_unchanged is True
    assert forced.saved_chunks == 0
    assert forced.chunk_count == 0
    assert await _chunk_ids(probe) == ids_before


async def test_same_hash_from_a_different_source_is_rewritten(pool, kit) -> None:
    await prepare_paper(_candidate(kit), _context(pool, kit, kit.FakeParser(source="pdf")))

    outcome = await prepare_paper(_candidate(kit), _context(pool, kit, kit.FakeParser(source="layout_pdf")))

    assert outcome.skipped_unchanged is False
    assert outcome.saved_fulltext == 1
    assert (await _state(pool) or ("", None, 0))[0] == "layout_pdf"


async def test_changed_text_replaces_chunks_and_their_embeddings(pool, probe, kit) -> None:
    await prepare_paper(_candidate(kit), _context(pool, kit))
    ids_before = await _chunk_ids(probe)
    changed = kit.FakeParser(sections_by_id={ARXIV_ID: kit.body_sections(ARXIV_ID, extra=" A new sentence.")})

    outcome = await prepare_paper(_candidate(kit), _context(pool, kit, changed))

    assert outcome.saved_chunks > 0
    assert set(await _chunk_ids(probe)).isdisjoint(ids_before)


async def test_chunk_write_failure_leaves_the_hash_unset_so_the_retry_rewrites(pool, kit, monkeypatch) -> None:
    original = fulltexts.save_chunks
    calls = {"count": 0}

    async def flaky(db: Any, arxiv_id: str, chunks: Any) -> bool:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("connection lost")
        return await original(db, arxiv_id, chunks)

    monkeypatch.setattr(fulltexts, "save_chunks", flaky)

    with pytest.raises(RuntimeError, match="connection lost"):
        await prepare_paper(_candidate(kit), _context(pool, kit))
    assert await _state(pool) == ("layout_pdf", None, 0)

    retried = await prepare_paper(_candidate(kit), _context(pool, kit))
    assert retried.skipped_unchanged is False
    assert retried.saved_chunks > 0
    assert await _state(pool) == ("layout_pdf", retried.content_hash, retried.chunk_count)

    again = await prepare_paper(_candidate(kit), _context(pool, kit))
    assert again.skipped_unchanged is True


async def test_download_failure_falls_back_to_the_abstract(pool, kit) -> None:
    outcome = await prepare_paper(
        _candidate(kit, summary="Fallback abstract text."),
        _context(pool, kit, pdfs=kit.FakePdfSource(missing={ARXIV_ID})),
    )

    assert outcome.fulltext_source == "fallback_abstract"
    assert outcome.fallback_used is True
    assert await _state(pool) == ("fallback_abstract", outcome.content_hash, 1)


async def test_surrogates_are_stripped_before_hashing_and_saving(pool, probe, kit) -> None:
    sections = [{"title": "Intro \ud835", "text": "Body \ud835 text with enough words to chunk."}]
    parser = kit.FakeParser(sections_by_id={ARXIV_ID: sections})

    outcome = await prepare_paper(_candidate(kit), _context(pool, kit, parser))

    stored = await probe.row("SELECT text, sections FROM paper_fulltexts WHERE arxiv_id = %s", (ARXIV_ID,))
    assert stored is not None
    assert "\ud835" not in stored["text"]
    assert stored["sections"] == [{"title": "Intro ", "text": "Body  text with enough words to chunk."}]
    assert outcome.content_hash == compute_content_hash(stored["text"], stored["sections"])


async def test_date_run_isolates_paper_failures_and_aggregates(pool, kit) -> None:
    payload = [kit.hf_item(f"2609.0000{number}") for number in range(1, 4)]
    await raw.save_daily_papers(pool, target_date=DAY, payload=payload)
    parser = kit.FakeParser(failing_ids={"2609.00002"})

    result = await prepare_date(DAY, _context(pool, kit, parser))

    assert result.status == "partial_failed"
    assert (result.success_count, result.failure_count) == (2, 1)
    assert result.prepared_arxiv_ids == ["2609.00001", "2609.00003"]
    assert result.recorded_failures == [
        {"arxiv_id": "2609.00002", "error": "RuntimeError: parser exploded on 2609.00002"}
    ]
    job_result = result.job_result()
    assert list(job_result) == [
        "saved_papers",
        "saved_fulltexts",
        "saved_chunks",
        "fallback_fulltexts",
        "skipped_lower_rank_overwrites",
        "skipped_unchanged",
        "unchanged_chunk_sets",
        "selected_candidate_count",
        "prepared_arxiv_count",
        "success_count",
        "failure_count",
        "failures",
    ]
    assert job_result["saved_papers"] == 2
    assert job_result["selected_candidate_count"] == 3
    assert job_result["prepared_arxiv_count"] == 2
    assert result.failure_summary() == (
        "1 of 3 paper(s) failed (success=2, failure=1); first error: RuntimeError: parser exploded on 2609.00002"
    )


async def test_date_without_raw_payload_is_an_empty_success(pool, kit) -> None:
    result = await prepare_date(DAY, _context(pool, kit))
    assert (result.status, result.success_count, result.failure_count) == ("success", 0, 0)
    assert result.job_result()["selected_candidate_count"] == 0


async def test_a_parse_failure_leaves_no_paper_row(pool, probe, kit) -> None:
    parser = kit.FakeParser(failing_ids={ARXIV_ID})

    with pytest.raises(RuntimeError, match="parser exploded"):
        await prepare_paper(_candidate(kit), _context(pool, kit, parser))

    assert await probe.scalar("SELECT count(*) FROM papers") == 0


async def test_an_hf_row_still_replaces_hf_metadata(pool, probe, kit) -> None:
    await prepare_paper(_candidate(kit, title="First title"), _context(pool, kit))

    await prepare_paper(_candidate(kit, title="Second title", upvotes=9), _context(pool, kit))

    row = await probe.row("SELECT title, upvotes, source FROM papers WHERE arxiv_id = %s", (ARXIV_ID,))
    assert row == {"title": "Second title", "upvotes": 9, "source": "hf_daily_papers_raw"}
