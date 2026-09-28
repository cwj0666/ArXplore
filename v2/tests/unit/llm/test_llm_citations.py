import asyncio
import logging

import pytest

from arxplore.llm.citations import (
    Citation,
    CitationSource,
    arxiv_id_from_url,
    collect_tool_hits,
    extract_markdown_links,
    extract_source_refs,
    normalize_cited_arxiv_id,
    paper_url,
    record_tool_hits,
    verify_agent_citations,
    verify_numbered_citations,
)

HITS = [
    CitationSource(
        arxiv_id="2401.00001", title="A", url="https://arxiv.org/abs/2401.00001", section_title="1 Intro", chunk_id=1
    ),
    CitationSource(
        arxiv_id="2401.00001", title="A", url="https://arxiv.org/abs/2401.00001", section_title="1 Intro", chunk_id=1
    ),
    CitationSource(arxiv_id="2402.00002", title="B", url="https://arxiv.org/pdf/2402.00002"),
]


def _summary(citations: list[Citation]) -> list[tuple[str, int | None, bool]]:
    return [(citation.arxiv_id, citation.chunk_id, citation.in_answer) for citation in citations]


def test_citation_payload_keeps_the_legacy_field_order() -> None:
    citation = Citation.from_source(
        CitationSource(arxiv_id="x", title="", url="", section_title="  ", chunk_id=None), in_answer=False
    )

    assert list(citation.to_payload().items()) == [
        ("arxiv_id", "x"),
        ("title", "제목 없음"),
        ("url", ""),
        ("section_title", None),
        ("chunk_id", None),
        ("in_answer", False),
    ]


def test_paper_url_prefers_the_pdf_link() -> None:
    assert paper_url("2401.00001", " https://arxiv.org/pdf/2401.00001 ") == "https://arxiv.org/pdf/2401.00001"
    assert paper_url("2401.00001", None) == "https://arxiv.org/abs/2401.00001"


def test_only_linked_hits_are_returned() -> None:
    assert _summary(verify_agent_citations("[A](https://arxiv.org/abs/2401.00001)", HITS)) == [("2401.00001", 1, True)]


def test_link_matches_by_arxiv_id_across_abs_pdf_and_version() -> None:
    citations = verify_agent_citations("[B](https://arxiv.org/abs/2402.00002v2)", HITS)

    assert [citation.arxiv_id for citation in citations] == ["2402.00002"]


def test_link_matches_by_normalized_url() -> None:
    citations = verify_agent_citations("[B](https://ARXIV.ORG/PDF/2402.00002/)", HITS)

    assert [citation.arxiv_id for citation in citations] == ["2402.00002"]


def test_no_links_returns_all_unique_hits_unverified() -> None:
    assert _summary(verify_agent_citations("링크 없는 답변", HITS)) == [
        ("2401.00001", 1, False),
        ("2402.00002", None, False),
    ]


def test_unknown_link_is_logged_and_the_answer_is_not_rewritten(caplog: pytest.LogCaptureFixture) -> None:
    answer = "[Fake](https://arxiv.org/abs/9999.99999) 그리고 [A](https://arxiv.org/abs/2401.00001)"

    with caplog.at_level(logging.WARNING, logger="arxplore.llm.citations"):
        citations = verify_agent_citations(answer, HITS)

    assert [citation.arxiv_id for citation in citations] == ["2401.00001"]
    assert "9999.99999" in caplog.text


def test_lookalike_domain_is_not_attributed_to_an_arxiv_hit() -> None:
    hits = [CitationSource(arxiv_id="2401.00001", title="A", url="https://arxiv.org/abs/2401.00001")]

    assert verify_agent_citations("[A](https://notarxiv.org/abs/2401.00001)", hits) == []


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://arxiv.org/abs/2401.00001", "2401.00001"),
        ("http://export.arxiv.org/pdf/2401.00001v3.pdf", "2401.00001"),
        ("https://www.arxiv.org/abs/2401.00001V2", "2401.00001V2"),
        ("https://arxiv.org/abs/hep-th/9901001", "hep-th/9901001"),
        ("https://notarxiv.org/abs/2401.00001", None),
        ("https://evil.example/?u=https://arxiv.org/abs/2401.00001", None),
        ("https://arxiv.org.evil.example/abs/2401.00001", None),
        ("javascript:arxiv.org/abs/2401.00001", None),
        ("https://arxiv.org/list/cs.CL", None),
    ],
)
def test_arxiv_id_from_url_requires_a_real_arxiv_host(url: str, expected: str | None) -> None:
    assert arxiv_id_from_url(url) == expected


def test_cited_id_normalization() -> None:
    assert normalize_cited_arxiv_id(" 2401.00001v12.PDF ") == "2401.00001"


def test_extract_helpers() -> None:
    assert extract_markdown_links("**[T](https://x.org/a)** [bad](ftp://x)") == [("T", "https://x.org/a")]
    assert extract_source_refs("근거 [2], [1, 3] 그리고 [링크](https://a) [2] [4\uff0c5] [6](x)") == [2, 1, 3, 4, 5]


SOURCES = [
    CitationSource(arxiv_id="p", title="T", url="u", section_title="Abstract", chunk_id=None),
    CitationSource(arxiv_id="p", title="T", url="u", section_title="3 Method", chunk_id=10),
    CitationSource(arxiv_id="p", title="T", url="u", section_title="4 Results", chunk_id=11),
]


def test_numbered_citations_keep_source_order() -> None:
    assert _summary(verify_numbered_citations("근거는 [3]과 [1]입니다.", SOURCES)) == [
        ("p", None, True),
        ("p", 11, True),
    ]


def test_answer_without_numbers_returns_every_source_unverified(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="arxplore.llm.citations"):
        citations = verify_numbered_citations("번호 없는 답변 [9]", SOURCES)

    assert _summary(citations) == [("p", None, False), ("p", 10, False), ("p", 11, False)]
    assert "[9]" in caplog.text


def test_tool_hits_are_collected_only_inside_the_block() -> None:
    record_tool_hits([HITS[0]])
    with collect_tool_hits() as hits:
        record_tool_hits([HITS[2]])
    record_tool_hits([HITS[1]])

    assert hits == [HITS[2]]


async def test_tool_hits_are_shared_with_tasks_started_inside_the_block() -> None:
    async def tool() -> None:
        record_tool_hits([HITS[0]])

    with collect_tool_hits() as hits:
        await asyncio.gather(tool(), tool())

    assert hits == [HITS[0], HITS[0]]
