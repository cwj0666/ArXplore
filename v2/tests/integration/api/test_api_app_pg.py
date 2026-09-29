import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import psycopg
import pytest
from fastapi import FastAPI

from arxplore.api.app import create_app
from arxplore.config import AppSettings
from arxplore.db.migrate import SchemaNotReadyError
from arxplore.llm.detail import ANALYSIS_PROMPT_VERSION
from arxplore.llm.summary_graph import SUMMARY_PROMPT_VERSION

pytestmark = pytest.mark.integration

MAIN_ID = "2401.00001"
OTHER_ID = "2401.00002"


@asynccontextmanager
async def serve(settings: AppSettings) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, client=("198.51.100.7", 50000))
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            yield client


async def post(client: httpx.AsyncClient, path: str, payload: Any = None) -> httpx.Response:
    token = (await client.get("/bootstrap.json")).cookies["csrftoken"]
    client.cookies.clear()
    content = None if payload is None else json.dumps(payload).encode()
    return await client.post(path, content=content, headers={"cookie": f"csrftoken={token}", "x-csrftoken": token})


def parse_sse(body: str) -> list[Any]:
    return [
        block.removeprefix("data: ") if block == "data: [DONE]" else json.loads(block.removeprefix("data: "))
        for block in body.split("\n\n")
        if block
    ]


async def test_startup_refuses_an_unmigrated_database(
    empty_database: str, database_settings: Callable[..., AppSettings]
) -> None:
    app = create_app(database_settings(OPENAI_API_KEY="sk-unused"))

    with pytest.raises(SchemaNotReadyError, match="pending migrations"):
        async with app.router.lifespan_context(app):
            pass


async def test_list_and_detail_read_the_database(api_database: str, api_settings: Callable[..., AppSettings]) -> None:
    async with serve(api_settings()) as client:
        listing = (await client.get("/papers/list.json?q=preference&sort=upvotes")).json()
        detail = (await client.get(f"/papers/{MAIN_ID}v2/detail.json")).json()

    assert [item["arxiv_id"] for item in listing["items"]] == [OTHER_ID, MAIN_ID]
    assert listing["items"][1]["published_at"].startswith("2024-01-0")
    assert detail["paper"]["arxiv_id"] == MAIN_ID
    related = detail["paper"]["related_papers"]
    assert related[0]["arxiv_id"] == OTHER_ID
    assert related[0]["source"] == "local"
    assert [paper["source"] for paper in related[1:]] == ["arxiv", "arxiv"]


async def test_generated_overview_and_summary_are_cached_in_postgres(
    api_database: str, api_settings: Callable[..., AppSettings], fake_openai: Any
) -> None:
    fake_openai.chat_handler = lambda request: (
        "- 첫 번째 핵심 포인트입니다" if "핵심 포인트" in request["messages"][0]["content"] else "생성된 결과"
    )

    async with serve(api_settings()) as client:
        overview = (await post(client, f"/papers/{MAIN_ID}/analyze/")).json()
        summary = (await post(client, f"/papers/{MAIN_ID}/summary/", {"model": "gpt-5"})).json()
    async with serve(api_settings(OPENAI_API_KEY="  ")) as client:
        cached_overview = (await post(client, f"/papers/{MAIN_ID}/analyze/")).json()
        cached_summary = (await post(client, f"/papers/{MAIN_ID}/summary/", {"model": "gpt-5"})).json()
        other_model = await post(client, f"/papers/{MAIN_ID}/summary/", {"model": "gpt-5-mini"})

    assert overview == {"overview": "생성된 결과", "key_findings": ["첫 번째 핵심 포인트입니다."], "cached": False}
    assert cached_overview == {**overview, "cached": True}
    assert summary == {"summary": "생성된 결과", "cached": False, "model": "gpt-5"}
    assert cached_summary == {**summary, "cached": True}
    assert other_model.status_code == 503
    async with await psycopg.AsyncConnection.connect(api_database) as connection:
        cursor = await connection.execute("SELECT model FROM paper_ai_overviews WHERE arxiv_id = %s", (MAIN_ID,))
        assert await cursor.fetchall() == [("gpt-5-mini",)]


async def test_results_cached_before_the_version_bump_are_regenerated(
    api_database: str, api_settings: Callable[..., AppSettings], fake_openai: Any
) -> None:
    fake_openai.chat_handler = lambda request: "새로 만든 결과"
    async with await psycopg.AsyncConnection.connect(api_database, autocommit=True) as connection:
        await connection.execute(
            "INSERT INTO paper_ai_overviews (arxiv_id, overview, model) VALUES (%s, 'old', 'gpt-5-mini')", (MAIN_ID,)
        )
        await connection.execute(
            "INSERT INTO paper_ai_detailed_summaries (arxiv_id, model, summary) VALUES (%s, 'gpt-5', 'old')", (MAIN_ID,)
        )

    async with serve(api_settings(OPENAI_API_KEY="  ")) as client:
        stale_overview = await post(client, f"/papers/{MAIN_ID}/analyze/")
        stale_summary = await post(client, f"/papers/{MAIN_ID}/summary/", {"model": "gpt-5"})
    async with serve(api_settings()) as client:
        overview = (await post(client, f"/papers/{MAIN_ID}/analyze/")).json()
        summary = (await post(client, f"/papers/{MAIN_ID}/summary/", {"model": "gpt-5"})).json()

    assert (stale_overview.status_code, stale_summary.status_code) == (503, 503)
    assert (overview["overview"], overview["cached"]) == ("새로 만든 결과", False)
    assert (summary["summary"], summary["cached"]) == ("새로 만든 결과", False)
    async with await psycopg.AsyncConnection.connect(api_database) as connection:
        overviews = await connection.execute(
            "SELECT overview, prompt_version FROM paper_ai_overviews WHERE arxiv_id = %s", (MAIN_ID,)
        )
        summaries = await connection.execute(
            "SELECT summary, prompt_version FROM paper_ai_detailed_summaries WHERE arxiv_id = %s", (MAIN_ID,)
        )
        assert await overviews.fetchall() == [("새로 만든 결과", ANALYSIS_PROMPT_VERSION)]
        assert await summaries.fetchall() == [("새로 만든 결과", SUMMARY_PROMPT_VERSION)]


async def test_paper_chat_streams_over_lexical_retrieval(
    api_database: str, api_settings: Callable[..., AppSettings], fake_openai: Any
) -> None:
    fake_openai.chat_handler = lambda request: "대조 손실은 [2]에 있습니다."

    async with serve(api_settings()) as client:
        response = await post(client, f"/papers/{MAIN_ID}/chat/stream/", {"message": "contrastive loss margin"})

    events = parse_sse(response.text)
    citations = events[-2]["citations"]
    assert "".join(event["chunk"] for event in events[:-2]) == "대조 손실은 [2]에 있습니다."
    assert [(c["section_title"], c["in_answer"]) for c in citations] == [("3 Method", True)]
    assert events[-1] == "[DONE]"
    prompt = fake_openai.requests_for("/v1/chat/completions")[0].body["messages"][-1]["content"]
    assert "[1] 섹션: Abstract | chunk_id: -\nWe revisit direct preference optimization for alignment." in prompt


async def test_agent_search_tool_reads_the_database(
    fakes: Any, api_database: str, api_settings: Callable[..., AppSettings], fake_openai: Any
) -> None:
    call = fakes.ToolCall(name="search_paper_chunks_tool", arguments=json.dumps({"query": "reference policy margin"}))
    fake_openai.chat_handler = lambda request: (
        f"[Direct Preference Optimization Revisited](https://arxiv.org/abs/{MAIN_ID})"
        if request["messages"][-1]["role"] == "tool"
        else call
    )

    async with serve(api_settings()) as client:
        payload = (await post(client, "/papers/assistant/chat/", {"message": "DPO 논문"})).json()

    assert payload["citations"][0]["arxiv_id"] == MAIN_ID
    assert payload["citations"][0]["in_answer"] is True
    tool_output = fake_openai.requests_for("/v1/chat/completions")[-1].body["messages"][-1]["content"]
    assert tool_output.startswith(f"[1] 제목: Direct Preference Optimization Revisited | arxiv_id: {MAIN_ID}")
