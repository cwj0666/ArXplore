import json
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from arxplore.api.app import create_app
from arxplore.api.commands import register, render_openapi
from arxplore.config import AppSettings

EXPECTED_OPERATIONS = {
    ("/bootstrap.json", "get"): "bootstrap",
    ("/papers/list.json", "get"): "listPapers",
    ("/papers/{arxiv_id}/detail.json", "get"): "getPaperDetail",
    ("/papers/{arxiv_id}/analyze/", "post"): "analyzePaper",
    ("/papers/{arxiv_id}/summary/", "post"): "summarizePaper",
    ("/papers/{arxiv_id}/chat/", "post"): "chatWithPaper",
    ("/papers/{arxiv_id}/chat/stream/", "post"): "streamPaperChat",
    ("/papers/assistant/chat/", "post"): "chatWithAssistant",
    ("/papers/assistant/stream/", "post"): "streamAssistantChat",
}


@pytest.fixture
def schema() -> dict[str, Any]:
    return create_app(AppSettings.model_validate({})).openapi()


def test_every_route_has_a_stable_operation_id(schema: dict[str, Any]) -> None:
    operations = {
        (path, method): operation["operationId"]
        for path, methods in schema["paths"].items()
        for method, operation in methods.items()
    }

    assert operations == EXPECTED_OPERATIONS


def test_request_bodies_and_stream_events_are_components(schema: dict[str, Any]) -> None:
    components = schema["components"]["schemas"]
    summary = schema["paths"]["/papers/{arxiv_id}/summary/"]["post"]["requestBody"]
    chat = schema["paths"]["/papers/assistant/stream/"]["post"]

    assert summary["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/SummaryRequest"}
    assert chat["requestBody"]["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/ChatRequest"}
    assert "text/event-stream" in chat["responses"]["200"]["content"]
    assert {"StreamChunkEvent", "StreamCitationsEvent", "StreamErrorEvent", "ChatHistoryMessage"} <= set(components)
    assert components["ChatRequest"]["properties"]["history"]["items"] == {
        "$ref": "#/components/schemas/ChatHistoryMessage"
    }


def test_nullable_fields_match_the_responses(schema: dict[str, Any]) -> None:
    components = schema["components"]["schemas"]

    def nullable(model: str, field: str) -> bool:
        return {"type": "null"} in components[model]["properties"][field].get("anyOf", [])

    assert nullable("PaperListItem", "published_at")
    assert nullable("PaperListItem", "pdf_url")
    assert nullable("CitationItem", "chunk_id")
    assert nullable("AnalysisResponse", "overview")
    assert not nullable("PaperListItem", "title")
    assert components["RateLimitedResponse"]["required"] == ["error", "retry_after"]


def test_validation_errors_are_not_documented(schema: dict[str, Any]) -> None:
    assert not {"HTTPValidationError", "ValidationError"} & set(schema["components"]["schemas"])
    assert all(
        "422" not in operation["responses"] for methods in schema["paths"].values() for operation in methods.values()
    )


def test_openapi_command_writes_sorted_deterministic_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cli = typer.Typer()
    register(cli)
    target = tmp_path / "out" / "openapi.json"

    result = CliRunner().invoke(cli, ["openapi", "--output", str(target)])

    assert result.exit_code == 0, result.output
    document = target.read_text(encoding="utf-8")
    assert document == render_openapi()
    assert document.endswith("}\n")
    parsed = json.loads(document)
    assert list(parsed) == sorted(parsed)
    assert parsed["info"]["title"] == "ArXplore API"
