import json
from typing import Any

from fastapi import Request
from starlette.responses import JSONResponse, Response

from arxplore.api.errors import INVALID_REQUEST_MESSAGE, ApiError, InvalidRequestError
from arxplore.api.services.detail import RelatedPaper, published_at_text
from arxplore.domain.papers import Paper
from arxplore.llm.citations import Citation

RELATION_SCORE_DIGITS = 3


def error_response(status_code: int, message: str, *, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code, headers=headers)


def api_error_response(error: ApiError) -> JSONResponse:
    return error_response(error.status_code, error.message)


def empty_response(status_code: int, *, headers: dict[str, str] | None = None) -> Response:
    return Response(status_code=status_code, headers=headers, media_type="text/html")


async def json_object(request: Request) -> dict[str, Any]:
    try:
        payload = json.loads(await request.body())
    except Exception as exc:
        raise InvalidRequestError(INVALID_REQUEST_MESSAGE) from exc
    if not isinstance(payload, dict):
        raise InvalidRequestError(INVALID_REQUEST_MESSAGE)
    return payload


def paper_item(paper: Paper) -> dict[str, Any]:
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title or "",
        "authors": paper.authors or [],
        "abstract": paper.abstract or "",
        "published_at": published_at_text(paper),
        "upvotes": paper.upvotes or 0,
        "pdf_url": paper.pdf_url,
    }


def related_item(paper: RelatedPaper) -> dict[str, Any]:
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title or "",
        "authors": paper.authors or [],
        "abstract": paper.abstract or "",
        "published_at": paper.published_at,
        "upvotes": paper.upvotes or 0,
        "pdf_url": paper.pdf_url,
        "source": paper.source or "local",
        "relation_score": round(float(paper.relation_score or 0), RELATION_SCORE_DIGITS),
    }


def citation_items(citations: tuple[Citation, ...] | list[Citation]) -> list[dict[str, Any]]:
    return [citation.to_payload() for citation in citations]
