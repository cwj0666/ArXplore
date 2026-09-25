from __future__ import annotations

from typing import Any

import requests

from src.integrations.paper_repository import PaperRepository
from src.integrations.paper_search import PaperSearchClient

from .tracing import build_pipeline_trace_config


def _merge_existing_and_enriched(existing: dict[str, Any], enriched: dict[str, Any]) -> dict[str, Any]:
    return {
        "arxiv_id": existing["arxiv_id"],
        "title": enriched.get("title") or existing.get("title", ""),
        "authors": enriched.get("authors") or existing.get("authors", []),
        "abstract": enriched.get("abstract") or existing.get("abstract", ""),
        "primary_category": enriched.get("primary_category") or existing.get("primary_category"),
        "categories": enriched.get("categories") or existing.get("categories", []),
        "pdf_url": enriched.get("pdf_url") or existing.get("pdf_url"),
        "published_at": enriched.get("published_at") or existing.get("published_at"),
        "updated_at": enriched.get("updated_at") or existing.get("updated_at"),
        "upvotes": existing.get("upvotes", 0),
        "github_url": existing.get("github_url"),
        "github_stars": existing.get("github_stars"),
        "citation_count": existing.get("citation_count"),
        "source": "hf_daily_papers+arxiv",
    }


def _build_soft_fail_result(
    *,
    status: str,
    runtime: str,
    user: str | None,
    candidates: list[dict[str, Any]],
    normalized_limit: int,
    error_message: str,
) -> dict[str, Any]:
    trace_config = build_pipeline_trace_config(
        stage="enrich_papers_metadata",
        runtime=runtime,
        user=user,
        extra_metadata={
            "candidate_count": len(candidates),
            "updated_count": 0,
            "skipped_count": 0,
            "max_papers": normalized_limit,
            "soft_failed": True,
            "soft_fail_status": status,
        },
    )

    return {
        "stage": "enrich_papers_metadata",
        "status": status,
        "candidate_count": len(candidates),
        "updated_count": 0,
        "skipped_count": 0,
        "max_papers": normalized_limit,
        "sample_updated": [],
        "failed_arxiv_ids": [paper["arxiv_id"] for paper in candidates[:5] if paper.get("arxiv_id")],
        "error": error_message,
        "trace_config": trace_config,
    }


def run_enrich_papers_metadata(
    *,
    runtime: str = "airflow",
    user: str | None = None,
    max_papers: int | str | None = 30,
    paper_repository: PaperRepository | None = None,
    search_client: PaperSearchClient | None = None,
) -> dict[str, Any]:
    """primary_category, categories, canonical pdf_url 등 arXiv 메타데이터를 후속 보강한다."""
    normalized_limit = 30 if max_papers in (None, "") else max(1, int(str(max_papers)))
    if paper_repository is None:
        paper_repository = PaperRepository()
        paper_repository.ensure_schema()
    search_client = search_client or PaperSearchClient()

    candidates = paper_repository.list_papers_missing_arxiv_metadata(limit=normalized_limit)
    arxiv_ids = [paper["arxiv_id"] for paper in candidates if paper.get("arxiv_id")]

    try:
        metadata_by_arxiv_id = search_client.fetch_arxiv_metadata(arxiv_ids)
    except requests.HTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code != 429:
            raise
        return _build_soft_fail_result(
            status="rate_limited",
            runtime=runtime,
            user=user,
            candidates=candidates,
            normalized_limit=normalized_limit,
            error_message=str(exc),
        )
    except requests.Timeout as exc:
        return _build_soft_fail_result(
            status="timed_out",
            runtime=runtime,
            user=user,
            candidates=candidates,
            normalized_limit=normalized_limit,
            error_message=str(exc),
        )

    updated_count = 0
    skipped_count = 0
    sample_updated: list[dict[str, Any]] = []

    for paper in candidates:
        arxiv_id = paper["arxiv_id"]
        enriched = metadata_by_arxiv_id.get(arxiv_id)
        if enriched is None:
            skipped_count += 1
            continue

        merged = _merge_existing_and_enriched(paper, enriched)
        paper_repository.save_paper(merged)
        updated_count += 1
        if len(sample_updated) < 5:
            sample_updated.append(
                {
                    "arxiv_id": arxiv_id,
                    "primary_category": merged.get("primary_category"),
                    "category_count": len(merged.get("categories") or []),
                    "pdf_url": merged.get("pdf_url"),
                }
            )

    trace_config = build_pipeline_trace_config(
        stage="enrich_papers_metadata",
        runtime=runtime,
        user=user,
        extra_metadata={
            "candidate_count": len(candidates),
            "updated_count": updated_count,
            "skipped_count": skipped_count,
            "max_papers": normalized_limit,
        },
    )

    return {
        "stage": "enrich_papers_metadata",
        "status": "success",
        "candidate_count": len(candidates),
        "updated_count": updated_count,
        "skipped_count": skipped_count,
        "max_papers": normalized_limit,
        "sample_updated": sample_updated,
        "trace_config": trace_config,
    }
