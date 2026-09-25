from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import date as date_cls
from datetime import timedelta
from typing import Any

from src.integrations.fulltext_parser import FulltextParser
from src.integrations.paper_repository import PaperRepository
from src.integrations.paper_search import PaperSearchClient
from src.integrations.prepare_job_repository import PrepareJobRepository
from src.integrations.raw_store import RawPaperStore

from .tracing import build_pipeline_trace_config

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_CATEGORIES = {"cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.RO", "stat.ML"}
FALLBACK_ABSTRACT_SOURCE = "fallback_abstract"
FULLTEXT_SOURCE_RANK = {"layout_pdf": 3, "pdf": 2, FALLBACK_ABSTRACT_SOURCE: 1}
MAX_RECORDED_FAILURES = 50
MAX_FAILURE_ERROR_CHARS = 500


class PrepareClaimLostError(RuntimeError):
    """처리 중 prepare job claim을 잃었을 때(재선점·stale reset) 남은 논문 처리를 중단하기 위해 사용한다."""


def _sum_result_values(results: list[dict[str, Any]], key: str) -> int:
    return sum(int(result.get(key, 0) or 0) for result in results)


def _build_sample_prepared(results: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "arxiv_id": result.get("arxiv_id"),
            "title": result.get("title", ""),
            "primary_category": result.get("primary_category"),
            "chunk_count": result.get("chunk_count", 0),
            "fulltext_source": result.get("fulltext_source"),
            "fallback_used": result.get("fallback_used"),
            "section_count": result.get("section_count", 0),
            "text_length": result.get("text_length", 0),
        }
        for result in results[:limit]
    ]


def _normalize_optional_date(value: str | None) -> str | None:
    normalized = (value or "").strip()
    return normalized or None


def _normalize_optional_positive_int(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    normalized = str(value).strip()
    if not normalized:
        return None
    parsed = int(normalized)
    return parsed if parsed > 0 else None


def _extract_candidate_arxiv_id(item: dict[str, Any]) -> str | None:
    paper = item.get("paper")
    if isinstance(paper, dict):
        for key in ("id", "arxivId", "arxiv_id"):
            value = paper.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("id", "arxivId", "arxiv_id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_hf_signals(item: dict[str, Any]) -> dict[str, Any]:
    paper = item.get("paper")
    if not isinstance(paper, dict):
        paper = {}
    upvotes = item.get("upvotes", paper.get("upvotes", 0))
    github = item.get("github", paper.get("github", {}))
    github_url = paper.get("githubRepo")
    github_stars = paper.get("githubStars")
    if isinstance(github, dict):
        github_url = github.get("url") or github.get("repoUrl") or github.get("repository") or github_url
        github_stars = github.get("stars") if github.get("stars") is not None else github_stars
    return {
        "upvotes": int(upvotes or 0),
        "github_url": github_url,
        "github_stars": int(github_stars) if github_stars is not None else None,
    }


def _extract_hf_authors(item: dict[str, Any]) -> list[str]:
    paper = item.get("paper")
    if not isinstance(paper, dict):
        return []
    authors = paper.get("authors")
    if not isinstance(authors, list):
        return []

    names: list[str] = []
    for author in authors:
        if not isinstance(author, dict):
            continue
        name = str(author.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _build_hf_pdf_url(arxiv_id: str) -> str:
    normalized = arxiv_id.strip()
    if not normalized:
        return ""
    return f"https://arxiv.org/pdf/{normalized}.pdf"


def _build_prepared_from_hf_item(arxiv_id: str, hf_item: dict[str, Any]) -> dict[str, Any]:
    paper = hf_item.get("paper")
    if not isinstance(paper, dict):
        paper = {}

    hf_signals = _extract_hf_signals(hf_item)
    return {
        "arxiv_id": arxiv_id,
        "title": str(paper.get("title") or "").strip(),
        "authors": _extract_hf_authors(hf_item),
        "abstract": str(paper.get("summary") or "").strip(),
        "primary_category": None,
        "categories": [],
        "pdf_url": _build_hf_pdf_url(arxiv_id),
        "published_at": paper.get("publishedAt"),
        "updated_at": paper.get("publishedAt"),
        **hf_signals,
        "source": "hf_daily_papers_raw",
    }


def _rebuild_fulltext_text_from_sections(sections: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for section in sections:
        title = str(section.get("title") or "").strip()
        text = str(section.get("text") or "").strip()
        if not text:
            continue
        if title in {"", "Front Matter", "Full Text"}:
            parts.append(text)
        else:
            parts.append(f"{title}\n{text}")
    return "\n\n".join(parts).strip()


def _repair_parsed_sections_with_metadata(fulltext, prepared: dict[str, Any]) -> None:
    metadata_abstract = " ".join(str(prepared.get("abstract") or "").split())
    if not metadata_abstract or not fulltext.sections:
        return

    repaired = False
    for section in fulltext.sections:
        title = str(section.get("title") or "").strip().lower()
        text = " ".join(str(section.get("text") or "").split())
        if title != "abstract" or not text:
            continue
        if text[0].islower():
            section["text"] = metadata_abstract
            repaired = True

    if not repaired:
        return

    fulltext.text = _rebuild_fulltext_text_from_sections(fulltext.sections) or fulltext.text
    fulltext.quality_metrics["text_length"] = len(fulltext.text)
    fulltext.quality_metrics["section_count"] = len(fulltext.sections)
    section_lengths = [len(str(section.get("text") or "")) for section in fulltext.sections]
    fulltext.quality_metrics["avg_section_chars"] = (
        round(sum(section_lengths) / len(section_lengths), 2) if section_lengths else 0
    )
    fulltext.quality_metrics["max_section_chars"] = max(section_lengths) if section_lengths else 0


def _is_allowed_category(metadata: dict[str, Any], *, allowed: set[str]) -> bool:
    categories = metadata.get("categories") or []
    if isinstance(categories, list):
        category_set = {str(value) for value in categories if value}
    else:
        category_set = set()

    primary = metadata.get("primary_category")
    if primary:
        category_set.add(str(primary))

    return any(category in allowed for category in category_set)


def load_prepare_candidates(
    target_date: str | None,
    max_papers: int | str | None,
    *,
    metadata_mode: str = "arxiv",
    allowed_categories: set[str] | None = None,
    raw_store: RawPaperStore | None = None,
    search_client: PaperSearchClient | None = None,
) -> dict[str, Any]:
    """전처리 후보 논문과 실행 메타데이터를 로드한다."""
    normalized_target_date = _normalize_optional_date(target_date)
    normalized_date = (
        date_cls.fromisoformat(normalized_target_date).isoformat() if normalized_target_date else date_cls.today().isoformat()
    )
    normalized_max_papers = _normalize_optional_positive_int(max_papers)
    allowed = allowed_categories or DEFAULT_ALLOWED_CATEGORIES

    raw_store = raw_store or RawPaperStore()
    search_client = search_client or PaperSearchClient()

    raw_payload = raw_store.load_daily_papers_response(date=normalized_date)
    raw_count = len(raw_payload)

    raw_ids = [
        candidate
        for item in raw_payload
        if isinstance(item, dict)
        for candidate in [_extract_candidate_arxiv_id(item)]
        if candidate
    ]
    normalized_ids = [search_client.normalize_arxiv_id(value) for value in raw_ids]
    deduplicated_ids = list(dict.fromkeys(normalized_ids))
    selected_ids = deduplicated_ids[:normalized_max_papers]
    hf_by_id: dict[str, dict[str, Any]] = {}
    for item in raw_payload:
        if not isinstance(item, dict):
            continue
        candidate = _extract_candidate_arxiv_id(item)
        if not candidate:
            continue
        normalized = search_client.normalize_arxiv_id(candidate)
        hf_by_id[normalized] = item

    candidates: list[dict[str, Any]] = []
    skipped_by_category = 0
    metadata_by_arxiv_id: dict[str, dict[str, Any]] = {}

    if metadata_mode == "arxiv":
        metadata_by_arxiv_id = search_client.fetch_arxiv_metadata(selected_ids)
        for arxiv_id, metadata in metadata_by_arxiv_id.items():
            if not _is_allowed_category(metadata, allowed=allowed):
                skipped_by_category += 1
                continue

            hf_item = hf_by_id.get(arxiv_id, {})
            hf_signals = _extract_hf_signals(hf_item)
            prepared = {
                **metadata,
                **hf_signals,
                "source": "hf_daily_papers+arxiv",
            }
            candidates.append(
                {
                    "arxiv_id": arxiv_id,
                    "prepared": prepared,
                }
            )
    elif metadata_mode == "hf_raw":
        for arxiv_id in selected_ids:
            hf_item = hf_by_id.get(arxiv_id)
            if not hf_item:
                continue
            candidates.append(
                {
                    "arxiv_id": arxiv_id,
                    "prepared": _build_prepared_from_hf_item(arxiv_id, hf_item),
                }
            )
    else:
        raise ValueError(f"지원하지 않는 metadata_mode입니다: {metadata_mode}")

    return {
        "metadata_mode": metadata_mode,
        "normalized_date": normalized_date,
        "raw_count": raw_count,
        "deduplicated_ids": deduplicated_ids,
        "selected_ids": selected_ids,
        "enriched_count": len(metadata_by_arxiv_id),
        "skipped_by_category": skipped_by_category,
        "candidates": candidates,
    }


def fulltext_source_rank(source: str | None) -> int:
    return FULLTEXT_SOURCE_RANK.get(str(source or ""), 0)


def compute_fulltext_content_hash(text: str, sections: list[dict[str, Any]]) -> str:
    """공백을 정규화한 본문과 섹션 제목 순서열의 sha256."""
    payload = {
        "text": " ".join(str(text or "").split()),
        "section_titles": [" ".join(str(section.get("title") or "").split()) for section in sections or []],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def prepare_single_paper(
    candidate: dict[str, Any],
    *,
    parser: FulltextParser | None = None,
    paper_repository: PaperRepository | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """후보 논문 1건을 저장, 파싱, 청크 생성까지 수행한다.

    force가 아니면 기존 본문보다 낮은 순위(layout_pdf > pdf > fallback_abstract)의 결과로 덮어쓰지 않고,
    source와 content_hash가 기존과 같고 청크가 저장돼 있으면 본문·청크 교체를 건너뛴다.
    content_hash는 청크 저장이 끝난 뒤에 기록한다.
    """
    parser = parser or FulltextParser()
    paper_repository = paper_repository or PaperRepository()

    arxiv_id = str(candidate.get("arxiv_id") or "").strip()
    prepared = candidate.get("prepared")
    if not arxiv_id:
        raise ValueError("candidate['arxiv_id']는 비어 있을 수 없습니다.")
    if not isinstance(prepared, dict):
        raise ValueError("candidate['prepared']는 dict여야 합니다.")

    paper_repository.save_paper(prepared)

    fulltext = parser.parse_from_pdf_url(
        prepared.get("pdf_url") or "",
        fallback_text=prepared.get("abstract") or "",
    )
    _repair_parsed_sections_with_metadata(fulltext, prepared)
    chunks = parser.build_chunks(
        fulltext.text or prepared.get("abstract", ""),
        sections=fulltext.sections,
    )
    fulltext_quality_metrics = {
        **fulltext.quality_metrics,
        **parser.summarize_chunks(chunks),
    }

    # 청크 DELETE는 임베딩까지 CASCADE 삭제하므로 일시적인 파서 장애로 인한 하향 교체와 동일 내용 재적재를 막는다.
    content_hash = compute_fulltext_content_hash(fulltext.text, fulltext.sections)
    existing = paper_repository.get_paper_fulltext_state(arxiv_id)
    existing_source = existing.get("source") if existing else None
    skipped_lower_rank_overwrite = False
    skipped_unchanged = False
    existing_chunk_count = int(existing.get("chunk_count") or 0) if existing else 0
    if existing and not force:
        if fulltext_source_rank(fulltext.source) < fulltext_source_rank(existing_source) and existing_chunk_count > 0:
            skipped_lower_rank_overwrite = True
        elif (
            existing_source == fulltext.source
            and existing.get("content_hash") == content_hash
            and existing_chunk_count > 0
        ):
            skipped_unchanged = True

    saved_fulltext = 0
    saved_chunks = 0
    chunks_unchanged = False
    if not (skipped_lower_rank_overwrite or skipped_unchanged):
        if fulltext.text:
            paper_repository.save_paper_fulltext(
                arxiv_id,
                text=fulltext.text,
                sections=fulltext.sections,
                source=fulltext.source,
                quality_metrics=fulltext_quality_metrics,
                artifacts=fulltext.artifacts,
                parser_metadata=fulltext.parser_metadata,
                content_hash=None,
            )
            saved_fulltext = 1
        if chunks:
            if paper_repository.save_paper_chunks(arxiv_id, chunks):
                saved_chunks = len(chunks)
            else:
                chunks_unchanged = True
        # 청크 저장 전에 실패하면 hash가 NULL로 남아 재시도 때 unchanged로 건너뛰지 않는다.
        if saved_fulltext:
            paper_repository.update_paper_fulltext_content_hash(arxiv_id, content_hash)

    result = {
        "arxiv_id": arxiv_id,
        "title": prepared.get("title", ""),
        "primary_category": prepared.get("primary_category"),
        "chunk_count": len(chunks) if (saved_chunks or skipped_unchanged) else (existing_chunk_count if skipped_lower_rank_overwrite else 0),
        "fulltext_source": fulltext.source,
        "fallback_used": fulltext_quality_metrics.get("fallback_used"),
        "section_count": fulltext_quality_metrics.get("section_count", 0),
        "text_length": fulltext_quality_metrics.get("text_length", 0),
        "saved_paper": 1,
        "saved_fulltext": saved_fulltext,
        "saved_chunks": saved_chunks,
        "content_hash": content_hash,
        "artifacts": fulltext.artifacts,
        "parser_metadata": fulltext.parser_metadata,
        "quality_metrics": fulltext_quality_metrics,
    }
    if skipped_lower_rank_overwrite:
        result["skipped_lower_rank_overwrite"] = True
        result["existing_fulltext_source"] = existing_source
    if skipped_unchanged:
        result["skipped_unchanged"] = True
    if chunks_unchanged:
        result["chunks_unchanged"] = True
    return result


def _format_failure_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message[:MAX_FAILURE_ERROR_CHARS]


def prepare_candidates(
    candidates: list[dict[str, Any]],
    *,
    parser: FulltextParser,
    paper_repository: PaperRepository,
    heartbeat: Callable[[], None] | None = None,
    force: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """후보를 1건씩 처리하고, 한 논문의 예외가 나머지 처리를 막지 않도록 실패를 수집한다.

    heartbeat는 논문마다 처리 전에 호출되며, 여기서 발생한 예외(PrepareClaimLostError 등)는 격리하지 않고 전파한다.
    """
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for candidate in candidates:
        if heartbeat is not None:
            heartbeat()
        try:
            results.append(
                prepare_single_paper(
                    candidate,
                    parser=parser,
                    paper_repository=paper_repository,
                    force=force,
                )
            )
        except Exception as exc:
            arxiv_id = str(candidate.get("arxiv_id") or "").strip() if isinstance(candidate, dict) else ""
            logger.exception("prepare_single_paper failed: arxiv_id=%s", arxiv_id)
            failures.append({"arxiv_id": arxiv_id, "error": _format_failure_error(exc)})
    return results, failures


def aggregate_prepare_results(
    results: list[dict[str, Any]],
    *,
    normalized_date: str,
    raw_count: int,
    deduplicated_ids: list[str],
    selected_ids: list[str],
    enriched_count: int,
    skipped_by_category: int,
    runtime: str,
    user: str | None,
    failures: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """prepare_single_paper 결과를 현재 파이프라인 반환 구조로 집계한다.

    status는 실패가 있고 성공이 0건이면 failed, 일부 실패면 partial_failed, 그 외 success다.
    """
    failures = failures or []
    success_count = len(results)
    failure_count = len(failures)
    if failure_count and not success_count:
        status = "failed"
    elif failure_count:
        status = "partial_failed"
    else:
        status = "success"
    skipped_lower_rank_overwrites = sum(1 for result in results if result.get("skipped_lower_rank_overwrite"))
    skipped_unchanged = sum(1 for result in results if result.get("skipped_unchanged"))
    unchanged_chunk_sets = sum(1 for result in results if result.get("chunks_unchanged"))
    fallback_fulltexts = sum(1 for result in results if result.get("fallback_used"))
    saved_papers = _sum_result_values(results, "saved_paper")
    saved_fulltexts = _sum_result_values(results, "saved_fulltext")
    saved_chunks = _sum_result_values(results, "saved_chunks")
    prepared_arxiv_ids = [str(result.get("arxiv_id") or "").strip() for result in results if result.get("arxiv_id")]

    return {
        "stage": "prepare_papers",
        "status": status,
        "target_date": normalized_date,
        "raw_payload_count": raw_count,
        "arxiv_candidate_count": len(deduplicated_ids),
        "selected_candidate_count": len(selected_ids),
        "enriched_count": enriched_count,
        "saved_papers": saved_papers,
        "saved_fulltexts": saved_fulltexts,
        "saved_chunks": saved_chunks,
        "prepared_arxiv_ids": prepared_arxiv_ids,
        "skipped_by_category": skipped_by_category,
        "fallback_fulltexts": fallback_fulltexts,
        "skipped_lower_rank_overwrites": skipped_lower_rank_overwrites,
        "skipped_unchanged": skipped_unchanged,
        "unchanged_chunk_sets": unchanged_chunk_sets,
        "success_count": success_count,
        "failure_count": failure_count,
        "failures": failures[:MAX_RECORDED_FAILURES],
        "sample_prepared": _build_sample_prepared(results),
        "trace_config": build_pipeline_trace_config(
            stage="prepare_papers",
            runtime=runtime,
            user=user,
            extra_metadata={
                "target_date": normalized_date,
                "raw_payload_count": raw_count,
                "selected_candidate_count": len(selected_ids),
                "saved_papers": saved_papers,
                "saved_chunks": saved_chunks,
                "fallback_fulltexts": fallback_fulltexts,
                "prepared_arxiv_count": len(prepared_arxiv_ids),
                "success_count": success_count,
                "failure_count": failure_count,
            },
        ),
    }


def run_prepare_papers(
    *,
    runtime: str = "airflow",
    user: str | None = None,
    target_date: str | None = None,
    allowed_categories: set[str] | None = None,
    max_papers: int | str | None = None,
    heartbeat: Callable[[], None] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """원본 payload를 읽어 arXiv 보강/본문 파싱/청크 생성/적재를 수행한다. force는 prepare_single_paper 참고."""
    prepare_context = load_prepare_candidates(
        target_date=target_date,
        max_papers=max_papers,
        metadata_mode="hf_raw",
        allowed_categories=allowed_categories,
    )
    paper_repository = PaperRepository()
    parser = FulltextParser()

    results, failures = prepare_candidates(
        prepare_context["candidates"],
        parser=parser,
        paper_repository=paper_repository,
        heartbeat=heartbeat,
        force=force,
    )

    return aggregate_prepare_results(
        results,
        failures=failures,
        normalized_date=prepare_context["normalized_date"],
        raw_count=prepare_context["raw_count"],
        deduplicated_ids=prepare_context["deduplicated_ids"],
        selected_ids=prepare_context["selected_ids"],
        enriched_count=prepare_context["enriched_count"],
        skipped_by_category=prepare_context["skipped_by_category"],
        runtime=runtime,
        user=user,
    )


def _build_prepare_backfill_dates(
    *,
    cursor_date: date_cls,
    oldest_date: date_cls,
    batch_days: int,
) -> list[date_cls]:
    dates: list[date_cls] = []
    current = cursor_date
    while current >= oldest_date and len(dates) < batch_days:
        dates.append(current)
        current -= timedelta(days=1)
    return dates


def _resolve_prepare_oldest_date(
    *,
    oldest_date: str | None,
    existing_state: dict[str, Any] | None,
    today: date_cls,
) -> date_cls:
    if oldest_date:
        return date_cls.fromisoformat(oldest_date)
    if existing_state and existing_state.get("oldest_date"):
        return date_cls.fromisoformat(str(existing_state["oldest_date"]))
    return today - timedelta(days=365)


def _resolve_prepare_cursor_date(
    *,
    cursor_date: str | None,
    existing_state: dict[str, Any] | None,
    today: date_cls,
) -> date_cls:
    if cursor_date:
        return date_cls.fromisoformat(cursor_date)
    if existing_state and existing_state.get("cursor_date"):
        return date_cls.fromisoformat(str(existing_state["cursor_date"]))
    return today - timedelta(days=1)


def _summarize_paper_failures(result: dict[str, Any]) -> str:
    failures = result.get("failures") or []
    failure_count = int(result.get("failure_count", len(failures)) or 0)
    success_count = int(result.get("success_count", 0) or 0)
    first_error = str(failures[0].get("error") or "") if failures else ""
    if success_count:
        headline = f"{failure_count} of {failure_count + success_count} paper(s) failed"
    else:
        headline = f"all {failure_count} paper(s) failed"
    return (
        f"{headline} (success={success_count}, failure={failure_count}); first error: {first_error}"
    )[:MAX_FAILURE_ERROR_CHARS]


def run_backfill_prepare_papers(
    *,
    runtime: str = "local",
    user: str | None = None,
    cursor_date: str | None = None,
    oldest_date: str | None = None,
    batch_days: int = 7,
    state_name: str = "default",
    max_papers: int | str | None = None,
    allowed_categories: set[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """로컬 GPU 환경에서 prepare_papers를 날짜 배치 단위로 순차 실행한다."""
    today = date_cls.today()
    raw_store = RawPaperStore()

    existing_state = raw_store.load_pipeline_state(
        pipeline="prepare_papers_backfill",
        name=state_name,
    )
    normalized_oldest_date = _resolve_prepare_oldest_date(
        oldest_date=oldest_date,
        existing_state=existing_state,
        today=today,
    )
    normalized_cursor_date = _resolve_prepare_cursor_date(
        cursor_date=cursor_date,
        existing_state=existing_state,
        today=today,
    )
    normalized_batch_days = max(1, int(batch_days or 7))

    if normalized_cursor_date < normalized_oldest_date:
        state = {
            "cursor_date": None,
            "oldest_date": normalized_oldest_date.isoformat(),
            "batch_days": normalized_batch_days,
            "status": "completed",
            "last_processed_dates": [],
            "last_failure": None,
        }
        raw_store.save_pipeline_state(
            pipeline="prepare_papers_backfill",
            name=state_name,
            state=state,
        )
        return {
            "stage": "backfill_prepare_papers",
            "status": "completed",
            "state_name": state_name,
            "cursor_date": normalized_cursor_date.isoformat(),
            "next_cursor_date": None,
            "oldest_date": normalized_oldest_date.isoformat(),
            "batch_days": normalized_batch_days,
            "success_count": 0,
            "skipped_missing_raw_count": 0,
            "failure_count": 0,
            "successes": [],
            "skipped_missing_raw": [],
            "failures": [],
            "stopped_reason": "cursor_exhausted",
            "trace_config": build_pipeline_trace_config(
                stage="backfill_prepare_papers",
                runtime=runtime,
                user=user,
            ),
        }

    target_dates = _build_prepare_backfill_dates(
        cursor_date=normalized_cursor_date,
        oldest_date=normalized_oldest_date,
        batch_days=normalized_batch_days,
    )
    successes: list[dict[str, Any]] = []
    skipped_missing_raw: list[str] = []
    failures: list[dict[str, str]] = []
    next_cursor_date: str | None = None
    stopped_reason = "batch_complete"

    for target in target_dates:
        target_str = target.isoformat()
        next_cursor_candidate = target - timedelta(days=1)

        if not raw_store.has_daily_papers_response(date=target_str):
            skipped_missing_raw.append(target_str)
            next_cursor_date = (
                next_cursor_candidate.isoformat() if next_cursor_candidate >= normalized_oldest_date else None
            )
            continue

        try:
            result = run_prepare_papers(
                runtime=runtime,
                user=user,
                target_date=target_str,
                max_papers=max_papers,
                allowed_categories=allowed_categories,
                force=force,
            )
        except Exception as exc:
            failures.append({"date": target_str, "error": str(exc)})
            stopped_reason = "prepare_failed"
            next_cursor_date = target_str
            break

        paper_failure_count = int(result.get("failure_count", 0) or 0)
        if paper_failure_count > 0 or result.get("status") in {"failed", "partial_failed"}:
            failures.append(
                {
                    "date": target_str,
                    "status": str(result.get("status") or "failed"),
                    "error": _summarize_paper_failures(result),
                    "paper_success_count": int(result.get("success_count", 0) or 0),
                    "paper_failure_count": paper_failure_count,
                    "paper_failures": list(result.get("failures") or []),
                    "prepared_arxiv_ids": [
                        str(value) for value in result.get("prepared_arxiv_ids", []) if str(value).strip()
                    ],
                }
            )
            stopped_reason = "prepare_failed"
            next_cursor_date = target_str
            break

        successes.append(
            {
                "date": target_str,
                "saved_papers": int(result.get("saved_papers", 0) or 0),
                "saved_fulltexts": int(result.get("saved_fulltexts", 0) or 0),
                "saved_chunks": int(result.get("saved_chunks", 0) or 0),
                "fallback_fulltexts": int(result.get("fallback_fulltexts", 0) or 0),
                "selected_candidate_count": int(result.get("selected_candidate_count", 0) or 0),
                "prepared_arxiv_ids": [str(value) for value in result.get("prepared_arxiv_ids", []) if str(value).strip()],
            }
        )
        next_cursor_date = next_cursor_candidate.isoformat() if next_cursor_candidate >= normalized_oldest_date else None

    status = "failed" if failures else ("completed" if next_cursor_date is None else "success")
    state = {
        "cursor_date": next_cursor_date,
        "oldest_date": normalized_oldest_date.isoformat(),
        "batch_days": normalized_batch_days,
        "status": status,
        "last_processed_dates": [item["date"] for item in successes] + skipped_missing_raw,
        "last_failure": failures[0] if failures else None,
    }
    raw_store.save_pipeline_state(
        pipeline="prepare_papers_backfill",
        name=state_name,
        state=state,
    )

    trace_config = build_pipeline_trace_config(
        stage="backfill_prepare_papers",
        runtime=runtime,
        user=user,
        extra_metadata={
            "state_name": state_name,
            "cursor_date": normalized_cursor_date.isoformat(),
            "next_cursor_date": next_cursor_date,
            "oldest_date": normalized_oldest_date.isoformat(),
            "batch_days": normalized_batch_days,
            "success_count": len(successes),
            "skipped_missing_raw_count": len(skipped_missing_raw),
            "failure_count": len(failures),
            "stopped_reason": stopped_reason,
        },
    )

    return {
        "stage": "backfill_prepare_papers",
        "status": status,
        "state_name": state_name,
        "cursor_date": normalized_cursor_date.isoformat(),
        "next_cursor_date": next_cursor_date,
        "oldest_date": normalized_oldest_date.isoformat(),
        "batch_days": normalized_batch_days,
        "success_count": len(successes),
        "skipped_missing_raw_count": len(skipped_missing_raw),
        "failure_count": len(failures),
        "successes": successes,
        "skipped_missing_raw": skipped_missing_raw,
        "failures": failures,
        "stopped_reason": stopped_reason,
        "trace_config": trace_config,
    }


def _build_job_heartbeat(
    prepare_job_repository: PrepareJobRepository,
    claim_token: dict[str, Any],
) -> Callable[[], None]:
    def heartbeat() -> None:
        try:
            alive = prepare_job_repository.heartbeat_prepare_job(**claim_token)
        except Exception:
            logger.warning("prepare job heartbeat failed: job_id=%s", claim_token["job_id"], exc_info=True)
            return
        if not alive:
            raise PrepareClaimLostError(f"prepare job claim lost: job_id={claim_token['job_id']}")

    return heartbeat


def _record_lost_claim(
    lost_claims: list[dict[str, Any]],
    *,
    target_date: str,
    claim_token: dict[str, Any],
    stage: str,
) -> None:
    logger.warning(
        "prepare job claim lost at %s: job_id=%s date=%s generation=%s",
        stage,
        claim_token["job_id"],
        target_date,
        claim_token["claim_generation"],
    )
    lost_claims.append(
        {
            "date": target_date,
            "job_id": claim_token["job_id"],
            "claim_generation": claim_token["claim_generation"],
            "stage": stage,
        }
    )


def run_consume_prepare_queue(
    *,
    runtime: str = "local",
    user: str | None = None,
    mode: str = "auto",
    worker_id: str = "local_prepare_worker",
    max_jobs_per_run: int = 1,
    max_papers: int | str | None = None,
    allowed_categories: set[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """prepare 큐를 소비해 날짜별 파싱/청킹 적재를 수행한다.

    complete/fail은 claim 토큰(job_id, worker_id, claim_generation)으로 fencing되며, claim을 잃은 작업은
    예외 없이 lost_claims에 기록된다. 논문이 1건이라도 실패하면 fail_prepare_job으로 backoff 재시도에 넘기고,
    재시도에서는 내용이 바뀌지 않은 성공분이 건너뛰어진다. force는 인자 또는 job payload의 "force"로 켠다.
    """
    prepare_job_repository = PrepareJobRepository()
    normalized_max_jobs = max(1, int(max_jobs_per_run or 1))
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    lost_claims: list[dict[str, Any]] = []
    claimed_dates: list[str] = []

    for _ in range(normalized_max_jobs):
        job = prepare_job_repository.claim_prepare_job(mode=mode, worker_id=worker_id)
        if not job:
            break

        claim_token = {
            "job_id": job["job_id"],
            "worker_id": job.get("worker_id") or worker_id,
            "claim_generation": int(job.get("claim_generation") or 0),
        }
        target_date = str(job.get("date") or "").strip()
        if not target_date:
            if not prepare_job_repository.fail_prepare_job(**claim_token, error="missing target_date"):
                _record_lost_claim(lost_claims, target_date="", claim_token=claim_token, stage="fail")
            failures.append({"date": "", "job_id": claim_token["job_id"], "error": "missing target_date"})
            continue
        claimed_dates.append(target_date)
        job_payload = job.get("payload")
        job_force = force or (isinstance(job_payload, dict) and job_payload.get("force") is True)
        try:
            result = run_prepare_papers(
                runtime=runtime,
                user=user,
                target_date=target_date,
                max_papers=max_papers,
                allowed_categories=allowed_categories,
                heartbeat=_build_job_heartbeat(prepare_job_repository, claim_token),
                force=job_force,
            )
            job_result = {
                "saved_papers": int(result.get("saved_papers", 0) or 0),
                "saved_fulltexts": int(result.get("saved_fulltexts", 0) or 0),
                "saved_chunks": int(result.get("saved_chunks", 0) or 0),
                "fallback_fulltexts": int(result.get("fallback_fulltexts", 0) or 0),
                "skipped_lower_rank_overwrites": int(result.get("skipped_lower_rank_overwrites", 0) or 0),
                "skipped_unchanged": int(result.get("skipped_unchanged", 0) or 0),
                "unchanged_chunk_sets": int(result.get("unchanged_chunk_sets", 0) or 0),
                "selected_candidate_count": int(result.get("selected_candidate_count", 0) or 0),
                "prepared_arxiv_count": len([str(value) for value in result.get("prepared_arxiv_ids", []) if str(value).strip()]),
                "success_count": int(result.get("success_count", 0) or 0),
                "failure_count": int(result.get("failure_count", 0) or 0),
                "failures": list(result.get("failures") or []),
            }
            if job_result["failure_count"] > 0 or result.get("status") == "failed":
                error_summary = _summarize_paper_failures(result)
                if not prepare_job_repository.fail_prepare_job(**claim_token, error=error_summary):
                    _record_lost_claim(lost_claims, target_date=target_date, claim_token=claim_token, stage="fail")
                failures.append(
                    {
                        "date": target_date,
                        "error": error_summary,
                        "prepared_arxiv_ids": [
                            str(value) for value in result.get("prepared_arxiv_ids", []) if str(value).strip()
                        ],
                    }
                )
                continue
            if not prepare_job_repository.complete_prepare_job(**claim_token, result=job_result):
                _record_lost_claim(lost_claims, target_date=target_date, claim_token=claim_token, stage="complete")
                continue
        except PrepareClaimLostError:
            _record_lost_claim(lost_claims, target_date=target_date, claim_token=claim_token, stage="heartbeat")
            continue
        except Exception as exc:
            if not prepare_job_repository.fail_prepare_job(**claim_token, error=str(exc)):
                _record_lost_claim(lost_claims, target_date=target_date, claim_token=claim_token, stage="fail")
            failures.append({"date": target_date, "error": str(exc)})
            continue

        successes.append(
            {
                "date": target_date,
                "saved_papers": int(result.get("saved_papers", 0) or 0),
                "saved_fulltexts": int(result.get("saved_fulltexts", 0) or 0),
                "saved_chunks": int(result.get("saved_chunks", 0) or 0),
                "fallback_fulltexts": int(result.get("fallback_fulltexts", 0) or 0),
                "selected_candidate_count": int(result.get("selected_candidate_count", 0) or 0),
                "prepared_arxiv_ids": [str(value) for value in result.get("prepared_arxiv_ids", []) if str(value).strip()],
                "paper_success_count": job_result["success_count"],
                "paper_failure_count": job_result["failure_count"],
                "paper_failures": job_result["failures"],
            }
        )

    if not claimed_dates and not failures and not lost_claims:
        status = "no_op"
    elif failures and not successes:
        status = "failed"
    elif failures or (lost_claims and successes):
        status = "partial_failed"
    elif lost_claims:
        status = "claim_lost"
    else:
        status = "success"

    trace_config = build_pipeline_trace_config(
        stage="consume_prepare_queue",
        runtime=runtime,
        user=user,
        extra_metadata={
            "mode": mode,
            "worker_id": worker_id,
            "claimed_dates": claimed_dates,
            "success_count": len(successes),
            "failure_count": len(failures),
            "lost_claim_count": len(lost_claims),
        },
    )

    return {
        "stage": "consume_prepare_queue",
        "status": status,
        "mode": mode,
        "worker_id": worker_id,
        "claimed_dates": claimed_dates,
        "claimed_count": len(claimed_dates),
        "success_count": len(successes),
        "failure_count": len(failures),
        "lost_claim_count": len(lost_claims),
        "successes": successes,
        "failures": failures,
        "lost_claims": lost_claims,
        "trace_config": trace_config,
    }
