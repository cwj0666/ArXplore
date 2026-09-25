from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from django.conf import settings
from django.contrib.auth import authenticate, get_user_model, login, logout
from django.contrib.auth.models import AbstractBaseUser, AnonymousUser
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.http import HttpRequest

from src.shared import get_settings, override_openai_runtime

from .models import DEFAULT_SUMMARY_MODEL, FavoritePaper, UserSettings
from .secret_box import InvalidToken, decrypt_secret, encrypt_secret

logger = logging.getLogger(__name__)

MAX_RECENT_PAPERS = 1500
PAPERS_PER_PAGE = 21
PAPER_CHUNK_LIMIT = 20
CHAT_HISTORY_MAX_MESSAGES = 20
CHAT_MESSAGE_MAX_CHARS = 4000
RELATED_PAPER_LIMIT = 5
RELATED_PAPER_CANDIDATE_LIMIT = 300
VALID_SEARCH_MODES = {"search", "ai"}
SESSION_API_KEY_KEY = "personal_openai_api_key"
OVERVIEW_MODEL = "gpt-5-mini"
AVAILABLE_SUMMARY_MODELS = ("gpt-5-mini", "gpt-5")


class PaperNotFoundError(LookupError):
    pass


class MissingApiKeyError(RuntimeError):
    pass


class AuthenticationRequiredError(PermissionError):
    pass


class InvalidRequestError(ValueError):
    """사용자에게 그대로 보여줘도 되는 입력 검증 오류."""


class InvalidSummaryModelError(InvalidRequestError):
    pass


class PasswordValidationError(InvalidRequestError):
    def __init__(self, messages: list[str]):
        super().__init__(" ".join(messages))
        self.messages = messages


@lru_cache(maxsize=1)
def get_paper_repository():
    from src.integrations.paper_repository import PaperRepository

    return PaperRepository()


def build_paper_list_payload(
    *, query: str, sort: str, mode: str, page: Any, user: AbstractBaseUser | AnonymousUser
) -> dict[str, Any]:
    papers = get_paper_repository().list_recent_papers(limit=MAX_RECENT_PAPERS)
    normalized_query = query.strip()
    normalized_mode = mode if mode in VALID_SEARCH_MODES else "search"
    normalized_sort = "upvotes" if sort == "upvotes" else "latest"

    if normalized_query:
        lowered_query = normalized_query.lower()
        papers = [
            paper
            for paper in papers
            if lowered_query in (paper.get("title") or "").lower()
            or lowered_query in (paper.get("abstract") or "").lower()
        ]

    if normalized_sort == "upvotes":
        papers.sort(key=lambda paper: paper.get("upvotes") or 0, reverse=True)

    paginator = Paginator(papers, PAPERS_PER_PAGE)
    page_obj = paginator.get_page(_parse_page_number(page))
    favorite_ids = _get_favorite_ids(user, [paper.get("arxiv_id") for paper in page_obj.object_list])

    return {
        "items": [_serialize_paper_for_list(paper, favorite_ids=favorite_ids) for paper in page_obj.object_list],
        "page": page_obj.number,
        "page_size": page_obj.paginator.per_page,
        "total_items": page_obj.paginator.count,
        "total_pages": page_obj.paginator.num_pages,
        "query": normalized_query,
        "sort": normalized_sort,
        "mode": normalized_mode,
    }


def build_paper_detail_payload(arxiv_id: str, *, user: AbstractBaseUser | AnonymousUser) -> dict[str, Any]:
    if not demo_mode_enabled():
        _require_authenticated_user(user)
    paper = _get_paper_or_raise(arxiv_id)
    related_papers = _build_related_papers(paper)
    favorite_ids = _get_favorite_ids(
        user,
        [paper.get("arxiv_id"), *(related_paper.get("arxiv_id") for related_paper in related_papers)],
    )
    return {
        "paper": {
            **_serialize_paper_for_detail(paper, favorite_ids=favorite_ids),
            "related_papers": [
                _serialize_related_paper(related_paper, favorite_ids=favorite_ids) for related_paper in related_papers
            ],
        },
    }


def build_bootstrap_payload(request: HttpRequest) -> dict[str, Any]:
    user = request.user
    preferred_model = DEFAULT_SUMMARY_MODEL
    username = ""
    if getattr(user, "is_authenticated", False):
        username = user.get_username()
        preferred_model = _get_preferred_summary_model(user)

    demo_mode = demo_mode_enabled()
    return {
        "is_authenticated": bool(getattr(user, "is_authenticated", False)),
        "username": username,
        "has_personal_api_key": has_personal_api_key(request),
        "preferred_summary_model": preferred_model,
        "available_summary_models": list(AVAILABLE_SUMMARY_MODELS),
        "demo_mode": demo_mode,
        "login_required_for": ["chat", "generate"] if demo_mode else ["detail", "chat", "generate"],
    }


def demo_mode_enabled() -> bool:
    return bool(getattr(settings, "DEMO_MODE", True))


def build_settings_payload(user: AbstractBaseUser | AnonymousUser) -> dict[str, Any]:
    _require_authenticated_user(user)
    return {
        "preferred_summary_model": _get_preferred_summary_model(user),
        "available_summary_models": list(AVAILABLE_SUMMARY_MODELS),
    }


def build_favorites_payload(user: AbstractBaseUser | AnonymousUser) -> dict[str, Any]:
    _require_authenticated_user(user)
    favorite_rows = list(
        FavoritePaper.objects.filter(user=user).order_by("-created_at").values_list("arxiv_id", flat=True)
    )
    repo = get_paper_repository()
    items: list[dict[str, Any]] = []
    for arxiv_id in favorite_rows:
        paper = repo.get_paper(arxiv_id)
        if not paper:
            continue
        items.append(_serialize_paper_for_list(paper, favorite_ids={arxiv_id}))
    return {"items": items}


def save_personal_api_key(request: HttpRequest, api_key: str) -> dict[str, Any]:
    _require_authenticated_user(request.user)
    normalized = api_key.strip()
    if not normalized:
        raise InvalidRequestError("API 키를 입력하세요.")
    request.session[SESSION_API_KEY_KEY] = encrypt_secret(normalized)
    request.session.modified = True
    return {"ok": True, "has_personal_api_key": True}


def clear_personal_api_key(request: HttpRequest) -> dict[str, Any]:
    _require_authenticated_user(request.user)
    request.session.pop(SESSION_API_KEY_KEY, None)
    request.session.modified = True
    return {"ok": True, "has_personal_api_key": False}


def get_session_api_key(request: HttpRequest) -> str | None:
    """세션의 암호문을 복호화한다. 복호화할 수 없는 값(이전 평문 저장분, 키 교체)은 지우고 없는 것으로 본다."""
    value = request.session.get(SESSION_API_KEY_KEY)
    if not value:
        return None
    try:
        normalized = decrypt_secret(value).strip()
    except InvalidToken:
        logger.warning("세션 API 키를 복호화할 수 없어 삭제합니다.")
        request.session.pop(SESSION_API_KEY_KEY, None)
        request.session.modified = True
        return None
    return normalized or None


def has_personal_api_key(request: HttpRequest) -> bool:
    return bool(get_session_api_key(request))


def register_user(*, username: str, password: str) -> dict[str, Any]:
    normalized_username = username.strip()
    if not normalized_username:
        raise InvalidRequestError("사용자 이름을 입력하세요.")
    if not password:
        raise InvalidRequestError("비밀번호를 입력하세요.")

    user_model = get_user_model()
    if user_model.objects.filter(username=normalized_username).exists():
        raise InvalidRequestError("이미 존재하는 사용자 이름입니다.")
    try:
        validate_password(password, user=user_model(username=normalized_username))
    except ValidationError as exc:
        raise PasswordValidationError([str(message) for message in exc.messages]) from exc

    user = user_model.objects.create_user(username=normalized_username, password=password)
    _get_or_create_user_settings(user)
    return {"ok": True, "username": user.get_username()}


def login_user(request: HttpRequest, *, username: str, password: str) -> dict[str, Any]:
    normalized_username = username.strip()
    if not normalized_username or not password:
        raise InvalidRequestError("사용자 이름과 비밀번호를 입력하세요.")

    user = authenticate(request, username=normalized_username, password=password)
    if user is None:
        raise InvalidRequestError("로그인에 실패했습니다.")

    login(request, user)
    _get_or_create_user_settings(user)
    return {"ok": True, "username": user.get_username()}


def logout_user(request: HttpRequest) -> dict[str, Any]:
    request.session.pop(SESSION_API_KEY_KEY, None)
    logout(request)
    return {"ok": True}


def build_auth_payload(user: AbstractBaseUser | AnonymousUser) -> dict[str, Any]:
    if not getattr(user, "is_authenticated", False):
        return {"is_authenticated": False, "username": ""}
    return {"is_authenticated": True, "username": user.get_username()}


def update_user_settings(*, user: AbstractBaseUser | AnonymousUser, preferred_summary_model: str) -> dict[str, Any]:
    _require_authenticated_user(user)
    normalized_model = preferred_summary_model.strip()
    if normalized_model not in AVAILABLE_SUMMARY_MODELS:
        raise InvalidSummaryModelError("지원하지 않는 모델입니다.")

    settings_obj = _get_or_create_user_settings(user)
    settings_obj.preferred_summary_model = normalized_model
    settings_obj.save(update_fields=["preferred_summary_model", "updated_at"])
    return {"ok": True, "preferred_summary_model": normalized_model}


def toggle_favorite_paper(*, user: AbstractBaseUser | AnonymousUser, arxiv_id: str) -> dict[str, Any]:
    _require_authenticated_user(user)
    _get_paper_or_raise(arxiv_id)
    favorite, created = FavoritePaper.objects.get_or_create(user=user, arxiv_id=arxiv_id)
    if created:
        return {"is_favorited": True}

    favorite.delete()
    return {"is_favorited": False}


def get_paper_analysis(
    arxiv_id: str,
    *,
    user: AbstractBaseUser | AnonymousUser,
    session_api_key: str | None,
) -> dict[str, Any]:
    """데모 모드에서는 캐시된 overview를 로그인·키 없이 반환한다. 새로 생성할 때만 로그인과 개인 키가 필요하다."""
    if not demo_mode_enabled():
        _require_generation_access(user, session_api_key)

    repo = get_paper_repository()
    paper = _get_paper_or_raise(arxiv_id)
    cached = repo.get_paper_overview(arxiv_id)

    if cached and cached.get("overview"):
        return {
            "overview": cached["overview"],
            "key_findings": cached.get("key_findings") or [],
            "cached": True,
        }

    api_key = _require_generation_access(user, session_api_key)

    fulltext = repo.get_paper_fulltext(arxiv_id) or {}
    chunks = repo.list_paper_chunks(arxiv_id, limit=PAPER_CHUNK_LIMIT)
    enriched_paper = dict(paper)
    enriched_paper["text"] = fulltext.get("text") or ""
    enriched_paper["sections"] = fulltext.get("sections") or []
    enriched_paper["chunks"] = chunks

    from src.core import analyze_paper_detail

    with override_openai_runtime(api_key=api_key, model=OVERVIEW_MODEL):
        summary_doc = analyze_paper_detail(enriched_paper, user=user.get_username())
    overview = summary_doc.overview if summary_doc else None
    key_findings = summary_doc.key_findings if summary_doc else []

    if overview:
        repo.upsert_paper_overview(arxiv_id, overview, key_findings, OVERVIEW_MODEL)

    return {
        "overview": overview,
        "key_findings": key_findings,
        "cached": False,
    }


def get_paper_summary(
    arxiv_id: str,
    *,
    model: str,
    user: AbstractBaseUser | AnonymousUser,
    session_api_key: str | None,
) -> dict[str, Any]:
    """데모 모드에서는 요청한 모델의 캐시된 요약을 로그인·키 없이 반환한다."""
    if not demo_mode_enabled():
        _require_generation_access(user, session_api_key)
    normalized_model = model.strip()
    if normalized_model not in AVAILABLE_SUMMARY_MODELS:
        raise InvalidSummaryModelError("지원하지 않는 모델입니다.")

    repo = get_paper_repository()
    paper = dict(_get_paper_or_raise(arxiv_id))
    cached = repo.get_detailed_summary(arxiv_id, normalized_model)

    if cached and cached.get("summary"):
        return {
            "summary": cached["summary"],
            "cached": True,
            "model": normalized_model,
        }

    api_key = _require_generation_access(user, session_api_key)

    fulltext = repo.get_paper_fulltext(arxiv_id) or {}
    paper["text"] = fulltext.get("text") or ""
    paper["sections"] = fulltext.get("sections") or []

    from src.core.translation_chains import build_summary

    summary_text = paper.get("text") or "\n\n".join(
        f"[{section.get('title', '')}]\n{section.get('text', '')}" for section in paper.get("sections", [])
    )
    with override_openai_runtime(api_key=api_key, model=normalized_model):
        result = build_summary(
            title=paper["title"],
            authors=paper["authors"],
            text=summary_text,
            sections=paper.get("sections"),
            user=user.get_username(),
        )

    if result:
        repo.upsert_detailed_summary(
            arxiv_id,
            result,
            normalized_model,
            created_by_user_id=getattr(user, "id", None),
        )

    return {
        "summary": result,
        "cached": False,
        "model": normalized_model,
    }


@dataclass(frozen=True)
class PreparedPaperChat:
    api_key: str = field(repr=False)
    message: str
    history: list[tuple[str, str]]
    username: str
    paper: dict[str, Any]


def prepare_paper_chat(
    arxiv_id: str,
    user_message: str,
    chat_history: Any,
    *,
    user: AbstractBaseUser | AnonymousUser,
    session_api_key: str | None,
) -> PreparedPaperChat:
    _require_authenticated_user(user)
    api_key = _require_personal_api_key(session_api_key)
    message, history = _validate_chat_input(user_message, chat_history)
    paper = _get_paper_or_raise(arxiv_id)
    return PreparedPaperChat(
        api_key=api_key,
        message=message,
        history=history,
        username=user.get_username(),
        paper=dict(paper),
    )


def stream_paper_chat(prepared: PreparedPaperChat):
    """`{"chunk"}` 이벤트들과 마지막 `{"citations"}` 이벤트를 내보낸다."""
    with override_openai_runtime(api_key=prepared.api_key):
        retrieval = _retrieve_paper_chat_sources(prepared)
        yield from _stream_paper_chat_answer(prepared, retrieval)


def answer_paper_chat(
    arxiv_id: str,
    user_message: str,
    chat_history: Any,
    *,
    user: AbstractBaseUser | AnonymousUser,
    session_api_key: str | None,
) -> dict[str, Any]:
    prepared = prepare_paper_chat(
        arxiv_id,
        user_message,
        chat_history,
        user=user,
        session_api_key=session_api_key,
    )
    with override_openai_runtime(api_key=prepared.api_key):
        retrieval = _retrieve_paper_chat_sources(prepared)
        answer, citations = _collect_stream(_stream_paper_chat_answer(prepared, retrieval))
    return {
        "answer": answer,
        "citations": citations,
        "retrieval_mode": retrieval.retrieval_mode,
    }


def _retrieve_paper_chat_sources(prepared: PreparedPaperChat):
    from src.core.agent.paper_chat import retrieve_paper_chat_sources
    from src.integrations.paper_retriever import PaperRetriever

    return retrieve_paper_chat_sources(
        prepared.paper,
        prepared.message,
        retriever=PaperRetriever(repository=get_paper_repository()),
    )


def _stream_paper_chat_answer(prepared: PreparedPaperChat, retrieval):
    from src.core.agent.paper_chat import stream_paper_chat_answer

    return stream_paper_chat_answer(
        prepared.message,
        paper=prepared.paper,
        retrieval=retrieval,
        chat_history=prepared.history,
        runtime=_trace_runtime(),
        user=prepared.username,
    )


def answer_agent_chat(
    user_message: str,
    chat_history: Any,
    *,
    user: AbstractBaseUser | AnonymousUser,
    session_api_key: str | None,
) -> dict[str, Any]:
    prepared = prepare_agent_chat(
        user_message,
        chat_history,
        user=user,
        session_api_key=session_api_key,
    )
    answer, citations = _collect_stream(stream_agent_chat(prepared))
    return {"answer": answer or "답변을 생성할 수 없습니다.", "citations": citations}


@dataclass(frozen=True)
class PreparedAgentChat:
    api_key: str = field(repr=False)
    message: str
    history: list[tuple[str, str]]
    username: str


def prepare_agent_chat(
    user_message: str,
    chat_history: Any,
    *,
    user: AbstractBaseUser | AnonymousUser,
    session_api_key: str | None,
) -> PreparedAgentChat:
    _require_authenticated_user(user)
    api_key = _require_personal_api_key(session_api_key)
    message, history = _validate_chat_input(user_message, chat_history)
    return PreparedAgentChat(
        api_key=api_key,
        message=message,
        history=history,
        username=user.get_username(),
    )


def stream_agent_chat(prepared: PreparedAgentChat):
    """`{"chunk"}` 이벤트들과 마지막 `{"citations"}` 이벤트를 내보낸다."""
    from src.core.agent.chatbot import stream_agent_search

    with override_openai_runtime(api_key=prepared.api_key):
        yield from stream_agent_search(
            prepared.message,
            chat_history=prepared.history,
            runtime=_trace_runtime(),
            user=prepared.username,
        )


def _collect_stream(events) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    citations: list[dict[str, Any]] = []
    for event in events:
        if "chunk" in event:
            parts.append(event["chunk"])
        elif "citations" in event:
            citations = event["citations"]
    return "".join(parts), citations


def _trace_runtime() -> str:
    from src.core.tracing import resolve_trace_runtime

    return resolve_trace_runtime(get_settings().app_runtime_mode)


def _validate_chat_input(user_message: str, chat_history: Any) -> tuple[str, list[tuple[str, str]]]:
    cleaned_message = user_message.strip()
    if not cleaned_message:
        raise InvalidRequestError("메시지를 입력하세요.")
    if len(cleaned_message) > CHAT_MESSAGE_MAX_CHARS:
        raise InvalidRequestError(f"메시지는 {CHAT_MESSAGE_MAX_CHARS:,}자 이하로 입력하세요.")
    if not isinstance(chat_history, list):
        raise InvalidRequestError("잘못된 요청입니다.")
    return cleaned_message, _build_history_tuples(chat_history, current_message=cleaned_message)


def _get_paper_or_raise(arxiv_id: str) -> dict[str, Any]:
    repo = get_paper_repository()
    paper = repo.get_paper(arxiv_id)
    if not paper:
        from src.integrations.paper_search import PaperSearchClient

        normalized_arxiv_id = PaperSearchClient.normalize_arxiv_id(arxiv_id)
        if normalized_arxiv_id != arxiv_id:
            paper = repo.get_paper(normalized_arxiv_id)
    if not paper:
        raise PaperNotFoundError("논문을 찾을 수 없습니다.")
    return paper


def _require_authenticated_user(user: AbstractBaseUser | AnonymousUser) -> None:
    if not getattr(user, "is_authenticated", False):
        raise AuthenticationRequiredError("로그인이 필요합니다.")


def _require_personal_api_key(session_api_key: str | None) -> str:
    normalized = (session_api_key or "").strip()
    if not normalized:
        raise MissingApiKeyError("개인 API 키를 먼저 등록하세요.")
    return normalized


def _require_generation_access(user: AbstractBaseUser | AnonymousUser, session_api_key: str | None) -> str:
    _require_authenticated_user(user)
    return _require_personal_api_key(session_api_key)


def _parse_page_number(raw_page: Any) -> int:
    try:
        page_number = int(raw_page)
    except (TypeError, ValueError):
        return 1
    return page_number if page_number > 0 else 1


def _build_history_tuples(
    chat_history: list[dict[str, Any]],
    *,
    current_message: str | None = None,
) -> list[tuple[str, str]]:
    """user/assistant 메시지만 남기고, 최근 CHAT_HISTORY_MAX_MESSAGES개와 메시지당 CHAT_MESSAGE_MAX_CHARS자로 자른다.

    마지막 user 메시지가 current_message와 같으면 현재 질문이 history에도 실려 온 것이므로 뺀다.
    """
    messages = [
        (message["role"], message["content"][:CHAT_MESSAGE_MAX_CHARS])
        for message in chat_history
        if isinstance(message, dict)
        and message.get("role") in ("user", "assistant")
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    if (
        current_message
        and messages
        and messages[-1][0] == "user"
        and messages[-1][1].strip() == current_message.strip()
    ):
        messages.pop()
    return messages[-CHAT_HISTORY_MAX_MESSAGES:]


def _get_preferred_summary_model(user: AbstractBaseUser) -> str:
    stored = UserSettings.objects.filter(user=user).values_list("preferred_summary_model", flat=True).first()
    return stored or DEFAULT_SUMMARY_MODEL


def _get_or_create_user_settings(user: AbstractBaseUser) -> UserSettings:
    settings_obj, _ = UserSettings.objects.get_or_create(
        user=user,
        defaults={"preferred_summary_model": DEFAULT_SUMMARY_MODEL},
    )
    return settings_obj


def _get_favorite_ids(
    user: AbstractBaseUser | AnonymousUser,
    arxiv_ids: list[str | None],
) -> set[str]:
    if not getattr(user, "is_authenticated", False):
        return set()
    normalized_ids = [arxiv_id for arxiv_id in arxiv_ids if arxiv_id]
    if not normalized_ids:
        return set()
    return set(FavoritePaper.objects.filter(user=user, arxiv_id__in=normalized_ids).values_list("arxiv_id", flat=True))


def _serialize_paper_for_list(paper: dict[str, Any], *, favorite_ids: set[str]) -> dict[str, Any]:
    arxiv_id = paper.get("arxiv_id")
    return {
        "arxiv_id": arxiv_id,
        "title": paper.get("title") or "",
        "authors": paper.get("authors") or [],
        "abstract": paper.get("abstract") or "",
        "published_at": paper.get("published_at"),
        "upvotes": paper.get("upvotes") or 0,
        "pdf_url": paper.get("pdf_url"),
        "is_favorited": bool(arxiv_id and arxiv_id in favorite_ids),
    }


def _serialize_paper_for_detail(paper: dict[str, Any], *, favorite_ids: set[str]) -> dict[str, Any]:
    return _serialize_paper_for_list(paper, favorite_ids=favorite_ids)


def _serialize_related_paper(paper: dict[str, Any], *, favorite_ids: set[str]) -> dict[str, Any]:
    serialized = _serialize_paper_for_list(paper, favorite_ids=favorite_ids)
    serialized["source"] = paper.get("source") or "local"
    serialized["relation_score"] = round(float(paper.get("relation_score") or 0), 3)
    return serialized


def _build_related_papers(paper: dict[str, Any], *, limit: int = RELATED_PAPER_LIMIT) -> list[dict[str, Any]]:
    repo = get_paper_repository()
    current_id = str(paper.get("arxiv_id") or "")
    seen_ids = {current_id}
    related: list[dict[str, Any]] = []

    for candidate in repo.list_recent_papers(limit=RELATED_PAPER_CANDIDATE_LIMIT):
        candidate_id = str(candidate.get("arxiv_id") or "")
        if not candidate_id or candidate_id in seen_ids:
            continue
        score = _score_related_paper(paper, candidate)
        if score <= 0:
            continue
        related_candidate = dict(candidate)
        related_candidate["relation_score"] = score
        related_candidate["source"] = "local"
        related.append(related_candidate)
        seen_ids.add(candidate_id)

    related.sort(
        key=lambda item: (
            float(item.get("relation_score") or 0),
            int(item.get("upvotes") or 0),
            str(item.get("published_at") or ""),
        ),
        reverse=True,
    )

    if len(related) < limit:
        related.extend(
            _search_external_related_papers(
                paper,
                seen_ids=seen_ids,
                limit=limit - len(related),
            )
        )

    return related[:limit]


def _score_related_paper(source: dict[str, Any], candidate: dict[str, Any]) -> float:
    source_categories = set(source.get("categories") or [])
    candidate_categories = set(candidate.get("categories") or [])
    category_overlap = len(source_categories & candidate_categories)
    same_primary = source.get("primary_category") and source.get("primary_category") == candidate.get(
        "primary_category"
    )

    source_title_tokens = _keyword_tokens(source.get("title") or "")
    candidate_title_tokens = _keyword_tokens(candidate.get("title") or "")
    source_abstract_tokens = _keyword_tokens(source.get("abstract") or "")
    candidate_abstract_tokens = _keyword_tokens(candidate.get("abstract") or "")

    title_overlap = len(source_title_tokens & candidate_title_tokens)
    abstract_overlap = len(source_abstract_tokens & candidate_abstract_tokens)

    return (
        category_overlap * 2.0 + (1.5 if same_primary else 0.0) + title_overlap * 0.8 + min(abstract_overlap, 12) * 0.12
    )


def _search_external_related_papers(
    paper: dict[str, Any],
    *,
    seen_ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    try:
        from src.integrations.paper_search import PaperSearchClient

        client = PaperSearchClient()
        query = " ".join(_keyword_tokens_in_order(paper.get("title") or "")[:8])
        if not query:
            return []
        external_papers = client.search_arxiv_papers(query, max_results=max(limit * 2, 5))
    except Exception:
        logger.warning("arXiv 관련 논문 검색 실패: arxiv_id=%s", paper.get("arxiv_id"), exc_info=True)
        return []

    related: list[dict[str, Any]] = []
    for external_paper in external_papers:
        arxiv_id = str(external_paper.get("arxiv_id") or "")
        if not arxiv_id or arxiv_id in seen_ids:
            continue
        external_paper["source"] = "arxiv"
        external_paper["relation_score"] = _score_related_paper(paper, external_paper)
        related.append(external_paper)
        seen_ids.add(arxiv_id)
        if len(related) >= limit:
            break
    return related


def _keyword_tokens(text: str) -> set[str]:
    return set(_keyword_tokens_in_order(text))


def _keyword_tokens_in_order(text: str) -> list[str]:
    stopwords = {
        "about",
        "after",
        "based",
        "between",
        "from",
        "into",
        "model",
        "models",
        "paper",
        "that",
        "the",
        "their",
        "this",
        "through",
        "using",
        "with",
    }
    tokens: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower()):
        if token in stopwords or token in seen:
            continue
        tokens.append(token)
        seen.add(token)
    return tokens
