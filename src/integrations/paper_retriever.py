from __future__ import annotations

import contextvars
import re
from concurrent.futures import ThreadPoolExecutor

from src.integrations.embedding_client import EmbeddingClient
from src.integrations.hybrid_fusion import (
    DEFAULT_HYBRID_FUSION,
    HybridFusionConfig,
    apply_paper_diversity,
    fuse_hybrid_candidates,
    query_tokens,
    to_float,
)
from src.integrations.paper_repository import PaperRepository
from src.integrations.pdf_parser.section_roles import is_references_section_title
from src.integrations.vector_repository import VectorRepository

_REFERENCE_INTENT_PATTERN = re.compile(
    r"\b(?:references?|bibliography|works\s+cited|cited\s+works)\b",
    re.IGNORECASE,
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def normalize_search_query(query: str) -> str:
    """NUL을 포함한 C0 제어 문자(\\n, \\t 제외)와 DEL을 지우고 공백을 한 칸으로 합친다."""
    return " ".join(_CONTROL_CHARACTERS.sub("", str(query or "")).split())


def candidate_fetch_limit(limit: int) -> int:
    """전체 검색에서 저장소에 요청하는 후보 수. 저장소가 논문당 상한을 적용한 뒤의 행 수다."""
    return max(max(1, int(limit)) * 5, 30)


def hybrid_branch_limit(limit: int) -> int:
    """hybrid 검색이 lexical/vector 경로 각각에서 받는 결과 수."""
    return max(max(1, int(limit)) * 3, 10)


class PaperRetriever:
    """논문 검색과 RAG용 문맥 구성을 담당하는 retrieval 경로.

    `parallel_channels`(기본 True)면 hybrid 검색이 lexical 경로와 vector 경로(질의 임베딩 → 벡터 SQL)를 동시에
    실행한다. False면 lexical → vector 순서로 실행한다(병렬화 전 동작. 지연 A/B 측정용).
    """

    def __init__(
        self,
        *,
        repository: PaperRepository | None = None,
        embedding_client: EmbeddingClient | None = None,
        vector_repository: VectorRepository | None = None,
        parallel_channels: bool = True,
    ) -> None:
        self.repository = repository or PaperRepository()
        self.embedding_client = embedding_client or EmbeddingClient()
        self.vector_repository = vector_repository or VectorRepository()
        self.parallel_channels = parallel_channels

    def search_paper_chunks(
        self,
        query: str,
        *,
        limit: int = 5,
        arxiv_id: str | None = None,
    ) -> list[dict]:
        """공용 반환 shape로 청크를 조회한다. 제어 문자를 지운 질의가 비면 DB를 조회하지 않고 []를 반환한다."""
        query = normalize_search_query(query)
        if not query:
            return []
        normalized_limit = max(1, limit)
        fetch_limit = normalized_limit if arxiv_id else candidate_fetch_limit(normalized_limit)
        candidates = self.repository.list_chunk_candidates_by_query(query, limit=fetch_limit, arxiv_id=arxiv_id)
        normalized_candidates = self._normalize_candidates(query, candidates, retrieval_method="lexical")
        reranked_candidates = self._rerank_lexical_candidates(query, normalized_candidates)
        filtered_candidates = self._filter_lexical_candidates(query, reranked_candidates)
        return self._apply_paper_diversity(filtered_candidates, limit=normalized_limit, arxiv_id=arxiv_id)

    def search_paper_chunks_by_vector(
        self,
        query: str,
        *,
        arxiv_id: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """벡터 검색 결과를 공용 반환 shape로 정규화해 반환한다. 제어 문자를 지운 질의가 비면 []를 반환한다."""
        query = normalize_search_query(query)
        if not query:
            return []
        normalized_limit = max(1, limit)
        fetch_limit = normalized_limit if arxiv_id else candidate_fetch_limit(normalized_limit)
        query_embedding = self.embedding_client.embed_texts([query])[0]
        candidates = self.vector_repository.search_paper_chunks(
            query_embedding,
            limit=fetch_limit,
            arxiv_id=arxiv_id,
        )
        normalized_candidates = self._normalize_candidates(query, candidates, retrieval_method="vector")
        reranked_candidates = self._rerank_vector_candidates(query, normalized_candidates)
        return self._apply_paper_diversity(reranked_candidates, limit=normalized_limit, arxiv_id=arxiv_id)

    def search_paper_chunks_by_hybrid(
        self,
        query: str,
        *,
        arxiv_id: str | None = None,
        limit: int = 5,
        lexical_limit: int | None = None,
        vector_limit: int | None = None,
    ) -> list[dict]:
        """lexical/vector 결과를 `DEFAULT_HYBRID_FUSION`(정규화 점수의 convex combination)으로 결합해 공용 retrieval
        shape로 반환한다. 제어 문자를 지운 질의가 비면 []를 반환한다."""
        query, lexical_candidates, vector_candidates = self.hybrid_fusion_inputs(
            query,
            arxiv_id=arxiv_id,
            limit=limit,
            lexical_limit=lexical_limit,
            vector_limit=vector_limit,
        )
        if not query:
            return []
        return self._merge_hybrid_candidates(
            query,
            lexical_candidates,
            vector_candidates,
            arxiv_id=arxiv_id,
            limit=max(1, limit),
        )

    def hybrid_fusion_inputs(
        self,
        query: str,
        *,
        arxiv_id: str | None = None,
        limit: int = 5,
        lexical_limit: int | None = None,
        vector_limit: int | None = None,
    ) -> tuple[str, list[dict], list[dict]]:
        """hybrid 융합에 들어가는 (정규화 질의, lexical 결과, vector 결과). 각 채널은 자기 경로의 정규화·rerank·
        필터·diversity를 거친 `hybrid_branch_limit(limit)`개다. 질의가 비면 DB를 조회하지 않고 빈 목록을 돌려준다.
        `scripts/eval_dump_candidates.py`가 같은 입력을 저장해 융합만 다시 재생한다.

        `parallel_channels`면 두 채널을 동시에 실행한다(`_run_channels_in_parallel`). 결과와 예외는 순차 실행과 같다."""
        query = normalize_search_query(query)
        if not query:
            return query, [], []
        normalized_limit = max(1, limit)
        if self.parallel_channels:
            lexical_candidates, vector_candidates = self._run_channels_in_parallel(
                query,
                arxiv_id=arxiv_id,
                lexical_limit=lexical_limit or hybrid_branch_limit(normalized_limit),
                vector_limit=vector_limit or hybrid_branch_limit(normalized_limit),
            )
            return query, lexical_candidates, vector_candidates
        lexical_candidates = self.search_paper_chunks(
            query,
            arxiv_id=arxiv_id,
            limit=lexical_limit or hybrid_branch_limit(normalized_limit),
        )
        vector_candidates = self.search_paper_chunks_by_vector(
            query,
            arxiv_id=arxiv_id,
            limit=vector_limit or hybrid_branch_limit(normalized_limit),
        )
        return query, lexical_candidates, vector_candidates

    def _run_channels_in_parallel(
        self,
        query: str,
        *,
        arxiv_id: str | None,
        lexical_limit: int,
        vector_limit: int,
    ) -> tuple[list[dict], list[dict]]:
        """vector 경로를 작업 스레드에서, lexical 경로를 호출 스레드에서 동시에 실행한다.

        - 대부분의 시간이 질의 임베딩 API 왕복이라 vector를 먼저 띄운다. 두 경로 모두 저장소 메서드 안에서
          풀 연결을 따로 빌렸다가 반납하므로 스레드끼리 연결을 공유하지 않는다(요청당 동시 연결은 최대 2개).
        - 작업 스레드는 `contextvars.copy_context()`로 실행해 요청 범위 키(`override_openai_runtime`)와
          LangSmith 추적 문맥을 그대로 본다.
        - 예외 우선순위는 순차 실행과 같다. lexical이 실패하면 vector 결과를 기다리지 않고 lexical 예외를 낸다
          (순차 실행에서는 vector가 아예 실행되지 않던 경우다. 이미 시작된 vector 호출은 끝까지 돌고 결과는 버린다).
          lexical이 성공하면 vector 예외(`OpenAIError` 등)를 그대로 다시 낸다.
        - 실행기는 호출마다 만든다(스레드 1개). 모듈 공용 실행기는 gthread 워커의 동시 요청이 작업 스레드 수에
          막혀 줄을 서게 되고, 스레드 생성 비용은 임베딩 왕복에 비해 무시할 만하다.
        """
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hybrid-vector")
        try:
            vector_future = executor.submit(
                contextvars.copy_context().run,
                self.search_paper_chunks_by_vector,
                query,
                arxiv_id=arxiv_id,
                limit=vector_limit,
            )
            lexical_candidates = self.search_paper_chunks(query, arxiv_id=arxiv_id, limit=lexical_limit)
            vector_candidates = vector_future.result()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return lexical_candidates, vector_candidates

    def search_paper_contexts(
        self,
        query: str,
        *,
        limit: int = 5,
        adjacency_window: int = 1,
        arxiv_id: str | None = None,
    ) -> list[dict]:
        """검색 hit 주변 청크까지 묶어 LLM 입력용 문맥 단위를 반환한다."""
        candidates = self.search_paper_chunks(query, limit=limit, arxiv_id=arxiv_id)
        return self._build_contexts(candidates, adjacency_window=adjacency_window)

    def search_paper_contexts_by_vector(
        self,
        query: str,
        *,
        arxiv_id: str | None = None,
        limit: int = 5,
        adjacency_window: int = 1,
    ) -> list[dict]:
        """벡터 검색 후 주변 문맥까지 묶어 반환한다. arxiv_id를 주면 해당 논문 내로 한정한다."""
        candidates = self.search_paper_chunks_by_vector(
            query,
            arxiv_id=arxiv_id,
            limit=limit,
        )
        return self._build_contexts(candidates, adjacency_window=adjacency_window)

    def search_paper_contexts_by_hybrid(
        self,
        query: str,
        *,
        arxiv_id: str | None = None,
        limit: int = 5,
        adjacency_window: int = 1,
        lexical_limit: int | None = None,
        vector_limit: int | None = None,
    ) -> list[dict]:
        """hybrid 검색 후 주변 문맥까지 묶어 반환한다."""
        candidates = self.search_paper_chunks_by_hybrid(
            query,
            arxiv_id=arxiv_id,
            limit=limit,
            lexical_limit=lexical_limit,
            vector_limit=vector_limit,
        )
        return self._build_contexts(candidates, adjacency_window=adjacency_window)

    def _build_contexts(self, candidates: list[dict], *, adjacency_window: int) -> list[dict]:
        """검색 결과를 주변 청크와 결합해 공용 context shape로 정규화한다."""
        normalized_window = max(0, adjacency_window)
        windows = self._fetch_chunk_windows(candidates, window=normalized_window)
        contexts: list[dict] = []
        for candidate, raw_context_chunks in zip(candidates, windows, strict=True):
            context_chunks = [self._normalize_context_chunk(chunk) for chunk in raw_context_chunks]
            contexts.append(
                {
                    **candidate,
                    "context_chunks": context_chunks,
                    "context_text": "\n\n".join(
                        chunk["chunk_text"] for chunk in context_chunks if chunk.get("chunk_text")
                    ),
                }
            )
        return contexts

    def _fetch_chunk_windows(self, candidates: list[dict], *, window: int) -> list[list[dict]]:
        """hit별 문맥 창을 한 번의 쿼리로 가져온다. 일괄 조회가 없는 저장소는 hit마다 조회한다."""
        if not candidates:
            return []
        centers = [(candidate["arxiv_id"], int(candidate["chunk_index"])) for candidate in candidates]
        list_chunk_windows = getattr(self.repository, "list_chunk_windows", None)
        if callable(list_chunk_windows):
            return list_chunk_windows(centers, window=window)
        return [self.repository.list_chunk_window(arxiv_id, index, window=window) for arxiv_id, index in centers]

    def _rerank_vector_candidates(self, query: str, candidates: list[dict]) -> list[dict]:
        """벡터 검색 결과를 섹션 prior와 lexical overlap으로 한 번 더 정렬한다."""
        query_tokens = self._query_tokens(query)
        query_lowered = query.lower()
        appendix_requested = any(
            keyword in query_lowered for keyword in ("appendix", "supplement", "additional analysis")
        )
        conclusion_requested = any(keyword in query_lowered for keyword in ("conclusion", "limitation", "discussion"))
        reference_requested = self._reference_intent_requested(query) or bool(
            re.search(r"\bcitations?\b", query_lowered)
        )
        section_intent_bonus = self._section_intent_bonus(query)

        reranked: list[dict] = []
        for candidate in candidates:
            section_title = str(candidate.get("section_title") or "")
            section_lowered = section_title.lower()
            chunk_text = str(candidate.get("chunk_text") or "")
            content_role = str(candidate.get("content_role") or "")
            overlap_bonus = self._lexical_overlap_bonus(query_tokens, f"{section_title} {chunk_text}")
            base_score = self._to_float(candidate.get("score") or candidate.get("similarity_score"))

            rerank_adjustment = overlap_bonus
            if not appendix_requested and any(
                keyword in section_lowered
                for keyword in (
                    "appendix",
                    "additional analysis",
                    "supplementary",
                    "experimental details",
                    "implementation details",
                )
            ):
                rerank_adjustment -= 0.08
            if not conclusion_requested and any(
                keyword in section_lowered for keyword in ("conclusion", "discussion", "limitations")
            ):
                rerank_adjustment -= 0.03
            if not reference_requested and (
                is_references_section_title(section_title) or "acknowledg" in section_lowered
            ):
                rerank_adjustment -= 0.18
            if content_role in {"front_matter", "table_like"}:
                rerank_adjustment -= 0.02
            if not reference_requested and self._looks_reference_like_text(chunk_text):
                rerank_adjustment -= 0.14
            rerank_adjustment += section_intent_bonus(section_lowered)

            reranked.append(
                {
                    **candidate,
                    "score": base_score + rerank_adjustment,
                    "similarity_score": base_score + rerank_adjustment,
                    "rerank_adjustment": rerank_adjustment,
                    "score_breakdown": {
                        **dict(candidate.get("score_breakdown") or {}),
                        "rerank_adjustment": rerank_adjustment,
                    },
                }
            )

        return sorted(
            reranked, key=lambda item: (self._to_float(item.get("score")), int(item.get("chunk_id") or 0)), reverse=True
        )

    def _normalize_candidates(self, query: str, candidates: list[dict], *, retrieval_method: str) -> list[dict]:
        """lexical/vector 후보를 공용 retrieval shape로 맞춘다."""
        return [
            self._normalize_candidate(query, candidate, retrieval_method=retrieval_method) for candidate in candidates
        ]

    def _rerank_lexical_candidates(self, query: str, candidates: list[dict]) -> list[dict]:
        """명시적 section-intent 질의에서는 lexical 결과도 해당 섹션을 약하게 우대한다."""
        section_intent_bonus = self._section_intent_bonus(query)
        reranked: list[dict] = []

        for candidate in candidates:
            section_lowered = str(candidate.get("section_title") or "").lower()
            bonus = section_intent_bonus(section_lowered)
            base_score = self._to_float(candidate.get("score"))
            reranked.append(
                {
                    **candidate,
                    "score": base_score + bonus,
                    "similarity_score": base_score + bonus,
                    "score_breakdown": {
                        **dict(candidate.get("score_breakdown") or {}),
                        "section_intent_bonus": bonus,
                    },
                }
            )

        return sorted(
            reranked, key=lambda item: (self._to_float(item.get("score")), int(item.get("chunk_id") or 0)), reverse=True
        )

    def _filter_lexical_candidates(self, query: str, candidates: list[dict]) -> list[dict]:
        """reference-like lexical 오염을 기본 경로에서 차단한다."""
        if self._reference_intent_requested(query):
            return candidates

        filtered_candidates: list[dict] = []
        for candidate in candidates:
            content_role = str(candidate.get("content_role") or "")
            section_title = str(candidate.get("section_title") or "")
            chunk_text = str(candidate.get("chunk_text") or "")

            if content_role == "references":
                continue
            if content_role == "front_matter":
                continue
            if is_references_section_title(section_title):
                continue
            if "front matter" in section_title.lower():
                continue
            if self._looks_reference_like_text(chunk_text):
                continue
            if self._looks_outline_like_text(chunk_text):
                continue

            filtered_candidates.append(candidate)

        return filtered_candidates

    def _merge_hybrid_candidates(
        self,
        query: str,
        lexical_candidates: list[dict],
        vector_candidates: list[dict],
        *,
        arxiv_id: str | None,
        limit: int,
        config: HybridFusionConfig = DEFAULT_HYBRID_FUSION,
    ) -> list[dict]:
        """lexical/vector 결과를 `config`(기본 `DEFAULT_HYBRID_FUSION`)로 병합(`fuse_hybrid_candidates`)한 뒤 논문 다양성을
        적용한다."""
        merged_candidates = fuse_hybrid_candidates(query, lexical_candidates, vector_candidates, config)
        return self._apply_paper_diversity(merged_candidates, limit=limit, arxiv_id=arxiv_id)

    def _apply_paper_diversity(
        self,
        candidates: list[dict],
        *,
        limit: int,
        arxiv_id: str | None,
        max_chunks_per_paper: int = 2,
    ) -> list[dict]:
        """논문당 청크 상한을 우선하는 다양성 규칙(`apply_paper_diversity`)."""
        return apply_paper_diversity(
            candidates, limit=limit, arxiv_id=arxiv_id, max_chunks_per_paper=max_chunks_per_paper
        )

    def _section_intent_bonus(self, query: str):
        """명시적 section-intent 질의에서만 해당 섹션을 밀어준다."""
        lowered = query.lower()
        checks: list[tuple[tuple[str, ...], tuple[str, ...], float]] = [
            (("limitation", "limitations"), ("limitation",), 0.14),
            (("conclusion", "conclusions"), ("conclusion",), 0.12),
            (("future work",), ("future work", "future directions"), 0.12),
            (("discussion",), ("discussion",), 0.1),
        ]

        active_rules = [
            (section_keywords, bonus)
            for query_keywords, section_keywords, bonus in checks
            if any(keyword in lowered for keyword in query_keywords)
        ]

        def resolve(section_title_lowered: str) -> float:
            for section_keywords, bonus in active_rules:
                if any(keyword in section_title_lowered for keyword in section_keywords):
                    return bonus
            return 0.0

        return resolve

    def _normalize_candidate(self, query: str, candidate: dict, *, retrieval_method: str) -> dict:
        """후보 하나를 공용 필드 집합으로 정규화한다."""
        score = self._to_float(candidate.get("score") or candidate.get("similarity_score"))
        content_role = str(candidate.get("content_role") or (candidate.get("metadata") or {}).get("content_role") or "")
        paper_title = str(candidate.get("paper_title") or "")
        paper_abstract = str(candidate.get("paper_abstract") or "")
        chunk_text = str(candidate.get("chunk_text") or "")

        return {
            **candidate,
            "paper_title": paper_title,
            "paper_abstract": paper_abstract,
            "chunk_text": chunk_text,
            "section_title": str(candidate.get("section_title") or ""),
            "content_role": content_role,
            "score": score,
            "similarity_score": score,
            "retrieval_method": retrieval_method,
            "score_source": retrieval_method,
            "snippet": str(
                candidate.get("snippet") or self._build_search_snippet(query, chunk_text, paper_abstract, paper_title)
            ),
        }

    @staticmethod
    def _normalize_context_chunk(chunk: dict) -> dict:
        """문맥 창의 chunk에도 공용 content_role 필드를 노출한다."""
        return {
            **chunk,
            "section_title": str(chunk.get("section_title") or ""),
            "content_role": str((chunk.get("metadata") or {}).get("content_role") or ""),
            "chunk_text": str(chunk.get("chunk_text") or ""),
        }

    @staticmethod
    def _build_search_snippet(query: str, chunk_text: str, abstract: str, title: str, max_chars: int = 280) -> str:
        """질의와 가장 가까운 텍스트 조각을 snippet으로 만든다."""
        terms = [term for term in re.split(r"\W+", query.lower()) if len(term) >= 3]
        candidates = [chunk_text, abstract, title]

        for candidate in candidates:
            if not candidate:
                continue
            lowered = candidate.lower()
            for term in terms:
                index = lowered.find(term)
                if index != -1:
                    start = max(0, index - max_chars // 3)
                    end = min(len(candidate), start + max_chars)
                    snippet = candidate[start:end].strip()
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(candidate):
                        snippet = snippet + "..."
                    return snippet

        fallback = next((candidate for candidate in candidates if candidate), "")
        compact = " ".join(fallback.split())
        return compact[:max_chars] + ("..." if len(compact) > max_chars else "")

    @staticmethod
    def _to_float(value: object) -> float:
        return to_float(value)

    @staticmethod
    def _query_tokens(query: str) -> set[str]:
        """짧은 영문 질의에서 의미 있는 토큰만 뽑는다."""
        return query_tokens(query)

    @staticmethod
    def _lexical_overlap_bonus(query_tokens: set[str], text: str) -> float:
        """질의 토큰이 chunk에 얼마나 직접 등장하는지 계산한다."""
        if not query_tokens:
            return 0.0

        text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        if not text_tokens:
            return 0.0

        overlap = len(query_tokens & text_tokens)
        if overlap == 0:
            return 0.0

        return min(0.12, 0.03 * overlap)

    @staticmethod
    def _looks_reference_like_text(text: str) -> bool:
        """본문 검색에서 제외해야 할 reference-like 청크를 감지한다.

        앞 1,200자(공백 정규화 기준)만 본다. 줄 머리가 `[n]`으로 시작하는 줄이 3개 이상이면 번호 참고문헌 목록으로
        본다. 문장 안의 인용 표지(`... [1], [2], [3] ...`)만으로는 판정하지 않는다: 서론과 결과 표도 인용 표지를 여러 개
        담기 때문이다. 인용 표지·학회명은 연도가 2개 이상 함께 있을 때만 참고문헌 신호로 센다.
        """
        compact_lines: list[str] = []
        total = 0
        for raw_line in str(text or "").splitlines():
            line = " ".join(raw_line.split())
            if not line:
                continue
            if total >= 1200:
                break
            compact_lines.append(line)
            total += len(line) + 1
        compact = " ".join(compact_lines)[:1200]
        if not compact:
            return False

        entry_markers = sum(1 for line in compact_lines if re.match(r"\[\d+\]", line))
        reference_markers = len(re.findall(r"\[\d+\]", compact))
        year_markers = len(re.findall(r"\b(?:19|20)\d{2}\b", compact))
        venue_markers = len(
            re.findall(
                r"\b(?:arXiv preprint|Proceedings|Conference|CVPR|ICCV|ECCV|NeurIPS|ICLR|ACL|EMNLP|AAAI)\b",
                compact,
                re.IGNORECASE,
            )
        )
        author_list_like = bool(
            re.match(
                r"^(?:[A-Z][A-Za-z'`.-]+,\s+[A-Z](?:\.[A-Z])?(?:,\s+[A-Z][A-Za-z'`.-]+,\s+[A-Z](?:\.[A-Z])?){1,}|(?:\[\d+\]\s*)?[A-Z][A-Za-z'`.-]+,)",
                compact,
            )
        )

        if entry_markers >= 3:
            return True
        if reference_markers >= 2 and year_markers >= 2:
            return True
        if venue_markers >= 2 and year_markers >= 2:
            return True
        if author_list_like and year_markers >= 1 and venue_markers >= 1:
            return True
        return False

    @staticmethod
    def _reference_intent_requested(query: str) -> bool:
        """사용자가 bibliography/reference 자체를 찾는 질의인지만 판별한다."""
        return bool(_REFERENCE_INTENT_PATTERN.search(query))

    @staticmethod
    def _looks_outline_like_text(text: str) -> bool:
        """서론 첫머리의 논문 구성 안내 같은 outline형 문장을 감지한다."""
        compact = " ".join(text.split()).lower()
        if not compact:
            return False

        outline_patterns = (
            "the remainder of this paper",
            "the rest of this paper",
            "the remainder of the paper",
            "the rest of the paper",
            "this paper is organized as follows",
            "the paper is organized as follows",
            "section 2 presents",
            "section 3 presents",
            "section 4 presents",
            "section 5 presents",
            "section 2 describes",
            "section 3 describes",
            "section 4 describes",
            "section 5 describes",
            "we conclude in section",
        )
        return any(pattern in compact for pattern in outline_patterns)
