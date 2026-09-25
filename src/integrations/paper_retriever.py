from __future__ import annotations

import re

from src.integrations.embedding_client import EmbeddingClient
from src.integrations.paper_repository import STRICT_MATCH_BONUS, PaperRepository
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
    """논문 검색과 RAG용 문맥 구성을 담당하는 retrieval 경로."""

    def __init__(
        self,
        *,
        repository: PaperRepository | None = None,
        embedding_client: EmbeddingClient | None = None,
        vector_repository: VectorRepository | None = None,
    ) -> None:
        self.repository = repository or PaperRepository()
        self.embedding_client = embedding_client or EmbeddingClient()
        self.vector_repository = vector_repository or VectorRepository()

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
        """lexical/vector 결과를 rank fusion으로 결합해 공용 retrieval shape로 반환한다.
        제어 문자를 지운 질의가 비면 []를 반환한다."""
        query = normalize_search_query(query)
        if not query:
            return []
        normalized_limit = max(1, limit)
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
        return self._merge_hybrid_candidates(
            query,
            lexical_candidates,
            vector_candidates,
            arxiv_id=arxiv_id,
            limit=normalized_limit,
        )

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
    ) -> list[dict]:
        """lexical/vector 결과를 reciprocal rank fusion으로 병합한다.

        vector 결과가 있으면 lexical 결과 중 일부 lexeme만 일치한 행(`strict_match`가 False)은 융합에서 뺀다.
        vector 결과가 없으면 lexical 결과를 모두 쓴다.
        """
        if vector_candidates:
            lexical_candidates = [
                candidate
                for candidate in lexical_candidates
                if (candidate.get("score_breakdown") or {}).get("strict_match") is not False
            ]
        rank_constant = 60.0
        method_weights = self._resolve_hybrid_method_weights(query, lexical_candidates, vector_candidates)
        merged: dict[int, dict] = {}
        fallback_key_seed = -1

        for method, candidates in (("lexical", lexical_candidates), ("vector", vector_candidates)):
            for index, candidate in enumerate(candidates):
                chunk_id = int(candidate.get("chunk_id") or fallback_key_seed)
                if not candidate.get("chunk_id"):
                    fallback_key_seed -= 1

                entry = merged.setdefault(
                    chunk_id,
                    {
                        **candidate,
                        "retrieval_method": "hybrid",
                        "score_source": "hybrid",
                        "matched_methods": [],
                        "score_breakdown": {},
                        "score": 0.0,
                        "similarity_score": 0.0,
                    },
                )

                rank = index + 1
                quality_weight = self._candidate_hybrid_quality_weight(method, candidate)
                rrf_score = (method_weights[method] * quality_weight) / (rank_constant + rank)
                method_score = self._to_float(candidate.get("score"))
                method_breakdown = dict(candidate.get("score_breakdown") or {})

                if method not in entry["matched_methods"]:
                    entry["matched_methods"].append(method)

                entry["score"] = self._to_float(entry.get("score")) + rrf_score
                entry["similarity_score"] = entry["score"]
                entry["score_breakdown"][f"{method}_rank"] = rank
                entry["score_breakdown"][f"{method}_rrf_score"] = rrf_score
                entry["score_breakdown"][f"{method}_score"] = method_score
                entry["score_breakdown"][f"{method}_weight"] = method_weights[method]
                entry["score_breakdown"][f"{method}_quality_weight"] = quality_weight
                entry["score_breakdown"][f"{method}_score_breakdown"] = method_breakdown

                if method == "vector" and "lexical" not in entry["matched_methods"]:
                    entry["snippet"] = candidate.get("snippet") or entry.get("snippet")
                elif method == "lexical":
                    entry["snippet"] = candidate.get("snippet") or entry.get("snippet")

        merged_candidates = list(merged.values())
        for candidate in merged_candidates:
            overlap_bonus = 0.015 if len(candidate.get("matched_methods") or []) > 1 else 0.0
            if overlap_bonus > 0:
                candidate["score"] = self._to_float(candidate.get("score")) + overlap_bonus
                candidate["similarity_score"] = candidate["score"]
                candidate["score_breakdown"]["cross_method_overlap_bonus"] = overlap_bonus

        merged_candidates.sort(
            key=lambda item: (
                self._to_float(item.get("score")),
                len(item.get("matched_methods") or []),
                int(item.get("chunk_id") or 0),
            ),
            reverse=True,
        )
        return self._apply_paper_diversity(merged_candidates, limit=limit, arxiv_id=arxiv_id)

    def _apply_paper_diversity(
        self,
        candidates: list[dict],
        *,
        limit: int,
        arxiv_id: str | None,
        max_chunks_per_paper: int = 2,
    ) -> list[dict]:
        """논문마다 앞에서부터 `max_chunks_per_paper`개까지만 고르고, 후보가 모자랄 때만
        상한을 넘긴 청크를 원래 순서대로 채운다. 논문 범위 검색이나 후보가 `limit` 이하면 순서대로 자른다."""
        normalized_limit = max(1, limit)
        if arxiv_id or len(candidates) <= normalized_limit:
            return candidates[:normalized_limit]

        selected: list[dict] = []
        overflow: list[dict] = []
        paper_counts: dict[str, int] = {}

        for candidate in candidates:
            candidate_arxiv_id = str(candidate.get("arxiv_id") or "")
            count = paper_counts.get(candidate_arxiv_id, 0)
            if candidate_arxiv_id and count >= max_chunks_per_paper:
                overflow.append(candidate)
                continue

            selected.append(candidate)
            if candidate_arxiv_id:
                paper_counts[candidate_arxiv_id] = count + 1
            if len(selected) >= normalized_limit:
                return selected[:normalized_limit]

        for candidate in overflow:
            selected.append(candidate)
            if len(selected) >= normalized_limit:
                break

        return selected[:normalized_limit]

    def _resolve_hybrid_method_weights(
        self,
        query: str,
        lexical_candidates: list[dict],
        vector_candidates: list[dict],
    ) -> dict[str, float]:
        """질의 성격과 lexical confidence를 보고 hybrid 가중치를 조정한다."""
        weights = {
            "lexical": 1.0,
            "vector": 1.0,
        }
        lexical_top_score = self._lexical_confidence(lexical_candidates[0]) if lexical_candidates else 0.0
        query_tokens = self._query_tokens(query)
        lexical_top_ids = {int(candidate.get("chunk_id") or 0) for candidate in lexical_candidates[:5]}
        vector_top_ids = {int(candidate.get("chunk_id") or 0) for candidate in vector_candidates[:5]}
        overlap_count = len(lexical_top_ids & vector_top_ids)

        if len(query_tokens) >= 5:
            weights["lexical"] *= 0.85
            weights["vector"] *= 1.05
        if lexical_top_score < 0.3:
            weights["lexical"] *= 0.45
            weights["vector"] *= 1.1
        elif lexical_top_score < 0.5:
            weights["lexical"] *= 0.7
            weights["vector"] *= 1.05
        if overlap_count == 0 and lexical_top_score < 0.4:
            weights["lexical"] *= 0.75
            weights["vector"] *= 1.08

        return {
            "lexical": max(0.2, weights["lexical"]),
            "vector": max(0.5, weights["vector"]),
        }

    def _candidate_hybrid_quality_weight(self, method: str, candidate: dict) -> float:
        """RRF에 후보 자체의 confidence를 반영한다."""
        if method != "lexical":
            return 1.0
        score = self._lexical_confidence(candidate)
        if score < 0.2:
            return 0.2
        if score < 0.3:
            return 0.4
        if score < 0.5:
            return 0.65
        if score < 0.8:
            return 0.85
        return 1.0

    def _lexical_confidence(self, candidate: dict) -> float:
        """hybrid 가중치 판단에 쓰는 lexical 점수. strict 일치 행은 `STRICT_MATCH_BONUS`를 뺀 점수,
        일부 lexeme만 일치한 행은 0이다. `strict_match`가 없는 후보는 점수를 그대로 쓴다."""
        score = self._to_float(candidate.get("score"))
        strict_match = (candidate.get("score_breakdown") or {}).get("strict_match")
        if strict_match is None:
            return score
        if not strict_match:
            return 0.0
        return score - STRICT_MATCH_BONUS

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
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _query_tokens(query: str) -> set[str]:
        """짧은 영문 질의에서 의미 있는 토큰만 뽑는다."""
        return {
            token
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if len(token) >= 3 and token not in {"the", "and", "for", "with", "from", "that", "this"}
        }

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
        """본문 검색에서 제외해야 할 reference-like 청크를 감지한다."""
        compact = " ".join(text.split())[:1200]
        if not compact:
            return False

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

        if reference_markers >= 3:
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
