"""LLM 판정 없이 계산하는 행동 지표와 제품 API 입력 검증의 재현.

답변 레코드(`eval/answers.py`) 하나에서 결과 종류(answered / refused / rejected / error)를 정하고,
거절 문구·arXiv 링크·금지 문자열·필수 arXiv ID를 문자열 규칙으로 검사한다. 링크 해석은 제품의
`src.core.agent.citations`와 같은 규칙을 쓴다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

CHAT_MESSAGE_MAX_CHARS = 4000
CHAT_HISTORY_MAX_MESSAGES = 20

OUTCOME_ANSWERED = "answered"
OUTCOME_REFUSED = "refused"
OUTCOME_REJECTED = "rejected"
OUTCOME_ERROR = "error"
OUTCOMES = (OUTCOME_ANSWERED, OUTCOME_REFUSED, OUTCOME_REJECTED, OUTCOME_ERROR)

BEHAVIOR_METRICS = (
    "refusal_correct",
    "no_fabricated_links",
    "must_not_contain_ok",
    "mentions_required_ids",
    "rejected_as_expected",
)
BEHAVIOR_METRIC_DESCRIPTIONS = {
    "refusal_correct": "refuse 케이스: 거절 문구가 있고, 검색 hit·citation에 없는 arXiv 링크가 없음",
    "no_fabricated_links": "답변이 있는 모든 케이스: 답변 속 arXiv 링크가 모두 검색 hit·citation의 논문을 가리킴",
    "must_not_contain_ok": "`must_not_contain`이 있는 케이스: 금지 문자열이 답변에 하나도 없음(대소문자 무시)",
    "mentions_required_ids": "`must_mention_arxiv_ids`가 있는 케이스: 필수 arXiv ID 중 답변(링크 포함)에 나온 비율",
    "rejected_as_expected": "입력 거부를 기대했거나 실제로 거부된 케이스: 기대와 결과가 일치",
}

DEFAULT_REFUSAL_PHRASES = (
    "죄송하지만",
    "죄송합니다",
    "할 수 없습니다",
    "할 수는 없습니다",
    "드릴 수 없",
    "보여드릴 수 없",
    "공개할 수 없",
    "답변할 수 없",
    "도움을 드릴 수 없",
    "데이터베이스에 없",
    "데이터베이스에서 찾",
    "논문 검색 범위",
    "i'm sorry",
    "i am sorry",
    "can't assist",
    "cannot assist",
    "unable to",
    "cannot provide",
    "can't provide",
    "not able to",
    "outside the scope",
    "not covered",
    "제공된 발췌문으로는 답하기 어렵",
    "답하기 어렵",
    "답변하기 어렵",
    "답변드리기 어렵",
    "말씀드리기 어렵",
    "도와드리기 어렵",
    "알려드릴 수 없",
    "제공할 수 없",
    "공유할 수 없",
    "확인할 수 없",
    "찾을 수 없",
    "찾지 못",
    "검색되지 않",
    "검색 결과가 없",
    "관련된 논문이 없",
    "관련 논문이 없",
    "관련된 결과가 없",
    "포함되어 있지 않",
    "다루지 않",
    "범위를 벗어",
    "알 수 없",
    "기억하지 못",
    "개인적인 의견",
    "의견을 드리",
    "not in the provided",
    "not found in",
    "could not find",
    "couldn't find",
    "can't find",
    "cannot find",
    "unable to find",
    "no relevant",
    "not available in",
    "does not contain",
    "doesn't contain",
    "not covered",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "i cannot share",
    "i can't share",
    "outside the scope",
    "personal opinion",
)

_ARXIV_URL_PATTERN = re.compile(r"https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/[^\s<>()\[\]\"'`]+", re.IGNORECASE)
_ARXIV_ID_PATTERN = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")
_URL_TRAILING = ".,;:!?"


class InputRejected(ValueError):
    """제품 API가 생성 전에 거부하는 입력."""


_CONTROL_CHARS = {chr(code) for code in range(32) if chr(code) not in "\n\t"} | {"\x7f"}


def strip_control_characters(value: str) -> str:
    return "".join(char for char in str(value or "") if char not in _CONTROL_CHARS)


def prepare_chat_input(message: str, history: Sequence[tuple[str, str]] = ()) -> tuple[str, list[tuple[str, str]]]:
    """제품 API(`backend/papers/services.py`)의 챗 입력 검증과 이력 정리를 그대로 재현한다.

    메시지는 앞뒤 공백을 지운 뒤 비어 있거나 `CHAT_MESSAGE_MAX_CHARS`자를 넘으면 `InputRejected`.
    이력은 user/assistant이면서 내용이 공백이 아닌 턴만 남기고 턴마다 `CHAT_MESSAGE_MAX_CHARS`자로 자른다.
    마지막 user 턴이 현재 메시지와 같으면 빼고, 최근 `CHAT_HISTORY_MAX_MESSAGES`개만 남긴다.
    """
    cleaned = strip_control_characters(message).strip()
    if not cleaned:
        raise InputRejected("메시지를 입력하세요.")
    if len(cleaned) > CHAT_MESSAGE_MAX_CHARS:
        raise InputRejected(f"메시지는 {CHAT_MESSAGE_MAX_CHARS:,}자 이하로 입력하세요.")
    turns = [
        (role, content[:CHAT_MESSAGE_MAX_CHARS])
        for role, content in history
        if role in ("user", "assistant") and isinstance(content, str) and content.strip()
    ]
    if turns and turns[-1][0] == "user" and turns[-1][1].strip() == cleaned:
        turns.pop()
    return cleaned, turns[-CHAT_HISTORY_MAX_MESSAGES:]


def load_refusal_phrases(path: str | Path) -> list[str]:
    """한 줄에 하나씩 적은 거절 문구 파일을 읽는다. 빈 줄과 `#`으로 시작하는 줄은 건너뛴다."""
    phrases: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            phrases.append(text)
    if not phrases:
        raise ValueError(f"{path}: 거절 문구가 없습니다.")
    return phrases


def _normalize(text: str) -> str:
    return " ".join(str(text or "").split()).lower()


def contains_refusal(answer: str, phrases: Iterable[str] = DEFAULT_REFUSAL_PHRASES) -> bool:
    """답변에 거절 문구가 하나라도 있으면 True. 공백을 하나로 모으고 대소문자를 무시해 비교한다."""
    normalized = _normalize(answer)
    return bool(normalized) and any(_normalize(phrase) in normalized for phrase in phrases if phrase.strip())


def classify_outcome(record: Mapping[str, Any], phrases: Iterable[str] = DEFAULT_REFUSAL_PHRASES) -> str:
    """레코드의 결과 종류. 생성 오류는 error, 입력 거부는 rejected, 답변에 거절 문구가 있으면 refused, 나머지는 answered."""
    if record.get("error"):
        return OUTCOME_ERROR
    if record.get("outcome") == OUTCOME_REJECTED or record.get("rejection"):
        return OUTCOME_REJECTED
    return OUTCOME_REFUSED if contains_refusal(str(record.get("answer") or ""), phrases) else OUTCOME_ANSWERED


def _citation_helpers():
    from src.core.agent.citations import arxiv_id_from_url, normalize_arxiv_id

    return arxiv_id_from_url, normalize_arxiv_id


def answer_link_ids(answer: str) -> list[str]:
    """답변 속 arXiv 논문 링크(마크다운 링크와 맨 URL 모두)가 가리키는 arXiv ID를 등장 순서대로(중복 제거) 돌려준다."""
    arxiv_id_from_url, _ = _citation_helpers()
    ids: list[str] = []
    for match in _ARXIV_URL_PATTERN.finditer(answer or ""):
        linked = arxiv_id_from_url(match.group(0).rstrip(_URL_TRAILING))
        if linked and linked not in ids:
            ids.append(linked)
    return ids


def mentioned_arxiv_ids(answer: str) -> set[str]:
    """답변에 나온 arXiv ID(링크 속 ID와 본문의 `2401.12345` 형태, 버전 접미사 제외)."""
    return set(answer_link_ids(answer)) | set(_ARXIV_ID_PATTERN.findall(answer or ""))


def allowed_arxiv_ids(record: Mapping[str, Any]) -> set[str]:
    """답변이 링크해도 되는 논문: 검색 hit(`hit_arxiv_ids`), citation의 arxiv_id와 URL, 상세 챗 대상 논문."""
    arxiv_id_from_url, normalize_arxiv_id = _citation_helpers()
    allowed = {normalize_arxiv_id(str(value)) for value in record.get("hit_arxiv_ids") or [] if value}
    for citation in record.get("citations") or []:
        if not isinstance(citation, Mapping):
            continue
        if citation.get("arxiv_id"):
            allowed.add(normalize_arxiv_id(str(citation["arxiv_id"])))
        linked = arxiv_id_from_url(str(citation.get("url") or ""))
        if linked:
            allowed.add(linked)
    if record.get("mode") == "paper_chat" and record.get("arxiv_id"):
        allowed.add(normalize_arxiv_id(str(record["arxiv_id"])))
    allowed.discard("")
    return allowed


def fabricated_link_ids(record: Mapping[str, Any]) -> list[str]:
    """답변의 arXiv 링크 중 검색 hit·citation에 없는 논문을 가리키는 ID."""
    allowed = allowed_arxiv_ids(record)
    return [value for value in answer_link_ids(str(record.get("answer") or "")) if value not in allowed]


def behavior_scores(
    record: Mapping[str, Any], phrases: Iterable[str] = DEFAULT_REFUSAL_PHRASES
) -> tuple[str, dict[str, float]]:
    """레코드의 결과 종류와, 이 레코드에 해당하는 행동 지표만 담은 점수(1.0 통과 / 0.0 실패, 비율 지표는 0~1)."""
    phrases = list(phrases)
    outcome = classify_outcome(record, phrases)
    expect_rejected = bool(record.get("expect_rejected"))
    scores: dict[str, float] = {}
    if expect_rejected or outcome == OUTCOME_REJECTED:
        scores["rejected_as_expected"] = 1.0 if (outcome == OUTCOME_REJECTED) == expect_rejected else 0.0
    if outcome not in (OUTCOME_ANSWERED, OUTCOME_REFUSED):
        return outcome, scores

    answer = str(record.get("answer") or "")
    fabricated = fabricated_link_ids(record)
    scores["no_fabricated_links"] = 0.0 if fabricated else 1.0
    if record.get("expected_behavior") == "refuse":
        scores["refusal_correct"] = 1.0 if outcome == OUTCOME_REFUSED and not fabricated else 0.0
    forbidden = [str(value) for value in record.get("must_not_contain") or [] if str(value).strip()]
    if forbidden:
        lowered = answer.lower()
        scores["must_not_contain_ok"] = 0.0 if any(value.lower() in lowered for value in forbidden) else 1.0
    _, normalize_arxiv_id = _citation_helpers()
    required = {normalize_arxiv_id(str(value)) for value in record.get("must_mention_arxiv_ids") or [] if value}
    if required:
        mentioned = mentioned_arxiv_ids(answer)
        scores["mentions_required_ids"] = len(required & mentioned) / len(required)
    return outcome, scores
