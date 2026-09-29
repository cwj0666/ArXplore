import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arxplore.api.errors import InvalidRequestError
from arxplore.api.services.chat import validate_chat_input
from arxplore.llm.citations import arxiv_id_from_url, normalize_cited_arxiv_id

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
    "mentions_required_ids": "`must_mention_arxiv_ids`가 있는 케이스: 필수 arXiv ID 중 답변에 ID·링크 또는 인용된 논문 제목으로 나온 비율",
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
    "cannot answer",
    "can't answer",
    "unable to answer",
    "i cannot share",
    "i can't share",
    "personal opinion",
)

MIN_TITLE_MATCH_CHARS = 12
TITLE_SUBTITLE_SEPARATOR = ":"

_ARXIV_URL_PATTERN = re.compile(r"https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/[^\s<>()\[\]\"'`]+", re.IGNORECASE)
_ARXIV_ID_PATTERN = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")
_URL_TRAILING = ".,;:!?"


class InputRejected(ValueError):
    pass


def prepare_chat_input(message: str, history: Sequence[tuple[str, str]] = ()) -> tuple[str, list[tuple[str, str]]]:
    payload = [{"role": role, "content": content} for role, content in history]
    try:
        chat = validate_chat_input(message, payload)
    except InvalidRequestError as exc:
        raise InputRejected(exc.message) from exc
    return chat.message, list(chat.history)


def load_refusal_phrases(path: str | Path) -> list[str]:
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
    normalized = _normalize(answer)
    return bool(normalized) and any(_normalize(phrase) in normalized for phrase in phrases if phrase.strip())


def classify_outcome(record: Mapping[str, Any], phrases: Iterable[str] = DEFAULT_REFUSAL_PHRASES) -> str:
    if record.get("error"):
        return OUTCOME_ERROR
    if record.get("outcome") == OUTCOME_REJECTED or record.get("rejection"):
        return OUTCOME_REJECTED
    return OUTCOME_REFUSED if contains_refusal(str(record.get("answer") or ""), phrases) else OUTCOME_ANSWERED


def answer_link_ids(answer: str) -> list[str]:
    ids: list[str] = []
    for match in _ARXIV_URL_PATTERN.finditer(answer or ""):
        linked = arxiv_id_from_url(match.group(0).rstrip(_URL_TRAILING))
        if linked and linked not in ids:
            ids.append(linked)
    return ids


def mentioned_arxiv_ids(answer: str) -> set[str]:
    return set(answer_link_ids(answer)) | set(_ARXIV_ID_PATTERN.findall(answer or ""))


def allowed_arxiv_ids(record: Mapping[str, Any]) -> set[str]:
    allowed = {normalize_cited_arxiv_id(str(value)) for value in record.get("hit_arxiv_ids") or [] if value}
    for citation in record.get("citations") or []:
        if not isinstance(citation, Mapping):
            continue
        if citation.get("arxiv_id"):
            allowed.add(normalize_cited_arxiv_id(str(citation["arxiv_id"])))
        linked = arxiv_id_from_url(str(citation.get("url") or ""))
        if linked:
            allowed.add(linked)
    if record.get("mode") == "paper_chat" and record.get("arxiv_id"):
        allowed.add(normalize_cited_arxiv_id(str(record["arxiv_id"])))
    allowed.discard("")
    return allowed


def fabricated_link_ids(record: Mapping[str, Any]) -> list[str]:
    allowed = allowed_arxiv_ids(record)
    return [value for value in answer_link_ids(str(record.get("answer") or "")) if value not in allowed]


def _title_seen(title: str, lowered_answer: str) -> bool:
    prefix = title.split(TITLE_SUBTITLE_SEPARATOR)[0].strip()
    return (len(title) >= MIN_TITLE_MATCH_CHARS and title in lowered_answer) or (
        len(prefix) >= MIN_TITLE_MATCH_CHARS and prefix in lowered_answer
    )


def _mentioned_required(record: Mapping[str, Any], answer: str, required: set[str]) -> float:
    mentioned = set(mentioned_arxiv_ids(answer))
    lowered_answer = answer.lower()
    for citation in record.get("citations") or []:
        cited_id = normalize_cited_arxiv_id(str(citation.get("arxiv_id") or ""))
        title = str(citation.get("title") or "").strip().lower()
        if cited_id in required and (citation.get("in_answer") or _title_seen(title, lowered_answer)):
            mentioned.add(cited_id)
    return len(required & mentioned) / len(required)


def behavior_scores(
    record: Mapping[str, Any], phrases: Iterable[str] = DEFAULT_REFUSAL_PHRASES
) -> tuple[str, dict[str, float]]:
    phrase_list = list(phrases)
    outcome = classify_outcome(record, phrase_list)
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
    required = {normalize_cited_arxiv_id(str(value)) for value in record.get("must_mention_arxiv_ids") or [] if value}
    if required:
        scores["mentions_required_ids"] = _mentioned_required(record, answer, required)
    return outcome, scores
