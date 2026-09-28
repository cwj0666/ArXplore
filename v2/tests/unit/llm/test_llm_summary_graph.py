import re
from collections import Counter
from typing import Any

import pytest

from arxplore.llm.models import ChatModelFactory
from arxplore.llm.summary_graph import (
    FALLBACK_TEXT_MAX_CHARS,
    MERGED_SUMMARY_MAX_CHARS,
    SECTION_TEXT_MAX_CHARS,
    TRUNCATION_SUFFIX,
    SummaryWriter,
    build_bucket_evidence,
    classify_section_bucket,
    format_summary_authors,
    group_summary_sections,
    merge_bucket_summaries,
    select_summary_sections,
    truncate_with_marker,
)
from arxplore.llm.tracing import Tracer

SECTION_TEMPERATURE = 0.1
BUCKET_TEMPERATURE = 0.15
FINAL_TEMPERATURE = 0.2


def _section(title: str, text: str = "Some section body text.") -> dict[str, Any]:
    return {"title": title, "text": text}


def _four_bucket_sections() -> list[dict[str, Any]]:
    return [
        *[_section(f"Introduction {index}") for index in range(4)],
        *[_section(f"Method {index}") for index in range(5)],
        *[_section(f"Experiments {index}") for index in range(5)],
        *[_section(f"Conclusion {index}") for index in range(5)],
        *[_section(f"Acknowledgments {index}") for index in range(3)],
    ]


@pytest.mark.parametrize(
    ("title", "bucket"),
    [
        ("Abstract", "background"),
        ("2 Related Work", "background"),
        ("Model Training", "method"),
        ("Results and Discussion", "experiments"),
        ("Ablation Analysis", "experiments"),
        ("Conclusion and Future Work", "limitations"),
        ("Acknowledgments", "other"),
        ("", "other"),
    ],
)
def test_bucket_rules_apply_in_order(title: str, bucket: str) -> None:
    assert classify_section_bucket(title) == bucket


def test_selection_caps_each_bucket_and_fills_with_other() -> None:
    titles = [section["title"] for section in select_summary_sections(_four_bucket_sections())]

    assert titles == [
        "Introduction 0",
        "Introduction 1",
        "Method 0",
        "Method 1",
        "Method 2",
        "Experiments 0",
        "Experiments 1",
        "Experiments 2",
        "Conclusion 0",
        "Conclusion 1",
        "Conclusion 2",
        "Acknowledgments 0",
    ]


def test_grouping_leaves_other_sections_out() -> None:
    grouped = group_summary_sections(select_summary_sections(_four_bucket_sections()))

    assert {bucket: len(items) for bucket, items in grouped.items()} == {
        "background": 2,
        "method": 3,
        "experiments": 3,
        "limitations": 3,
    }


def test_other_only_input_is_capped_at_twelve_and_never_grouped() -> None:
    selected = select_summary_sections([_section(f"Appendix {index}") for index in range(20)])

    assert len(selected) == 12
    assert all(not items for items in group_summary_sections(selected).values())


def test_selection_skips_non_mapping_and_blank_sections() -> None:
    sections = ["not a dict", _section("Method", "   "), _section("Method", "body"), None]

    assert select_summary_sections(sections) == [_section("Method", "body")]


def test_truncation_marker_and_limits() -> None:
    assert (SECTION_TEXT_MAX_CHARS, FALLBACK_TEXT_MAX_CHARS, MERGED_SUMMARY_MAX_CHARS) == (3200, 12000, 22000)
    assert truncate_with_marker("abc", 3) == "abc"
    assert truncate_with_marker("abcd", 3) == "abc" + TRUNCATION_SUFFIX


def test_evidence_uses_two_sections_of_900_characters() -> None:
    evidence = build_bucket_evidence(
        [_section("Method 0", "alpha " * 400), _section("", "beta"), _section("Method 2", "gamma")]
    )

    first, second = evidence.split("\n\n")
    assert first.startswith("[근거 1] Method 0\n")
    assert len(first.split("\n", 1)[1]) == 900
    assert second == "[근거 2] Section 2\nbeta"


def test_merge_labels_buckets_and_truncates_at_22000() -> None:
    merged = merge_bucket_summaries(
        {"background": "b" * 9000, "method": "m" * 9000, "experiments": "e" * 9000, "limitations": "l" * 9000},
        {},
        "",
    )

    assert merged.startswith("[배경/문제 해설]\n")
    assert len(merged) == 22000 + len(TRUNCATION_SUFFIX)
    assert merged.endswith(TRUNCATION_SUFFIX)


def test_merge_attaches_evidence_and_falls_back() -> None:
    grouped = {"method": [_section("Method 0", "alpha"), _section("Method 1", "beta"), _section("Method 2", "gamma")]}

    merged = merge_bucket_summaries({"method": "  method\n text "}, grouped, "fallback")

    assert merged == "[방법 해설]\nmethod text\n\n[방법 원문 근거]\n[근거 1] Method 0\nalpha\n\n[근거 2] Method 1\nbeta"
    assert merge_bucket_summaries({}, {}, "fallback") == "fallback"


@pytest.mark.parametrize(
    ("authors", "expected"),
    [
        (None, "저자 미상"),
        ("  Ada  ", "Ada"),
        ("   ", "저자 미상"),
        (["Ada", {"name": "Ben"}, {"name": " "}, 3], "Ada, Ben"),
        ({"name": "not a list"}, "저자 미상"),
    ],
)
def test_summary_author_line(authors: Any, expected: str) -> None:
    assert format_summary_authors(authors) == expected


def _temperatures(fake_openai: Any) -> Counter[float]:
    return Counter(request.body["temperature"] for request in fake_openai.requests_for("/v1/chat/completions"))


def _user_texts(fake_openai: Any) -> list[str]:
    return [request.body["messages"][-1]["content"] for request in fake_openai.requests_for("/v1/chat/completions")]


async def test_full_four_bucket_input_makes_sixteen_calls(
    fake_openai: Any, models: ChatModelFactory, tracer: Tracer
) -> None:
    result = await SummaryWriter(models, tracer).summarize(
        title="Paper", authors=["A. Author"], text="", sections=_four_bucket_sections(), model="gpt-4o"
    )

    assert result.startswith("Fake reply ")
    assert _temperatures(fake_openai) == {SECTION_TEMPERATURE: 11, BUCKET_TEMPERATURE: 4, FINAL_TEMPERATURE: 1}
    final_prompt = _user_texts(fake_openai)[-1]
    for label in ("배경/문제", "방법", "실험/결과", "한계/결론"):
        assert f"[{label} 원문 근거]" in final_prompt
    assert "Acknowledgments" not in final_prompt
    assert "저자: A. Author" in final_prompt


async def test_section_prompt_gets_3200_characters_and_the_bucket_name(
    fake_openai: Any, models: ChatModelFactory, tracer: Tracer
) -> None:
    await SummaryWriter(models, tracer).summarize(
        title="", authors=[], text="", sections=[_section("Method", "word " * 2000)], model="gpt-4o"
    )

    section_prompt = _user_texts(fake_openai)[0]
    body = section_prompt.split("섹션 본문:\n", 1)[1].split("\n\n요구사항:", 1)[0]
    assert len(body) == 3200
    assert body.endswith("…")
    assert "제목: 제목 없음\n섹션명: Method\n섹션 분류: method" in section_prompt


async def test_section_summaries_are_collapsed_before_the_bucket_call(
    fake_openai: Any, models: ChatModelFactory, tracer: Tracer
) -> None:
    fake_openai.chat_handler = lambda request: (
        "첫 문단입니다.\n\n둘째   문단입니다." if "섹션 분류" in request["messages"][-1]["content"] else "요약"
    )

    await SummaryWriter(models, tracer).summarize(
        title="Paper", authors="A", text="", sections=[_section("Method", "body")], model="gpt-4o"
    )

    bucket_prompt = next(text for text in _user_texts(fake_openai) if re.search(r"^bucket: method$", text, re.M))
    assert "[Method]\n첫 문단입니다. 둘째 문단입니다." in bucket_prompt


async def test_empty_buckets_skip_calls(fake_openai: Any, models: ChatModelFactory, tracer: Tracer) -> None:
    await SummaryWriter(models, tracer).summarize(
        title="Paper", authors="A", text="", sections=[_section("Method"), _section("Conclusion")], model="gpt-4o"
    )

    assert _temperatures(fake_openai) == {SECTION_TEMPERATURE: 2, BUCKET_TEMPERATURE: 2, FINAL_TEMPERATURE: 1}


async def test_unclassified_sections_use_the_fallback_text_only(
    fake_openai: Any, models: ChatModelFactory, tracer: Tracer
) -> None:
    result = await SummaryWriter(models, tracer).summarize(
        title="Paper", authors="A", text=" fallback body ", sections=[_section("Appendix")], model="gpt-4o"
    )

    assert result
    assert _temperatures(fake_openai) == {FINAL_TEMPERATURE: 1}
    assert "본문:\nfallback body\n---" in _user_texts(fake_openai)[0]


async def test_fallback_text_is_truncated_to_12000_characters(
    fake_openai: Any, models: ChatModelFactory, tracer: Tracer
) -> None:
    await SummaryWriter(models, tracer).summarize(title="T", authors="A", text="x" * 20000, sections=[], model="gpt-4o")

    assert "x" * 12000 + TRUNCATION_SUFFIX + "\n---" in _user_texts(fake_openai)[0]
    assert "x" * 12001 not in _user_texts(fake_openai)[0]


async def test_nothing_to_summarize_makes_no_call(fake_openai: Any, models: ChatModelFactory, tracer: Tracer) -> None:
    result = await SummaryWriter(models, tracer).summarize(title="T", authors="A", text="", sections=[], model="gpt-4o")

    assert result == ""
    assert fake_openai.requests == []


async def test_gpt5_summary_calls_carry_no_temperature(
    fake_openai: Any, models: ChatModelFactory, tracer: Tracer
) -> None:
    await SummaryWriter(models, tracer).summarize(
        title="Paper", authors="A", text="", sections=[_section("Method")], model="gpt-5"
    )

    bodies = [request.body for request in fake_openai.requests_for("/v1/chat/completions")]
    assert len(bodies) == 3
    assert all(body["model"] == "gpt-5" and "temperature" not in body for body in bodies)
