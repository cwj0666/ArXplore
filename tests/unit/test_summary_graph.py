from __future__ import annotations

import threading
from collections import Counter
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.core import summary_graph

SECTION_TEMPERATURE = 0.1
BUCKET_TEMPERATURE = 0.15
FINAL_TEMPERATURE = 0.2
TRUNCATION_SUFFIX = "\n\n[이하 생략]"


class _CallRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[tuple[float, list[Any]]] = []

    def record(self, temperature: float, messages: list[Any]) -> None:
        with self._lock:
            self.calls.append((temperature, list(messages)))

    def count_by_stage(self) -> Counter:
        return Counter(temperature for temperature, _ in self.calls)

    def human_texts(self, temperature: float) -> list[str]:
        return [str(messages[-1].content) for temp, messages in self.calls if temp == temperature]


class _RecordingFakeChatModel(FakeListChatModel):
    recorder: Any
    temperature: float

    def _call(self, messages, *args, **kwargs):
        self.recorder.record(self.temperature, messages)
        return super()._call(messages, *args, **kwargs)


@pytest.fixture
def recorder(monkeypatch) -> _CallRecorder:
    recorder = _CallRecorder()

    def fake_build_llm(temperature: float = 0.1):
        return _RecordingFakeChatModel(
            responses=[f"summary@{temperature}"],
            recorder=recorder,
            temperature=temperature,
        )

    monkeypatch.setattr(summary_graph, "_build_llm", fake_build_llm)
    return recorder


def _section(title: str, text: str = "Some section body text.") -> dict[str, Any]:
    return {"title": title, "text": text}


def _four_bucket_sections() -> list[dict[str, Any]]:
    return [
        *[_section(f"Introduction {i}") for i in range(4)],
        *[_section(f"Method {i}") for i in range(5)],
        *[_section(f"Experiments {i}") for i in range(5)],
        *[_section(f"Conclusion {i}") for i in range(5)],
        *[_section(f"Acknowledgments {i}") for i in range(3)],
    ]


class TestSectionSelection:
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
    def test_bucket_rules_apply_in_order(self, title, bucket):
        assert summary_graph._classify_section_bucket(title) == bucket

    def test_caps_per_bucket_and_fills_with_other(self):
        selected = summary_graph._select_sections(_four_bucket_sections())

        titles = [section["title"] for section in selected]
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

    def test_group_sections_drops_other(self):
        grouped = summary_graph._group_sections(summary_graph._select_sections(_four_bucket_sections()))

        assert set(grouped) == {"background", "method", "experiments", "limitations"}
        assert {bucket: len(items) for bucket, items in grouped.items()} == {
            "background": 2,
            "method": 3,
            "experiments": 3,
            "limitations": 3,
        }

    def test_other_only_input_is_capped_at_twelve(self):
        sections = [_section(f"Appendix {i}") for i in range(20)]

        selected = summary_graph._select_sections(sections)

        assert len(selected) == 12
        assert all(not items for items in summary_graph._group_sections(selected).values())

    def test_skips_non_dict_and_blank_sections(self):
        sections = ["not a dict", _section("Method", "   "), _section("Method", "body"), None]

        assert summary_graph._select_sections(sections) == [_section("Method", "body")]


class TestLimits:
    def test_constants(self):
        assert summary_graph._SECTION_TEXT_MAX_CHARS == 3200
        assert summary_graph._MERGED_SUMMARY_MAX_CHARS == 22000
        assert summary_graph._FALLBACK_TEXT_MAX_CHARS == 12000

    def test_section_text_is_compacted_to_3200_chars(self, recorder):
        long_text = "word " * 2000

        summary_graph._summarize_bucket(
            title="T",
            bucket="method",
            sections=[_section("Method", long_text)],
            runtime="test",
            user="tester",
            quality_score=None,
        )

        (section_prompt,) = recorder.human_texts(SECTION_TEMPERATURE)
        section_body = section_prompt.split("섹션 본문:\n", 1)[1].split("\n\n요구사항:", 1)[0]
        assert len(section_body) == 3200
        assert section_body.endswith("…")

    def test_fallback_text_is_truncated_to_12000_chars(self):
        state = summary_graph._normalize_input(
            {"title": "T", "authors": "A", "fallback_text": "x" * 20000, "runtime": "test"}
        )

        assert state["fallback_text"] == "x" * 12000 + TRUNCATION_SUFFIX

    def test_merged_summaries_are_truncated_to_22000_chars(self):
        state = {
            "background_summary": "b" * 9000,
            "method_summary": "m" * 9000,
            "experiments_summary": "e" * 9000,
            "limitations_summary": "l" * 9000,
            "grouped_sections": {},
        }

        merged = summary_graph._merge_section_summaries_node(state)["merged_section_summaries"]

        assert len(merged) == 22000 + len(TRUNCATION_SUFFIX)
        assert merged.endswith(TRUNCATION_SUFFIX)

    def test_merge_attaches_up_to_two_evidence_sections(self):
        grouped = {
            "method": [_section("Method 0", "alpha"), _section("Method 1", "beta"), _section("Method 2", "gamma")]
        }

        merged = summary_graph._merge_section_summaries_node(
            {"method_summary": "method text", "grouped_sections": grouped}
        )["merged_section_summaries"]

        assert "[근거 1] Method 0\nalpha" in merged
        assert "[근거 2] Method 1\nbeta" in merged
        assert "gamma" not in merged

    def test_merge_falls_back_to_fallback_text(self):
        merged = summary_graph._merge_section_summaries_node({"fallback_text": "fallback", "grouped_sections": {}})

        assert merged == {"merged_section_summaries": "fallback"}


class TestGraphRun:
    def _run(self, sections, text: str = "") -> str:
        return summary_graph.generate_summary_via_graph(
            title="Paper",
            authors="A. Author",
            text=text,
            sections=sections,
            runtime="test",
            user="tester",
            quality_score=None,
        )

    def test_full_four_bucket_input_makes_sixteen_llm_calls(self, recorder):
        result = self._run(_four_bucket_sections())

        assert result == f"summary@{FINAL_TEMPERATURE}"
        assert len(recorder.calls) == 16
        assert recorder.count_by_stage() == {
            SECTION_TEMPERATURE: 11,
            BUCKET_TEMPERATURE: 4,
            FINAL_TEMPERATURE: 1,
        }

    def test_final_prompt_receives_merged_bucket_summaries(self, recorder):
        self._run(_four_bucket_sections())

        (final_prompt,) = recorder.human_texts(FINAL_TEMPERATURE)
        for label in ("배경/문제", "방법", "실험/결과", "한계/결론"):
            assert f"[{label} 원문 근거]" in final_prompt
        assert "Acknowledgments" not in final_prompt

    def test_empty_buckets_skip_llm_calls(self, recorder):
        self._run([_section("Method", "body"), _section("Conclusion", "body")])

        assert recorder.count_by_stage() == {
            SECTION_TEMPERATURE: 2,
            BUCKET_TEMPERATURE: 2,
            FINAL_TEMPERATURE: 1,
        }

    def test_no_classified_sections_uses_fallback_text_only(self, recorder):
        result = self._run([_section("Appendix", "body")], text="fallback body")

        assert result == f"summary@{FINAL_TEMPERATURE}"
        assert recorder.count_by_stage() == {FINAL_TEMPERATURE: 1}
        assert "fallback body" in recorder.human_texts(FINAL_TEMPERATURE)[0]

    def test_nothing_to_summarize_skips_llm(self, recorder):
        assert self._run([], text="") == ""
        assert recorder.calls == []
