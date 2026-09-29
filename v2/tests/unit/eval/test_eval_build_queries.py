import json
from typing import Any

import pytest

from arxplore_eval.build_queries import (
    ChunkQuestionOutput,
    ChunkSample,
    KnownItemOutput,
    PaperSample,
    all_match_placeholders,
    attach_catalogue,
    attach_preview_lines,
    candidate_sql,
    choose_distinct,
    chunk_lang_for,
    combine_query_files,
    dry_run_lines,
    generate_queries,
    load_placeholder_registry,
    merge_with_existing,
    parse_overrides,
    read_case_payloads,
    sample_ids,
)
from arxplore_eval.dataset import DatasetError, EvalQuery, dump_queries, find_placeholders, load_queries

PAPERS = {"0000.0000a": ("2405.01234", "Direct Preference Optimization Revisited For Everyone")}


class TestSampling:
    def test_sampling_is_deterministic_and_streams_are_independent(self) -> None:
        ids = list(range(100))
        first = sample_ids(ids, 5, seed=42, stream="papers")
        assert first == sample_ids(ids, 5, seed=42, stream="papers")
        assert first != sample_ids(ids, 5, seed=42, stream="chunks")
        assert first != sample_ids(ids, 5, seed=7, stream="papers")
        assert sample_ids(ids[:3], 10, seed=42, stream="papers") == sample_ids(ids[:3], 3, seed=42, stream="papers")

    def test_chunk_language_policy(self) -> None:
        assert [chunk_lang_for(index, "alternate") for index in range(3)] == ["en", "ko", "en"]
        assert chunk_lang_for(1, "en") == "en"


class FakeStructuredModel:
    def __init__(self, responses: list[dict[str, Any]], seen: list[tuple[type, Any]]) -> None:
        self.responses = responses
        self.seen = seen
        self.schema: type = object

    async def ainvoke(self, messages: Any) -> Any:
        self.seen.append((self.schema, messages))
        return self.responses.pop(0)


class FakeModel:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.seen: list[tuple[type, Any]] = []

    def with_structured_output(self, schema: type) -> FakeStructuredModel:
        structured = FakeStructuredModel(self.responses, self.seen)
        structured.schema = schema
        return structured


class TestGenerateQueries:
    async def test_rejects_verbatim_and_empty_outputs(self) -> None:
        abstract = "We introduce a routing scheme that keeps expert load balanced without any auxiliary loss term."
        model = FakeModel(
            [
                {"ko": "보조 손실 없이 전문가 부하를 고르게 맞추는 MoE 라우팅", "en": abstract},
                {"question": "   "},
                {"question": "Which batch size was used for the ablation runs?"},
            ]
        )
        paper = PaperSample(arxiv_id="2401.00001", title="Balanced MoE", abstract=abstract, key_findings=())
        chunks = [
            ChunkSample(1, "2401.00001", 0, "Method", "text", "Balanced MoE", "en"),
            ChunkSample(2, "2401.00001", 5, "Experiments", "We use batch size 512.", "Balanced MoE", "en"),
        ]
        logs: list[str] = []

        records, rejected = await generate_queries([paper], chunks, model=model, max_shared_words=5, log=logs.append)

        assert [record.id for record in records] == ["ki-2401.00001-ko", "cs-2"]
        assert records[1].relevant_chunk_ids == (2,)
        assert [line.split(":")[0] for line in rejected] == ["ki-2401.00001-en", "cs-1"]
        assert [schema for schema, _ in model.seen] == [KnownItemOutput, ChunkQuestionOutput, ChunkQuestionOutput]
        assert model.seen[0][1][0][0] == "system"
        assert logs[0] == "[known_item 1/1] 2401.00001"

    def test_dry_run_prints_the_first_prompt_of_each_kind(self) -> None:
        paper = PaperSample(arxiv_id="2401.00001", title="T", abstract="A" * 10, key_findings=("finding",))
        chunk = ChunkSample(7, "2401.00001", 3, "Method", "passage", "T", "ko")
        lines = dry_run_lines([paper, paper], [chunk], show_prompts=False)
        assert lines[0] == "known_item 표본 2편 → 최대 4개 질의 (ko/en)"
        assert sum(1 for line in lines if line.endswith("--- prompt ---")) == 2
        assert "abstract+key_findings" in lines[1]
        assert lines[-1].startswith("\ndry-run")


class TestFiles:
    def test_merge_with_existing_keeps_old_ids(self, tmp_path) -> None:
        path = tmp_path / "queries.generated.jsonl"
        old = EvalQuery(id="ki-1-ko", query="old", lang="ko", relevant_arxiv_ids=("1",), source="known_item")
        path.write_text(dump_queries([old]), encoding="utf-8")
        new = EvalQuery(id="ki-2-ko", query="new", lang="ko", relevant_arxiv_ids=("2",), source="known_item")
        again = EvalQuery(id="ki-1-ko", query="dup", lang="ko", relevant_arxiv_ids=("1",), source="known_item")

        merged, skipped = merge_with_existing(path, [new, again], append=True)

        assert [query.id for query in merged] == ["ki-1-ko", "ki-2-ko"]
        assert merged[0].query == "old" and skipped == 1
        assert merge_with_existing(path, [new], append=False) == ([new], 0)

    def test_combine_writes_generated_then_cases(self, tmp_path) -> None:
        generated = tmp_path / "queries.generated.jsonl"
        cases = tmp_path / "queries.cases.jsonl"
        out = tmp_path / "queries.jsonl"
        first = EvalQuery(id="g", query="g", lang="en", relevant_arxiv_ids=("1",), source="known_item")
        second = EvalQuery(id="c", query="c", lang="en", relevant_arxiv_ids=("2",), source="manual")
        generated.write_text(dump_queries([first]), encoding="utf-8")
        cases.write_text(dump_queries([second]), encoding="utf-8")

        assert [query.id for query in combine_query_files(generated, cases, out)] == ["g", "c"]
        assert out.read_bytes() == generated.read_bytes() + cases.read_bytes()

        cases.write_text(dump_queries([first]), encoding="utf-8")
        with pytest.raises(DatasetError, match="duplicate"):
            combine_query_files(generated, cases, out)
        assert [query.id for query in load_queries(out)] == ["g", "c"]
        assert not (tmp_path / "queries.jsonl.tmp").exists()


class TestAttachIds:
    def test_choose_distinct_avoids_reusing_a_paper(self) -> None:
        chosen = choose_distinct(
            {"0000.0000a": [("1", "One"), ("2", "Two")], "0000.0000b": [("1", "One"), ("3", "Three")], "0000.0000c": []}
        )
        assert chosen == {"0000.0000a": ("1", "One"), "0000.0000b": ("3", "Three")}

    def test_choose_distinct_keeps_every_candidate_for_set_placeholders(self) -> None:
        chosen = choose_distinct(
            {"0000.0000a": [("1", "One")], "0000.0000r": [("1", "One"), ("2", "Two")], "0000.0000s": []},
            all_matches={"0000.0000r", "0000.0000s"},
        )
        assert chosen == {"0000.0000a": ("1", "One"), "0000.0000r": [("1", "One"), ("2", "Two")]}

    def test_attach_catalogue_drops_unresolved_and_present_absent_cases(self) -> None:
        payloads = [
            {"id": "keep", "query": "{{title:0000.0000a}}", "relevant_arxiv_ids": ["0000.0000a"]},
            {"id": "unresolved", "query": "x", "relevant_arxiv_ids": ["0000.0000b"]},
            {"id": "absent", "query": "AlexNet top-5?", "relevant_arxiv_ids": []},
        ]
        attached, dropped = attach_catalogue(payloads, PAPERS, absent_present={"absent": "1 AlexNet"})
        assert [payload["id"] for payload in attached] == ["keep"]
        assert attached[0]["relevant_arxiv_ids"] == ["2405.01234"]
        assert len(dropped) == 2 and dropped[0].startswith("unresolved") and dropped[1].startswith("absent")

    def test_candidate_sql(self) -> None:
        sql, params = candidate_sql(
            {"title_keyword": "Expert", "min_chunks": 25, "section_keyword": "Limitation", "content_role": "appendix"},
            5,
        )
        assert sql.count("%s") == len(params)
        assert params == ["%Expert%", 25, "%Limitation%", "appendix", 5]
        sql, params = candidate_sql({"title_keyword": "", "chunk_pattern": "Figure 1"}, 3)
        assert params == ["%%", 1, "%Figure 1%", 3]
        sql, params = candidate_sql(
            {"title_keywords": ["Quantization", "4-bit"], "exclude_title_keywords": ["Tokenizer"], "all_matches": True},
            50,
        )
        assert "(p.title ILIKE %s OR p.title ILIKE %s)" in sql and "p.title NOT ILIKE %s" in sql
        assert params == ["%Quantization%", "%4-bit%", "%Tokenizer%", 1, 50]

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            ({"title_keyword": "x", "title_keywords": "Diffusion"}, "title_keywords"),
            ({"title_keyword": "x", "exclude_title_keywords": [""]}, "exclude_title_keywords"),
            ({"title_keyword": "x", "all_matches": "yes"}, "all_matches"),
            ({"title_keyword": "x", "mystery": 1}, "알 수 없는 키"),
        ],
    )
    def test_registry_rejects_bad_fields(self, tmp_path, entry: dict[str, Any], message: str) -> None:
        path = tmp_path / "placeholders.json"
        path.write_text(json.dumps({"placeholders": {"0000.0000a": entry}, "absent": {}}), encoding="utf-8")
        with pytest.raises(DatasetError, match=message):
            load_placeholder_registry(path)

    def test_overrides(self) -> None:
        assert parse_overrides(["0000.0000a=2405.01234"]) == {"0000.0000a": "2405.01234"}
        with pytest.raises(DatasetError, match="--set"):
            parse_overrides(["a=b"])

    def test_attached_catalogue_is_valid(self, tmp_path, eval_dir) -> None:
        registry = load_placeholder_registry(eval_dir / "cases" / "placeholders.json")
        papers: dict[str, Any] = {}
        for index, placeholder in enumerate(sorted(registry["placeholders"]), start=1):
            paper = (f"2501.{index:05d}", f"Paper number {index} about things")
            if registry["placeholders"][placeholder].get("all_matches"):
                papers[placeholder] = [paper, (f"2502.{index:05d}", f"Second paper {index}")]
            else:
                papers[placeholder] = paper
        payloads = read_case_payloads(eval_dir / "cases")

        attached, dropped = attach_catalogue(payloads, papers, absent_present={})

        assert dropped == []
        out = tmp_path / "queries.cases.jsonl"
        out.write_text(
            "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in attached), encoding="utf-8"
        )
        loaded = load_queries(out)
        assert len(loaded) == len(load_queries(eval_dir / "cases"))
        assert not find_placeholders([query.to_dict() for query in loaded])
        by_id = {query.id: query for query in loaded}
        assert by_id["qf-exact-title"].query == papers["0000.0000g"][1]
        assert by_id["qf-partial-title"].query.startswith("Paper number 9 about ")
        assert len(by_id["qf-single-word-en"].relevant_arxiv_ids) == 2
        assert by_id["qf-single-word-en"].retrieval_eligible

        needed = sorted(find_placeholders(payloads))
        sets = all_match_placeholders(needed, registry, {})
        assert sets and all(registry["placeholders"][name].get("all_matches") for name in sets)
        preview = attach_preview_lines(needed, registry, {name: [] for name in needed}, papers)
        assert preview[0].startswith("자리표시자 → 선택한 논문")
        assert any("집합 2편" in line for line in preview)
