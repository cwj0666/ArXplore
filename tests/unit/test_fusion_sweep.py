from __future__ import annotations

import gzip
import json
import random
import statistics
from collections import Counter
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from eval.candidate_cache import (
    CachedQuery,
    CacheError,
    capture_query,
    compact_candidate,
    compare_live,
    gate,
    read_cache,
    recorded_fusion_config,
    replay,
    restore_candidate,
    write_cache,
)
from eval.dataset import EvalQuery
from eval.fusion_sweep import (
    CONVEX,
    CONVEX_ALPHAS,
    CONVEX_KEEP,
    CURRENT,
    PLAIN_RRF_K60,
    VECTOR_ONLY,
    BootstrapResult,
    Selection,
    SweepSpec,
    bootstrap_draws,
    build_convex_specs,
    build_specs,
    cluster_key,
    cluster_strata,
    clusters_of,
    convex_groups,
    count_parameters,
    cross_validate,
    degenerate_normalizations,
    fold_plan,
    paired_bootstrap,
    paper_rank,
    pre_registered_decision,
    query_differences,
    render_convex_sections,
    run_sweep,
    select_one_se,
    stratified_folds,
)
from eval.runner import run_query
from src.integrations.hybrid_fusion import (
    DEFAULT_HYBRID_FUSION,
    LEGACY_RULES_FUSION,
    STRICT_LEXICAL_SCORE_FLOOR,
    HybridFusionConfig,
)
from tests.unit.test_eval_runner import _load_script, _real_retriever
from tests.unit.test_retrieval_fusion import _retriever, synthetic_fusion_case

SOURCES = ("known_item", "manual", "llm_synth")


def _synthetic_cached(
    seed: int, *, k: int = 5, query_id: str | None = None, config: HybridFusionConfig = DEFAULT_HYBRID_FUSION
) -> CachedQuery:
    """합성 융합 입력 + 제품 `_merge_hybrid_candidates`(`config`, 기본은 제품 기본값) 결과를 live로 담은 캐시 질의.
    정답은 후보 논문 중 하나다."""
    case = synthetic_fusion_case(seed)
    rng = random.Random(seed)
    papers = sorted({row["arxiv_id"] for row in [*case["lexical"], *case["vector"]]}) or ["2409.00000"]
    query = EvalQuery(
        id=query_id or f"q{seed:03d}",
        query=case["query"],
        lang=rng.choice(("ko", "en")),
        relevant_arxiv_ids=(rng.choice(papers),),
        source=SOURCES[seed % 3],
    )
    live = _retriever()._merge_hybrid_candidates(
        case["query"], case["lexical"], case["vector"], arxiv_id=None, limit=k, config=config
    )
    return CachedQuery(
        query=query,
        normalized_query=case["query"],
        lexical=tuple(case["lexical"]),
        vector=tuple(case["vector"]),
        live={"hybrid": tuple(live)},
    )


def _roundtrip(
    tmp_path, queries: list[CachedQuery], *, k: int = 5, name: str = "cache.jsonl.gz", header: dict | None = None
):
    path = tmp_path / name
    write_cache(path, {"k": k, "git_revision": "test", **(header or {})}, queries)
    return path, *read_cache(path)


class TestCandidateCache:
    def test_compact_keeps_only_fusion_fields(self):
        candidate = {
            "chunk_id": 7,
            "arxiv_id": "2409.00001",
            "score": 1.25,
            "content_role": "body",
            "section_title": "Method",
            "chunk_text": "long text",
            "score_breakdown": {"strict_match": False, "coverage": 0.4},
        }

        lexical = compact_candidate(candidate, channel="lexical")
        vector = compact_candidate(candidate, channel="vector")

        assert set(lexical) == {"chunk_id", "arxiv_id", "score", "content_role", "section_title", "strict_match"}
        assert "strict_match" not in vector
        assert restore_candidate(lexical)["score_breakdown"] == {"strict_match": False}
        assert restore_candidate(vector)["score_breakdown"] == {}

    def test_roundtrip_preserves_replay_and_is_byte_stable(self, tmp_path):
        queries = [_synthetic_cached(seed) for seed in range(8)]
        path, header, loaded = _roundtrip(tmp_path, queries)
        second, _, _ = _roundtrip(tmp_path, queries, name="again.jsonl.gz")

        assert header["k"] == 5 and header["format"] == 1
        assert [cached.query for cached in loaded] == [cached.query for cached in queries]
        for original, restored in zip(queries, loaded, strict=True):
            assert [hit["chunk_id"] for hit in replay(original, k=5)] == [
                hit["chunk_id"] for hit in replay(restored, k=5)
            ]
        assert path.read_bytes() == second.read_bytes()

    def test_gate_passes_on_faithful_cache_and_reports_tampering(self, tmp_path):
        _, _, loaded = _roundtrip(tmp_path, [_synthetic_cached(seed) for seed in range(12)])
        assert gate(loaded, k=5) == []

        tampered = replace(loaded[3], live={"hybrid": tuple(reversed(loaded[3].live["hybrid"]))})
        mismatches = gate([*loaded[:3], tampered, *loaded[4:]], k=5)

        assert [item.query_id for item in mismatches] == [tampered.id]
        assert mismatches[0].live != mismatches[0].replayed

    def test_gate_replays_the_fusion_config_recorded_in_the_header(self, tmp_path):
        legacy = [_synthetic_cached(seed, config=LEGACY_RULES_FUSION) for seed in range(15)]
        _, header, loaded = _roundtrip(tmp_path, legacy, header={"fusion_default": asdict(LEGACY_RULES_FUSION)})

        assert recorded_fusion_config(header) == LEGACY_RULES_FUSION
        assert gate(loaded, k=5, config=recorded_fusion_config(header)) == []
        assert gate(loaded, k=5) != []

    def test_recorded_config_defaults_and_rejects_unknown_fields(self):
        assert recorded_fusion_config({"k": 5}) == DEFAULT_HYBRID_FUSION
        old_header = {
            name: value
            for name, value in asdict(LEGACY_RULES_FUSION).items()
            if not name.startswith(("convex", "score"))
        }
        assert recorded_fusion_config({"fusion_default": old_header}) == LEGACY_RULES_FUSION
        with pytest.raises(CacheError):
            recorded_fusion_config({"fusion_default": {"weighting": "rules", "mystery": 1}})

    def test_gate_fails_when_live_is_missing(self):
        cached = replace(_synthetic_cached(0), live={})
        assert [item.query_id for item in gate([cached], k=5)] == [cached.id]

    def test_replay_changes_with_config(self):
        cached = next(
            cached
            for cached in (_synthetic_cached(seed) for seed in range(40))
            if len(cached.lexical) > 5 and len(cached.vector) > 5
        )
        vector_heavy = HybridFusionConfig(weighting="static", static_weights=(0.0, 1.0), quality_weight=False)
        lexical_heavy = HybridFusionConfig(weighting="static", static_weights=(1.0, 0.0), quality_weight=False)
        assert replay(cached, k=5, config=vector_heavy) != replay(cached, k=5, config=lexical_heavy)

    def test_rejects_unknown_format(self, tmp_path):
        path = tmp_path / "bad.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": "header", "format": 99}) + "\n")
        with pytest.raises(CacheError, match="형식"):
            read_cache(path)

    def test_capture_matches_eval_hybrid_on_the_product_retriever(self, tmp_path):
        query = EvalQuery(id="q", query="benchmark evaluation", lang="en", relevant_arxiv_ids=("A",), source="manual")
        retriever = _real_retriever()

        cached = capture_query(retriever, query, k=3)
        _, _, (loaded,) = _roundtrip(tmp_path, [cached], k=3)

        assert [hit["chunk_id"] for hit in cached.live["hybrid"]] == run_query(
            retriever, "hybrid", query, k=3
        ).retrieved_chunk_ids
        assert [hit["chunk_id"] for hit in cached.live["vector"]] == run_query(
            retriever, "vector", query, k=3
        ).retrieved_chunk_ids
        assert len(cached.lexical) <= 10 and len(cached.vector) <= 10
        assert gate([loaded], k=3) == []
        assert compare_live([loaded], k=3, channel="vector") == []


class TestSpecs:
    def test_grid_sizes_and_unique_names(self):
        specs = build_specs()
        families = Counter(spec.family for spec in specs)
        assert families == {"R0_vector": 1, "R0_lexical": 1, "F0": 5, "F1": 220, "F2": 720, "C": 1, "CF": 8, "REF": 2}
        assert len({spec.name for spec in specs}) == len(specs)

    def test_factorial_corners_are_current_and_plain_rrf(self):
        by_name = {spec.name: spec for spec in build_specs()}
        assert by_name["CF_m1q1b1"].config == by_name[CURRENT].config == LEGACY_RULES_FUSION
        assert by_name["CF_m0q0b0"].config == by_name[PLAIN_RRF_K60].config
        assert by_name[PLAIN_RRF_K60].config.drop_partial_lexical
        assert not by_name["REF_plainrrf_ablation"].config.drop_partial_lexical

    def test_parameter_counts(self):
        by_name = {spec.name: spec for spec in build_specs()}
        assert by_name[CURRENT].n_params == 25
        assert by_name[PLAIN_RRF_K60].n_params == 1
        assert by_name["F1_w0.3_k60_q0_b0"].n_params == 2
        assert by_name["F1_w1.0_k60_q0_b0"].n_params == 1
        assert by_name["F2_wmin0.5_tau0.3_k20_q1_b0.015"].n_params == 1 + 2 + 8 + 1
        assert by_name["CF_m1q0b0"].n_params == 16
        assert by_name[VECTOR_ONLY].n_params == 0
        assert (
            count_parameters(
                HybridFusionConfig(weighting="static", static_weights=(0.5, 0.5), quality_weight=False, overlap_bonus=0)
            )
            == 3
        )


def _query_layout() -> list[SimpleNamespace]:
    """실제 질의셋과 같은 구성: known_item 쌍 27 + 혼자 3, manual 쌍 10 + 혼자 23, llm_synth 혼자 11 = 111개, 클러스터 74개."""
    queries: list[SimpleNamespace] = []
    for number in range(27):
        queries += [SimpleNamespace(id=f"ki-{number}-{lang}", source="known_item", lang=lang) for lang in ("ko", "en")]
    queries += [SimpleNamespace(id=f"ki-solo-{number}", source="known_item", lang="ko") for number in range(3)]
    for number in range(10):
        queries += [SimpleNamespace(id=f"lang-{number}-{lang}", source="manual", lang=lang) for lang in ("ko", "en")]
    queries += [
        SimpleNamespace(id=f"ev-{number}", source="manual", lang=("ko", "en")[number % 2]) for number in range(23)
    ]
    queries += [
        SimpleNamespace(id=f"cs-{number}", source="llm_synth", lang=("ko", "en")[number % 2]) for number in range(11)
    ]
    return queries


class TestClusters:
    def test_cluster_key_strips_only_a_trailing_language_suffix(self):
        assert cluster_key("ki-2609.26355-ko") == cluster_key("ki-2609.26355-en") == "ki-2609.26355"
        assert cluster_key("lang-ko-typo") == "lang-ko-typo"
        assert cluster_key("lang-ko") == "lang"
        assert cluster_key("cs-1892") == "cs-1892"

    def test_cluster_strata_use_pair_or_language(self):
        groups, strata = cluster_strata(_query_layout())
        assert len(groups) == 111 and len(set(groups)) == 74
        assert Counter(strata) == {
            "known_item:pair": 54,
            "known_item:ko": 3,
            "manual:pair": 20,
            "manual:ko": 12,
            "manual:en": 11,
            "llm_synth:ko": 6,
            "llm_synth:en": 5,
        }

    def test_clusters_of_keeps_first_appearance_order(self):
        assert clusters_of(["a", "b", "a", "c"]) == [[0, 2], [1], [3]]


class TestFolds:
    def _plan(self, *, repeats: int = 3, seed: int = 7) -> tuple[list[str], list[str], list[list[int]]]:
        groups, strata = cluster_strata(_query_layout())
        return groups, strata, fold_plan(groups, strata, n_folds=5, repeats=repeats, seed=seed)

    def test_pairs_are_never_split_across_folds(self):
        groups, _, plan = self._plan(repeats=10)
        for folds in plan:
            by_group: dict[str, set[int]] = {}
            for group, fold in zip(groups, folds, strict=True):
                by_group.setdefault(group, set()).add(fold)
            assert all(len(assigned) == 1 for assigned in by_group.values())

    def test_folds_are_balanced_and_each_stratum_is_spread(self):
        _, strata, plan = self._plan(repeats=10)
        for folds in plan:
            sizes = Counter(folds)
            assert len(sizes) == 5
            assert max(sizes.values()) - min(sizes.values()) <= 2
            for name in set(strata):
                counts = Counter(fold for fold, value in zip(folds, strata, strict=True) if value == name)
                per_fold = [counts.get(fold, 0) for fold in range(5)]
                assert max(per_fold) - min(per_fold) <= 2

    def test_plan_is_deterministic_and_repeats_differ(self):
        _, _, first = self._plan()
        assert first == self._plan()[2]
        assert first[0] != first[1]
        assert first != self._plan(seed=8)[2]

    def test_cluster_spanning_strata_is_rejected(self):
        with pytest.raises(ValueError, match="spans"):
            stratified_folds(["a", "a"], ["x", "y"], 2, random.Random(0))

    def test_cross_validation_of_a_single_config_is_its_mean(self):
        rng = random.Random(3)
        _, _, plan = self._plan(repeats=4, seed=1)
        scores = {"only": [rng.random() for _ in range(111)]}

        result = cross_validate("x", [SweepSpec("only", "x", config=LEGACY_RULES_FUSION)], scores, plan)

        assert result.mean == pytest.approx(statistics.fmean(scores["only"]))
        assert result.se > 0
        assert result.choices == (("only", 20),)

    def test_tuning_picks_the_train_best_and_breaks_ties_by_simplicity(self):
        simple = SweepSpec("simple", "x", config=HybridFusionConfig(weighting="static"))
        complex_ = SweepSpec("complex", "x", config=LEGACY_RULES_FUSION)
        worse = SweepSpec("worse", "x", config=HybridFusionConfig(weighting="static", quality_weight=False))
        scores = {"simple": [0.5] * 111, "complex": [0.5] * 111, "worse": [0.4] * 111}
        _, _, plan = self._plan(repeats=2, seed=1)

        result = cross_validate("x", [complex_, worse, simple], scores, plan)

        assert result.choices == (("simple", 10),)
        assert result.mean == pytest.approx(0.5)


class TestBootstrap:
    def test_identical_systems_give_a_zero_interval(self):
        values = [random.Random(1).random() for _ in range(30)]
        assert paired_bootstrap(values, values, bootstrap_draws(30, 200, seed=1)) == (0.0, 0.0, 0.0)

    def test_constant_shift_is_recovered_exactly(self):
        base = [random.Random(2).random() for _ in range(25)]
        shifted = [value + 0.1 for value in base]
        delta, low, high = paired_bootstrap(shifted, base, bootstrap_draws(25, 300, seed=2))
        assert (delta, low, high) == pytest.approx((0.1, 0.1, 0.1))

    def test_interval_contains_the_mean_and_is_deterministic(self):
        rng = random.Random(4)
        a = [float(rng.random() < 0.7) for _ in range(80)]
        b = [float(rng.random() < 0.5) for _ in range(80)]
        draws = bootstrap_draws(80, 2000, seed=11)

        delta, low, high = paired_bootstrap(a, b, draws)

        assert low < delta < high
        assert (delta, low, high) == paired_bootstrap(a, b, bootstrap_draws(80, 2000, seed=11))
        assert 0.0 < high - low < 0.6

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            paired_bootstrap([1.0], [1.0, 0.0], [[0]])

    def test_clusters_must_partition_the_queries(self):
        with pytest.raises(ValueError, match="partition"):
            paired_bootstrap([1.0, 0.0], [0.0, 0.0], [[0]], clusters=[[0]])

    def test_cluster_resampling_uses_the_per_query_mean(self):
        a = [1.0, 1.0, 0.0]
        b = [0.0, 0.0, 0.0]
        clusters = [[0, 1], [2]]

        assert paired_bootstrap(a, b, [[0, 0]], clusters=clusters) == (pytest.approx(2 / 3), 1.0, 1.0)
        assert paired_bootstrap(a, b, [[0, 1]], clusters=clusters)[1] == pytest.approx(2 / 3)
        assert paired_bootstrap(a, b, [[1, 1]], clusters=clusters)[1:] == (0.0, 0.0)

    def test_cluster_bootstrap_is_deterministic_and_wider_for_correlated_pairs(self):
        rng = random.Random(9)
        pair_deltas = [float(rng.random() < 0.6) for _ in range(40)]
        a = [value for value in pair_deltas for _ in range(2)]
        b = [0.0] * 80
        clusters = [[2 * index, 2 * index + 1] for index in range(40)]

        clustered = paired_bootstrap(a, b, bootstrap_draws(40, 3000, seed=5), clusters=clusters)
        naive = paired_bootstrap(a, b, bootstrap_draws(80, 3000, seed=5))

        assert clustered == paired_bootstrap(a, b, bootstrap_draws(40, 3000, seed=5), clusters=clusters)
        assert clustered[0] == naive[0]
        assert clustered[2] - clustered[1] > naive[2] - naive[1]


class TestOneStandardError:
    def _specs(self):
        return [
            SweepSpec("C", "C", config=LEGACY_RULES_FUSION),
            SweepSpec("mid", "F1", config=HybridFusionConfig(weighting="static", static_weights=(0.5, 1.0))),
            SweepSpec("plain", "F0", config=HybridFusionConfig(weighting="static", quality_weight=False)),
        ]

    def test_picks_the_fewest_parameters_within_one_se(self):
        scores = {"C": 0.80, "mid": 0.79, "plain": 0.775}
        selection = select_one_se(self._specs(), scores, lambda name: 0.03)
        assert selection.best == "C"
        assert selection.threshold == pytest.approx(0.77)
        assert selection.pick == "plain"
        assert selection.within == ("plain", "mid", "C")

    def test_keeps_the_best_when_nothing_simpler_is_close(self):
        scores = {"C": 0.80, "mid": 0.70, "plain": 0.60}
        selection = select_one_se(self._specs(), scores, lambda name: 0.01)
        assert selection.pick == "C"
        assert selection.within == ("C",)


def _bootstrap(label: str, subset: str, low: float, high: float) -> BootstrapResult:
    return BootstrapResult(label, subset, "mrr@10", 50, 40, (low + high) / 2, low, high)


class TestDecision:
    SELECTION = Selection("best", 0.8, 0.02, 0.78, "F0_rrf_k40", 0.79, ("F0_rrf_k40",))

    def test_rule_one_replaces_when_interval_includes_zero(self):
        lines = pre_registered_decision(
            self.SELECTION,
            [
                _bootstrap("pick − C", "all", -0.02, 0.01),
                _bootstrap("pick − C", "manual", -0.05, 0.03),
                _bootstrap("pick − vector_only", "all", 0.01, 0.04),
            ],
            primary="mrr@10",
        )
        assert lines[0].startswith("규칙 (1)")
        assert "유의하게 낫다" in lines[-1]

    def test_rule_two_keeps_current_and_checks_it_against_vector(self):
        lines = pre_registered_decision(
            self.SELECTION,
            [
                _bootstrap("pick − C", "all", -0.05, -0.01),
                _bootstrap("pick − C", "manual", -0.05, 0.01),
                _bootstrap("C − vector_only", "all", -0.01, 0.02),
            ],
            primary="mrr@10",
        )
        assert lines[0].startswith("규칙 (2)")
        assert lines[-1].startswith("규칙 (3)") and CURRENT in lines[-1]

    def test_manual_regression_blocks_replacement(self):
        lines = pre_registered_decision(
            self.SELECTION,
            [_bootstrap("pick − C", "all", -0.01, 0.02), _bootstrap("pick − C", "manual", -0.09, -0.01)],
            primary="mrr@10",
        )
        assert lines[0].startswith("manual 부분집합 비회귀 조건 실패")


def _small_specs() -> list[SweepSpec]:
    wanted = {VECTOR_ONLY, "lexical_only", CURRENT, PLAIN_RRF_K60, "F0_rrf_k10", "F1_w0.5_k60_q0_b0", "CF_m0q1b1"}
    return [spec for spec in build_specs() if spec.name in wanted or spec.family == "REF"]


class TestRunSweep:
    def test_end_to_end_is_deterministic(self):
        queries = [_synthetic_cached(seed) for seed in range(30)]
        kwargs = {"k": 5, "specs": _small_specs(), "seed": 3, "n_folds": 3, "repeats": 2, "n_resamples": 200}

        first = run_sweep(queries, **kwargs)
        second = run_sweep(queries, **kwargs)

        assert first.selection == second.selection
        assert first.comparisons == second.comparisons
        assert first.cv == second.cv
        assert first.selection.pick in {spec.name for spec in kwargs["specs"] if spec.selectable}
        assert {row.subset for row in first.comparisons} == {"all", "manual"}
        assert "REF" not in first.cv

    def test_pairs_form_bootstrap_clusters(self):
        queries = [
            _synthetic_cached(seed, query_id=f"p{seed // 2}-{'ko' if seed % 2 == 0 else 'en'}" if seed < 20 else None)
            for seed in range(30)
        ]
        report = run_sweep(queries, k=5, specs=_small_specs(), n_folds=3, repeats=2, n_resamples=100)

        overall = [row for row in report.comparisons if row.subset == "all"]
        assert {(row.n, row.n_clusters) for row in overall} == {(30, 20)}
        assert report.subset_sizes["all"] == (30, 20)

    def test_current_config_matches_the_recorded_live_metrics(self):
        """C는 2026-09-29까지의 제품 규칙이다. 그 규칙으로 기록한 live와 C 재생 순위가 같다."""
        queries = [_synthetic_cached(seed, config=LEGACY_RULES_FUSION) for seed in range(30)]
        report = run_sweep(queries, k=5, specs=_small_specs(), n_folds=3, repeats=1, n_resamples=50)
        live_chunks = [[hit["chunk_id"] for hit in cached.live["hybrid"]] for cached in queries]
        assert [row.retrieved_chunk_ids for row in report.results[CURRENT]] == live_chunks


class TestFusionSweepScript:
    def test_gate_uses_the_recorded_config_of_a_legacy_cache(self, tmp_path):
        script = _load_script("eval_fusion_sweep")
        legacy = [_synthetic_cached(seed, config=LEGACY_RULES_FUSION) for seed in range(15)]
        recorded, _, _ = _roundtrip(tmp_path, legacy, header={"fusion_default": asdict(LEGACY_RULES_FUSION)})
        unrecorded, _, _ = _roundtrip(tmp_path, legacy, name="no_header.jsonl.gz")

        assert script.main(["gate", "--cache", str(recorded)]) == 0
        assert script.main(["gate", "--cache", str(unrecorded)]) == 1

    def test_gate_and_sweep_commands(self, tmp_path, capsys):
        script = _load_script("eval_fusion_sweep")
        queries = [_synthetic_cached(seed) for seed in range(15)]
        good, _, _ = _roundtrip(tmp_path, queries)
        tampered = [*queries[:-1], replace(queries[-1], live={"hybrid": tuple(reversed(queries[-1].live["hybrid"]))})]
        bad, _, _ = _roundtrip(tmp_path, tampered, name="bad.jsonl.gz")
        out_dir = tmp_path / "results"

        assert script.main(["gate", "--cache", str(good)]) == 0
        assert script.main(["gate", "--cache", str(bad)]) == 1
        assert "불일치 q014" in capsys.readouterr().err
        assert script.main(["sweep", "--cache", str(bad), "--out-dir", str(out_dir)]) == 1
        assert not out_dir.exists()
        assert script.main(["gate", "--cache", str(tmp_path / "missing.jsonl.gz")]) == 2

    def test_sweep_writes_markdown_and_csv(self, tmp_path, monkeypatch):
        script = _load_script("eval_fusion_sweep")
        monkeypatch.setattr(
            script, "run_sweep", lambda queries, **kwargs: run_sweep(queries, specs=_small_specs(), **kwargs)
        )
        good, _, _ = _roundtrip(tmp_path, [_synthetic_cached(seed) for seed in range(15)])
        out_dir = tmp_path / "results"

        code = script.main(
            ["sweep", "--cache", str(good), "--out-dir", str(out_dir), "--repeats", "2", "--resamples", "100"]
        )

        assert code == 0
        markdown = next(out_dir.glob("fusion_*.md")).read_text(encoding="utf-8")
        assert "## 사전 등록 규칙 판정" in markdown and "## 1-SE 선택" in markdown
        assert "게이트: 기록 시점 설정(weighting=convex) 재생 = live hybrid, 15/15" in markdown
        rows = next(out_dir.glob("fusion_*.csv")).read_text(encoding="utf-8").splitlines()
        assert len(rows) == 1 + len(_small_specs())


class TestDumpCandidatesScript:
    def test_dump_writes_a_cache_that_passes_the_gate(self, tmp_path, monkeypatch, capsys):
        import types

        import src.integrations.embedding_client as embedding_module
        import src.integrations.paper_retriever as retriever_module
        import src.shared
        from eval.dataset import dump_queries

        settings = types.SimpleNamespace(
            openai_api_key="sk-test",
            openai_embedding_model="text-embedding-3-large",
            openai_embedding_dimensions=1536,
            vector_min_similarity=0.0,
        )
        monkeypatch.setattr(src.shared, "get_settings", lambda: settings)
        script = _load_script("eval_dump_candidates")
        corpus = {"papers": 4, "papers_with_chunks": 4, "chunks": 11, "embeddings": 11, "fulltext_sources": {"pdf": 4}}
        monkeypatch.setattr(script, "preflight", lambda settings, queries, needs_embeddings: corpus)
        monkeypatch.setattr(embedding_module, "EmbeddingClient", lambda: None)
        monkeypatch.setattr(retriever_module, "PaperRetriever", lambda embedding_client: _real_retriever())
        queries = [
            EvalQuery(id="q1", query="benchmark evaluation", lang="en", relevant_arxiv_ids=("A",), source="manual"),
            EvalQuery(id="q2", query="평가 benchmark", lang="ko", relevant_arxiv_ids=("B",), source="known_item"),
            EvalQuery(
                id="r1",
                query="x",
                lang="en",
                relevant_arxiv_ids=(),
                source="manual",
                expected_behavior="refuse",
                mode="agent",
            ),
        ]
        queries_path = tmp_path / "queries.jsonl"
        queries_path.write_text(dump_queries(queries), encoding="utf-8")
        out = tmp_path / "cache" / "c.jsonl.gz"

        code = script.main(["--queries", str(queries_path), "--k", "3", "--out", str(out)])

        assert code == 0
        header, loaded = read_cache(out)
        assert [cached.id for cached in loaded] == ["q1", "q2"]
        assert header["k"] == 3 and header["branch_limit"] == 10 and header["branch_fetch_limit"] == 50
        assert header["queries"]["skipped_ids"] == ["r1"]
        assert header["corpus"] == corpus
        assert header["fusion_default"]["rank_constant"] == 60.0
        assert gate(loaded, k=3) == []
        assert "모두 같습니다" in capsys.readouterr().err

    def test_memo_embedding_client_calls_once_per_text(self):
        script = _load_script("eval_dump_candidates")
        calls: list[list[str]] = []

        class Inner:
            def embed_texts(self, texts):
                calls.append(list(texts))
                return [[float(len(text))] for text in texts]

        client = script.MemoEmbeddingClient(Inner())
        assert client.embed_texts(["a"]) == client.embed_texts(["a"]) == [[1.0]]
        client.embed_texts(["bb"])
        assert calls == [["a"], ["bb"]]


class TestConvexSpecs:
    def test_families_grid_and_names(self):
        specs = build_convex_specs()
        assert Counter(spec.family for spec in specs) == {CONVEX: 60, CONVEX_KEEP: 40}
        names = {spec.name for spec in specs}
        assert len(names) == len(specs)
        assert not names & {spec.name for spec in build_specs()}
        assert CONVEX_ALPHAS[0] == 0.0 and CONVEX_ALPHAS[-1] == 0.95 and len(CONVEX_ALPHAS) == 20
        assert all(spec.config.convex_alpha < 1.0 for spec in specs)

    def test_primary_family_drops_partial_rows_and_is_the_only_selectable_one(self):
        for spec in build_convex_specs():
            assert spec.config.weighting == "convex"
            assert spec.config.drop_partial_lexical == (spec.family == CONVEX)
            assert spec.selectable == (spec.family == CONVEX)
            assert spec.n_params == 1

    def test_normalization_floors(self):
        by_name = {spec.name: spec.config for spec in build_convex_specs()}
        assert by_name["CC_mm_a0.20"].score_normalization == "minmax"
        assert by_name["CC_tmm_a0.20"].score_normalization == "theoretical"
        assert by_name["CC_tmm_a0.20"].score_floors == (0.0, -1.0)
        assert by_name["CC_tmms_a0.20"].score_floors == (STRICT_LEXICAL_SCORE_FLOOR, -1.0)
        assert "CCK_tmms_a0.20" not in by_name

    def test_groups_split_by_normalization(self):
        groups = convex_groups(build_convex_specs())
        assert list(groups) == ["CC_mm", "CC_tmm", "CC_tmms", "CCK_mm", "CCK_tmm"]
        assert {len(members) for members in groups.values()} == {20}

    def test_product_default_is_the_pre_registered_pick(self):
        by_name = {spec.name: spec.config for spec in build_convex_specs()}
        assert by_name["CC_mm_a0.35"] == DEFAULT_HYBRID_FUSION

    def test_convex_config_counts_one_parameter(self):
        assert count_parameters(HybridFusionConfig(weighting="convex", convex_alpha=0.3)) == 1


class TestRrfCheck:
    SELECTION = Selection("CC_mm_a0.30", 0.8, 0.02, 0.78, "CC_mm_a0.30", 0.8, ("CC_mm_a0.30",))

    @pytest.mark.parametrize(
        ("low", "high", "expected"),
        [(0.01, 0.03, "재현했다"), (-0.03, -0.01, "반대다"), (-0.01, 0.02, "재현하지 못했다")],
    )
    def test_rule_four_reads_the_pick_against_standard_rrf(self, low, high, expected):
        comparisons = [
            _bootstrap("pick − C", "all", -0.01, 0.02),
            _bootstrap("pick − C", "manual", -0.02, 0.03),
            _bootstrap("pick − F0(k=60)", "all", low, high),
        ]
        lines = pre_registered_decision(self.SELECTION, comparisons, primary="mrr@10", rrf_check=True)
        assert lines[0].startswith("규칙 (1)")
        assert lines[-1].startswith("규칙 (4)") and expected in lines[-1]
        assert not any(
            line.startswith("규칙 (4)")
            for line in pre_registered_decision(self.SELECTION, comparisons, primary="mrr@10")
        )


def _convex_run(queries: list[CachedQuery]):
    specs = _small_specs() + build_convex_specs()
    return run_sweep(
        queries,
        k=5,
        specs=specs,
        n_folds=3,
        repeats=2,
        n_resamples=100,
        selectable_families=(CONVEX,),
        cv_groups=convex_groups(specs),
        rrf_check=True,
    )


class TestConvexSweep:
    def test_pick_comes_from_the_convex_family(self):
        queries = [_synthetic_cached(seed) for seed in range(30)]
        report = _convex_run(queries)

        assert report.selection.pick.startswith("CC_")
        assert set(report.selection.within) <= {spec.name for spec in build_convex_specs() if spec.family == CONVEX}
        assert {"CC", "CCK", "CC_mm", "CC_tmm", "CC_tmms", "CCK_mm", "CCK_tmm"} <= set(report.cv)
        assert report.selectable_families == (CONVEX,)
        assert report.decision[-1].startswith("규칙 (4)")

    def test_cv_group_names_must_not_shadow_families(self):
        queries = [_synthetic_cached(seed) for seed in range(12)]
        with pytest.raises(ValueError):
            run_sweep(
                queries, k=5, specs=_small_specs(), n_folds=3, repeats=1, n_resamples=10, cv_groups={"C": [CURRENT]}
            )

    def test_query_differences_and_ranks_agree_with_the_metrics(self):
        queries = [_synthetic_cached(seed) for seed in range(30)]
        report = _convex_run(queries)
        pick = report.selection.pick

        differences = query_differences(report, pick, CURRENT)
        changed = {
            left.query_id
            for left, right in zip(report.results[pick], report.results[CURRENT], strict=True)
            if left.paper_metrics["mrr@5"] != right.paper_metrics["mrr@5"]
        }
        assert {item.query_id for item in differences} == changed
        assert query_differences(report, CURRENT, CURRENT) == []
        for row in report.results[CURRENT]:
            rank = paper_rank(row)
            assert (rank is None) == (row.paper_metrics["mrr@5"] == 0)
            if rank is not None:
                assert row.paper_metrics["mrr@5"] == pytest.approx(1 / rank)

    def test_degenerate_counts(self):
        vector = (
            {"chunk_id": 3, "arxiv_id": "a", "score": 0.9, "score_breakdown": {}},
            {"chunk_id": 4, "arxiv_id": "b", "score": 0.5, "score_breakdown": {}},
        )
        single = {"chunk_id": 1, "arxiv_id": "c", "score": 1.2, "score_breakdown": {"strict_match": True}}
        partial = {"chunk_id": 2, "arxiv_id": "d", "score": 0.4, "score_breakdown": {"strict_match": False}}
        base = _synthetic_cached(0)
        queries = [
            replace(base, lexical=(single, partial), vector=vector),
            replace(base, lexical=(partial,), vector=vector),
        ]
        by_name = {spec.name: spec.config for spec in build_convex_specs()}

        assert degenerate_normalizations(queries, by_name["CC_mm_a0.30"]) == {
            "lexical": 1,
            "vector": 0,
            "empty_lexical": 1,
        }
        assert degenerate_normalizations(queries, by_name["CC_tmm_a0.30"])["lexical"] == 0
        assert degenerate_normalizations(queries, by_name["CCK_mm_a0.30"]) == {
            "lexical": 1,
            "vector": 0,
            "empty_lexical": 0,
        }

    def test_convex_sections_render(self):
        queries = [_synthetic_cached(seed) for seed in range(30)]
        report = _convex_run(queries)

        text = render_convex_sections(report, queries, watch_queries=["q003", "missing"], focus_query="q004")

        for heading in (
            "## Convex combination 요약",
            "## 정규화별 하위 가족 교차검증",
            "## α 곡선",
            "## 퇴화 정규화",
            "## 질의별 차이: pick",
            "## 기준선에서 hybrid와 vector가 갈린 질의",
            "## q004: CC 설정별 정답 논문 순위",
        ):
            assert heading in text
        assert "| missing |" not in text
        assert (
            render_convex_sections(
                run_sweep(queries, k=5, specs=_small_specs(), n_folds=3, repeats=1, n_resamples=10), queries
            )
            == ""
        )


class TestConvexScript:
    def test_convex_command_writes_the_convex_report(self, tmp_path, monkeypatch):
        script = _load_script("eval_fusion_sweep")
        monkeypatch.setattr(script, "build_specs", _small_specs)
        good, _, _ = _roundtrip(tmp_path, [_synthetic_cached(seed) for seed in range(15)])
        out_dir = tmp_path / "results"

        code = script.main(
            ["convex", "--cache", str(good), "--out-dir", str(out_dir), "--repeats", "2", "--resamples", "100"]
        )

        assert code == 0
        markdown = next(out_dir.glob("fusion_*.md")).read_text(encoding="utf-8")
        assert markdown.startswith("# Convex combination fusion sweep")
        assert "## Convex combination 요약" in markdown and "사전 등록: docs/worklog/phase-4/2026-09-29_01" in markdown
        assert "규칙 (4)" in markdown
        rows = next(out_dir.glob("fusion_*.csv")).read_text(encoding="utf-8").splitlines()
        assert len(rows) == 1 + len(_small_specs()) + 100
        header = rows[0].split(",")
        assert {"alpha", "normalization", "score_floors"} <= set(header)
