import random
import statistics
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from arxplore_eval.fusion_sweep import (
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
    render_markdown,
    run_sweep,
    select_one_se,
    stratified_folds,
    write_csv,
)
from arxplore_eval.research_fusion import (
    LEGACY_RULES_FUSION,
    PRODUCT_FUSION,
    STRICT_LEXICAL_SCORE_FLOOR,
    HybridFusionConfig,
)


class TestSpecs:
    def test_grid_sizes_and_unique_names(self) -> None:
        specs = build_specs()
        families = Counter(spec.family for spec in specs)
        assert families == {"R0_vector": 1, "R0_lexical": 1, "F0": 5, "F1": 220, "F2": 720, "C": 1, "CF": 8, "REF": 2}
        assert len({spec.name for spec in specs}) == len(specs)

    def test_factorial_corners_are_current_and_plain_rrf(self) -> None:
        by_name = {spec.name: spec for spec in build_specs()}
        assert by_name["CF_m1q1b1"].config == by_name[CURRENT].config == LEGACY_RULES_FUSION
        assert by_name["CF_m0q0b0"].config == by_name[PLAIN_RRF_K60].config
        assert by_name[PLAIN_RRF_K60].config is not None and by_name[PLAIN_RRF_K60].config.drop_partial_lexical
        plain = by_name["REF_plainrrf_ablation"].config
        assert plain is not None and not plain.drop_partial_lexical

    def test_parameter_counts(self) -> None:
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
    def test_cluster_key_strips_only_a_trailing_language_suffix(self) -> None:
        assert cluster_key("ki-2609.26355-ko") == cluster_key("ki-2609.26355-en") == "ki-2609.26355"
        assert cluster_key("lang-ko-typo") == "lang-ko-typo"
        assert cluster_key("lang-ko") == "lang"
        assert cluster_key("cs-1892") == "cs-1892"

    def test_cluster_strata_use_pair_or_language(self) -> None:
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

    def test_clusters_of_keeps_first_appearance_order(self) -> None:
        assert clusters_of(["a", "b", "a", "c"]) == [[0, 2], [1], [3]]


class TestFolds:
    def _plan(self, *, repeats: int = 3, seed: int = 7) -> tuple[list[str], list[str], list[list[int]]]:
        groups, strata = cluster_strata(_query_layout())
        return groups, strata, fold_plan(groups, strata, n_folds=5, repeats=repeats, seed=seed)

    def test_pairs_are_never_split_across_folds(self) -> None:
        groups, _, plan = self._plan(repeats=10)
        for folds in plan:
            by_group: dict[str, set[int]] = {}
            for group, fold in zip(groups, folds, strict=True):
                by_group.setdefault(group, set()).add(fold)
            assert all(len(assigned) == 1 for assigned in by_group.values())

    def test_folds_are_balanced_and_each_stratum_is_spread(self) -> None:
        _, strata, plan = self._plan(repeats=10)
        for folds in plan:
            sizes = Counter(folds)
            assert len(sizes) == 5
            assert max(sizes.values()) - min(sizes.values()) <= 2
            for name in set(strata):
                counts = Counter(fold for fold, value in zip(folds, strata, strict=True) if value == name)
                per_fold = [counts.get(fold, 0) for fold in range(5)]
                assert max(per_fold) - min(per_fold) <= 2

    def test_plan_is_deterministic_and_repeats_differ(self) -> None:
        _, _, first = self._plan()
        assert first == self._plan()[2]
        assert first[0] != first[1]
        assert first != self._plan(seed=8)[2]

    def test_cluster_spanning_strata_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="spans"):
            stratified_folds(["a", "a"], ["x", "y"], 2, random.Random(0))

    def test_cross_validation_of_a_single_config_is_its_mean(self) -> None:
        rng = random.Random(3)
        _, _, plan = self._plan(repeats=4, seed=1)
        scores = {"only": [rng.random() for _ in range(111)]}

        result = cross_validate("x", [SweepSpec("only", "x", config=LEGACY_RULES_FUSION)], scores, plan)

        assert result.mean == pytest.approx(statistics.fmean(scores["only"]))
        assert result.se > 0
        assert result.choices == (("only", 20),)

    def test_tuning_picks_the_train_best_and_breaks_ties_by_simplicity(self) -> None:
        simple = SweepSpec("simple", "x", config=HybridFusionConfig(weighting="static"))
        complex_ = SweepSpec("complex", "x", config=LEGACY_RULES_FUSION)
        worse = SweepSpec("worse", "x", config=HybridFusionConfig(weighting="static", quality_weight=False))
        scores = {"simple": [0.5] * 111, "complex": [0.5] * 111, "worse": [0.4] * 111}
        _, _, plan = self._plan(repeats=2, seed=1)

        result = cross_validate("x", [complex_, worse, simple], scores, plan)

        assert result.choices == (("simple", 10),)
        assert result.mean == pytest.approx(0.5)


class TestBootstrap:
    def test_identical_systems_give_a_zero_interval(self) -> None:
        values = [random.Random(1).random() for _ in range(30)]
        assert paired_bootstrap(values, values, bootstrap_draws(30, 200, seed=1)) == (0.0, 0.0, 0.0)

    def test_constant_shift_is_recovered_exactly(self) -> None:
        base = [random.Random(2).random() for _ in range(25)]
        shifted = [value + 0.1 for value in base]
        assert paired_bootstrap(shifted, base, bootstrap_draws(25, 300, seed=2)) == pytest.approx((0.1, 0.1, 0.1))

    def test_interval_contains_the_mean_and_is_deterministic(self) -> None:
        rng = random.Random(4)
        a = [float(rng.random() < 0.7) for _ in range(80)]
        b = [float(rng.random() < 0.5) for _ in range(80)]

        delta, low, high = paired_bootstrap(a, b, bootstrap_draws(80, 2000, seed=11))

        assert low < delta < high
        assert (delta, low, high) == paired_bootstrap(a, b, bootstrap_draws(80, 2000, seed=11))
        assert 0.0 < high - low < 0.6

    def test_invalid_inputs_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            paired_bootstrap([1.0], [1.0, 0.0], [[0]])
        with pytest.raises(ValueError, match="partition"):
            paired_bootstrap([1.0, 0.0], [0.0, 0.0], [[0]], clusters=[[0]])
        with pytest.raises(ValueError, match="resample"):
            paired_bootstrap([1.0], [0.0], [])

    def test_cluster_resampling_uses_the_per_query_mean(self) -> None:
        a = [1.0, 1.0, 0.0]
        b = [0.0, 0.0, 0.0]
        clusters = [[0, 1], [2]]

        assert paired_bootstrap(a, b, [[0, 0]], clusters=clusters) == (pytest.approx(2 / 3), 1.0, 1.0)
        assert paired_bootstrap(a, b, [[0, 1]], clusters=clusters)[1] == pytest.approx(2 / 3)
        assert paired_bootstrap(a, b, [[1, 1]], clusters=clusters)[1:] == (0.0, 0.0)

    def test_cluster_bootstrap_is_wider_for_correlated_pairs(self) -> None:
        rng = random.Random(9)
        pair_deltas = [float(rng.random() < 0.6) for _ in range(40)]
        a = [value for value in pair_deltas for _ in range(2)]
        b = [0.0] * 80
        clusters = [[2 * index, 2 * index + 1] for index in range(40)]

        clustered = paired_bootstrap(a, b, bootstrap_draws(40, 3000, seed=5), clusters=clusters)
        naive = paired_bootstrap(a, b, bootstrap_draws(80, 3000, seed=5))

        assert clustered[0] == naive[0]
        assert clustered[2] - clustered[1] > naive[2] - naive[1]


class TestOneStandardError:
    def _specs(self) -> list[SweepSpec]:
        return [
            SweepSpec("C", "C", config=LEGACY_RULES_FUSION),
            SweepSpec("mid", "F1", config=HybridFusionConfig(weighting="static", static_weights=(0.5, 1.0))),
            SweepSpec("plain", "F0", config=HybridFusionConfig(weighting="static", quality_weight=False)),
        ]

    def test_picks_the_fewest_parameters_within_one_se(self) -> None:
        selection = select_one_se(self._specs(), {"C": 0.80, "mid": 0.79, "plain": 0.775}, lambda name: 0.03)
        assert selection.best == "C"
        assert selection.threshold == pytest.approx(0.77)
        assert selection.pick == "plain"
        assert selection.within == ("plain", "mid", "C")

    def test_keeps_the_best_when_nothing_simpler_is_close(self) -> None:
        selection = select_one_se(self._specs(), {"C": 0.80, "mid": 0.70, "plain": 0.60}, lambda name: 0.01)
        assert selection.pick == "C"
        assert selection.within == ("C",)


def _bootstrap(label: str, subset: str, low: float, high: float) -> BootstrapResult:
    return BootstrapResult(label, subset, "mrr@10", 50, 40, (low + high) / 2, low, high)


class TestDecision:
    SELECTION = Selection("best", 0.8, 0.02, 0.78, "F0_rrf_k40", 0.79, ("F0_rrf_k40",))

    def test_rule_one_replaces_when_interval_includes_zero(self) -> None:
        lines = pre_registered_decision(
            self.SELECTION,
            [
                _bootstrap("pick - C", "all", -0.02, 0.01),
                _bootstrap("pick - C", "manual", -0.05, 0.03),
                _bootstrap("pick - vector_only", "all", 0.01, 0.04),
            ],
            primary="mrr@10",
        )
        assert lines[0].startswith("규칙 (1)")
        assert "유의하게 낫다" in lines[-1]

    def test_rule_two_keeps_current_and_checks_it_against_vector(self) -> None:
        lines = pre_registered_decision(
            self.SELECTION,
            [
                _bootstrap("pick - C", "all", -0.05, -0.01),
                _bootstrap("pick - C", "manual", -0.05, 0.01),
                _bootstrap("C - vector_only", "all", -0.01, 0.02),
            ],
            primary="mrr@10",
        )
        assert lines[0].startswith("규칙 (2)")
        assert lines[-1].startswith("규칙 (3)") and CURRENT in lines[-1]

    def test_manual_regression_blocks_replacement(self) -> None:
        lines = pre_registered_decision(
            self.SELECTION,
            [_bootstrap("pick - C", "all", -0.01, 0.02), _bootstrap("pick - C", "manual", -0.09, -0.01)],
            primary="mrr@10",
        )
        assert lines[0].startswith("manual 부분집합 비회귀 조건 실패")

    def test_pick_equal_to_current_and_significant_win(self) -> None:
        current = Selection("best", 0.8, 0.02, 0.78, CURRENT, 0.8, (CURRENT,))
        assert pre_registered_decision(current, [], primary="mrr@10")[0].startswith("1-SE 선택이 가중 규칙")
        lines = pre_registered_decision(self.SELECTION, [_bootstrap("pick - C", "all", 0.01, 0.03)], primary="mrr@10")
        assert lines[0].startswith("pick이 C보다 유의하게 낫다")

    @pytest.mark.parametrize(
        ("low", "high", "expected"),
        [(0.01, 0.03, "재현했다"), (-0.03, -0.01, "반대다"), (-0.01, 0.02, "재현하지 못했다")],
    )
    def test_rule_four_reads_the_pick_against_standard_rrf(self, low: float, high: float, expected: str) -> None:
        selection = Selection("CC_mm_a0.30", 0.8, 0.02, 0.78, "CC_mm_a0.30", 0.8, ("CC_mm_a0.30",))
        comparisons = [
            _bootstrap("pick - C", "all", -0.01, 0.02),
            _bootstrap("pick - C", "manual", -0.02, 0.03),
            _bootstrap("pick - F0(k=60)", "all", low, high),
        ]
        lines = pre_registered_decision(selection, comparisons, primary="mrr@10", rrf_check=True)
        assert lines[0].startswith("규칙 (1)")
        assert lines[-1].startswith("규칙 (4)") and expected in lines[-1]
        assert not any(
            line.startswith("규칙 (4)") for line in pre_registered_decision(selection, comparisons, primary="mrr@10")
        )


def _small_specs() -> list[SweepSpec]:
    wanted = {VECTOR_ONLY, "lexical_only", CURRENT, PLAIN_RRF_K60, "F0_rrf_k10", "F1_w0.5_k60_q0_b0", "CF_m0q1b1"}
    return [spec for spec in build_specs() if spec.name in wanted or spec.family == "REF"]


class TestRunSweep:
    def test_end_to_end_is_deterministic(self, synthetic_cached, tmp_path) -> None:
        queries = [synthetic_cached(seed) for seed in range(30)]
        kwargs: dict[str, Any] = {
            "k": 5,
            "specs": _small_specs(),
            "seed": 3,
            "n_folds": 3,
            "repeats": 2,
            "n_resamples": 200,
        }

        first = run_sweep(queries, **kwargs)
        second = run_sweep(queries, **kwargs)

        assert first.selection == second.selection
        assert first.comparisons == second.comparisons
        assert first.cv == second.cv
        assert first.selection.pick in {spec.name for spec in kwargs["specs"] if spec.selectable}
        assert {row.subset for row in first.comparisons} == {"all", "manual"}
        assert "REF" not in first.cv

        markdown = render_markdown(first, title="t", meta={"커밋": "x"})
        assert "## 사전 등록 규칙 판정" in markdown and "## 1-SE 선택" in markdown
        write_csv(first, tmp_path / "sweep.csv")
        assert len((tmp_path / "sweep.csv").read_text(encoding="utf-8").splitlines()) == 1 + len(kwargs["specs"])

    def test_pairs_form_bootstrap_clusters(self, synthetic_cached) -> None:
        queries = [
            synthetic_cached(seed, query_id=f"p{seed // 2}-{'ko' if seed % 2 == 0 else 'en'}" if seed < 20 else None)
            for seed in range(30)
        ]
        report = run_sweep(queries, k=5, specs=_small_specs(), n_folds=3, repeats=2, n_resamples=100)

        overall = [row for row in report.comparisons if row.subset == "all"]
        assert {(row.n, row.n_clusters) for row in overall} == {(30, 20)}
        assert report.subset_sizes["all"] == (30, 20)

    def test_current_config_matches_the_recorded_live_metrics(self, synthetic_cached) -> None:
        queries = [synthetic_cached(seed, config=LEGACY_RULES_FUSION) for seed in range(30)]
        report = run_sweep(queries, k=5, specs=_small_specs(), n_folds=3, repeats=1, n_resamples=50)
        live_chunks = [
            [hit["chunk_id"] for hit in cached.live["hybrid"] if hit["chunk_id"] is not None] for cached in queries
        ]
        assert [row.retrieved_chunk_ids for row in report.results[CURRENT]] == live_chunks

    def test_required_specs_and_unique_names(self, synthetic_cached) -> None:
        queries = [synthetic_cached(seed) for seed in range(6)]
        with pytest.raises(ValueError, match="must include"):
            run_sweep(queries, k=5, specs=[spec for spec in _small_specs() if spec.name != CURRENT])
        with pytest.raises(ValueError, match="unique"):
            run_sweep(queries, k=5, specs=[*_small_specs(), _small_specs()[0]])


class TestConvexSpecs:
    def test_families_grid_and_names(self) -> None:
        specs = build_convex_specs()
        assert Counter(spec.family for spec in specs) == {CONVEX: 60, CONVEX_KEEP: 40}
        names = {spec.name for spec in specs}
        assert len(names) == len(specs)
        assert not names & {spec.name for spec in build_specs()}
        assert CONVEX_ALPHAS[0] == 0.0 and CONVEX_ALPHAS[-1] == 0.95 and len(CONVEX_ALPHAS) == 20

    def test_primary_family_drops_partial_rows_and_is_the_only_selectable_one(self) -> None:
        for spec in build_convex_specs():
            assert spec.config is not None and spec.config.weighting == "convex"
            assert spec.config.drop_partial_lexical == (spec.family == CONVEX)
            assert spec.selectable == (spec.family == CONVEX)
            assert spec.n_params == 1

    def test_normalization_floors(self) -> None:
        by_name = {spec.name: spec.config for spec in build_convex_specs()}
        assert by_name["CC_mm_a0.20"] is not None and by_name["CC_mm_a0.20"].score_normalization == "minmax"
        assert by_name["CC_tmm_a0.20"] is not None and by_name["CC_tmm_a0.20"].score_floors == (0.0, -1.0)
        assert by_name["CC_tmms_a0.20"] is not None
        assert by_name["CC_tmms_a0.20"].score_floors == (STRICT_LEXICAL_SCORE_FLOOR, -1.0)
        assert "CCK_tmms_a0.20" not in by_name

    def test_groups_split_by_normalization(self) -> None:
        groups = convex_groups(build_convex_specs())
        assert list(groups) == ["CC_mm", "CC_tmm", "CC_tmms", "CCK_mm", "CCK_tmm"]
        assert {len(members) for members in groups.values()} == {20}

    def test_product_fusion_is_the_pre_registered_pick(self) -> None:
        by_name = {spec.name: spec.config for spec in build_convex_specs()}
        assert by_name["CC_mm_a0.35"] == PRODUCT_FUSION


def _convex_run(queries: list[Any]) -> Any:
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
    def test_pick_comes_from_the_convex_family(self, synthetic_cached) -> None:
        report = _convex_run([synthetic_cached(seed) for seed in range(30)])

        assert report.selection.pick.startswith("CC_")
        assert set(report.selection.within) <= {spec.name for spec in build_convex_specs() if spec.family == CONVEX}
        assert {"CC", "CCK", "CC_mm", "CC_tmm", "CC_tmms", "CCK_mm", "CCK_tmm"} <= set(report.cv)
        assert report.selectable_families == (CONVEX,)
        assert report.decision[-1].startswith("규칙 (4)")

    def test_cv_group_names_must_not_shadow_families(self, synthetic_cached) -> None:
        queries = [synthetic_cached(seed) for seed in range(12)]
        with pytest.raises(ValueError):
            run_sweep(
                queries, k=5, specs=_small_specs(), n_folds=3, repeats=1, n_resamples=10, cv_groups={"C": [CURRENT]}
            )

    def test_query_differences_and_ranks_agree_with_the_metrics(self, synthetic_cached) -> None:
        report = _convex_run([synthetic_cached(seed) for seed in range(30)])
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

    def test_degenerate_counts(self, synthetic_cached) -> None:
        vector = (
            {"chunk_id": 3, "arxiv_id": "a", "score": 0.9, "score_breakdown": {}},
            {"chunk_id": 4, "arxiv_id": "b", "score": 0.5, "score_breakdown": {}},
        )
        single = {"chunk_id": 1, "arxiv_id": "c", "score": 1.2, "score_breakdown": {"strict_match": True}}
        partial = {"chunk_id": 2, "arxiv_id": "d", "score": 0.4, "score_breakdown": {"strict_match": False}}
        base = synthetic_cached(0)
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

    def test_convex_sections_render(self, synthetic_cached) -> None:
        queries = [synthetic_cached(seed) for seed in range(30)]
        report = _convex_run(queries)

        text = render_convex_sections(report, queries, watch_queries=["q003", "missing"], focus_query="q004")

        for heading in (
            "## Convex combination 요약",
            "## 정규화별 하위 가족 교차검증",
            "## alpha 곡선",
            "## 퇴화 정규화",
            "## 질의별 차이: pick",
            "## 기준선에서 hybrid와 vector가 갈린 질의",
            "## q004: CC 설정별 정답 논문 순위",
        ):
            assert heading in text
        assert "| missing |" not in text
        plain = run_sweep(queries, k=5, specs=_small_specs(), n_folds=3, repeats=1, n_resamples=10)
        assert render_convex_sections(plain, queries) == ""
