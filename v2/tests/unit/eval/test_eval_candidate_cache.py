import gzip
import json
from dataclasses import asdict, replace
from typing import Any

import pytest

from arxplore_eval.candidate_cache import (
    CacheError,
    cache_digest,
    compact_candidate,
    compare_live,
    gate,
    product_check,
    read_cache,
    recorded_fusion_config,
    replay,
    replay_product,
    restore_candidate,
    write_cache,
)
from arxplore_eval.research_fusion import LEGACY_RULES_FUSION, PRODUCT_FUSION, HybridFusionConfig

COMMITTED_CACHE = "cache/candidates_20260928-040013.jsonl.gz"


def _roundtrip(tmp_path, queries, *, k: int = 5, name: str = "cache.jsonl.gz", header: dict[str, Any] | None = None):
    path = tmp_path / name
    write_cache(path, {"k": k, "git_revision": "test", **(header or {})}, queries)
    return path, *read_cache(path)


class TestCandidateRecords:
    def test_compact_keeps_only_fusion_fields(self) -> None:
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


class TestCacheFile:
    def test_roundtrip_preserves_replay_and_is_byte_stable(self, tmp_path, synthetic_cached) -> None:
        queries = [synthetic_cached(seed) for seed in range(8)]
        path, header, loaded = _roundtrip(tmp_path, queries)
        second, _, _ = _roundtrip(tmp_path, queries, name="again.jsonl.gz")

        assert header["k"] == 5 and header["format"] == 1
        assert [cached.query for cached in loaded] == [cached.query for cached in queries]
        for original, restored in zip(queries, loaded, strict=True):
            assert [hit["chunk_id"] for hit in replay(original, k=5)] == [
                hit["chunk_id"] for hit in replay(restored, k=5)
            ]
        assert path.read_bytes() == second.read_bytes()
        assert cache_digest(path) == cache_digest(second)

    def test_rejects_unknown_format_bad_headers_and_duplicates(self, tmp_path, synthetic_cached) -> None:
        path = tmp_path / "bad.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": "header", "format": 99}) + "\n")
        with pytest.raises(CacheError, match="형식"):
            read_cache(path)

        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": "query"}) + "\n")
        with pytest.raises(CacheError, match="머리말"):
            read_cache(path)

        cached = synthetic_cached(0)
        duplicate = tmp_path / "duplicate.jsonl.gz"
        write_cache(duplicate, {"k": 5}, [cached, cached])
        with pytest.raises(CacheError, match="중복"):
            read_cache(duplicate)


class TestGate:
    def test_gate_passes_on_a_faithful_cache_and_reports_tampering(self, tmp_path, synthetic_cached) -> None:
        _, _, loaded = _roundtrip(tmp_path, [synthetic_cached(seed) for seed in range(12)])
        assert gate(loaded, k=5) == []

        tampered = replace(loaded[3], live={"hybrid": tuple(reversed(loaded[3].live["hybrid"]))})
        mismatches = gate([*loaded[:3], tampered, *loaded[4:]], k=5)

        assert [item.query_id for item in mismatches] == [tampered.id]
        assert mismatches[0].live != mismatches[0].replayed

    def test_gate_replays_the_fusion_config_recorded_in_the_header(self, tmp_path, synthetic_cached) -> None:
        legacy = [synthetic_cached(seed, config=LEGACY_RULES_FUSION) for seed in range(15)]
        _, header, loaded = _roundtrip(tmp_path, legacy, header={"fusion_default": asdict(LEGACY_RULES_FUSION)})

        assert recorded_fusion_config(header) == LEGACY_RULES_FUSION
        assert gate(loaded, k=5, config=recorded_fusion_config(header)) == []
        assert gate(loaded, k=5) != []

    def test_recorded_config_defaults_and_rejects_unknown_fields(self) -> None:
        assert recorded_fusion_config({"k": 5}) == PRODUCT_FUSION
        old_header = {
            name: value
            for name, value in asdict(LEGACY_RULES_FUSION).items()
            if not name.startswith(("convex", "score"))
        }
        assert recorded_fusion_config({"fusion_default": old_header}) == LEGACY_RULES_FUSION
        with pytest.raises(CacheError):
            recorded_fusion_config({"fusion_default": {"weighting": "rules", "mystery": 1}})

    def test_gate_fails_when_live_is_missing(self, synthetic_cached) -> None:
        cached = replace(synthetic_cached(0), live={})
        assert [item.query_id for item in gate([cached], k=5)] == [cached.id]

    def test_replay_changes_with_config_and_channel(self, synthetic_cached) -> None:
        cached = next(
            item
            for item in (synthetic_cached(seed) for seed in range(40))
            if len(item.lexical) > 5 and len(item.vector) > 5
        )
        vector_heavy = HybridFusionConfig(weighting="static", static_weights=(0.0, 1.0), quality_weight=False)
        lexical_heavy = HybridFusionConfig(weighting="static", static_weights=(1.0, 0.0), quality_weight=False)
        assert replay(cached, k=5, config=vector_heavy) != replay(cached, k=5, config=lexical_heavy)
        assert [hit["chunk_id"] for hit in replay(cached, k=3, channel="vector")] == [
            hit["chunk_id"] for hit in cached.vector[:3]
        ]
        with pytest.raises(ValueError):
            replay(cached, k=3, channel="bm25")

    def test_product_replay_matches_the_research_product_config(self, synthetic_cached) -> None:
        queries = [
            item
            for item in (synthetic_cached(seed) for seed in range(60))
            if all(candidate["chunk_id"] is not None for candidate in item.lexical)
        ]
        assert len(queries) > 40
        assert product_check(queries, k=5) == []
        assert product_check(queries[:10], k=5, config=LEGACY_RULES_FUSION) != []
        assert [hit.chunk_id for hit in replay_product(queries[0], k=3)] == [
            hit["chunk_id"] for hit in replay(queries[0], k=3)
        ]


class TestCommittedCache:
    @pytest.fixture
    def committed(self, eval_dir):
        path = eval_dir / COMMITTED_CACHE
        if not path.is_file():
            pytest.skip(f"committed candidate cache missing: {path}")
        return read_cache(path)

    def test_gate_on_the_recorded_legacy_config(self, committed) -> None:
        header, queries = committed
        config = recorded_fusion_config(header)

        assert len(queries) == 111 and header["k"] == 10
        assert config.weighting == "rules"
        assert gate(queries, k=10, config=config) == []

    def test_product_fusion_matches_the_research_product_config_on_every_query(self, committed) -> None:
        _, queries = committed
        assert product_check(queries, k=10) == []

    def test_single_channel_replays_differ_from_live_only_by_candidate_counts(self, committed) -> None:
        _, queries = committed
        assert len(compare_live(queries, k=10, channel="vector")) < len(queries)
        assert len(compare_live(queries, k=10, channel="lexical")) < len(queries)
