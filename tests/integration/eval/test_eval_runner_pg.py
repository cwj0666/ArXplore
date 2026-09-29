from collections import Counter

import pytest

from arxplore.domain.retrieval import RetrievalContext
from arxplore.retrieval.diversity import apply_paper_diversity
from arxplore.retrieval.service import Retriever
from arxplore_eval.candidate_cache import capture_query, compare_live, gate, product_check
from arxplore_eval.corpus import (
    PreconditionError,
    corpus_summary,
    fetch_chunk_texts,
    generation_preflight,
    retrieval_preflight,
)
from arxplore_eval.dataset import EvalQuery
from arxplore_eval.embedders import MemoEmbedder
from arxplore_eval.latency import (
    CONTEXT_STAGE,
    EMBEDDING_STAGE,
    LEXICAL_STAGE,
    VECTOR_STAGE,
    BenchRetriever,
    StageRecorder,
    build_paths,
)
from arxplore_eval.research_fusion import LEGACY_RULES_FUSION, STANDARD_RRF_FUSION, fuse_hits
from arxplore_eval.runner import METHODS, aggregate, run_evaluation

pytestmark = pytest.mark.integration

APPENDIX_ONLY = "2401.00004"
DPO = "2401.00001"
POLICY_QUERY = "policy loss"
REFERENCE_QUERY = "Rafailov"


def _ids(contexts: list[RetrievalContext]) -> list[int]:
    return [context.hit.chunk_id for context in contexts]


def _papers(contexts: list[RetrievalContext]) -> Counter[str]:
    return Counter(context.hit.arxiv_id for context in contexts)


async def _search(retriever: Retriever, method: str, query: str, k: int) -> list[RetrievalContext]:
    return await METHODS[method].search(retriever, query, k, 1)


class TestMethodsOnPostgres:
    async def test_base_methods_are_the_product_context_searches(self, retriever: Retriever) -> None:
        assert await _search(retriever, "lexical", POLICY_QUERY, 3) == await retriever.search_lexical_contexts(
            POLICY_QUERY, limit=3
        )
        assert await _search(retriever, "vector", POLICY_QUERY, 3) == await retriever.search_vector_contexts(
            POLICY_QUERY, limit=3
        )
        assert await _search(retriever, "hybrid", POLICY_QUERY, 3) == await retriever.search_hybrid_contexts(
            POLICY_QUERY, limit=3
        )

    async def test_nodiv_lets_one_paper_fill_the_top(self, retriever: Retriever) -> None:
        lexical = await _search(retriever, "lexical", POLICY_QUERY, 3)
        lexical_nodiv = await _search(retriever, "lexical_nodiv", POLICY_QUERY, 3)
        assert _papers(lexical)[DPO] == 2
        assert _papers(lexical_nodiv)[DPO] == 3
        assert _ids(lexical)[:2] == _ids(lexical_nodiv)[:2]

        vector = await _search(retriever, "vector", POLICY_QUERY, 4)
        vector_nodiv = await _search(retriever, "vector_nodiv", POLICY_QUERY, 4)
        assert _papers(vector)[DPO] == 2
        assert _papers(vector_nodiv)[DPO] == 3

        hybrid = await _search(retriever, "hybrid", POLICY_QUERY, 4)
        hybrid_nodiv = await _search(retriever, "hybrid_nodiv", POLICY_QUERY, 4)
        assert _papers(hybrid)[DPO] == 2
        assert _papers(hybrid_nodiv)[DPO] == 3
        assert _ids(hybrid)[:3] == _ids(hybrid_nodiv)[:3]

    async def test_nofilter_keeps_reference_like_chunks(self, retriever: Retriever) -> None:
        assert await _search(retriever, "lexical", REFERENCE_QUERY, 3) == []
        kept = await _search(retriever, "lexical_nofilter", REFERENCE_QUERY, 3)
        assert [context.hit.section_title for context in kept] == ["5 Notes"]

    async def test_norerank_keeps_the_sql_order(self, retriever: Retriever) -> None:
        reranked = await _search(retriever, "vector", POLICY_QUERY, 3)
        sql_order = await _search(retriever, "vector_norerank", POLICY_QUERY, 3)
        assert sql_order[0].hit.arxiv_id == APPENDIX_ONLY
        assert reranked[0].hit.arxiv_id == DPO
        assert set(_ids(sql_order)) == set(_ids(reranked))

    async def test_research_fusions_run_on_the_product_inputs(self, retriever: Retriever) -> None:
        inputs = await retriever.hybrid_inputs(POLICY_QUERY, limit=3)
        for method, config in (("hybrid_rules", LEGACY_RULES_FUSION), ("hybrid_plainrrf", STANDARD_RRF_FUSION)):
            contexts = await _search(retriever, method, POLICY_QUERY, 3)
            assert _ids(contexts) == [hit.chunk_id for hit in fuse_hits(inputs, config, limit=3)]
            assert all(context.text for context in contexts)

    async def test_every_method_scores_without_errors(self, retriever: Retriever) -> None:
        queries = [
            EvalQuery(id="dpo-en", query=POLICY_QUERY, lang="en", relevant_arxiv_ids=(DPO,), source="manual"),
            EvalQuery(
                id="sparse-en",
                query="sparse attention kernels",
                lang="en",
                relevant_arxiv_ids=("2401.00002",),
                source="manual",
            ),
        ]

        results = await run_evaluation(retriever, queries, list(METHODS), k=3)

        assert len(results) == 2 * len(METHODS)
        assert all(result.ok for result in results)
        overall = {row.method: row for row in aggregate(results) if row.subset == "all"}
        assert overall["lexical"].paper["hit@1"] == 1.0
        assert overall["vector"].paper["mrr@3"] == 1.0


class TestCandidateCapture:
    async def test_captured_inputs_replay_the_live_product_hybrid(self, eval_pool, query_vectors) -> None:
        retriever = Retriever(eval_pool, MemoEmbedder(query_vectors))
        query = EvalQuery(
            id="dpo-en", query=f"  {POLICY_QUERY} ", lang="en", relevant_arxiv_ids=(DPO,), source="manual"
        )

        cached = await capture_query(retriever, query, k=3)

        assert cached.normalized_query == POLICY_QUERY
        assert gate([cached], k=3) == []
        assert product_check([cached], k=3) == []
        assert compare_live([cached], k=3, channel="vector") == []
        assert all("strict_match" in candidate["score_breakdown"] for candidate in cached.lexical)
        assert query_vectors.calls == [[POLICY_QUERY]]


class TestBenchRetriever:
    async def test_sequential_and_parallel_paths_return_the_same_contexts(self, eval_pool, query_vectors) -> None:
        recorder = StageRecorder()
        sequential = BenchRetriever(eval_pool, query_vectors, sequential=True, recorder=recorder)
        parallel = BenchRetriever(eval_pool, query_vectors, sequential=False)
        paired, controls = build_paths(k=3, adjacency_window=1, product_limit=5)

        for fn in (*paired.values(), *controls.values()):
            assert await fn(sequential, POLICY_QUERY) == await fn(parallel, POLICY_QUERY)

        recorder.reset()
        await paired["hybrid_eval"](sequential, POLICY_QUERY)
        stages = recorder.snapshot_ms()
        assert {EMBEDDING_STAGE, LEXICAL_STAGE, VECTOR_STAGE, CONTEXT_STAGE} <= set(stages)
        assert stages[VECTOR_STAGE] >= stages[EMBEDDING_STAGE]

    async def test_sequential_inputs_match_the_product_inputs(self, eval_pool, query_vectors) -> None:
        sequential = BenchRetriever(eval_pool, query_vectors, sequential=True)
        product = Retriever(eval_pool, query_vectors)
        assert await sequential.hybrid_inputs(POLICY_QUERY, limit=3) == await product.hybrid_inputs(
            POLICY_QUERY, limit=3
        )
        empty = await sequential.hybrid_inputs("\x00 ", limit=3)
        assert (empty.query, empty.lexical, empty.vector) == ("", (), ())


class TestCorpus:
    async def test_summary_preflight_and_chunk_texts(self, eval_pool, chunk_ids) -> None:
        async with eval_pool.connection() as connection:
            summary = await corpus_summary(connection)
            assert (summary.papers, summary.papers_with_chunks, summary.chunks, summary.embeddings) == (4, 4, 11, 11)
            assert summary.fulltext_sources == {"layout_pdf": 4}

            good = EvalQuery(
                id="g",
                query="q",
                lang="en",
                relevant_arxiv_ids=(DPO,),
                relevant_chunk_ids=(chunk_ids[(DPO, 1)],),
                source="manual",
            )
            assert await retrieval_preflight(connection, [good], needs_embeddings=True) == summary

            missing = EvalQuery(
                id="m",
                query="q",
                lang="en",
                relevant_arxiv_ids=("2499.99999",),
                relevant_chunk_ids=(10**9,),
                source="manual",
            )
            with pytest.raises(PreconditionError) as excinfo:
                await retrieval_preflight(connection, [missing], needs_embeddings=False)
            assert "2499.99999" in str(excinfo.value) and "chunk_id" in str(excinfo.value)

            chat = EvalQuery(id="c", query="q", lang="en", relevant_arxiv_ids=("2499.99999",), source="manual")
            with pytest.raises(PreconditionError, match="상세 챗 대상"):
                await generation_preflight(connection, [chat], modes=["paper_chat"])
            assert await generation_preflight(connection, [chat], modes=["agent"]) == summary

            texts = await fetch_chunk_texts(connection, [chunk_ids[(DPO, 1)]])
            assert texts == {
                chunk_ids[(DPO, 1)]: "The policy loss compares chosen and rejected responses with a logistic loss."
            }


async def test_product_diversity_is_used_by_the_default_hybrid(retriever: Retriever) -> None:
    inputs = await retriever.hybrid_inputs(POLICY_QUERY, limit=3)
    hybrid = await retriever.search_hybrid(POLICY_QUERY, limit=3)
    assert len(hybrid) == 3
    assert apply_paper_diversity(hybrid, limit=3) == hybrid
    assert {hit.chunk_id for hit in hybrid} <= {hit.chunk_id for hit in (*inputs.lexical, *inputs.vector)}
