import importlib.util
import random
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from arxplore.domain.retrieval import LexicalScores, RetrievalContext, RetrievalHit, RetrievalMethod, VectorScores
from arxplore.retrieval.lexical import STRICT_MATCH_BONUS
from arxplore_eval.candidate_cache import CachedQuery
from arxplore_eval.dataset import EvalQuery
from arxplore_eval.research_fusion import (
    PRODUCT_FUSION,
    HybridFusionConfig,
    apply_paper_diversity,
    fuse_hybrid_candidates,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = REPO_ROOT / "eval"
FAKE_SERVER_PATH = REPO_ROOT / "tests" / "fakes" / "openai_server.py"
FAKE_SERVER_MODULE = "arxplore_eval_tests_fake_openai_server"

GOLDEN_QUERIES = (
    "policy loss",
    "direct preference optimization policy loss reward",
    "강화학습 보상 모델의 한계는 무엇인가",
    "LoRA",
    "What limitations does the paper discuss about sparse attention kernels",
)
GOLDEN_SCORE_POINTS = (0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 0.95)


def candidate(chunk_id: int | None, score: float, *, arxiv_id: str | None = None, snippet: str = "") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "arxiv_id": arxiv_id or f"2401.{(chunk_id or 0):05d}",
        "chunk_index": 0,
        "chunk_text": "body",
        "section_title": "Method",
        "content_role": "body",
        "score": score,
        "snippet": snippet,
    }


def lexical_candidate(chunk_id: int, score: float, *, strict: bool, coverage: float = 1.0) -> dict[str, Any]:
    return {**candidate(chunk_id, score), "score_breakdown": {"strict_match": strict, "coverage": coverage}}


def synthetic_fusion_case(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    pool = list(range(1, 81))
    rng.shuffle(pool)
    papers = [f"2409.{number:05d}" for number in range(rng.randint(3, 12))]

    def row(chunk_id: int, score: float, breakdown: dict[str, Any], snippet: str) -> dict[str, Any]:
        return {
            **candidate(chunk_id, score, arxiv_id=papers[chunk_id % len(papers)], snippet=snippet),
            "score_breakdown": breakdown,
        }

    def lexical_row(chunk_id: int) -> dict[str, Any]:
        confidence = rng.choice(GOLDEN_SCORE_POINTS) if rng.random() < 0.5 else round(rng.random(), 4)
        tier = rng.random()
        if tier < 0.6:
            return row(chunk_id, STRICT_MATCH_BONUS + confidence, {"strict_match": True}, f"lexical {chunk_id}")
        if tier < 0.9:
            return row(chunk_id, confidence, {"strict_match": False, "coverage": 0.5}, f"lexical {chunk_id}")
        return row(chunk_id, confidence, {}, f"lexical {chunk_id}")

    def vector_row(chunk_id: int) -> dict[str, Any]:
        snippet = f"vector {chunk_id}" if rng.random() < 0.8 else ""
        return row(chunk_id, round(rng.uniform(0.2, 0.9), 4), {"rerank_adjustment": 0.0}, snippet)

    lexical_ids = pool[: rng.randint(0, 30)]
    shared = rng.randint(0, len(lexical_ids))
    vector_ids = rng.sample(lexical_ids, shared) + pool[30 : 30 + rng.randint(0, 30 - shared)]
    rng.shuffle(vector_ids)
    lexical = sorted((lexical_row(chunk_id) for chunk_id in lexical_ids), key=lambda item: item["score"], reverse=True)
    vector = sorted((vector_row(chunk_id) for chunk_id in vector_ids), key=lambda item: item["score"], reverse=True)
    if lexical and rng.random() < 0.1:
        lexical[-1]["chunk_id"] = None
    return {
        "query": rng.choice(GOLDEN_QUERIES),
        "lexical": lexical,
        "vector": vector,
        "limit": rng.choice((1, 3, 5, 10)),
    }


def build_hit(
    chunk_id: int,
    *,
    method: RetrievalMethod = "lexical",
    score: float = 0.5,
    arxiv_id: str | None = None,
    section_title: str = "Method",
    content_role: str = "body",
    strict: bool | None = True,
    chunk_index: int = 0,
    text: str = "body",
) -> RetrievalHit:
    lexical = None
    vector = None
    if method == "lexical" and strict is not None:
        lexical = LexicalScores(
            fts_score=score,
            ilike_bonus=0.0,
            content_role_adjustment=0.0,
            section_boost=0.0,
            structural_adjustment=0.0,
            coverage=1.0,
            strict_match=strict,
        )
    elif method == "vector":
        vector = VectorScores(raw_similarity_score=score, content_role_adjustment=0.0, section_boost=0.0)
    return RetrievalHit(
        chunk_id=chunk_id,
        arxiv_id=arxiv_id if arxiv_id is not None else f"2401.{chunk_id:05d}",
        chunk_index=chunk_index,
        paper_title=f"Paper {chunk_id}",
        paper_abstract="",
        chunk_text=text,
        section_title=section_title,
        content_role=content_role,
        method=method,
        score=score,
        snippet=f"{method} {chunk_id}",
        lexical=lexical,
        vector=vector,
    )


def build_context(hit: RetrievalHit, text: str | None = None) -> RetrievalContext:
    return RetrievalContext(hit=hit, chunks=(), text=text if text is not None else f"context {hit.chunk_id}")


@pytest.fixture
def make_candidate() -> Callable[..., dict[str, Any]]:
    return candidate


@pytest.fixture
def make_lexical_candidate() -> Callable[..., dict[str, Any]]:
    return lexical_candidate


@pytest.fixture
def synthetic_case() -> Callable[[int], dict[str, Any]]:
    return synthetic_fusion_case


@pytest.fixture
def make_hit() -> Callable[..., RetrievalHit]:
    return build_hit


@pytest.fixture
def make_context() -> Callable[..., RetrievalContext]:
    return build_context


@pytest.fixture
def eval_dir() -> Path:
    return EVAL_DIR


def load_fake_server_module() -> ModuleType:
    if FAKE_SERVER_MODULE in sys.modules:
        return sys.modules[FAKE_SERVER_MODULE]
    if not FAKE_SERVER_PATH.is_file():
        pytest.skip(f"fake OpenAI server not found at {FAKE_SERVER_PATH}")
    spec = importlib.util.spec_from_file_location(FAKE_SERVER_MODULE, FAKE_SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[FAKE_SERVER_MODULE] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def fakes() -> ModuleType:
    return load_fake_server_module()


@pytest.fixture
def fake_openai(fakes: ModuleType) -> Iterator[Any]:
    with fakes.FakeOpenAIServer() as server:
        yield server


CACHE_SOURCES = ("known_item", "manual", "llm_synth")


def build_synthetic_cached(
    seed: int, *, k: int = 5, query_id: str | None = None, config: HybridFusionConfig = PRODUCT_FUSION
) -> CachedQuery:
    case = synthetic_fusion_case(seed)
    rng = random.Random(seed)
    papers = sorted({row["arxiv_id"] for row in [*case["lexical"], *case["vector"]]}) or ["2409.00000"]
    query = EvalQuery(
        id=query_id or f"q{seed:03d}",
        query=case["query"],
        lang=rng.choice(("ko", "en")),
        relevant_arxiv_ids=(rng.choice(papers),),
        source=CACHE_SOURCES[seed % 3],
    )
    fused = fuse_hybrid_candidates(case["query"], case["lexical"], case["vector"], config)
    live = apply_paper_diversity(fused, limit=k)
    return CachedQuery(
        query=query,
        normalized_query=case["query"],
        lexical=tuple(case["lexical"]),
        vector=tuple(case["vector"]),
        live={"hybrid": tuple(live)},
    )


@pytest.fixture
def synthetic_cached() -> Callable[..., CachedQuery]:
    return build_synthetic_cached
