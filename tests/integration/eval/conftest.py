import importlib.util
import json
import math
import sys
from collections import Counter
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from arxplore.db.migrate import migrate
from arxplore.db.pool import Pool, create_pool, open_pool
from arxplore.db.repositories.embeddings import vector_literal
from arxplore.retrieval.service import RetrievalConfig, Retriever

REPO_ROOT = Path(__file__).resolve().parents[3]
FAKE_SERVER_PATH = REPO_ROOT / "tests" / "fakes" / "openai_server.py"
FAKE_SERVER_MODULE = "arxplore_eval_integration_fake_openai_server"
EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSIONS = 1536
DPO = "2401.00001"
SPARSE = "2401.00002"
RAG = "2401.00003"
APPENDIX_ONLY = "2401.00004"
POLICY_QUERY = "policy loss"
REFERENCE_QUERY = "Rafailov"
REFERENCE_TEXT = (
    "[1] Rafailov, R. Direct preference optimization. NeurIPS 2023.\n"
    "[2] Ouyang, L. Training language models with human feedback. NeurIPS 2022.\n"
    "[3] Christiano, P. Deep reinforcement learning from human preferences. NeurIPS 2017."
)


def basis_vector(*weights: tuple[int, float]) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for index, value in weights:
        vector[index] = value
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


CORPUS: tuple[dict[str, Any], ...] = (
    {
        "arxiv_id": DPO,
        "title": "Direct preference optimization revisited",
        "abstract": "We revisit direct preference optimization and its policy loss for alignment.",
        "chunks": [
            ("Abstract", "We revisit direct preference optimization and its policy loss for alignment.", "body"),
            ("3 Method", "The policy loss compares chosen and rejected responses with a logistic loss.", "body"),
            ("3 Method", "A reference policy regularizes the policy loss through a KL penalty term.", "body"),
            ("4 Experiments", "The policy loss improves win rates on dialogue benchmarks over PPO.", "body"),
            ("5 Notes", REFERENCE_TEXT, "body"),
        ],
    },
    {
        "arxiv_id": SPARSE,
        "title": "Sparse attention kernels for long inputs",
        "abstract": "Block sparse attention kernels speed up long context inference.",
        "chunks": [
            ("Abstract", "Block sparse attention kernels speed up long context inference.", "body"),
            ("2 Method", "Each block keeps the top scoring keys and drops the rest of the sequence.", "body"),
            ("5 Discussion", "A policy loss is not used; the kernels only change attention computation.", "body"),
        ],
    },
    {
        "arxiv_id": RAG,
        "title": "Retrieval augmented generation with chunk windows",
        "abstract": "Chunk windows give the generator neighbouring passages.",
        "chunks": [
            ("Abstract", "Chunk windows give the generator neighbouring passages.", "body"),
            ("3 Approach", "Neighbouring chunks are joined into one context window for each hit.", "body"),
        ],
    },
    {
        "arxiv_id": APPENDIX_ONLY,
        "title": "Supplementary tables",
        "abstract": "Extra tables for reproducibility.",
        "chunks": [("Appendix B", "Hyperparameters for every run are listed in the tables below.", "body")],
    },
)
CHUNK_VECTORS: dict[tuple[str, int], list[float]] = {
    (DPO, 0): basis_vector((1, 0.6), (0, 0.4)),
    (DPO, 1): basis_vector((0, 0.8), (1, 0.6)),
    (DPO, 2): basis_vector((0, 0.8), (2, 0.6)),
    (DPO, 3): basis_vector((0, 0.7), (3, 0.71)),
    (DPO, 4): basis_vector((4, 1.0)),
    (SPARSE, 0): basis_vector((5, 1.0)),
    (SPARSE, 1): basis_vector((5, 0.8), (6, 0.6)),
    (SPARSE, 2): basis_vector((0, 0.5), (5, 0.87)),
    (RAG, 0): basis_vector((7, 1.0)),
    (RAG, 1): basis_vector((7, 0.8), (8, 0.6)),
    (APPENDIX_ONLY, 0): basis_vector((0, 1.0)),
}
QUERY_VECTORS: dict[str, list[float]] = {
    POLICY_QUERY: basis_vector((0, 1.0)),
    "sparse attention kernels": basis_vector((5, 1.0)),
    "chunk windows": basis_vector((7, 1.0)),
}


@dataclass
class QueryVectors:
    vectors: Mapping[str, Sequence[float]] = field(default_factory=lambda: QUERY_VECTORS)
    default: Sequence[float] = field(default_factory=lambda: basis_vector((9, 1.0)))
    calls: list[list[str]] = field(default_factory=list)

    def is_available(self) -> bool:
        return True

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [list(self.vectors.get(text, self.default)) for text in texts]


async def seed_corpus(connection: psycopg.AsyncConnection[Any]) -> dict[tuple[str, int], int]:
    ids: dict[tuple[str, int], int] = {}
    async with connection.cursor(row_factory=dict_row) as cursor:
        for paper in CORPUS:
            await cursor.execute(
                "INSERT INTO papers (arxiv_id, title, abstract) VALUES (%s, %s, %s)",
                (paper["arxiv_id"], paper["title"], paper["abstract"]),
            )
            await cursor.execute(
                "INSERT INTO paper_fulltexts (arxiv_id, text, sections, source) VALUES (%s, %s, '[]'::jsonb, %s)",
                (paper["arxiv_id"], paper["abstract"], "layout_pdf"),
            )
            for index, (section, text, role) in enumerate(paper["chunks"]):
                await cursor.execute(
                    """
                    INSERT INTO paper_chunks (arxiv_id, chunk_index, chunk_text, section_title, token_count, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING id
                    """,
                    (paper["arxiv_id"], index, text, section, len(text.split()), json.dumps({"content_role": role})),
                )
                row = await cursor.fetchone()
                assert row is not None
                ids[(paper["arxiv_id"], index)] = int(row["id"])
        for key, vector in CHUNK_VECTORS.items():
            await cursor.execute(
                "INSERT INTO paper_embeddings (chunk_id, embedding, model_name) VALUES (%s, %s::vector, %s)",
                (ids[key], vector_literal(vector), EMBEDDING_MODEL),
            )
    return ids


@pytest.fixture
async def chunk_ids(empty_database: str) -> dict[tuple[str, int], int]:
    async with await psycopg.AsyncConnection.connect(empty_database, autocommit=True) as connection:
        await migrate(connection)
        return await seed_corpus(connection)


@pytest.fixture
async def eval_pool(empty_database: str, chunk_ids: dict[tuple[str, int], int]) -> AsyncIterator[Pool]:
    pool = create_pool(empty_database, role="test", max_size=4, timeout=5, connect_timeout=5)
    async with open_pool(pool):
        yield pool


@pytest.fixture
def query_vectors() -> QueryVectors:
    return QueryVectors()


@pytest.fixture
def retriever(eval_pool: Pool, query_vectors: QueryVectors) -> Retriever:
    return Retriever(eval_pool, query_vectors, RetrievalConfig(mode="hybrid", embedding_model=EMBEDDING_MODEL))


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
    vectors = fakes.EmbeddingVectors({fakes.embedding_key(text): vector for text, vector in QUERY_VECTORS.items()})
    with fakes.FakeOpenAIServer(embedding_handler=vectors) as server:
        yield server


RAGAS_FAIL_MARKER = "RAGAS-JUDGE-FAIL"
RAGAS_FLAKY_MARKER = "RAGAS-JUDGE-FLAKY"
INVALID_JUDGE_REPLY = "this is not the requested JSON"
SCHEMA_START = "json_schema:"
SCHEMA_END = "Make sure to return"
TRANSLATION_START = "Statements to translate:"
INPUT_START = "input: "
OUTPUT_START = "Output:"


def _system_text(request: Mapping[str, Any]) -> str:
    return next(str(message["content"]) for message in request["messages"] if message["role"] == "system")


def _user_text(request: Mapping[str, Any]) -> str:
    return str(request["messages"][-1]["content"])


def _prompt_input(user: str) -> dict[str, Any]:
    payload = json.loads(user[user.rindex(INPUT_START) + len(INPUT_START) : user.rindex(OUTPUT_START)])
    assert isinstance(payload, dict)
    return payload


@dataclass
class RagasJudge:
    titles: Counter[str] = field(default_factory=Counter)
    unknown: list[str] = field(default_factory=list)
    failed: int = 0
    flaked: int = 0

    def schema_title(self, request: Mapping[str, Any]) -> str:
        system = _system_text(request)
        schema = json.loads(system[system.index(SCHEMA_START) + len(SCHEMA_START) : system.index(SCHEMA_END)])
        return str(schema["title"])

    def reply(self, title: str, user: str) -> dict[str, Any] | None:
        if title == "_TranslatedStrings":
            return {"statements": json.loads(user[user.index(TRANSLATION_START) + len(TRANSLATION_START) :])}
        if title == "StatementGeneratorOutput":
            return {"statements": ["The answer makes a first claim.", "The answer makes a second claim."]}
        if title == "NLIStatementOutput":
            statements = _prompt_input(user)["statements"]
            return {
                "statements": [
                    {"statement": statement, "reason": "Scripted verdict.", "verdict": index % 2}
                    for index, statement in enumerate(statements)
                ]
            }
        if title == "AnswerRelevanceOutput":
            return {"question": "What does the answer describe?", "noncommittal": 0}
        if title == "ContextPrecisionOutput":
            return {"reason": "The context supports the reference.", "verdict": 1}
        if title == "ContextRecallOutput":
            return {
                "classifications": [
                    {"statement": "The reference makes one claim.", "reason": "It is in the context.", "attributed": 1}
                ]
            }
        return None

    def __call__(self, request: Mapping[str, Any]) -> str:
        title = self.schema_title(request)
        self.titles[title] += 1
        user = _user_text(request)
        if any(RAGAS_FAIL_MARKER in str(message["content"]) for message in request["messages"]):
            self.failed += 1
            return INVALID_JUDGE_REPLY
        if RAGAS_FLAKY_MARKER in user:
            self.flaked += 1
            return INVALID_JUDGE_REPLY
        payload = self.reply(title, user)
        if payload is None:
            self.unknown.append(title)
            return "{}"
        return json.dumps(payload, ensure_ascii=False)


@pytest.fixture
def ragas_judge() -> RagasJudge:
    return RagasJudge()
